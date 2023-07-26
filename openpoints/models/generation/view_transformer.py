from inspect import isfunction
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from ..build import MODELS
from ...dataset.data_util import rotate_point_clouds_batch


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerBlockSA(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        x = self.attn1(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


@MODELS.register_module()
class ViewTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 depth,
                 channels_per_head,
                 drop_rate,
                 obj_size,
                 img_size,
                 img_ds_ratio,
                 query_component={'point_grid': 3, 'direction_norm': 3, 'query_pos': 2},
                 **kwargs
                 ):
        super().__init__()
        # basic args
        self.obj_size = obj_size
        self.img_size = img_size
        self.ds_ratio = img_ds_ratio
        self.query_mesh, self.query_pos = self.make_meshgrid()
        self.pad_memory = nn.Parameter(torch.zeros(feat_channels), requires_grad=True)

        self.query_component = query_component
        self.query_channels = 0
        for key in list(query_component.keys()):
            self.query_channels += query_component[key]

        self.feat_mlp = Mlp(in_features=in_channels + 3, out_features=feat_channels)
        self.query_mlp = Mlp(in_features=self.query_channels, out_features=feat_channels)

        # build transformer 
        self.transformer = nn.ModuleList([
            BasicTransformerBlock(
                dim=feat_channels,
                n_heads=(channels_per_head),
                d_head=channels_per_head,
                dropout=drop_rate,
                checkpoint=False,
            )
            for _ in range(depth)
        ])

    def make_meshgrid(self):
        img_feat_size = self.img_size // self.ds_ratio
        hs, ws = torch.meshgrid(torch.linspace(0, img_feat_size, img_feat_size+1), torch.linspace(0, img_feat_size, img_feat_size+1))
        query_meshgrid = torch.stack([hs[:-1, :-1], ws[:-1, :-1]], dim=-1).reshape(-1, 2)
        query_meshgrid_center = query_meshgrid + torch.tensor([[1 / 2, 1 / 2]])
        query_meshgrid_center = query_meshgrid_center * self.ds_ratio

        hs_pos, ws_pos = torch.meshgrid(torch.linspace(0, 1, img_feat_size), torch.linspace(0, 1, img_feat_size))
        query_pos_meshgrid = torch.stack([hs_pos, ws_pos], dim=-1).reshape(-1, 2)
        return query_meshgrid_center, query_pos_meshgrid
    
    def cal_scale_bias(self, pos_pc, view):
        B, V, _, _ = view.shape
        N = pos_pc.shape[1]
        pos_rotate = rotate_point_clouds_batch(pos_pc.unsqueeze(1).expand(-1, V, -1, -1).reshape(B*V, N, 3),
                                               view.reshape(B*V, 3, 3))

        # calculate range
        pc_min = pos_rotate.min(dim=1)[0][:, :2]
        pc_range = pos_rotate.max(dim=1)[0] - pos_rotate.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.obj_size - 3)  # B,
        idx_xy = torch.floor((pos_rotate[:, :, :2] - pc_min.unsqueeze(dim=1)) / grid_size.unsqueeze(dim=1).unsqueeze(dim=2))  # B N 2
        idx_xy_center = torch.floor((idx_xy.max(dim=1)[0] + idx_xy.min(dim=1)[0]) / 2).int()
        offset_x = self.obj_size / 2 - idx_xy_center[:, 0:1] - 1
        offset_y = self.obj_size / 2 - idx_xy_center[:, 1:2] - 1
        offset = torch.cat([offset_x, offset_y], dim=1)
        pad_size = (self.img_size - self.obj_size) // 2

        scale = grid_size.unsqueeze(1)
        bias = pc_min - scale * (offset + pad_size)
        return scale, bias

    def build_memory(self, feats_pc, pos_feats, num_views):
        feats = self.feat_mlp(torch.cat([feats_pc, pos_feats], dim=-1))
        B, N, C = feats.shape
        feats_memory = feats.unsqueeze(1).expand(-1, num_views, -1, -1).reshape(B*num_views, N, C)
        return feats_memory

    def build_query(self, pos_pc, view):
        scale, bias = self.cal_scale_bias(pos_pc, view)
        view = view.reshape(-1, 3, 3)
        point_grid = torch.einsum('bcd,bnd->bnc', [torch.linalg.inv(view)[:, :, :2].float(),
                                                (scale.unsqueeze(1) * self.query_mesh.unsqueeze(0).expand(view.shape[0], -1, -1).to(pos_pc.device) + bias.unsqueeze(1))])
        B, N, _ = point_grid.shape
        direction_norm = F.normalize(view[:, :, 2], p=2, dim=1).unsqueeze(1).expand(-1, N, -1)
        
        query_list = []
        query_pos = self.query_pos.unsqueeze(0).expand(B, -1, -1).to(pos_pc.device)
        for component in list(self.query_component.keys()):
            query_list.append(eval(component))
        query = torch.cat(query_list, dim=-1)
        return query.float()

    def forward(self, feats_pc, pos_feats, pos_pc, view):
        feats_memory = self.build_memory(feats_pc, pos_feats, view.shape[1])
        query = self.build_query(pos_pc, view)
        feats_query = self.query_mlp(query)

        for block in self.transformer:
            feats_query = block(feats_query, context=feats_memory, mask=None)

        B, N, C = feats_query.shape
        feat_size = self.img_size // self.ds_ratio
        feats_img = feats_query.reshape(B, feat_size, feat_size, C).permute(0, 3, 1, 2)
        return feats_img