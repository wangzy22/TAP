import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS, build_model_from_cfg


@MODELS.register_module()
class ViewGenBase(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 generator_args=None,
                 decoder_args=None,
                 loss_args=None,
                 **kwargs):
        super().__init__()
        self.weight_fg = loss_args.weight_fg
        self.weight_bg = loss_args.weight_bg

        self.encoder = build_model_from_cfg(encoder_args)
        self.generator = build_model_from_cfg(generator_args)
        self.decoder = build_model_from_cfg(decoder_args)
        

    def forward(self, p0, f0=None):
        feats_pc, coords_pc = self.encoder.forward_cls_feat(p0, f0)
        feats_img = self.generator(feats_pc, coords_pc, p0['pos'], p0['views'])
        recon_img = self.decoder(feats_img)
        
        H, W = recon_img.shape[-2:]
        img_gt = p0['imgs'].reshape(-1, 3, H, W)
        img_mask = ((img_gt < 1).sum(1) > 0).unsqueeze(1).expand(-1, 3, -1, -1)
        loss = F.mse_loss(recon_img[img_mask], img_gt[img_mask]) * self.weight_fg + F.mse_loss(recon_img[~img_mask], img_gt[~img_mask]) * self.weight_bg
        return loss, recon_img