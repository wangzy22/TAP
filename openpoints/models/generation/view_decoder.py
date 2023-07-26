import torch.nn as nn

from ..build import MODELS



@MODELS.register_module()
class ViewDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 5, stride=4, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 8, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        

    def forward(self, feats_img):
        feats = self.layer1(feats_img)
        feats = self.layer2(feats)
        feats = self.layer3(feats)
        img = self.layer4(feats)
        return img