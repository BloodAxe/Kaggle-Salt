import torch
from torch import nn
from torch.nn import functional as F

class HyperColumn(nn.Module):
    def __init__(self, mode='bilinear', align_corners=True):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, *features):
        layers = []
        dst_size = features[0].size()[-2:]

        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))

        return torch.cat(layers, dim=1)


class DeepSupervision(nn.Module):
    """
    https://arxiv.org/pdf/1409.5185.pdf
    https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eed71108da598cba69ab4c14dac2fdc0688516c0/models/networks/unet_CT_multi_att_dsv_3D.py#L52
    """
    def __init__(self, in_filters, out_filters, mode='bilinear', align_corners=True):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels, out_channels in zip(in_filters, out_filters)])

    def forward(self, *features):
        layers = []
        dst_size = features[0].size()[-2:]

        for conv, f in zip(self.convs, features):
            f = conv(f)
            f = F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners)
            layers.append(f)

        return torch.cat(layers, dim=1)
