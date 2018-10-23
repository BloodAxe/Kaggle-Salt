import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.abn import ABN, ACT_LEAKY_RELU, ACT_RELU
from models.modules.conv_bn_act import CABN
from models.modules.coord_conv import append_coords
from models.modules.gate2d import ChannelGate2d, SpatialGate2d


class UnetDecoderBlock(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, middle_channels=None, abn_block=ABN, activation=ACT_RELU, up=True, pre_dropout_rate=0., post_dropout_rate=0.):
        super(UnetDecoderBlock, self).__init__()
        self.up = up
        if middle_channels is None:
            middle_channels = (in_channels + out_channels) // 2

        self.conv1 = CABN(in_channels, middle_channels, kernel_size=3, stride=1, padding=1, abn_block=abn_block, activation=activation)
        self.conv2 = CABN(middle_channels, out_channels, kernel_size=3, stride=1, padding=1, abn_block=abn_block, activation=activation)
        self.pre_drop = nn.Dropout2d(pre_dropout_rate)
        self.post_drop = nn.Dropout2d(post_dropout_rate)

    def forward(self, x, enc=None):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        if enc is not None:
            x = torch.cat([x, enc], 1)

        x = self.pre_drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.post_drop(x)
        return x


class UnetDecoderBlockSE(UnetDecoderBlock):
    """
    Decoder with Squeeze & Excitation block
    """

    def __init__(self, in_channels, out_channels, middle_channels=None, abn_block=ABN, activation=ACT_RELU, up=True, pre_dropout_rate=0., post_dropout_rate=0.):
        super().__init__(in_channels, out_channels, middle_channels, abn_block, activation, up, pre_dropout_rate, post_dropout_rate)
        self.channel_gate = ChannelGate2d(out_channels)
        self.spatial_gate = SpatialGate2d(out_channels)

    def forward(self, x, enc=None):
        x = super().forward(x, enc)
        x = self.spatial_gate(x) + self.channel_gate(x)
        return x


class UnetDecoderBlockSECoord(UnetDecoderBlockSE):
    """
    Decoder with Squeeze & Excitation block and CoordConv
    """

    def __init__(self, in_channels, out_channels, middle_channels=None, abn_block=ABN, activation=ACT_RELU, up=True, pre_dropout_rate=0., post_dropout_rate=0.):
        super().__init__(in_channels + 2, out_channels, middle_channels, abn_block, activation, up, pre_dropout_rate, post_dropout_rate)

    def forward(self, x, enc=None):
        x_coord = append_coords(x)
        x = super().forward(x_coord, enc)
        return x
