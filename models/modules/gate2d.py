from torch import nn, Tensor
from torch.nn import functional as F


class ChannelGate2d(nn.Module):
    """
    Channel Squeeze and Spatial Excitation module from https://arxiv.org/pdf/1803.02579.pdf
    """

    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        module_input = x
        x = self.squeeze(x)
        x = x.sigmoid()
        return module_input * x


class SpatialGate2d(nn.Module):
    """
    Spatial squeeze and Channel Excitation module from https://arxiv.org/pdf/1803.02579.pdf
    Aka SEModule
    """

    def __init__(self, channels, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        module_input = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2d(channels)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)
