from torch import nn

from models.modules.abn import ABN


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride, bias=bias, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthwiseSeparableConvNormAct2d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, abn_block=ABN):
        super(DepthwiseSeparableConvNormAct2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=False)
        self.abn_block = abn_block(nout)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.abn_block(out)
        return out
