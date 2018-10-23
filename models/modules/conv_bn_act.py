import torch.nn as nn

from models.modules.abn import ABN, ACT_LEAKY_RELU


class CABN(nn.Module):
    """
    Convolution + BatchNorm + Activation
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=1,
                 dilation=1,
                 padding=0, stride=1,
                 groups=1,
                 bias=False,
                 abn_block=ABN,
                 activation=ACT_LEAKY_RELU,
                 slope=0.01):
        super(CABN, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation,
                              stride=stride,
                              groups=groups,
                              bias=bias)

        self.abn = abn_block(out_channels, activation=activation, slope=slope)
        # self.reset_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x

    # def reset_weights(self):
    #     nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity=self.activation)
    #     if self.conv.bias is not None:
    #         nn.init.constant_(self.conv.bias, 0)
    #
    #     nn.init.constant_(self.abn.weight, 1)
    #     nn.init.constant_(self.abn.bias, 0)
