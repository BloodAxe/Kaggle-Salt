"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""
import os

import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import Sequential
from collections import OrderedDict

from lib.common import count_parameters
from models.modules.abn import ABN
from models.wider_resnet import WiderResNet


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)


class TernausNetV2(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 is_deconv=False,
                 pretrained=True,
                 abn_block=ABN,
                 num_channels=3,
                 **kwargs):
        """

        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_channels: Number of channels in the input images.
        """
        super(TernausNetV2, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], abn_block=abn_block, classes=1000)
        if pretrained:
            checkpoint = torch.load(os.path.join('pretrain', 'wide_resnet38_ipabn_lr_256.pth.tar'))

            # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            encoder.load_state_dict(new_state_dict)

        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv2d(num_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        # nn.init.kaiming_normal_(self.final.weight, mode='fan_out', nonlinearity='linear')
        # nn.init.constant_(self.final.bias, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)

    def set_fine_tune(self, fine_tune_enabled):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = bool(not fine_tune_enabled)

    def set_encoder_training_enabled(self, enabled):
        # First layer is trainable since we use 1-channel image instead of 3-channel
        layers = [self.conv2, self.conv3, self.conv4, self.conv5]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = bool(enabled)


if __name__ == '__main__':
    net = TernausNetV2(num_classes=1, num_channels=1)
    net = net.eval()
    print(count_parameters(net))

    x = torch.rand((4, 1, 128, 128))
    y = net(x)
    print(x.size(), y.size())

    x = torch.rand((4, 1, 224, 224))
    y = net(x)
    print(x.size(), y.size())

    x = torch.rand((4, 1, 256, 256))
    y = net(x)
    print(x.size(), y.size())
