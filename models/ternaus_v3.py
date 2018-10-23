import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.nn import Sequential

from lib.common import count_parameters, logit, logit_logit_gate
from models.modules.abn import ABN, ACT_LEAKY_RELU, ACT_RELU, ACT_ELU
from models.modules.conv_bn_act import CABN
from models.modules.coord_conv import append_coords
from models.modules.gap import GlobalAvgPool2d, GlobalAvgMaxPool2d
from models.modules.gate2d import ChannelSpatialGate2d
from models.wider_resnet import WiderResNet
from models.wrn_oc import ASP_OC_Module


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvAct(nn.Module):
    def __init__(self, in_: int, out: int, activation=ACT_RELU):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)

        if self.activation == ACT_RELU:
            x = F.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            x = F.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            x = F.elu(x, inplace=True)

        return x


class DecoderBlockV3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, up=True, abn_block=ABN, activation=ACT_RELU, batch_norm=True):
        super(DecoderBlockV3, self).__init__()
        self.in_channels = in_channels
        self.up = up

        if batch_norm:
            self.block = nn.Sequential(
                CABN(in_channels, middle_channels, abn_block=abn_block, activation=activation),
                CABN(middle_channels, out_channels, abn_block=abn_block, activation=activation)
            )
        else:
            self.block = nn.Sequential(
                ConvAct(in_channels, middle_channels, activation=activation),
                ConvAct(middle_channels, out_channels, activation=activation)
            )

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.block(x)
        return x


class DecoderBlockV3SE(DecoderBlockV3):
    def __init__(self, in_channels, middle_channels, out_channels, up=True, abn_block=ABN, activation=ACT_RELU, batch_norm=True):
        super(DecoderBlockV3SE, self).__init__(in_channels, middle_channels, out_channels, up, abn_block, activation, batch_norm)
        self.scse = ChannelSpatialGate2d(out_channels)

    def forward(self, x, e=None):
        x = super().forward(x, e)
        x = self.scse(x)
        return x


class TernausNetV3(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self,
                 num_classes=1,
                 pretrained=True,
                 abn_block=ABN,
                 activation=ACT_LEAKY_RELU,
                 filters=64,
                 classifier_classes=1,
                 num_channels=3,
                 use_dropout=True,
                 batch_norm=True,
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
        super(TernausNetV3, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.use_dropout = use_dropout
        self.classifier_classes = classifier_classes
        self.filters = filters

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

        if num_channels == 1:
            self.conv1 = encoder.mod1
        else:
            self.conv1 = Sequential(OrderedDict([('conv1', nn.Conv2d(num_channels + 2, 64, 3, padding=1, bias=False))]))

        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        # Decoder head
        self.center = nn.Sequential(
            CABN(1024, 512, kernel_size=3, padding=1),
            CABN(512, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.dec5 = DecoderBlockV3SE(1024 + 256, 512, 64, abn_block=abn_block, activation=activation, batch_norm=batch_norm)
        self.dec4 = DecoderBlockV3SE(512 + 64, 512, 64, abn_block=abn_block, activation=activation, batch_norm=batch_norm)
        self.dec3 = DecoderBlockV3SE(256 + 64, 256, 64, abn_block=abn_block, activation=activation, batch_norm=batch_norm)
        self.dec2 = DecoderBlockV3SE(128 + 64, 128, 64, abn_block=abn_block, activation=activation, batch_norm=batch_norm)
        self.dec1 = DecoderBlockV3SE(64, 32, 64, abn_block=abn_block, activation=activation, batch_norm=batch_norm)

        self.mask_logit    = nn.Sequential(
            nn.Conv2d(64*5+1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  1, kernel_size=1, padding=0),
        )

        # Classification head
        self.avg_pool = GlobalAvgMaxPool2d()
        self.type_fuse = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(256 * 2 + 1, 64)),  # Center features + depth channel
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(64, self.classifier_classes))
        ]))

    def forward(self, batch):
        image = batch['image']
        depth = batch['depth']

        image = self.append_depth(image)

        batch_size, channels, height, width = image.size()

        # Encoder path
        conv1 = self.conv1(image)
        e2 = self.conv2(self.pool(self.maybe_drop(conv1, 0.)))
        e3 = self.conv3(self.pool(self.maybe_drop(e2, 0.)))
        e4 = self.conv4(self.pool(self.maybe_drop(e3, 0.)))
        e5 = self.conv5(self.pool(self.maybe_drop(e4, 0.)))

        c = self.center(e5)

        # Decoder path
        d5 = self.dec5(c, e5)
        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2)

        f = torch.cat([
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=True),
            depth.view(batch_size, 1, 1, 1).repeat(1, 1, height, width)
        ], 1)

        mask = self.mask_logit(f)

        # Classification head
        type_features = self.avg_pool(c)
        type_features = self.maybe_drop(type_features, p=0.25)
        type_features = type_features.view(batch_size, -1)
        type_features = torch.cat([type_features, depth.view(batch_size, 1)], dim=1)  # Attach depth
        mask_type = self.type_fuse(type_features)

        return {
            'mask': mask,
            'class': mask_type,
        }

    def append_depth(self, image):

        batch_size, _, x_dim, y_dim = image.size()

        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        yy_channel = yy_channel.float() / (y_dim - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.type_as(image)

        return torch.cat([image, yy_channel, image * yy_channel], dim=1)

    def maybe_drop(self, x, p=0.5):
        if self.use_dropout:
            x = F.dropout(x, p, training=self.training)
        return x

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
    net = TernausNetV3(num_classes=1, num_channels=1)
    net = net.eval()
    print(count_parameters(net))

    x = {'image': torch.rand((4, 1, 128, 128)),
         'depth': torch.rand((4))
         }
    y = net(x)
    print(y['mask'].size())
