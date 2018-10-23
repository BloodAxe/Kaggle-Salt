"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.autograd import Variable
from torch.nn import Sequential

from lib.common import count_parameters
from models.modules.abn import ABN, ACT_LEAKY_RELU, ACT_RELU, ACT_ELU
from models.modules.conv_bn_act import CABN
from models.modules.coord_conv import append_coords
from models.modules.gap import GlobalAvgPool2d
from models.modules.gate2d import ChannelSpatialGate2d
from models.wider_resnet import WiderResNet


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

    def forward(self, x):
        x = self.block(x)
        return x


class DecoderBlockV3SE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, abn_block=ABN, activation=ACT_RELU, batch_norm=True):
        super(DecoderBlockV3SE, self).__init__()
        self.in_channels = in_channels
        self.scse = ChannelSpatialGate2d(out_channels)

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

    def forward(self, x):
        x = self.block(x)
        x = self.scse(x)
        return x


class DeepSupervision(nn.Module):
    def __init__(self, in_size, out_size, num_classes, scale_factor):
        super(DeepSupervision, self).__init__()
        self.dsn = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(out_size, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )

    def forward(self, input):
        return self.dsn(input)


def neareset_upsample(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='nearest')


def bilinear_upsample(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ABN(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            ABN(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            ABN(out_channels),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ABN(out_features),
                                     BaseOC_Context_Module(in_channels=out_features, out_channels=out_features, key_channels=out_features // 2, value_channels=out_features,
                                                           dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   ABN(out_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   ABN(out_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   ABN(out_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   ABN(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ABN(out_features),
            nn.Dropout2d(0.1)
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output


class TernausNetOC(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self,
                 num_classes=1,
                 pretrained=True,
                 abn_block=ABN,
                 activation=ACT_RELU,
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
        super(TernausNetOC, self).__init__()
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

        # decoder_out = [self.filters, self.filters, self.filters, self.filters, self.filters, self.filters]
        decoder_out = [self.filters, self.filters, self.filters, self.filters, self.filters, self.filters]

        self.center = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            # BaseOC_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, dropout=0.05, sizes=([1])),
            ASP_OC_Module(features=512, out_features=self.filters, dilations=(2, 4, 8))
        )

        self.dec4 = DecoderBlockV3SE(512 + decoder_out[4], decoder_out[3], decoder_out[3], abn_block=abn_block, activation=activation, batch_norm=batch_norm)
        self.dec3 = DecoderBlockV3SE(256 + decoder_out[3], decoder_out[2], decoder_out[2], abn_block=abn_block, activation=activation, batch_norm=batch_norm)
        self.dec2 = DecoderBlockV3SE(128 + decoder_out[2], decoder_out[1], decoder_out[1], abn_block=abn_block, activation=activation, batch_norm=batch_norm)
        self.dec1 = DecoderBlockV3SE(64 + decoder_out[1], decoder_out[0], decoder_out[0], abn_block=abn_block, activation=activation, batch_norm=batch_norm)

        # Deep supervision
        self.dsv = DeepSupervision(256, 128, num_classes, scale_factor=4)

        self.mask_logit = nn.Conv2d(decoder_out[0], num_classes, kernel_size=1, padding=0)

        # Classification head
        self.avg_pool = GlobalAvgPool2d()
        self.type_fuse = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.filters + 1, self.filters)),  # Center features + depth channel
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(self.filters, self.classifier_classes))
        ]))

    def forward(self, batch):
        image = batch['image']
        depth = batch['depth']

        image = append_coords(image)

        batch_size, channels, height, width = image.size()

        # Encoder path
        conv1 = self.conv1(image)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        # Decoder head
        center = self.center(conv5)

        # Attach global features
        # center = torch.cat([center,
        #                     mask_type.view(batch_size, self.classifier_classes, 1, 1).repeat(1, 1, center.size(2), center.size(3)),
        #                     depth.view(batch_size, 1, 1, 1).repeat(1, 1, center.size(2), center.size(3))],
        #                    dim=1)

        dec4 = self.dec4(torch.cat([bilinear_upsample(center, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([bilinear_upsample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([bilinear_upsample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([bilinear_upsample(dec2, scale_factor=2), conv1], 1))

        # Classification head
        type_features = self.avg_pool(center)
        type_features = type_features.view(batch_size, -1)
        type_features = self.maybe_drop(type_features, p=0.25)
        type_features = torch.cat([type_features, depth.view(batch_size, 1)], dim=1)  # Attach depth
        mask_type = self.type_fuse(type_features)

        # # Deep Supervision
        dsv = self.dsv(conv3)

        mask = self.mask_logit(dec1)

        # gate = mask_type.view(batch_size, 1, 1, 1)
        #
        # # Gate mask with global mask presence flag
        # mask = logit_logit_gate(mask, gate)

        return {
            'mask': mask,
            'class': mask_type,
            'dsv': dsv,
        }

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
    net = TernausNetOC(num_classes=1, num_channels=1)
    net = net.eval()
    print(count_parameters(net))

    x = {'image': torch.rand((4, 1, 128, 128)),
         'depth': torch.rand((4))
         }
    y = net(x)
    print(y['mask'].size())
