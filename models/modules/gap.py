import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return F.avg_pool2d(inputs, kernel_size=in_size[2:])


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return F.max_pool2d(inputs, kernel_size=in_size[2:])


class GlobalAvgMaxPool2d(nn.Module):
    def __init__(self):
        """Global average & max pooling over the input's spatial dimensions"""
        super(GlobalAvgMaxPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        global_max = F.max_pool2d(inputs, kernel_size=in_size[2:])
        global_avg = F.avg_pool2d(inputs, kernel_size=in_size[2:])
        return torch.cat([global_avg, global_max], dim=1)