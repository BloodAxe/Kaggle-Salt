import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MaxAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super(MaxAvgPool2d, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        mp = self.max_pool(x)
        ap = self.avg_pool(x)
        return torch.cat([mp, ap], dim=1)
