import torch
from torch import nn

from models.modules.gap import GlobalAvgPool2d, GlobalMaxPool2d


class ClassificationModule(nn.Module):
    def __init__(self, features, num_classes=1):
        super().__init__()
        self.avgpool = GlobalAvgPool2d()
        self.maxpool = GlobalMaxPool2d()

        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(features * 2, num_classes)

    def forward(self, x):
        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x