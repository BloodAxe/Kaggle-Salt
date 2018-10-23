import os

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable


def logit(x):
    eps = 1e-6
    x = x.clamp(eps, 1. - eps)
    p = torch.reciprocal(x) - 1.
    return -torch.log(p)


def logit_logit_gate(s, g):
    """
    :param s:
    :param g:
    :return: logit(sigmoid(m) * sigmoid(g))
    """

    s = torch.sigmoid(s) * torch.sigmoid(g)
    return logit(s)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        raise ValueError('Unsupported type')
    return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') == 0:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_mask_class(y_true: Tensor):
    """
    Computes index [0;4] for masks. 0 - <20%, 1 - 20-40%, 2 - 40-60%, 3 - 60-80%, 4 - 80-100%
    :param y_true:
    :return:
    """
    y_true = y_true.detach().cpu()
    batch_size = y_true.size(0)
    num_classes = y_true.size(1)
    if num_classes == 1:
        y_true = y_true.view(batch_size, -1)
    elif num_classes == 2:
        y_true = y_true[:, 1, ...].contiguous().view(batch_size, -1)  # Take salt class
    else:
        raise ValueError('Unknown num_classes')

    img_area = float(y_true.size(1))
    percentage = y_true.sum(dim=1) / img_area
    class_index = (percentage * 4).round().byte()
    return class_index


is_sorted = lambda a: np.all(a[:-1] <= a[1:])
