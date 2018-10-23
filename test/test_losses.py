import math

import torch
import numpy as np

from lib.common import logit


def test_gate():
    x = torch.rand((8))
    g = torch.rand((1))

    mask1 = torch.log(torch.exp(x + g) / (x.exp() + g.exp() + 1))
    mask2 = logit(torch.sigmoid(x) * torch.sigmoid(g))

    print(mask1)
    print(mask2)

    assert np.allclose(mask1.numpy(), mask2.numpy())
