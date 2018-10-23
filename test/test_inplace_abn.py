import torch

from models.modules.abn_inplace import InPlaceABN


def test_inplace_abn():
    abn = InPlaceABN(64)
    x = torch.rand((4,64,128,128))
    x = abn(x)
