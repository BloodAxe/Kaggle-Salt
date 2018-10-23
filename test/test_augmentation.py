import cv2
import numpy as np
import torch


def test_rezize():
    x = torch.rand((101, 101)).numpy()
    assert np.array_equal(x, cv2.resize(x, (101, 101), interpolation=cv2.INTER_NEAREST))
    assert np.array_equal(x, cv2.resize(x, (101, 101), interpolation=cv2.INTER_LINEAR))
    assert np.array_equal(x, cv2.resize(x, (101, 101), interpolation=cv2.INTER_CUBIC))
