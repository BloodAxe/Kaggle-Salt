import matplotlib as mpl

from nnn import ssim_cv

mpl.use('module://backend_interagg')

import numpy as np


def test_ssim_normalization():
    assert ssim_cv(np.zeros((101, 101), dtype=np.uint8),
                   np.zeros((101, 101), dtype=np.uint8)) == 1.0

    assert ssim_cv(np.ones((101, 101), dtype=np.uint8) * 255,
                   np.ones((101, 101), dtype=np.uint8) * 255) == 1.0

    assert ssim_cv(np.zeros((101, 101), dtype=np.uint8),
                   np.ones((101, 101), dtype=np.uint8) * 255) < 0.0001

    assert ssim_cv(np.ones((101, 101), dtype=np.uint8) * 127,
                   np.ones((101, 101), dtype=np.uint8) * 255) > 0.5

    one_black = np.ones((101, 101), dtype=np.uint8) * 255
    one_black[1, 1] = 0

    assert ssim_cv(one_black,
                   np.ones((101, 101), dtype=np.uint8) * 255) > 0.99
