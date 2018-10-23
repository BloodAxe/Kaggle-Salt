import numpy as np
import torch
from torch import Tensor


def average_classes(predictions, n):
    res = []
    for i in range(0, len(predictions), n):
        view = predictions[i:i + n]
        mean = np.mean(view, axis=0)
        res.append(mean)

    return np.array(res)


def tta_fliplr_aug(images, depths):
    is_tensor = isinstance(images, Tensor)
    if is_tensor:
        images = [np.moveaxis(x.numpy(), 0, -1) for x in images]
        depths = [d for d in depths]

    res = []
    dep = []

    for image,d in zip(images, depths):
        res.extend([
            image,
            np.fliplr(image),
        ])

        dep.append(d)
        dep.append(d)

    images = np.array(res)
    depths = np.array(dep)

    if is_tensor:
        images = torch.from_numpy(np.moveaxis(images, -1, 1))
        depths = torch.from_numpy(depths)

    return images, depths


def tta_fliplr_deaug(masks, types):
    assert len(masks) == len(types)
    assert len(masks) % 2 == 0

    masks_list = []
    types_list = []
    one_over_2 = float(1. / 2.)

    for i in range(0, len(masks), 2):
        img  = masks[i + 0] + np.fliplr(masks[i + 1])
        type = types[i + 0] + types[i + 1]

        masks_list.append(img * one_over_2)
        types_list.append(type * one_over_2)

    return np.array(masks_list), np.array(types_list)
