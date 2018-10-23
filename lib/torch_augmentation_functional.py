import torch
from albumentations.augmentations.functional import get_center_crop_coords
from torch.nn import functional as F


def pad_if_needed(img, min_width, min_height, border_mode='reflect'):
    """

    :param img: A tensor of [NCHW] channels
    :param min_width:
    :param min_height:
    :param border_mode:
    :return:
    """
    height, width = img.size()[-2:]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    padding = w_pad_left, w_pad_right, h_pad_top, h_pad_bottom
    img = F.pad(img, padding, border_mode)

    assert img.size()[-2] == max(min_height, height)
    assert img.size()[-1] == max(min_width, width)

    return img


def central_crop(img, crop_width, crop_height):
    """

    :param img: A tensor of [NCHW] channels
    :param crop_width:
    :param crop_height:
    :return:
    """
    batch_size, channels, height, width = img.size()
    if height == crop_height and width == crop_width:
        return img

    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[..., y1:y2, x1:x2].contiguous()
    return img


if __name__ == '__main__':
    x = torch.rand((4, 1, 101, 102))
    print(x.size())

    x_pad = pad_if_needed(x, 128, 130)
    print(x_pad.size())

    x_center = central_crop(x_pad, 102, 101)
    print(x_center.size())

    assert (x == x_center).all()

    central_crop(torch.rand((4,1,128,128)), 128, 128)
    print(x_center.size())
