import random

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.functional import clipped


@clipped
def random_contrast_gray(img, alpha):
    gray = ((1.0 - alpha) / img.size) * np.sum(img)
    return alpha * img + gray


class RandomContrastGray(A.ImageOnlyTransform):
    """Randomly change contrast of the input image.

    Args:
        limit ((float, float) or float): factor range for changing contrast. If limit is a single float, the range
            will be (-limit, limit). Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, limit=.2, p=.5):
        super(RandomContrastGray, self).__init__(p)
        self.limit = A.to_tuple(limit)

    def apply(self, img, alpha=0.2, **params):
        return random_contrast_gray(img, alpha)

    def get_params(self):
        return {'alpha': 1.0 + random.uniform(self.limit[0], self.limit[1])}


class AxisShear(A.DualTransform):
    def __init__(self, sx=0.1, sy=0.1, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
        super(AxisShear, self).__init__(p)
        self.sx = sx
        self.sy = sy
        self.interpolation = interpolation
        self.border_mode = border_mode

    def get_params(self):
        return {"cx": random.uniform(0, 1),
                "cy": random.uniform(0, 1),
                "sx": random.uniform(- self.sx, self.sx),
                "sy": random.uniform(- self.sx, self.sx)}

    def apply(self, img, cx=0.5, cy=0.5, sx=0, sy=0, interpolation=cv2.INTER_LINEAR, **params):
        center = np.eye(3, 3)
        center[0, 2] = cx * img.shape[1]
        center[1, 2] = cy * img.shape[0]

        inv_center = np.eye(3, 3)
        inv_center[0, 2] = -center[0, 2]
        inv_center[1, 2] = -center[1, 2]

        shear = np.eye(3, 3)
        shear[0, 1] = sx
        shear[1, 0] = sy

        m = np.matmul(np.matmul(center, shear), inv_center)
        return cv2.warpAffine(img, m[:2, ...], dsize=(img.shape[1], img.shape[0]), flags=interpolation, borderMode=self.border_mode)


class AxisScale(A.DualTransform):
    def __init__(self, sx=0.1, sy=0.1, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
        super(AxisScale, self).__init__(p)
        self.sx = sx
        self.sy = sy
        self.interpolation = interpolation
        self.border_mode = border_mode

    def get_params(self):
        return {"cx": random.uniform(0, 1),
                "cy": random.uniform(0, 1),
                "sx": random.uniform(1 - self.sx, 1 + self.sx),
                "sy": random.uniform(1 - self.sx, 1 + self.sx)}

    def apply(self, img, cx=0.5, cy=0.5, sx=1, sy=1, interpolation=cv2.INTER_LINEAR, **params):
        center = np.eye(3, 3)
        center[0, 2] = cx * img.shape[1]
        center[1, 2] = cy * img.shape[0]

        inv_center = np.eye(3, 3)
        inv_center[0, 2] = -center[0, 2]
        inv_center[1, 2] = -center[1, 2]

        scale = np.eye(3, 3)
        scale[0, 0] = sx
        scale[1, 1] = sy

        m = np.matmul(np.matmul(center, scale), inv_center)
        return cv2.warpAffine(img, m[:2, ...], dsize=(img.shape[1], img.shape[0]), flags=interpolation, borderMode=self.border_mode)
