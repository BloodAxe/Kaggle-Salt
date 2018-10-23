import os
from typing import Optional

import os
from typing import Optional

import albumentations as A
import albumentations.augmentations.functional as AF
import cv2
import numpy as np
import pandas as pd
import torch
# from albumentations.core.transforms_interface import NoOp
from skimage.morphology import remove_small_objects, remove_small_holes
from sklearn.utils import check_random_state
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

import lib.augmentations as AA
from lib import torch_augmentation_functional as TAF
from lib import train_utils as U
from lib.common import find_in_dir, is_sorted

DATA_ROOT = 'data'
N_FOLDS = 5
ORIGINAL_SIZE = 101


def all_train_ids() -> np.ndarray:
    """
    Return all train ids
    :return: Numpy array of ids
    """
    return np.array(sorted([id_from_fname(fname) for fname in find_in_dir(os.path.join(DATA_ROOT, 'train', 'images'))]))


def all_test_ids() -> np.ndarray:
    """
    Return all test ids
    :return: Numpy array of ids
    """
    return np.array(sorted([id_from_fname(fname) for fname in find_in_dir(os.path.join(DATA_ROOT, 'test', 'images'))]))


def read_train_image(sample_id) -> np.ndarray:
    return cv2.imread(os.path.join(DATA_ROOT, 'train', 'images', '%s.png' % sample_id), cv2.IMREAD_GRAYSCALE)


def read_test_image(sample_id) -> np.ndarray:
    return cv2.imread(os.path.join(DATA_ROOT, 'test', 'images', '%s.png' % sample_id), cv2.IMREAD_GRAYSCALE)


def read_train_mask(sample_id) -> np.ndarray:
    mask = cv2.imread(os.path.join(DATA_ROOT, 'train', 'masks', '%s.png' % sample_id), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    return mask


def read_train_images(ids) -> np.ndarray:
    """
    Reads train images. Returns numpy array of shape [N;H;W], where N is number of images, H - height, W - width.
    Images read as np.uint8 type with range in [0.255]
    :param ids: List of image ids.
    :return: Numpy array
    """
    if not is_sorted(ids):
        raise ValueError('Array ids must be sorted')

    images = [read_train_image(sample_id) for sample_id in ids]
    images = np.array(images, dtype=np.uint8)
    return images


def read_test_images(ids) -> np.ndarray:
    """
    Reads test images. Returns numpy array of shape [N;H;W], where N is number of images, H - height, W - width
    Images read as np.uint8 type with range in [0.255]
    :param ids: List of image ids.
    :return: Numpy array
    """
    if not is_sorted(ids):
        raise ValueError('Array ids must be sorted')

    images = [read_test_image(sample_id) for sample_id in ids]
    images = np.array(images, dtype=np.uint8)
    return images


def read_train_masks(ids) -> np.ndarray:
    """
    Reads train masks. Returns numpy array of shape [N;H;W], where N is number of images, H - height, W - width
    :param ids: List of image ids.
    :return: Numpy array with values {0,1}
    """
    if not is_sorted(ids):
        raise ValueError('Array ids must be sorted')

    images = [read_train_mask(sample_id) for sample_id in ids]
    images = np.array(images, dtype=np.uint8)
    return images


def read_depths(ids):
    if not is_sorted(ids):
        raise ValueError('Array ids must be sorted')

    df = pd.read_csv(os.path.join(DATA_ROOT, 'depths.csv'))
    df['z'] = df['z'].astype(np.float32)
    df['z'] = df['z'] / df['z'].max()

    depths = []
    for sample_id in ids:
        z = df[df['id'] == sample_id].iloc[0]['z']
        depths.append(z)
    return np.array(depths)


def get_selection_mask(ids: np.ndarray, query: np.ndarray):
    if not is_sorted(ids):
        raise ValueError('Array ids must be sorted')

    if not is_sorted(query):
        raise ValueError('Array subset must be sorted')

    if not np.in1d(query, ids, assume_unique=True).all():
        raise ValueError("Some elements of subset are not in ids")

    # mask2 = []
    # for sample_id in subset:
    #     index = np.argwhere(ids == sample_id)[0, 0]
    #     mask2.append(index)
    # mask2 = np.array(mask2)
    # return mask2

    mask = np.array([sample_id in query for sample_id in ids])
    return mask


def drop_some(images, masks, drop_black=True, drop_vstrips=False, drop_empty=False, drop_few=None) -> np.ndarray:
    skips = []

    dropped_blacks = 0
    dropped_vstrips = 0
    dropped_few = 0
    dropped_empty = 0

    for image, mask in zip(images, masks):
        should_keep = True

        if drop_black and is_black(image, mask):
            should_keep = False
            dropped_blacks += 1

        if drop_vstrips and is_vertical_strips(image, mask):
            should_keep = False
            dropped_vstrips += 1

        if drop_few and is_salt_less_than(image, mask, int(drop_few)) and not is_salt_less_than(image, mask, 1):
            should_keep = False
            dropped_few += 1

        if drop_empty and is_salt_less_than(image, mask, 1):
            should_keep = False
            dropped_empty += 1

        skips.append(should_keep)

    print(f'Dropped {dropped_blacks} black images; {dropped_vstrips} vertical strips; {dropped_empty}  empty masks; {dropped_few} few-pixel salt')
    return np.array(skips)


def cumsum(img, axis=0) -> np.ndarray:
    """
    https://www.kaggle.com/bguberfain/unet-with-depth#360485
    For what I know about seismic imaging, the cumsum on the depth axis will (remotely) approximate an inversion operation* (that is, convert from interface transition to interface properly).
    :param axis:
    :param img: Single-channel image
    :return:
    """
    x_mean = img.mean()
    x_csum = (np.float32(img) - x_mean).cumsum(axis=axis)
    x_csum -= x_csum.mean()
    x_csum /= max(1e-3, x_csum.std())
    return x_csum


def id_from_fname(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def harder_augmentations(target_size, border_mode=cv2.BORDER_CONSTANT):
    border_mode = U.get_border_mode(border_mode)
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.OneOf([A.IAAAdditiveGaussianNoise(), A.GaussNoise()], p=0.3),
        A.OneOf([A.MotionBlur(p=.1), A.MedianBlur(blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1)], p=0.2),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=border_mode),
            A.RandomSizedCrop((int(target_size * 0.8), target_size), target_size, target_size, p=0.5),
        ], p=0.5),
        A.OneOf([
            AA.AxisShear(sx=0.1, sy=0.1, border_mode=border_mode, p=0.5),
            A.ElasticTransform(alpha=0.5, sigma=0.5, alpha_affine=10, border_mode=border_mode),
            A.GridDistortion(border_mode=border_mode),
            A.IAAPerspective(p=0.3),
        ], p=0.3),
        A.Cutout()
    ])
    return aug


def hard_augmentations(target_size, border_mode=cv2.BORDER_CONSTANT):
    border_mode = U.get_border_mode(border_mode)
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([A.RandomBrightness(p=0.5), A.RandomGamma(gamma_limit=(80, 120), p=0.5)]),
        A.OneOf([A.IAAAdditiveGaussianNoise(), A.GaussNoise()], p=0.3),
        A.OneOf([A.MotionBlur(p=.1), A.MedianBlur(blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1)], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.075, rotate_limit=10, p=0.5, border_mode=border_mode),
        A.OneOf([
            A.ElasticTransform(alpha=0.5, sigma=0.5, alpha_affine=10, border_mode=border_mode, p=0.1),
            A.GridDistortion(border_mode=border_mode, p=0.1),
        ], p=0.3),
        A.Cutout()
    ])

    return aug


def medium_augmentations(target_size, border_mode=cv2.BORDER_CONSTANT):
    border_mode = U.get_border_mode(border_mode)
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.25),
        AA.RandomContrastGray(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=5, p=0.5, border_mode=border_mode),
        A.OneOf([
            A.ElasticTransform(alpha=0.5, sigma=0.5, alpha_affine=10, border_mode=border_mode, p=0.1),
            A.GridDistortion(border_mode=border_mode, p=0.1),
            A.NoOp(p=0.5)
        ]),
        A.OneOf([A.IAAAdditiveGaussianNoise(), A.GaussNoise()], p=0.3),
    ])
    return aug


def light_augmentations(target_size, border_mode=cv2.BORDER_CONSTANT):
    border_mode = U.get_border_mode(border_mode)
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=0, p=0.5, border_mode=border_mode),
        A.OneOf([
            A.ElasticTransform(alpha=0.5, sigma=0.5, alpha_affine=10, border_mode=border_mode, p=0.1),
            A.GridDistortion(border_mode=border_mode, p=0.1),
            A.NoOp(p=0.5)
        ]),
    ])
    return aug


def flip_augmentations(target_size, border_mode=cv2.BORDER_CONSTANT):
    border_mode = U.get_border_mode(border_mode)
    aug = A.HorizontalFlip(p=0.5)
    return aug


def none_augmentations(target_size, border_mode=cv2.BORDER_CONSTANT):
    return A.NoOp(p=1)


class DatasetResizePad:
    def __init__(self, resize_size, target_size, border_mode=cv2.BORDER_CONSTANT, border_fill=0, interpolation=cv2.INTER_LINEAR, **kwargs):
        self.resize_size = resize_size
        self.target_size = target_size
        self.border_mode = border_mode
        self.border_fill = border_fill
        self.interpolation = interpolation

        self.t_forward = A.Compose([
            A.Resize(resize_size, resize_size, interpolation=interpolation),
            A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=border_mode)])

    def forward(self, **kwargs):
        return self.t_forward(**kwargs)

    def backward(self, x):
        if isinstance(x, torch.Tensor):
            x = TAF.central_crop(x, self.resize_size, self.resize_size)
        elif isinstance(x, np.ndarray):
            x = AF.center_crop(x, self.resize_size, self.resize_size).ascontiguousarray()
        return x

    def __repr__(self):
        return f'DatasetResizePad(resize_size={self.resize_size}, target_size={self.target_size}, border_mode={self.border_mode}, border_fill={self.border_fill}, interpolation={self.interpolation})'


def get_prepare_fn(name, border_mode=cv2.BORDER_DEFAULT, **kwargs):
    border_mode = U.get_border_mode(border_mode)
    if name is None or name == 'None':
        return None
    if name == '128':
        return DatasetResizePad(resize_size=128, target_size=128, border_mode=border_mode, **kwargs)
    if name == '224':
        return DatasetResizePad(resize_size=224, target_size=224, border_mode=border_mode, **kwargs)
    if name == '256':
        return DatasetResizePad(resize_size=256, target_size=256, border_mode=border_mode, **kwargs)
    if name == '128pad':
        return DatasetResizePad(resize_size=ORIGINAL_SIZE, target_size=128, border_mode=border_mode, **kwargs)
    if name == '224pad':
        return DatasetResizePad(resize_size=ORIGINAL_SIZE * 2, target_size=224, border_mode=border_mode, **kwargs)
    if name == '256pad':
        return DatasetResizePad(resize_size=ORIGINAL_SIZE * 2, target_size=256, border_mode=border_mode, **kwargs)

    raise ValueError('Unsupported prepare fn')


class ImageAndMaskDataset(Dataset):
    """
    Creates a dataset object.
    :param images - List of images
    :param masks - List of masks
    :param depths - List of depths

    """

    def __init__(self, ids: np.ndarray, images: np.ndarray, masks: Optional[np.ndarray], depths: np.ndarray,
                 prepare_fn: DatasetResizePad = None,
                 normalize=A.Normalize(mean=0.5, std=0.224, max_pixel_value=255.0),
                 augment=None):

        if not is_sorted(ids):
            raise ValueError('Array ids must be sorted')

        self.ids = ids
        self.images = images
        self.masks = masks
        self.depths = depths
        self.augment = augment
        self.normalize = normalize
        self.num_channels = 1
        self.resize_fn = prepare_fn

        if prepare_fn is not None:
            self.images = np.array([self.resize_fn.forward(image=x)['image'] for x in self.images])
            if self.masks is not None:
                self.masks = np.array([self.resize_fn.forward(image=x, mask=x)['mask'] for x in self.masks])

    def __getitem__(self, index):

        data = {'image': self.images[index].copy()}
        if self.masks is not None:
            data['mask'] = self.masks[index].copy()

        if self.augment is not None:
            data = self.augment(**data)

        data = self.normalize(**data)

        image = np.expand_dims(data['image'], 0)
        image = torch.from_numpy(image).float()

        sample = {
            'index': index,
            'id': self.ids[index],
            'image': image,
            'depth': self.depths[index],
        }

        mask = data.get('mask', None)
        if mask is not None:
            if not np.isin(mask, [0, 1]).all():
                raise RuntimeError(f'A mask after augmentation contains values other than {{0;1}}: {np.unique(mask)}')

            # BCE problem, so float target
            mask_class = np.array((mask > 0).any(), dtype=np.float32)
            mask_class = np.expand_dims(mask_class, 0)

            mask = np.expand_dims(mask, 0)
            mask = torch.from_numpy(mask).float()

            sample['mask'] = mask
            sample['class'] = mask_class

        return sample

    def channels(self):
        return self.num_channels

    def __len__(self):
        return len(self.images)


def get_folds_vector(kind, images, masks, depths, n_folds=N_FOLDS, random_state=None):
    n = len(depths)
    folds = np.array(list(range(n_folds)) * n)[:n]

    rnd = check_random_state(random_state)

    if kind == 'coverage' or kind == 'area':
        coverage = np.array([cv2.countNonZero(x) for x in masks], dtype=np.int)
        sorted_indexes = np.argsort(coverage)
    elif kind == 'depth':
        sorted_indexes = np.argsort(depths)
    elif kind == 'resolution':
        resolution = np.array([cv2.Laplacian(image, cv2.CV_32F, borderType=cv2.BORDER_REFLECT101).std() for image in images])
        sorted_indexes = np.argsort(resolution)
    else:
        sorted_indexes = list(range(n))
        rnd.shuffle(sorted_indexes)

    return folds[sorted_indexes]


def get_train_test_split_for_fold(stratify, fold, ids):
    folds = pd.read_csv(os.path.join('data', f'folds_by_{stratify}.csv'))
    folds = np.array([folds[folds['id'] == id].iloc[0]['fold'] for id in ids])
    return folds != fold, folds == fold


def fix_mask(mask):
    """
    Tries to 'fix' a mask by filling gaps and removing single-pixel noise
    :param mask:
    :return:
    """
    mask = mask.astype(np.bool)
    mask = remove_small_holes(mask, area_threshold=12, connectivity=1)
    mask = remove_small_objects(mask, min_size=12, connectivity=1)

    # kernel = np.ones((5, 5), dtype=np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REFLECT101)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT101)
    return mask.astype(np.uint8)


def fix_masks(masks, train_ids):
    changed_masks = []
    changed_ids = []

    for image_id, mask in zip(train_ids, masks):
        new_mask = fix_mask(mask)
        changed_masks.append(new_mask)

        if not np.array_equal(new_mask, mask):
            changed_ids.append(image_id)

    masks = np.array(changed_masks)
    return masks, changed_ids


def is_black(image, mask):
    return image.sum() == 0


def is_vertical_strips(image, mask):
    colsum = np.sum(mask, axis=0)
    uniq = np.unique(colsum)
    return len(uniq) == 2 and uniq.min() == 0 and uniq.max() == mask.shape[0]


def is_salt_less_than(image, mask, threshold):
    return mask.sum() < threshold


def is_salt_greater_than(image, mask, threshold):
    return mask.sum() > threshold


AUGMENTATION_MODES = {
    'harder': harder_augmentations,
    'hard': hard_augmentations,
    'medium': medium_augmentations,
    'light': light_augmentations,
    'flip': flip_augmentations,
    'none': none_augmentations,
}
