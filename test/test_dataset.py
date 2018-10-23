import os

import matplotlib as mpl
import torch

mpl.use('module://backend_interagg')
import matplotlib.pyplot as plt

import cv2

import albumentations as A
import albumentations.augmentations.functional as AF

import numpy as np
import lib.dataset as D
import lib.metrics as M


def test_fix_mask():
    mask = D.read_train_mask('000e218f21')
    new_mask = D.fix_mask(mask)
    assert np.array_equal(mask, new_mask)


def test_fix_masks():
    train_ids = D.all_train_ids()
    masks = D.read_train_masks(train_ids)
    new_masks, changed_ids = D.fix_masks(masks, train_ids)
    print(len(changed_ids))

    dst = 'test/out/test_fix_masks'
    os.makedirs(dst, exist_ok=True)

    idx = D.get_selection_mask(train_ids, changed_ids)

    for id, old, new in zip(changed_ids, masks[idx], new_masks[idx]):
        image = np.concatenate((old, new), 1)
        fname = f'{id}.png'
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(dst, fname), image * 255)


def test_folds_coverage():
    train_ids = D.all_train_ids()
    depths = D.read_depths(train_ids)
    images = D.read_train_images(train_ids)
    masks = D.read_train_masks(train_ids)

    n_folds = 10
    coverage = np.array([cv2.countNonZero(x) for x in masks], dtype=np.int)
    folds_d = D.get_folds_vector('coverage', images, masks, depths, n_folds=n_folds)

    f, ax = plt.subplots(1, 2)

    for fold in range(n_folds):
        train = coverage[folds_d != fold]
        val = coverage[folds_d == fold]

        ax[0].hist(train, label=f'Fold {fold}')
        ax[1].hist(val, label=f'Fold {fold}')

    f.show()


def test_map():
    train_id = D.all_train_ids()
    masks = D.read_train_masks(train_id)
    print(M.threshold_mining(masks, masks))


def test_get_selection_mask():
    train_ids = D.all_train_ids()
    train_images = D.read_train_images(train_ids)
    black_images, no_salt, full_salt, vstrips, one_pixel_salt, few_salt, few_non_salt = D.find_problematic_masks(train_ids, few_pixels_threshold=8)

    mask = D.get_selection_mask(train_ids, vstrips)
    vstrips_images = train_images[mask]
    print(len(vstrips_images))

def test_no_resize_pad():
    mask = AF.to_float(cv2.imread('data/train/masks/0b73b427d1.png', cv2.IMREAD_GRAYSCALE))

    processor = D.DatasetResizePad(101, 128, border_mode=cv2.BORDER_REFLECT101, interpolation=cv2.INTER_LINEAR)
    mask_pad = processor.forward(image=mask, mask=mask)['image']
    mask_torch = torch.from_numpy(mask_pad).unsqueeze(0).unsqueeze(0).float()
    mask_back = processor.backward(mask_torch).numpy().squeeze()

    f, ax = plt.subplots(3, 1, figsize=(12, 20))
    f.suptitle('test_resize_pad', fontsize=16)

    ax[0].imshow(mask)
    ax[1].imshow(mask_pad)
    ax[2].imshow(mask_back)

    f.tight_layout()
    f.subplots_adjust(top=0.88)
    f.show()

    assert np.allclose(mask, mask_back)

def test_resize_pad():
    mask = AF.to_float(cv2.imread('data/train/masks/0b73b427d1.png', cv2.IMREAD_GRAYSCALE))

    processor = D.DatasetResizePad(202, 224, border_mode=cv2.BORDER_REFLECT101, interpolation=cv2.INTER_LINEAR)
    mask_pad = processor.forward(image=mask, mask=mask)['image']
    mask_torch = torch.from_numpy(mask_pad).unsqueeze(0).unsqueeze(0).float()
    mask_back = processor.backward(mask_torch).numpy().squeeze()

    f, ax = plt.subplots(3, 1, figsize=(12, 20))
    f.suptitle('test_resize_pad', fontsize=16)

    ax[0].imshow(mask)
    ax[1].imshow(mask_pad)
    ax[2].imshow(mask_back)

    f.tight_layout()
    f.subplots_adjust(top=0.88)
    f.show()

    assert np.allclose(mask, mask_back)
