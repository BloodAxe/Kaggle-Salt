import matplotlib as mpl

mpl.use('module://backend_interagg')

import json
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.metrics import do_kaggle_metric, threshold_mining
from lib.postprocess import zero_masks_inplace
from test import get_test_dataset, sigmoid

import matplotlib.pyplot as plt

import cv2
import numpy as np

from lib.submission import decode_submission, create_submission
from lib.train_utils import auto_file, get_model
from lib import dataset as D


def test_rle_encode_decode():
    train_pred = auto_file('Oct10_20_28_dpn_128_medium_wonderful_goldberg_val_lb.pth_train_predictions.npz')
    train_pred = np.load(train_pred)

    train_ids = D.all_train_ids()
    true_masks = D.read_train_masks(train_ids)
    pred_masks = np.array([train_pred[id] for id in train_ids])
    pred_masks = (pred_masks > 0.45).astype(np.uint8)


    submit = create_submission(train_ids, pred_masks)
    submit.to_csv('test_rle_encode_decode.csv.gz', compression='gzip', index=False)

    decoded_ids, decoded_masks = decode_submission('test_rle_encode_decode.csv.gz')
    decoded_masks = dict(zip(decoded_ids, decoded_masks))
    assert set(decoded_ids) == set(train_ids)

    decoded_masks = np.array([decoded_masks[id] for id in train_ids])

    p1, r1, _ = do_kaggle_metric(pred_masks, true_masks)
    p2, r2, _ = do_kaggle_metric(decoded_masks, true_masks)

    assert np.array_equal(p1,p2)
    assert np.array_equal(r1,r2)
    print(np.mean(p1),np.mean(p2))


def test_prediction_pipeline_tta_pre():
    from lib import tta

    device = 'cuda'
    config = auto_file('infallible_lamport.json')
    snapshot = auto_file('Oct09_23_17_wider_unet_224pad_medium_infallible_lamport_val_lb.pth')

    config = json.load(open(config))
    snapshot = torch.load(snapshot)

    prepare_fn = D.get_prepare_fn(config['prepare'], **config)
    dataset = get_test_dataset(dataset=config['dataset'],
                               prepare=prepare_fn,
                               test_or_train='train')

    model = get_model(config['model'],
                      num_classes=config['num_classes'],
                      num_channels=dataset.channels(),
                      pretrained=False).to(device)

    if device == 'cpu':
        warnings.warn('Using CPU for prediction. It will be SLOW.')

    model.load_state_dict(snapshot['model'])
    model.eval()

    batch_size = config['batch_size']
    collate_fn = tta.tta_fliplr_collate
    batch_size = max(1, batch_size // 2)

    pred_masks = []
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)
        for images, image_ids in tqdm(loader, total=len(loader), desc=f'Predicting'):
            images = images.to(device, non_blocking=True)

            output = model(images)
            is_raw_mask = isinstance(output, torch.Tensor)
            is_mask_and_class = isinstance(output, tuple) and len(output) == 2

            if is_raw_mask:
                masks = output
            elif is_mask_and_class:
                masks, presence = output
            else:
                raise RuntimeError('Unknown output type')

            masks = dataset.resize_fn.backward(masks)
            masks = np.array([np.squeeze(x) for x in masks.cpu().numpy()])
            masks = tta.tta_fliplr_deaug(masks)
            masks = sigmoid(masks)

            if is_mask_and_class:
                presence = presence.softmax(dim=1).cpu().numpy()
                presence = tta.average_classes(presence, 2)
                presence = np.argmax(presence, axis=1)
                masks = zero_masks_inplace(masks, presence == 0)

            for mask, image_id in zip(masks, image_ids):
                mask = cv2.resize(mask, (D.ORIGINAL_SIZE, D.ORIGINAL_SIZE), interpolation=cv2.INTER_LANCZOS4)
                pred_masks.append(mask)

    del model, loader

    pred_masks = np.array(pred_masks)
    true_masks = D.read_train_masks(dataset.ids)

    plt.figure()
    binarization_thresholds, scores = threshold_mining(pred_masks, true_masks, min_threshold=0, max_threshold=1)
    plt.plot(binarization_thresholds, scores)
    plt.title("test_prediction_pipeline_tta_pre")
    plt.show()
    return pred_masks, dataset


def test_prediction_pipeline_tta_post():
    from lib import tta

    device = 'cuda'
    config = auto_file('infallible_lamport.json')
    snapshot = auto_file('Oct09_23_17_wider_unet_224pad_medium_infallible_lamport_val_lb.pth')

    config = json.load(open(config))
    snapshot = torch.load(snapshot)

    prepare_fn = D.get_prepare_fn(config['prepare'], **config)
    dataset = get_test_dataset(dataset=config['dataset'],
                               prepare=prepare_fn,
                               test_or_train='train')

    model = get_model(config['model'],
                      num_classes=config['num_classes'],
                      num_channels=dataset.channels(),
                      pretrained=False).to(device)

    if device == 'cpu':
        warnings.warn('Using CPU for prediction. It will be SLOW.')

    model.load_state_dict(snapshot['model'])
    model.eval()

    batch_size = config['batch_size']
    collate_fn = tta.tta_fliplr_collate
    batch_size = max(1, batch_size // 2)

    pred_masks = []
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)
        for images, image_ids in tqdm(loader, total=len(loader), desc=f'Predicting'):
            images = images.to(device, non_blocking=True)

            output = model(images)
            is_raw_mask = isinstance(output, torch.Tensor)
            is_mask_and_class = isinstance(output, tuple) and len(output) == 2

            if is_raw_mask:
                masks = output
            elif is_mask_and_class:
                masks, presence = output
            else:
                raise RuntimeError('Unknown output type')

            masks = dataset.resize_fn.backward(masks)
            masks = masks.sigmoid()
            masks = np.array([np.squeeze(x) for x in masks.cpu().numpy()])
            masks = tta.tta_fliplr_deaug(masks)
            # masks = sigmoid(masks)

            if is_mask_and_class:
                presence = presence.softmax(dim=1).cpu().numpy()
                presence = tta.average_classes(presence, 2)
                presence = np.argmax(presence, axis=1)
                masks = zero_masks_inplace(masks, presence == 0)

            for mask, image_id in zip(masks, image_ids):
                mask = cv2.resize(mask, (D.ORIGINAL_SIZE, D.ORIGINAL_SIZE), interpolation=cv2.INTER_LANCZOS4)
                pred_masks.append(mask)

    del model, loader

    pred_masks = np.array(pred_masks)
    true_masks = D.read_train_masks(dataset.ids)

    plt.figure()
    binarization_thresholds, scores = threshold_mining(pred_masks, true_masks, min_threshold=0, max_threshold=1)
    plt.plot(binarization_thresholds, scores)
    plt.title("test_prediction_pipeline_tta_pos")
    plt.show()
    return pred_masks, dataset


def test_prediction_pipeline_no_tta():
    from lib import tta

    device = 'cuda'
    config = auto_file('infallible_lamport.json')
    snapshot = auto_file('Oct09_23_17_wider_unet_224pad_medium_infallible_lamport_val_lb.pth')

    config = json.load(open(config))
    snapshot = torch.load(snapshot)

    prepare_fn = D.get_prepare_fn(config['prepare'], **config)
    dataset = get_test_dataset(dataset=config['dataset'],
                               prepare=prepare_fn,
                               test_or_train='train')

    model = get_model(config['model'],
                      num_classes=config['num_classes'],
                      num_channels=dataset.channels(),
                      pretrained=False).to(device)

    if device == 'cpu':
        warnings.warn('Using CPU for prediction. It will be SLOW.')

    model.load_state_dict(snapshot['model'])
    model.eval()

    batch_size = config['batch_size']

    pred_masks = []
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
        for images, image_ids in tqdm(loader, total=len(loader), desc=f'Predicting'):
            images = images.to(device, non_blocking=True)

            output = model(images)
            is_raw_mask = isinstance(output, torch.Tensor)
            is_mask_and_class = isinstance(output, tuple) and len(output) == 2

            if is_raw_mask:
                masks = output
            elif is_mask_and_class:
                masks, presence = output
            else:
                raise RuntimeError('Unknown output type')

            masks = dataset.resize_fn.backward(masks)
            masks = masks.sigmoid()
            masks = np.array([np.squeeze(x) for x in masks.cpu().numpy()])

            if is_mask_and_class:
                presence = presence.softmax(dim=1).cpu().numpy()
                presence = tta.average_classes(presence, 2)
                presence = np.argmax(presence, axis=1)
                masks = zero_masks_inplace(masks, presence == 0)

            for mask, image_id in zip(masks, image_ids):
                mask = cv2.resize(mask, (D.ORIGINAL_SIZE, D.ORIGINAL_SIZE), interpolation=cv2.INTER_LANCZOS4)
                pred_masks.append(mask)

    del model, loader

    pred_masks = np.array(pred_masks)
    true_masks = D.read_train_masks(dataset.ids)

    plt.figure()
    binarization_thresholds, scores = threshold_mining(pred_masks, true_masks, min_threshold=0, max_threshold=1)
    plt.plot(binarization_thresholds, scores)
    plt.title("test_prediction_pipeline_no_tta")
    plt.show()
    return pred_masks, dataset
