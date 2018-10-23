import json
import math
import os
import warnings

import cv2
import numpy as np
import torch
import numpy as np
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib import dataset as D
from lib.metrics import threshold_mining
from lib.postprocess import morphology_postprocess, zero_masks_inplace
from lib.submission import create_submission
from lib.train_utils import auto_file, get_model


def sigmoid(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.sigmoid().numpy()
    return x


def get_test_dataset(dataset: str, prepare, test_or_train='test'):
    if test_or_train == 'test':
        ids = D.all_test_ids()
        images = D.read_test_images(ids)
    else:
        ids = D.all_train_ids()
        images = D.read_train_images(ids)

    depths = D.read_depths(ids)

    use_cumsum = (dataset == 'image_depth_cumsum' or dataset == 'image_cumsum')
    use_depth = (dataset == 'image_depth' or dataset == 'image_depth_cumsum')

    dataset = D.ImageAndMaskDataset(ids, images, None, depths,
                                    prepare_fn=prepare)
    return dataset


def predict_masks_auto(config: str, snapshot: str, test_or_train='test', device='cuda'):
    from lib import tta

    config = json.load(open(config))
    snapshot = torch.load(snapshot)

    prepare_fn = D.get_prepare_fn(config['prepare'], **config)
    dataset = get_test_dataset(dataset=config['dataset'],
                               prepare=prepare_fn,
                               test_or_train=test_or_train)

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
        for batch in tqdm(loader, total=len(loader), desc=f'Predicting on {test_or_train}'):
            image_ids = batch['id']

            # TTA
            batch['image'], batch['depth'] = tta.tta_fliplr_aug(batch['image'], batch['depth'])

            # Move all data to GPU
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.cuda(non_blocking=True)

            output = model(batch)
            masks = output['mask']
            mask_class = output['class']

            masks = dataset.resize_fn.backward(masks)
            masks = np.array([np.squeeze(x) for x in masks.cpu().numpy()])
            masks, mask_class = tta.tta_fliplr_deaug(masks, mask_class)

            masks = sigmoid(masks)
            mask_class = sigmoid(mask_class)

            masks = zero_masks_inplace(masks, mask_class < 0.15)

            for mask, image_id in zip(masks, image_ids):
                mask = cv2.resize(mask, (D.ORIGINAL_SIZE, D.ORIGINAL_SIZE), interpolation=cv2.INTER_LANCZOS4)
                pred_masks.append(mask)

    del model, loader

    pred_masks = np.array(pred_masks)

    return pred_masks, dataset


def convert_predictions_to_images(predictions_files, output_directory):
    """Converts a predictions saved as numpy array to folder with PNG files"""
    os.makedirs(output_directory, exist_ok=True)
    predictions = np.load(predictions_files)
    for image_id, image in predictions.items():
        mask_fname = os.path.join(output_directory, f'{image_id}.png')
        image = np.clip(image, 0, 1)
        cv2.imwrite(mask_fname, (image * 255).astype(np.uint8))


def generate_model_submission(snapshot_name: str, config_name: str, postprocess=morphology_postprocess, mine_on_val=True, export_png=False):
    print('Generating model submission for session', snapshot_name)

    snapshot_basename = os.path.splitext(os.path.basename(snapshot_name))[0]
    config_file = auto_file(config_name)
    save_file = auto_file(snapshot_name)
    working_dir = os.path.dirname(config_file)

    # OOF
    # stratify = config['stratify']
    # fold = config['fold']
    #
    # train_ids = D.all_train_ids()
    # train_indexes, test_indexes = D.get_train_test_split_for_fold(stratify, fold, train_ids)
    # train_predictions = np.load(train_predictions)

    # Predictions for train dataset
    train_predictions, train_dataset = predict_masks_auto(config_file, save_file, test_or_train='train')
    train_predictions_file = os.path.join(working_dir, f'{snapshot_basename}_train_predictions.npz')
    np.savez_compressed(train_predictions_file, **dict((image_id, image) for image_id, image in zip(train_dataset.ids, train_predictions)))

    # Predictions for test dataset
    test_predictions, test_dataset = predict_masks_auto(config_file, save_file, test_or_train='test')
    test_predictions_file = os.path.join(working_dir, f'{snapshot_basename}_test_predictions.npz')
    np.savez_compressed(test_predictions_file, **dict((image_id, image) for image_id, image in zip(test_dataset.ids, test_predictions)))

    # Save prediction as unit8 masks
    if export_png:
        convert_predictions_to_images(train_predictions_file, os.path.join(working_dir, 'train_predictions'))
        convert_predictions_to_images(test_predictions_file, os.path.join(working_dir, 'test_predictions'))

    # Threshold mining
    if mine_on_val:
        config = json.load(open(config_file))
        valid_ids = np.array(config['valid_set'])
        valid_mask = D.get_selection_mask(train_dataset.ids, valid_ids)
        true_masks = D.read_train_masks(valid_ids)
        threshold, lb_score = threshold_mining(train_predictions[valid_mask], true_masks, min_threshold=0.15, max_threshold=0.85, step=0.005)
    else:
        true_masks = D.read_train_masks(train_dataset.ids)
        threshold, lb_score = threshold_mining(train_predictions, true_masks, min_threshold=0.15, max_threshold=0.85, step=0.005)

    i = np.argmax(lb_score)
    threshold, lb_score = float(threshold[i]), float(lb_score[i])

    suffix = '_mine_on_val' if mine_on_val else ''
    submit_file = os.path.join(working_dir, '{}_LB{:.4f}_TH{:.4f}{}.csv.gz'.format(snapshot_basename, lb_score, threshold, suffix))

    test_predictions = test_predictions > threshold

    if postprocess is not None:
        final_masks = []
        for image, mask in zip(D.read_test_images(test_dataset.ids), test_predictions):
            mask = postprocess(image, mask)
            final_masks.append(mask)
        test_predictions = np.array(final_masks)

    create_submission(test_dataset.ids, test_predictions).to_csv(submit_file, compression='gzip', index=False)
    print('Saved submission to ', working_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--snapshot', required=True, default=None, type=str, help='')
    parser.add_argument('-c', '--config', required=True, default=None, type=str, help='')
    parser.add_argument('-mov', '--mine-on-val', action='store_true')

    args = parser.parse_args()

    snapshot_file = auto_file(args.snapshot)
    config_file = auto_file(args.config)

    generate_model_submission(snapshot_file, config_file, mine_on_val=args.mine_on_val)


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
