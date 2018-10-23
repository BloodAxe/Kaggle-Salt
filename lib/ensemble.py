import json
import os

import cv2

import numpy as np
from tqdm import tqdm

from lib import dataset as D
from lib.common import find_in_dir
from lib.metrics import threshold_mining
from lib.postprocess import morphology_postprocess
from lib.submission import create_submission
from lib.train_utils import auto_file


class ArithmeticMean:
    def __init__(self):
        self.accumulator = None
        self.len = 0

    def update(self, predict, weight=1.0):
        predict = np.clip(predict, 0, 1)

        if self.accumulator is None:
            self.accumulator = np.zeros(predict.shape, dtype=np.float32)

        self.accumulator += predict.astype(np.float32, copy=False) * weight
        self.len += 1

    def value(self):
        return np.divide(self.accumulator, self.len)


class GeometricMean:
    def __init__(self):
        self.accumulator = None
        self.len = 0

    def update(self, predict, weight=1.0):
        predict = np.clip(predict, 0, 1)
        if self.accumulator is None:
            self.accumulator = np.ones(predict.shape, dtype=np.float64)

        self.accumulator *= predict.astype(np.float64) * weight
        self.len += 1

    def value(self):
        return np.float_power(self.accumulator, 1. / self.len)


class HarmonicMean:
    def __init__(self):
        self.accumulator = None
        self.len = 0

    def update(self, predict, weight=1.0):
        eps = 1e-5
        predict = np.clip(predict, 0 + eps, 1 - eps)

        if self.accumulator is None:
            self.accumulator = np.ones(predict.shape, dtype=np.float64)

        self.accumulator += 1. / predict.astype(np.float64)
        self.len += 1

    def value(self):
        return np.divide(self.len, self.accumulator)


def extract_oof_predictions(model) -> dict:
    test_predictions = auto_file(f'{model}_test_predictions.npz')
    train_predictions = auto_file(f'{model}_train_predictions.npz')

    experiment_dir = os.path.dirname(test_predictions)

    json_config = [fname for fname in sorted(os.listdir(experiment_dir)) if os.path.splitext(fname)[1] == '.json']
    json_config = auto_file(json_config[0])

    config = json.load(open(json_config))
    stratify = config['stratify']
    fold = config['fold']

    train_ids = D.all_train_ids()
    train_indexes, test_indexes = D.get_train_test_split_for_fold(stratify, fold, train_ids)
    train_predictions = np.load(train_predictions)

    valid_ids = train_ids[test_indexes]
    valid_predictions = np.array([train_predictions[id] for id in valid_ids])
    oof_predictions = dict(zip(valid_ids, valid_predictions))

    np.savez_compressed(os.path.join(experiment_dir, f'{model}_oof_predictions.npz'), **oof_predictions)
    return oof_predictions


def merge_oof(prediction_files, ids) -> np.ndarray:
    all_predictions = {}

    for pred_file in prediction_files:
        data_dict = np.load(pred_file)

        for id, mask in data_dict.items():
            if id in all_predictions:
                raise ValueError(f"Predictions has overlapping id {id}")
            all_predictions[id] = mask

    if all_predictions.keys() != set(ids):
        raise ValueError("Some ids missing in OOF predictions ")

    masks = [all_predictions[id] for id in ids]
    return np.array(masks)


def ensemble(prediction_files, ids, weights=None, averaging=ArithmeticMean) -> np.ndarray:
    n_items = len(prediction_files)
    if weights is None:
        weights = np.ones(n_items)

    acc = averaging()
    for pred_file, w in tqdm(zip(prediction_files, weights), desc='Ensembling', total=len(prediction_files)):
        data_dict = np.load(pred_file)
        predictions = np.array([data_dict[id] for id in ids])
        acc.update(predictions, w)

    ensemble_predictions = acc.value()
    return ensemble_predictions


def make_cv_submit(inputs, prefix, output_dir='submits'):
    os.makedirs(output_dir, exist_ok=True)

    test_predictions = [auto_file(f'{model}_test_predictions.npz') for model in inputs]
    oof_predictions = [auto_file(f'{model}_oof_predictions.npz') for model in inputs]

    train_ids = D.all_train_ids()
    true_masks = D.read_train_masks(train_ids)
    test_ids = D.all_test_ids()

    pred_masks = merge_oof(oof_predictions, train_ids)
    threshold, lb_score = threshold_mining(pred_masks, true_masks, min_threshold=0.1, max_threshold=0.9, step=0.001)

    i = np.argmax(lb_score)
    threshold, lb_score = float(threshold[i]), float(lb_score[i])
    print('Threshold', threshold, 'CV score', lb_score)

    # Arithmetic
    ensembled_test_pred = ensemble(test_predictions, test_ids, averaging=ArithmeticMean)
    ensembled_test_pred = ensembled_test_pred > threshold

    submit_file = f'{prefix}_a_mean_CV_{lb_score:.4f}_TH{threshold:.4f}.csv.gz'
    create_submission(test_ids, ensembled_test_pred).to_csv(os.path.join(output_dir, submit_file), compression='gzip', index=False)
    print('Saved submission', submit_file)

    postprocess = morphology_postprocess
    if postprocess is not None:
        final_masks = []
        for image, mask in zip(D.read_test_images(test_ids), ensembled_test_pred):
            mask = postprocess(image, mask)
            final_masks.append(mask)
        test_predictions = np.array(final_masks)

        submit_file = f'{prefix}_a_mean_PPC_CV_{lb_score:.4f}_TH{threshold:.4f}.csv.gz'
        create_submission(test_ids, test_predictions).to_csv(os.path.join(output_dir, submit_file), compression='gzip', index=False)
        print('Saved submission', submit_file)

    # Geometric
    # ensembled_test_pred = ensemble(test_inputs, test_ids, averaging=GeometricMean)
    # ensembled_test_pred = ensembled_test_pred > threshold
    #
    # submit_file = f'{prefix}_g_mean_CV_{lb_score:.4f}_TH{threshold:.4f}.csv.gz'
    # create_submission(test_ids, ensembled_test_pred).to_csv(os.path.join(output_dir, submit_file), compression='gzip', index=False)
    # print('Saved submission', submit_file)

    # Harmonic
    # ensembled_test_pred = ensemble(test_inputs, test_ids, averaging=HarmonicMean)
    # ensembled_test_pred = ensembled_test_pred > threshold
    #
    # submit_file = f'{prefix}_h_mean_CV_{lb_score:.4f}_TH{threshold:.4f}.csv.gz'
    # create_submission(test_ids, ensembled_test_pred).to_csv(os.path.join(output_dir, submit_file), compression='gzip', index=False)
    # print('Saved submission', submit_file)
