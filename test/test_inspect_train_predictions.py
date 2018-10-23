import os

import matplotlib as mpl

mpl.use('module://backend_interagg')

import cv2

from lib.metrics import do_kaggle_metric
import numpy as np
import matplotlib.pyplot as plt

import lib.dataset as D
from lib.train_utils import auto_file
from test import threshold_mining, convert_predictions_to_images
import json


def test_inspect_train_predictions():
    train_ids = D.all_train_ids()
    train_images = D.read_train_images(train_ids)
    train_masks = D.read_train_masks(train_ids)
    print(train_ids.shape, train_images.shape, train_masks.shape)

    CONFIG = auto_file('wonderful_goldberg.json')
    WEIGHT_TRAIN = auto_file('Oct10_20_28_dpn_128_medium_wonderful_goldberg_val_lb.pth_train_predictions.npz')
    WEIGHT_TEST = auto_file('Oct10_20_28_dpn_128_medium_wonderful_goldberg_val_lb.pth_test_predictions.npz')

    convert_predictions_to_images(WEIGHT_TEST, os.path.join('test', 'test_predictions'))
    convert_predictions_to_images(WEIGHT_TRAIN, os.path.join('test', 'train_predictions'))

    train_predictions = auto_file(WEIGHT_TRAIN)
    train_predictions = np.load(train_predictions)

    # image = train_predictions['0aab0afa9c']

    train_predictions = np.array([train_predictions[id] for id in train_ids])
    print(train_predictions.shape)

    threshold, lb_score = threshold_mining(train_predictions, train_masks, min_threshold=0.15, max_threshold=0.85, step=0.005)

    plt.figure()
    plt.plot(threshold, lb_score)
    plt.tight_layout()

    i = np.argmax(lb_score)
    best_threshold, best_lb_score = float(threshold[i]), float(lb_score[i])
    print(best_threshold, best_lb_score)

    config_file = auto_file(CONFIG)

    config = json.load(open(config_file))
    valid_ids = np.array(config['valid_set'])
    valid_mask = D.get_selection_mask(train_ids, valid_ids)
    val_threshold, val_lb_score = threshold_mining(train_predictions[valid_mask], train_masks[valid_mask], min_threshold=0.15, max_threshold=0.85, step=0.005)

    plt.figure()
    plt.plot(val_threshold, val_lb_score)
    plt.tight_layout()
    plt.show()

    val_i = np.argmax(val_lb_score)
    val_th = val_threshold[val_i]
    print(val_threshold[val_i], val_lb_score[val_i])

    precision, result, threshold = do_kaggle_metric(train_predictions, train_masks, val_th)

    x = []
    y = []
    for prec, true_mask in zip(precision, train_masks):
        x.append(prec)
        y.append(cv2.countNonZero(true_mask))

    plt.figure()
    plt.scatter(x, y)
    plt.tight_layout()
    plt.show()

    # visualize_predictions(train_images[valid_mask],
    #                       train_predictions[valid_mask],
    #                       train_masks[valid_mask])
    # plt.show()
