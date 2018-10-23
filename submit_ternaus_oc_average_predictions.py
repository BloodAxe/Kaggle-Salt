import os

import numpy as np
from tqdm import tqdm

from lib.ensemble import extract_oof_predictions, make_cv_submit
from lib.train_utils import auto_file


def average_predictions(predictions, dst_file=None):
    predictions = [auto_file(p) for p in predictions]
    dir = os.path.dirname(predictions[0])

    scale = float(1. / len(predictions))

    avg = {}

    pred = np.load(predictions[0])
    ids = pred.keys()
    for id in ids:
        avg[id] = pred[id] * scale

    for pred_file in predictions[1:]:
        pred = np.load(pred_file)

        for id in ids:
            avg[id] += pred[id] * scale

    if dst_file is not None:
        np.savez_compressed(os.path.join(dir, dst_file), **avg)
        print(f'Saved {dst_file}')
    return avg


# Fold 0
train0 = [
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_train_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_0_train_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_1_train_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_2_train_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_3_train_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_4_train_predictions.npz',
]

test0 = [
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_test_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_0_test_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_1_test_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_2_test_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_3_test_predictions.npz',
    'ternaus_oc_128_light_jolly_gates_fold_salt_0_val_lb_snapshot_4_test_predictions.npz',
]

# Fold 1
train1 = [
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_train_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_0_train_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_1_train_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_2_train_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_3_train_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_4_train_predictions.npz',
]

test1 = [
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_test_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_0_test_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_1_test_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_2_test_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_3_test_predictions.npz',
    'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_val_lb_snapshot_4_test_predictions.npz',
]

# Fold 2
train2 = [
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_train_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_0_train_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_1_train_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_2_train_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_3_train_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_4_train_predictions.npz',

]

test2 = [
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_test_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_0_test_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_1_test_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_2_test_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_3_test_predictions.npz',
    'ternaus_oc_128_light_reverent_swartz_fold_salt_2_val_lb_snapshot_4_test_predictions.npz',
]

# Fold 3
train3 = [
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_train_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_0_train_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_1_train_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_2_train_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_3_train_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_4_train_predictions.npz',
]

test3 = [
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_test_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_0_test_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_1_test_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_2_test_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_3_test_predictions.npz',
    'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_val_lb_snapshot_4_test_predictions.npz',
]

# Fold 4
train4 = [
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_train_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_0_train_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_1_train_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_2_train_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_3_train_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_4_train_predictions.npz',
]

test4 = [
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_test_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_0_test_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_1_test_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_2_test_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_3_test_predictions.npz',
    'ternaus_oc_128_light_serene_perlman_fold_salt_4_val_lb_snapshot_4_test_predictions.npz',
]


if __name__ == '__main__':
    # average_predictions(train0,
    #                     'ternaus_oc_128_light_jolly_gates_fold_salt_0_avg_train_predictions.npz')
    #
    # average_predictions(test0,
    #                     'ternaus_oc_128_light_jolly_gates_fold_salt_0_avg_test_predictions.npz')
    #
    # average_predictions(train1,
    #                     'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_avg_train_predictions.npz')
    #
    # average_predictions(test1,
    #                     'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_avg_test_predictions.npz')
    #
    # average_predictions(train2,
    #                     'ternaus_oc_128_light_reverent_swartz_fold_salt_2_avg_train_predictions.npz')
    #
    # average_predictions(test2,
    #                     'ternaus_oc_128_light_reverent_swartz_fold_salt_2_avg_test_predictions.npz')
    #
    #
    # average_predictions(train3,
    #                     'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_avg_train_predictions.npz')
    #
    # average_predictions(test3,
    #                     'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_avg_test_predictions.npz')

    # average_predictions(train4,
    #                     'ternaus_oc_128_light_serene_perlman_fold_salt_4_avg_train_predictions.npz')
    #
    # average_predictions(test4,
    #                     'ternaus_oc_128_light_serene_perlman_fold_salt_4_avg_test_predictions.npz')

    # for input in tqdm(inputs, desc='Extracting OOF', total=len(inputs)):
    #     extract_oof_predictions(input)

    #
    inputs = [
        'ternaus_oc_128_light_jolly_gates_fold_salt_0_avg',
        'ternaus_oc_128_light_lucid_visvesvaraya_fold_salt_1_avg',
        'ternaus_oc_128_light_reverent_swartz_fold_salt_2_avg',
        'ternaus_oc_128_light_clever_dijkstra_fold_salt_3_avg',
        'ternaus_oc_128_light_serene_perlman_fold_salt_4_avg',
    ]

    make_cv_submit(inputs, 'ternaus_oc_128_light')
