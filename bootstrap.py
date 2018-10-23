from multiprocessing.pool import Pool

import cv2
import numpy as np
import lib.dataset as D
from tqdm import tqdm

from lib.common import find_in_dir


def _comute_mean_std_proc(image_fname):
    one_over_255 = float(1. / 255.)
    x = cv2.imread(image_fname) * one_over_255
    mean, stddev = cv2.meanStdDev(x)
    count = x.shape[0] * x.shape[1]
    return np.squeeze(mean), np.squeeze(stddev), count


def _parallel_mean_variance(avg_a, count_a, var_a, avg_b, count_b, var_b):
    delta = avg_b - avg_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)

    new_mean = (avg_a * count_a + avg_b * count_b) / (count_a + count_b)
    new_var = M2 / (count_a + count_b - 1)
    return new_mean, new_var


def compute_mean_std(dataset):
    """
    https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    """
    global_mean = np.zeros(3, dtype=np.float64)
    global_var = np.zeros(3, dtype=np.float64)
    global_count = 0

    n_items = len(dataset)

    with Pool(6) as wp:
        for local_mean, local_std, local_count in tqdm(wp.imap_unordered(_comute_mean_std_proc, dataset), total=n_items):
            local_var = local_std ** 2

            if global_count == 0:
                global_mean = local_mean
                global_var = local_var
                global_count = 0
            else:
                global_mean, global_var = _parallel_mean_variance(global_mean, global_count, global_var, local_mean, local_count, local_var)
                global_count += local_count

    return global_mean, np.sqrt(global_var)


def main():
    one_over_255 = float(1. / 255.)

    print(compute_mean_std(find_in_dir('data/train/images')))
    print(compute_mean_std(find_in_dir('data/test/images')))
    print(compute_mean_std(find_in_dir('data/train/images') + (find_in_dir('data/test/images'))))

    train = one_over_255 * D.read_train_images(D.all_train_ids())
    test = one_over_255 * D.read_test_images(D.all_test_ids())
    print(train.mean(), train.std())
    print(test.mean(), test.std())
    all = np.concatenate([train, test], axis=0)
    print(all.mean(), all.std())

    mean = train.mean()
    std = train.std()

    train -= mean
    train /= std
    print(train.mean(), train.std())


if __name__ == '__main__':
    main()
