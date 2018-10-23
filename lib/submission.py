from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.train_utils import auto_file


def rle_encode_fast(img, format=True):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_arr, h=101, w=101):
    if isinstance(rle_arr, str):
        rle_arr = [int(x) for x in str.split(rle_arr, ' ')]
    else:
        rle_arr = None

    indices = []
    if rle_arr is not None and len(rle_arr) > 0:
        for idx, cnt in zip(rle_arr[0::2], rle_arr[1::2]):
            indices.extend(list(range(idx - 1, idx + cnt - 1)))  # RLE is 1-based index
    mask = np.zeros(h * w, dtype=np.uint8)
    mask[indices] = 1
    return mask.reshape((w, h)).T


def _create_submission_proc(args):
    id, mask = args
    rle = rle_encode_fast(mask)
    return id, rle


def create_submission(ids: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    if predictions.dtype != np.uint8 and predictions.dtype != np.bool:
        raise ValueError()

    dict_data = {
        'id': [],
        'rle_mask': []
    }

    with Pool(8) as wp:
        for id, rle in tqdm(wp.imap_unordered(_create_submission_proc, zip(ids, predictions)), total=len(predictions)):
            dict_data['id'].append(id)
            dict_data['rle_mask'].append(rle)

    df = pd.DataFrame.from_dict(dict_data)
    df = df.sort_values(by='id')
    return df


def decode_submission(submission):
    if isinstance(submission, str):
        submission = pd.read_csv(auto_file(submission))

    submission = submission.sort_values('id')
    images = []
    ids = []

    for index, row in submission.iterrows():
        rle_mask = row['rle_mask']


        images.append(rle_decode(rle_mask))
        ids.append(row['id'])

    return ids, images
