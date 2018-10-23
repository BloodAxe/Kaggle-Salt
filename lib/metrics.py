import numpy as np
from tensorboardX import SummaryWriter
from torch import Tensor

MAP_THRESHOLDS = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], dtype=np.float32)


def do_kaggle_metric(predict, truth, threshold=0.5):
    if len(predict) != len(truth):
        raise ValueError()

    N = len(predict)
    smooth = 1e-3

    predict = predict.reshape(N, -1)
    truth = truth.reshape(N, -1)

    predict = predict > threshold
    truth = truth > 0.5

    intersection = truth & predict
    union = truth | predict
    iou = intersection.sum(1) / (union.sum(1) + smooth)

    result = []
    precision = []
    is_empty_truth = (truth.sum(1) == 0)
    is_empty_predict = (predict.sum(1) == 0)

    # threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in MAP_THRESHOLDS:
        # p = iou >= t

        tp = (~is_empty_truth) & (~is_empty_predict) & (iou > t)
        fp = (~is_empty_truth) & (~is_empty_predict) & (iou <= t)
        fn = (~is_empty_truth) & (is_empty_predict)
        fp_empty = (is_empty_truth) & (~is_empty_predict)
        tn_empty = (is_empty_truth) & (is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append(np.column_stack((tp, fp, fn, tn_empty, fp_empty)))
        precision.append(p)

    result = np.array(result).transpose(1, 2, 0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold


def threshold_mining(y_pred: np.ndarray, y_true: np.ndarray, min_threshold=0.3, max_threshold=0.7, step=0.01):
    """Computes optimal binarization threshold"""
    binarization_thresholds = np.arange(min_threshold, max_threshold, step, dtype=np.float32)
    scores = np.zeros_like(binarization_thresholds)

    for i, prob_threshold in enumerate(binarization_thresholds):
        precision, result, threshold = do_kaggle_metric(y_pred, y_true, prob_threshold)
        scores[i] = np.mean(precision)

    return binarization_thresholds, scores


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def extend(self, losses):
        for loss_val in losses:
            self.update(loss_val)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '%.3f' % self.avg


class JaccardIndex:
    def __init__(self, threshold=None):
        self.threshold = threshold
        self.scores_per_image = []

    def reset(self):
        self.scores_per_image = []

    def update(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_true.size(0)

        y_pred = y_pred.detach().view(batch_size, -1)
        y_true = y_true.detach().view(batch_size, -1)

        if self.threshold is not None:
            y_pred = y_pred > float(self.threshold)

        y_true = y_true.float()
        y_pred = y_pred.float()

        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)
        iou = intersection / (union - intersection + 1e-7)

        iou = iou[y_true.sum(dim=1) > 0]  # IoU defined only for non-empty masks
        self.scores_per_image.extend(iou.cpu().numpy())

    def __str__(self):
        return '%.4f' % self.value()

    def value(self):
        if len(self.scores_per_image) == 0:
            return 0
        return np.mean(self.scores_per_image)

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        saver.add_scalar(prefix + '/value', self.value(), step)
        saver.add_histogram(prefix + '/histogram', np.array(self.scores_per_image), step)


class PixelAccuracy:
    def __init__(self, threshold=0.5):
        self.scores_per_image = []
        self.threshold = threshold

    def reset(self):
        self.scores_per_image = []

    def update(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_true.size(0)

        y_pred = y_pred.detach().view(batch_size, -1) > self.threshold
        y_true = y_true.detach().view(batch_size, -1) > 0.5

        correct = (y_true == y_pred).float()
        accuracy = correct.sum(1) / y_true.size(1)

        self.scores_per_image.extend(accuracy.cpu().numpy())

    def __str__(self):
        return '%.4f' % self.value()

    def value(self):
        return np.mean(self.scores_per_image)

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        saver.add_scalar(prefix + '/value', self.value(), step)
        saver.add_histogram(prefix + '/histogram', np.array(self.scores_per_image), step)


class IncorrectClassCounter:
    def __init__(self):
        self.labels_true = []
        self.labels_pred = []

    def reset(self):
        self.labels_true = []
        self.labels_pred = []

    def update(self, y_pred, y_true):
        self.labels_pred.extend(y_pred.detach().cpu().numpy())
        self.labels_true.extend(y_true.detach().cpu().numpy())

    def value(self):
        labels_true = np.array(self.labels_true)
        labels_pred = np.array(self.labels_pred)
        mismatches = (labels_true != labels_pred).sum()
        return mismatches

    def __str__(self):
        return '%.4f' % self.value()

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        saver.add_scalar(prefix, self.value(), step)
