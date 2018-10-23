import torch
from tensorboardX import SummaryWriter
import numpy as np

from lib import dataset as D
from lib import metrics as M

#https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044#379515
# def castF(x):
#     return K.cast(x, K.floatx())
#
# def castB(x):
#     return K.cast(x, bool)
#
# def iou_loss_core(true,pred):
#     intersection = true * pred
#     notTrue = 1 - true
#     union = true + (notTrue * pred)
#
#     return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())
# #
# def competitionMetric2(true, pred): #any shape can go
#
#     tresholds = [0.5 + (i*.05)  for i in range(10)]
#
#     #flattened images (batch, pixels)
#     true = K.batch_flatten(true)
#     pred = K.batch_flatten(pred)
#     pred = castF(K.greater(pred, 0.5))
#
#     #total white pixels - (batch,)
#     trueSum = K.sum(true, axis=-1)
#     predSum = K.sum(pred, axis=-1)
#
#     #has mask or not per image - (batch,)
#     true1 = castF(K.greater(trueSum, 1))
#     pred1 = castF(K.greater(predSum, 1))
#
#     #to get images that have mask in both true and pred
#     truePositiveMask = castB(true1 * pred1)
#
#     #separating only the possible true positives to check iou
#     testTrue = tf.boolean_mask(true, truePositiveMask)
#     testPred = tf.boolean_mask(pred, truePositiveMask)
#
#     #getting iou and threshold comparisons
#     iou = iou_loss_core(testTrue,testPred)
#     truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]
#
#     #mean of thressholds for true positives and total sum
#     truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
#     truePositives = K.sum(truePositives)
#
#     #to get images that don't have mask in both true and pred
#     trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
#     trueNegatives = K.sum(trueNegatives)
#
#     return (truePositives + trueNegatives) / castF(K.shape(true)[0])
from lib.metrics import do_kaggle_metric


def test_kaggle_metric():
    ids = D.all_train_ids()
    y_true = D.read_train_masks(ids)
    # y_pred = np.load('experiments/Sep14_18_14_ternaus_netv3_naughty_roentgen/Sep14_18_14_ternaus_netv3_naughty_roentgen_best_lb.pth_train_predictions.npz')
    # y_pred = np.array([y_pred[x] for x in ids])
    # y_pred = y_true.copy()
    # print(y_pred.min(), y_pred.max())

    # print(np.count_nonzero(y_pred > 0), np.count_nonzero(y_true))
    # print(np.sum(y_true == (y_pred > 0)) / float(np.prod(y_true.shape)))


    precision, result, threshold = do_kaggle_metric(y_true, y_true, 0.5)
    print(np.mean(precision))
    # map_mine = M.precision_at(y_pred, y_true, 0.5)
    # map_heng = np.mean(precision)

    # print(map_mine)
    # print(map_heng)



def test_pixel_acc():
    ids = D.all_train_ids()
    y_true = D.read_train_masks(ids)
    y_pred = np.load('experiments/Sep14_18_14_ternaus_netv3_naughty_roentgen/Sep14_18_14_ternaus_netv3_naughty_roentgen_best_lb.pth_train_predictions.npz')
    y_pred = np.array([y_pred[x] for x in ids])

    acc = M.PixelAccuracy()
    acc.update(torch.from_numpy(y_pred), torch.from_numpy(y_true))
    print(acc.value())
