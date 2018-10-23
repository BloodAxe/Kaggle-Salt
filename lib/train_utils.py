import argparse
import glob
import os
import random
from functools import partial
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from lib.adamw import AdamW
from lib.loss import BCEAndLovaszLoss, BCELoss, FocalLoss, BCEAndJaccardLoss, FocalAndJaccardLoss, LovaszHingeLoss
from lib.lr_schedules import OnceCycleLR
from models.deep_residual_unet import DeepResidualUNet
from models.deep_residual_unet_se import DeepResidualUNetSE
from models.frog_net import FrogNet
from models.modules.abn import ACT_LEAKY_RELU, ACT_RELU
from models.ternaus_v2 import TernausNetV2
from models.ternaus_v3 import TernausNetV3
from models.ternaus_v3_hyp import TernausNetV3Hyp
from models.unet_resnet34 import UNetResNet
from models.unet_resnext50 import UNetSEResnext50
from models.vanilla_unet import UNet

from models.wrn_oc import TernausNetOC
from pytorch_zoo.unet import dpn_unet


def auto_file(filename, where='.') -> str:
    """
    Helper function to find a unique filename in subdirectory without specifying fill path to it
    :param where:
    :param filename:
    :return:
    """

    if os.path.isabs(filename):
        return filename

    prob = os.path.join(where, filename)
    if os.path.exists(prob) and os.path.isfile(prob):
        return prob

    files = list(glob.iglob(os.path.join(where, '**', filename), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError('Given file could not be found with recursive search:' + filename)

    if len(files) > 1:
        raise FileNotFoundError('More than one file matches given filename. Please specify it explicitly' + filename)

    return files[0]


def set_manual_seed(seed):
    """ If manual seed is not specified, choose a random one and communicate it to the user.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))


def should_quit(experiment_dir: str):
    # Magic check to stop training by placing a file STOP in experiment dir
    if os.path.exists(os.path.join(experiment_dir, 'STOP')):
        os.remove(os.path.join(experiment_dir, 'STOP'))
        return True


def log_learning_rate(writer, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar('train/lr/%d' % i, param_group['lr'], global_step=epoch)


def is_better(score, best_score, mode):
    if mode == 'max':
        return score > best_score
    if mode == 'min':
        return score < best_score

    raise ValueError(mode)


def save_checkpoint(snapshot_file: str, model: torch.nn.Module, epoch: int, train_history: pd.DataFrame, optimizer=None, multi_gpu=False, **kwargs):
    dto = {
        'model': model.module.state_dict() if multi_gpu else model.state_dict(),
        'epoch': epoch,
        'train_history': train_history.to_dict(),
        'torch_rng': torch.get_rng_state(),
        'torch_rng_cuda': torch.cuda.get_rng_state_all(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
        'optimizer': optimizer
    }
    dto.update(**kwargs)

    torch.save(dto, snapshot_file)


def restore_checkpoint(snapshot_file: str, model: torch.nn.Module, optimizer: Optional[Optimizer] = None):
    checkpoint = torch.load(snapshot_file)
    start_epoch = checkpoint['epoch'] + 1
    metric_score = checkpoint.get('metric_score', None)

    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError:
        model.load_state_dict(checkpoint['model'], strict=False)
        print('Loaded model with strict=False mode')

    try:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        print('Optimizer state not loaded')

    try:
        torch_rng = checkpoint['torch_rng']
        torch.set_rng_state(torch_rng)
        print('Set torch rng state')

        torch_rng_cuda = checkpoint['torch_rng_cuda']
        torch.cuda.set_rng_state(torch_rng_cuda)
        print('Set torch rng cuda state')
    except:
        pass

    try:
        numpy_rng = checkpoint['numpy_rng']
        np.random.set_state(numpy_rng)
        print('Set numpy rng state')
    except:
        pass

    try:
        python_rng = checkpoint['python_rng']
        random.setstate(python_rng)
        print('Set python rng state')
    except:
        pass

    train_history = pd.DataFrame.from_dict(checkpoint['train_history'])

    return start_epoch, train_history, metric_score


def get_random_name():
    from lib import namesgenerator as ng
    return ng.get_random_name()


def get_lr_scheduler(name, optimizer, epochs, step_size=20, gamma=0.5):
    if name == 'step':
        return StepLR(optimizer, step_size=100, gamma=gamma)

    if name == 'auto':
        return ReduceLROnPlateau(optimizer, mode='max', patience=step_size, factor=0.5, min_lr=1e-5)

    if name == '1cycle':
        return OnceCycleLR(optimizer, epochs, min_lr_factor=0.05)

    raise ValueError(f'Unsupported LR scheduler name {name}')


def get_border_mode(border_mode):
    if border_mode == 'constant':
        return cv2.BORDER_CONSTANT

    if border_mode == 'reflect':
        return cv2.BORDER_REFLECT101

    if border_mode == 'replicate':
        return cv2.BORDER_REPLICATE

    if isinstance(border_mode, str):
        raise ValueError('Unsupported border mode')

    return int(border_mode)


def get_abn_block(abn):
    if abn == 'default':
        from models.modules.abn import ABN
        return ABN

    if abn == 'inplace':
        from models.modules.abn_inplace import InPlaceABN
        return InPlaceABN

    if abn == 'inplace_sync':
        from models.modules.abn_inplace import InPlaceABNSync
        return InPlaceABNSync

    raise ValueError('Parameter abn can be one of: default, inplace, inplace_sync')


def get_loss(loss_name):
    if loss_name == 'bce':
        return BCELoss(per_image=True)

    if loss_name == 'focal':
        return FocalLoss(gamma=1, per_image=True)

    if loss_name == 'lovasz':
        return LovaszHingeLoss(per_image=True)

    if loss_name == 'focal_jaccard':
        return FocalAndJaccardLoss(focal_weight=1, jaccard_weight=0.5, per_image=True)

    if loss_name == 'bce_jaccard':
        return BCEAndJaccardLoss(bce_weight=1, jaccard_weight=0.5, per_image=True)

    if loss_name == 'bce_lovasz':
        return BCEAndLovaszLoss(bce_weight=0.1, lovasz_weight=1, per_image=True)

    raise ValueError(loss_name)


def get_optimizer(optimizer_name, model_parameters, learning_rate, **kwargs):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, **kwargs)

    if optimizer_name == 'rms':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate, **kwargs)

    if optimizer_name == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate, amsgrad=True, **kwargs)

    if optimizer_name == 'adamw':
        return AdamW(model_parameters, lr=learning_rate, **kwargs)

    raise ValueError(optimizer_name)


MODEL_ZOO = {
    'dpn': dpn_unet,
    'ternaus_v2': TernausNetV2,
    'ternaus_v3': partial(TernausNetV3, activation=ACT_RELU),
    'ternaus_oc': TernausNetOC
}


def get_model(model_name, num_classes, num_channels, abn='default', **kwargs):
    model_name = str.lower(model_name)
    abn_block = get_abn_block(abn)
    model_builder = MODEL_ZOO[model_name]
    model = model_builder(num_classes=num_classes, num_channels=num_channels, abn_block=abn_block, **kwargs)
    return model


def logit_to_prob(logits: Tensor, criterion):
    """
    Helper function to get probabilities from prediction output.
    This function respects the loss function (e.g lovasz that is hingle loss, and applying sigmoid is not right)
    :param logits:
    :param criterion:
    :return:
    """
    logits = logits.detach()

    if logits.size(1) == 2:
        # return logits.softmax()[:, 0:1, ...]  # Take probabilities of the first class (salt)
        return logits.argmax(dim=1).float()
    elif logits.size(1) == 1:
        if isinstance(criterion, LovaszHingeLoss) or criterion == 'lovasz':
            return (logits > 0).float()
        else:
            return logits.sigmoid()

    raise ValueError('Unsupported number of channels')


def get_argparser():
    """
    Gets the default argument parser
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Run on extremely reduced dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split-seed', type=int, default=1234, help='Random seed for train/val split')
    parser.add_argument('-a', '--augmentation', default='none', help='Augmentation used. Possible values: hard, medium, light, safe, none')
    parser.add_argument('-abn', '--abn', default='default', help='Use of activate + batch_norm block. Values: default, inplace, inplace_sync')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-bm', '--border-mode', type=str, default='reflect', help='Border mode. Either constant|reflect')
    parser.add_argument('-d', '--dataset', type=str, default='image_only', help='image_only, image_depth, image_cumsum, image_depth_cumsum')
    parser.add_argument('-de', '--drop-empty', action='store_true')
    parser.add_argument('-df', '--drop-few', default=None, type=int)
    parser.add_argument('-dv', '--drop-vstrips', action='store_true')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='Epoch to run')
    parser.add_argument('-es', '--early-stopping', type=int, default=None, help='Maximum number of epochs without improvement')
    parser.add_argument('-f', '--fold', default=None, type=int, help='Fold to train')
    parser.add_argument('-fe', '--freeze-encoder', type=int, default=0, help='Freeze encoder parameters for N epochs')
    parser.add_argument('-fm', '--fix-masks', action='store_true')
    parser.add_argument('-ft', '--fine-tune', action='store_true')
    parser.add_argument('-l', '--loss', type=str, default='bce', help='Loss (lovasz, bce_iou)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('-lrs', '--lr-scheduler', default=None, help='LR scheduler')
    parser.add_argument('-m', '--model', required=True, type=str, help='Name of the model')
    parser.add_argument('-multi-gpu', '--multi-gpu', action='store_true')
    parser.add_argument('-nc', '--num-classes', default=1, type=int, help='Run on extremely reduced dataset')
    parser.add_argument('-nd', '--no-dropout', action='store_true', help='Disable dropout (if model has it)')
    parser.add_argument('-npt', '--no-pretrain', action='store_true', help='Disables use of pretrain weights for encoders')
    parser.add_argument('-o', '--optimizer', default='Adam', help='Name of the optimizer')
    parser.add_argument('-p', '--prepare', type=str, default='128', help='Possible tile preparations (128, 128pad, 224, 224pad, 256, 256pad)')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Checkpoint filename to resume')
    parser.add_argument('-re', '--restart-every', type=int, default=-1, help='Restart optimizer every N epochs')
    parser.add_argument('-s', '--stratify', default=None, type=str, help='Stratification class. One of: coverage, depth')
    parser.add_argument('-tm', '--target-metric', type=str, default='val_lb', help='Target metric to use for storing snapshots')
    parser.add_argument('-w', '--workers', default=0, type=int, help='Num workers')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0, help='L2 weight decay')
    parser.add_argument('-x', '--experiment', type=str, help='Name of the experiment')

    return parser
