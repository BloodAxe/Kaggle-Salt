import json
import os.path
from datetime import datetime

import albumentations as A
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from lib import dataset as D
from lib import train_utils as U
from lib.common import count_parameters, is_sorted
from lib.loss import CELoss
from lib.metrics import JaccardIndex, AverageMeter, PixelAccuracy, threshold_mining, do_kaggle_metric
from lib.train_utils import logit_to_prob
from test import generate_model_submission

tqdm.monitor_interval = 0  # Workaround for https://github.com/tqdm/tqdm/issues/481


def main():
    parser = U.get_argparser()
    args = parser.parse_args()
    U.set_manual_seed(args.seed)

    train_session_args = vars(args)
    train_session = U.get_random_name()
    current_time = datetime.now().strftime('%b%d_%H_%M')
    prefix = f'{current_time}_{args.model}_{args.prepare}_{args.augmentation}_{train_session}'
    if args.fold is not None:
        prefix += f'_fold_{args.stratify}_{args.fold}'

    log_dir = os.path.join('runs', prefix)
    exp_dir = os.path.join('experiments', args.model, args.prepare, args.augmentation, prefix)
    os.makedirs(exp_dir, exist_ok=True)

    train_ids = D.get_train_ids(drop_black=True, drop_vstrips=args.drop_vstrips, drop_empty=args.drop_empty, drop_few=args.drop_few, fast=args.fast)
    depths = D.read_depths(train_ids)
    images = D.read_train_images(train_ids)
    masks = D.read_train_masks(train_ids)

    if args.fix_masks:
        masks, changed_ids = D.fix_masks(masks, train_ids)
        with open(os.path.join(exp_dir, 'fixed_masks.txt'), 'w') as f:
            for sample_id in changed_ids:
                f.write(sample_id)
                f.write('\n')
        print(f'Fixed {len(changed_ids)} masks')

    if args.fold is not None:
        train_indexes, test_indexes = D.get_train_test_split_for_fold(args.stratify, args.fold, train_ids)
    else:
        train_indexes, test_indexes = train_test_split(np.arange(len(train_ids)), shuffle=False, random_state=args.split_seed, test_size=0.2)

    ids_train, ids_test = train_ids[train_indexes], train_ids[test_indexes]
    if not is_sorted(ids_train):
        raise RuntimeError("ids_train is not sorted")
    if not is_sorted(ids_test):
        raise RuntimeError("ids_test_sorted is not sorted")

    img_train, img_test = images[train_indexes], images[test_indexes]
    mask_train, mask_test = masks[train_indexes], masks[test_indexes]
    depth_train, depth_test = depths[train_indexes], depths[test_indexes]

    prepare_fn = D.get_prepare_fn(args.prepare, **train_session_args)

    # This line valid if we apply prepare_fn first and then do augmentation
    target_size = prepare_fn.target_size if prepare_fn is not None else D.ORIGINAL_SIZE
    # target_size = D.ORIGINAL_SIZE

    build_augmentation_fn = D.AUGMENTATION_MODES[args.augmentation]
    aug = build_augmentation_fn(target_size, border_mode=args.border_mode)

    train_transform_list = []
    valid_transform_list = []
    if prepare_fn is not None:
        train_transform_list.append(prepare_fn.t_forward)
        valid_transform_list.append(prepare_fn.t_forward)

    train_transform_list.append(aug)

    trainset = D.ImageAndMaskDataset(ids_train, img_train, mask_train, depth_train,
                                     augment=A.Compose(train_transform_list))

    validset = D.ImageAndMaskDataset(ids_test, img_test, mask_test, depth_test,
                                     augment=A.Compose(valid_transform_list))

    trainloader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             num_workers=args.workers,
                             pin_memory=True,
                             drop_last=True,
                             shuffle=True)

    validloader = DataLoader(validset,
                             batch_size=args.batch_size,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    # Save train/val split for future use
    train_session_args.update({
        'train_set': list(ids_train),
        'valid_set': list(ids_test)
    })

    # Declare variables we will use during training
    start_epoch = 0
    train_history = pd.DataFrame()
    scheduler = None
    optimizer = None

    target_metric = args.target_metric
    target_metric_mode = 'max'
    best_metric_val = 0
    best_lb_checkpoint = os.path.join(exp_dir, f'{prefix}_{target_metric}.pth')

    model = U.get_model(args.model,
                        num_classes=args.num_classes,
                        num_channels=trainset.channels(),
                        abn=args.abn,
                        use_dropout=not args.no_dropout,
                        pretrained=not args.no_pretrain).cuda()

    print('Train set size :', len(trainloader), 'batch size', trainloader.batch_size)
    print('Valid set size :', len(validloader), 'batch size', validloader.batch_size)
    print('Tile transform :', prepare_fn if prepare_fn is not None else "None")
    print('Model          :', args.model, count_parameters(model))
    print('Augmentations  :', args.augmentation, args.border_mode)
    print('Input channels :', trainset.channels())
    print('Output classes :', args.num_classes)
    print('Criterion      :', args.loss),
    print('Optimizer      :', args.optimizer, args.learning_rate, args.weight_decay)
    print('Use of dropout :', not args.no_dropout)
    print('Train session  :', train_session)
    print('Freeze encoder :', args.freeze_encoder)
    print('Seed           :', args.seed, args.split_seed)
    print('Restart every  :', args.restart_every)
    print('Fold           :', args.fold, args.stratify)
    print('Fine-tune      :', args.fine_tune)
    print('ABN Mode       :', args.abn)
    print('Fix masks      :', args.fix_masks)

    if args.resume:
        fname = U.auto_file(args.resume)
        start_epoch, train_history, best_score = U.restore_checkpoint(fname, model)
        print(train_history)
        print('Resuming training from epoch', start_epoch, ' and score', best_score, args.resume)

    segmentation_loss = U.get_loss(args.loss)

    if args.fine_tune and args.freeze_encoder > 0:
        raise ValueError('Incompatible options --fune-tune and --freeze-encoder')

    writer = SummaryWriter(log_dir)
    writer.add_text('train/params', '```' + json.dumps(train_session_args, indent=2) + '```', 0)

    config_fname = os.path.join(exp_dir, f'{train_session}.json')
    with open(config_fname, 'w') as f:
        f.write(json.dumps(train_session_args, indent=2))

    # Start training loop
    no_improvement_epochs = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # On Epoch begin
        if U.should_quit(exp_dir) or (args.early_stopping is not None and no_improvement_epochs > args.early_stopping):
            break

        epochs_trained = epoch - start_epoch
        should_restart_optimizer = (args.restart_every > 0 and epochs_trained % args.restart_every == 0) or (epochs_trained == args.freeze_encoder) or optimizer is None

        if should_restart_optimizer:
            del optimizer
            if args.fine_tune:
                model.set_fine_tune(args.fine_tune)
            else:
                model.set_encoder_training_enabled(epochs_trained >= args.freeze_encoder)

            trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = U.get_optimizer(args.optimizer, trainable_parameters, args.learning_rate, weight_decay=args.weight_decay)

            print('Restarting optimizer state', epoch, count_parameters(model))

            if args.lr_scheduler:
                scheduler = U.get_lr_scheduler(args.lr_scheduler, optimizer, args.epochs)

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epochs_trained)

        U.log_learning_rate(writer, optimizer, epoch)

        # Epoch
        train_metrics = process_epoch(model, segmentation_loss, optimizer, trainloader, epoch, True, writer, mask_postprocess=prepare_fn.backward)
        valid_metrics = process_epoch(model, segmentation_loss, None, validloader, epoch, False, writer, mask_postprocess=prepare_fn.backward)

        all_metrics = {}
        all_metrics.update(train_metrics)
        all_metrics.update(valid_metrics)

        # On Epoch End
        summary = {
            'epoch': [int(epoch)],
            'lr': [float(optimizer.param_groups[0]['lr'])]
        }
        for k, v in all_metrics.items():
            summary[k] = [v]

        train_history = train_history.append(pd.DataFrame.from_dict(summary), ignore_index=True)
        print(epoch, summary)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(all_metrics[target_metric], epochs_trained)

        if U.is_better(all_metrics[target_metric], best_metric_val, target_metric_mode):
            best_metric_val = all_metrics[target_metric]
            U.save_checkpoint(best_lb_checkpoint, model, epoch, train_history,
                              metric_name=target_metric,
                              metric_score=best_metric_val)
            print('Checkpoint saved', epoch, best_metric_val, best_lb_checkpoint)
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

    print('Training finished')

    generate_model_submission(best_lb_checkpoint, config_fname, mine_on_val=True)
    # generate_model_submission(best_lb_checkpoint, config_fname, mine_on_val=False)


def compute_mask_class(y_true: Tensor):
    """
    Computes index [0;4] for masks. 0 - <20%, 1 - 20-40%, 2 - 40-60%, 3 - 60-80%, 4 - 80-100%
    :param y_true:
    :return:
    """
    batch_size = y_true.size(0)
    num_classes = y_true.size(1)
    if num_classes == 1:
        y_true = y_true.view(batch_size, -1)
    elif num_classes == 2:
        y_true = y_true[:, 1, ...].contiguous().view(batch_size, -1)  # Take salt class
    else:
        raise ValueError('Unknown num_classes')

    img_area = float(y_true.size(1))
    percentage = y_true.sum(dim=1) / img_area
    class_index = (percentage * 4).round().byte()
    return class_index


def noop(x):
    return x


def process_epoch(model, seg_criterion, optimizer, dataloader, epoch: int, is_train, summary_writer, mask_postprocess=noop, tag=None) -> dict:
    avg_seg_loss = AverageMeter()

    metrics = {
        'iou': JaccardIndex(0.5),
        'acc': PixelAccuracy(),
    }

    if tag is None:
        tag = 'train' if is_train else 'val'

    epoch_ids = []

    epoch_seg_losses = []
    epoch_msk_labels = []

    epoch_image = []

    epoch_mask_pred = []
    epoch_mask_true = []

    with torch.set_grad_enabled(is_train):
        if is_train:
            model.train()
        else:
            model.eval()

        n_batches = len(dataloader)
        with tqdm(total=n_batches) as tq:
            tq.set_description(f'{tag} epoch %d' % epoch)

            image = None
            mask_true = None
            msk_pred = None
            seg_loss = None

            for batch_index, (image, mask_true, sample_ids) in enumerate(dataloader):
                mask_true = mask_postprocess(mask_true)

                epoch_ids.extend(sample_ids)
                epoch_image.extend(image.detach().numpy()[:, 0:1, :, :])
                epoch_mask_true.extend(mask_true.detach().numpy())

                mask_class_labels = compute_mask_class(mask_true)
                image, mask_true = image.cuda(non_blocking=True), mask_true.cuda(non_blocking=True)

                if isinstance(seg_criterion, CELoss):
                    mask_true = mask_true.long().squeeze()

                if is_train:
                    with torch.autograd.detect_anomaly():
                        optimizer.zero_grad()
                        msk_pred = mask_postprocess(model(image))
                        seg_loss = seg_criterion(msk_pred, mask_true)
                        seg_loss.mean().backward()
                        optimizer.step()
                else:
                    msk_pred = mask_postprocess(model(image))
                    seg_loss = seg_criterion(msk_pred, mask_true)

                mask_pred_activate = logit_to_prob(msk_pred, seg_criterion)

                seg_loss_np = seg_loss.detach().cpu().numpy()
                epoch_mask_pred.extend(mask_pred_activate.cpu().numpy())
                epoch_seg_losses.extend(seg_loss_np)
                epoch_msk_labels.extend(mask_class_labels.numpy())

                # Log metrics
                for name, metric in metrics.items():
                    metric.update(mask_pred_activate, mask_true)

                if is_train and batch_index == 0:
                    # Log gradients at the first batch of epoch
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            summary_writer.add_histogram(f'{tag}/grad/{name}', param.grad.cpu(), epoch)

                avg_seg_loss.extend(seg_loss_np)

                tq.set_postfix(seg_loss='{:.3f}'.format(avg_seg_loss.avg))
                tq.update()

            del image, mask_true, msk_pred, seg_loss

    for key, metric in metrics.items():
        metric.log_to_tensorboard(summary_writer, f'{tag}/epoch/' + key, epoch)

    epoch_ids = np.array(epoch_ids)
    epoch_image = np.array(epoch_image)
    epoch_mask_true = np.array(epoch_mask_true)
    epoch_mask_pred = np.array(epoch_mask_pred)
    epoch_seg_losses = np.array(epoch_seg_losses)
    epoch_msk_labels = np.array(epoch_msk_labels)

    # End of train epoch

    # Mine thresholds on val
    if not is_train:
        thresholds, scores = threshold_mining(epoch_mask_pred, epoch_mask_true)
        i = np.argmax(scores)
        optimal_threshold = float(thresholds[i])
        lb_at_optimal_threshold = float(scores[i])
        summary_writer.add_scalar(f'{tag}/epoch/lb/optimal_threshold', optimal_threshold, epoch)
        summary_writer.add_scalar(f'{tag}/epoch/lb/optimal_score', lb_at_optimal_threshold, epoch)

    precision, result, threshold = do_kaggle_metric(epoch_mask_pred, epoch_mask_true, 0.50)
    lb_50 = np.mean(precision)
    summary_writer.add_scalar(f'{tag}/epoch/lb', lb_50, epoch)

    # Log losses
    summary_writer.add_scalar(f'{tag}/epoch/seg_loss', epoch_seg_losses.mean(), epoch)
    summary_writer.add_histogram(f'{tag}/epoch/seg_loss/histogram', epoch_seg_losses, epoch)

    for cls in np.unique(epoch_msk_labels):
        summary_writer.add_scalar(f'{tag}/epoch/seg_loss_class_{cls}', np.mean(epoch_seg_losses[epoch_msk_labels == cls]), epoch)

    # Plot segmentation negatives (loss)
    seg_losses_desc = np.argsort(-epoch_seg_losses)[:64]
    seg_negatives = pd.DataFrame.from_dict({
        'id': epoch_ids[seg_losses_desc],
        'seg_loss': epoch_seg_losses[seg_losses_desc]
    })

    summary_writer.add_image(f'{tag}/hard_negatives/loss/image', make_grid(torch.from_numpy(epoch_image[seg_losses_desc]), nrow=4, normalize=True), epoch)
    summary_writer.add_image(f'{tag}/hard_negatives/loss/y_true', make_grid(torch.from_numpy(epoch_mask_true[seg_losses_desc]), normalize=False, nrow=4), epoch)
    summary_writer.add_image(f'{tag}/hard_negatives/loss/y_pred', make_grid(torch.from_numpy(epoch_mask_pred[seg_losses_desc]), normalize=False, nrow=4), epoch)
    summary_writer.add_text(f'{tag}/hard_negatives/loss/ids', '```' + seg_negatives.to_csv(index=False) + '```', epoch)

    # Plot negative examples (LB)
    iou_losses_desc = np.argsort(precision)[:64]
    iou_negatives = pd.DataFrame.from_dict({
        'id': epoch_ids[iou_losses_desc],
        'iou_score': epoch_seg_losses[iou_losses_desc]
    })
    summary_writer.add_image(f'{tag}/hard_negatives/lb/image', make_grid(torch.from_numpy(epoch_image[iou_losses_desc]), nrow=4, normalize=True), epoch)
    summary_writer.add_image(f'{tag}/hard_negatives/lb/y_true', make_grid(torch.from_numpy(epoch_mask_true[iou_losses_desc]), normalize=False, nrow=4), epoch)
    summary_writer.add_image(f'{tag}/hard_negatives/lb/y_pred', make_grid(torch.from_numpy(epoch_mask_pred[iou_losses_desc]), normalize=False, nrow=4), epoch)
    summary_writer.add_text(f'{tag}/hard_negatives/lb/ids', '```' + iou_negatives.to_csv(index=False) + '```', epoch)

    if is_train:
        # Plot histogram of parameters after each epoch
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_data = param.data.cpu().numpy()
                summary_writer.add_histogram('model/' + name, param_data, epoch)

    metric_scores = {f'{tag}_seg_loss': epoch_seg_losses.mean(),
                     f'{tag}_lb': lb_50}

    for key, metric in metrics.items():
        metric_scores[f'{tag}_{key}'] = metric.value()

    return metric_scores


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
