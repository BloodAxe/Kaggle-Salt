import json
import os.path
from datetime import datetime

import albumentations as A
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import Tensor, nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from lib import dataset as D
from lib import train_utils as U
from lib.common import count_parameters, is_sorted, compute_mask_class, to_numpy
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

    train_ids = D.all_train_ids()
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
    img_train, img_test = images[train_indexes], images[test_indexes]
    mask_train, mask_test = masks[train_indexes], masks[test_indexes]
    depth_train, depth_test = depths[train_indexes], depths[test_indexes]

    # Here we can exclude some images from training, but keep in validation
    train_mask = D.drop_some(img_train, mask_train, drop_black=True, drop_vstrips=args.drop_vstrips, drop_few=args.drop_few)
    ids_train = ids_train[train_mask]
    img_train = img_train[train_mask]
    mask_train = mask_train[train_mask]
    depth_train = depth_train[train_mask]

    if not is_sorted(ids_train):
        raise RuntimeError("ids_train is not sorted")
    if not is_sorted(ids_test):
        raise RuntimeError("ids_test_sorted is not sorted")

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

    print('Train set size :', len(ids_train), 'batch size', trainloader.batch_size)
    print('Valid set size :', len(ids_test), 'batch size', validloader.batch_size)
    print('Tile transform :', prepare_fn if prepare_fn is not None else "None")
    print('Model          :', args.model, count_parameters(model))
    print('Augmentations  :', args.augmentation, args.border_mode)
    print('Input channels :', trainset.channels())
    print('Output classes :', args.num_classes)
    print('Optimizer      :', args.optimizer, 'wd', args.weight_decay)
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

    if args.fine_tune and args.freeze_encoder > 0:
        raise ValueError('Incompatible options --fune-tune and --freeze-encoder')

    writer = SummaryWriter(log_dir)
    writer.add_text('train/params', '```' + json.dumps(train_session_args, indent=2) + '```', 0)

    config_fname = os.path.join(exp_dir, f'{train_session}.json')
    with open(config_fname, 'w') as f:
        f.write(json.dumps(train_session_args, indent=2))

    weights = {
        'mask': 1.0,
        'class': 0.05,
        'dsv': 0.1,
    }

    bce = U.get_loss('bce')
    bce_lovasz = U.get_loss('bce_lovasz')
    bce_jaccard = U.get_loss('bce_jaccard')

    losses = {
        'warmup': {
            'mask': bce,
            'class': bce,
            'dsv': bce,
        },
        'main': {
            'mask': bce_jaccard,
            'class': bce,
            'dsv': bce,
        },
        'annealing': {
            'mask': bce_lovasz,
            'class': bce,
            'dsv': bce,
        }
    }

    epochs = {
        'warmup': 50,
        'main': 250,
        'annealing': 50
    }

    if args.fast:
        for key in epochs.keys():
            epochs[key] = 1

    learning_rates = {
        'warmup': args.learning_rate,
        'main': 1e-3,
        'annealing': 1e-2
    }

    # Warmup phase
    if epochs['warmup']:
        print(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_cached())
        trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = U.get_optimizer(args.optimizer, trainable_parameters, learning_rates['warmup'], weight_decay=args.weight_decay)
        scheduler = None# StepLR(optimizer, gamma=0.5, step_size=50)

        train_history, best_metric_val, start_epoch = train(model, losses['warmup'], weights, optimizer, scheduler, trainloader, validloader, writer, start_epoch,
                                                            epochs=epochs['warmup'],
                                                            early_stopping=args.early_stopping,
                                                            train_history=train_history,
                                                            experiment_dir=exp_dir,
                                                            target_metric=target_metric,
                                                            best_metric_val=best_metric_val,
                                                            target_metric_mode=target_metric_mode,
                                                            checkpoint_filename=best_lb_checkpoint)
        U.save_checkpoint(os.path.join(exp_dir, f'{prefix}_warmup.pth'), model, start_epoch, train_history,
                          metric_name=target_metric,
                          metric_score=best_metric_val)

        del trainable_parameters, optimizer, scheduler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print('Finished warmup phase. Main train loop.')

    # Main training phase
    print(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_cached())
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = U.get_optimizer(args.optimizer, trainable_parameters, learning_rates['main'], weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=50, factor=0.5, min_lr=1e-5)

    train_history, best_metric_val, start_epoch = train(model, losses['main'], weights, optimizer, scheduler, trainloader, validloader, writer, start_epoch,
                                                        epochs=epochs['main'],
                                                        early_stopping=args.early_stopping,
                                                        train_history=train_history,
                                                        experiment_dir=exp_dir,
                                                        target_metric=target_metric,
                                                        best_metric_val=best_metric_val,
                                                        target_metric_mode=target_metric_mode,
                                                        checkpoint_filename=best_lb_checkpoint)
    del trainable_parameters, optimizer, scheduler
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    snapshots = [best_lb_checkpoint]

    U.save_checkpoint(os.path.join(exp_dir, f'{prefix}_main.pth'), model, start_epoch, train_history,
                      metric_name=target_metric,
                      metric_score=best_metric_val)

    print('Finished train phase.')

    # Cosine annealing
    if epochs['annealing']:

        for snapshot in range(5):
            print(f'Starting annealing phase {snapshot}')
            print(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_cached())
            # model.set_fine_tune(True)
            trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = U.get_optimizer('sgd', trainable_parameters, learning_rates['annealing'], weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, epochs['annealing'], eta_min=1e-7)

            snapshot_name = os.path.join(exp_dir, f'{prefix}_{target_metric}_snapshot_{snapshot}.pth')
            snapshots.append(snapshot_name)
            train_history, best_metric_val, start_epoch = train(model, losses['annealing'], weights, optimizer, scheduler, trainloader, validloader, writer, start_epoch,
                                                                epochs=epochs['annealing'],
                                                                early_stopping=args.early_stopping,
                                                                train_history=train_history,
                                                                experiment_dir=exp_dir,
                                                                target_metric=target_metric,
                                                                best_metric_val=0,
                                                                target_metric_mode=target_metric_mode,
                                                                checkpoint_filename=snapshot_name)
            del trainable_parameters, optimizer, scheduler
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print('Training finished')
    train_history.to_csv(os.path.join(exp_dir, 'train_history.csv'), index=False)

    for snapshot_file in snapshots:
        generate_model_submission(snapshot_file, config_fname, mine_on_val=True)


def noop(x):
    return x


def train(model: nn.Module, criterions: dict, criterion_weights, optimizer: Optimizer, scheduler, trainloader: DataLoader, validloader: DataLoader, writer: SummaryWriter, start_epoch: int, epochs: int,
          early_stopping,
          train_history: pd.DataFrame, experiment_dir: str,
          target_metric: str,
          best_metric_val: float,
          target_metric_mode: str,
          checkpoint_filename: str):
    # Start training loop
    no_improvement_epochs = 0
    epochs_trained = 0

    model.zero_grad()
    for epoch in range(start_epoch, start_epoch + epochs):
        # On Epoch begin
        if U.should_quit(experiment_dir) or (early_stopping is not None and no_improvement_epochs > early_stopping):
            break

        epochs_trained = epoch - start_epoch
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epochs_trained)

        U.log_learning_rate(writer, optimizer, epoch)

        # Epoch
        train_metrics = process_epoch(model, criterions, criterion_weights, optimizer, trainloader, epoch, True, writer)
        valid_metrics = process_epoch(model, criterions, criterion_weights, None, validloader, epoch, False, writer)

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
            U.save_checkpoint(checkpoint_filename, model, epoch, train_history,
                              metric_name=target_metric,
                              metric_score=best_metric_val)
            print('Checkpoint saved', epoch, best_metric_val, checkpoint_filename)
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

    model.zero_grad()
    return train_history, best_metric_val, epochs_trained + start_epoch + 1


def compute_total_loss(losses: dict, weights: dict = None):
    total_loss = None

    for name, loss in losses.items():

        if weights is not None:
            loss = loss * float(weights[name])

        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss

    return total_loss


def target_for_dsv(name):
    if str.startswith(name, 'dsv'):
        return 'mask'
    return name


def process_epoch(model, criterions: dict, criterion_weights: dict, optimizer, dataloader, epoch: int, is_train, summary_writer, tag=None) -> dict:
    avg_loss = AverageMeter()

    metrics = {
        'iou': JaccardIndex(0.5),
        'acc': PixelAccuracy(),
    }

    if tag is None:
        tag = 'train' if is_train else 'val'

    epoch_ids = []
    epoch_image = []
    epoch_mask_labels = []
    epoch_mask_pred = []
    epoch_mask_true = []
    epoch_losses = {}
    for key, _ in criterions.items():
        epoch_losses[key] = []

    with torch.set_grad_enabled(is_train):
        if is_train:
            model.train()
        else:
            model.eval()

        n_batches = len(dataloader)
        with tqdm(total=n_batches) as tq:
            tq.set_description(f'{tag} epoch %d' % epoch)

            batch = None
            total_loss = None

            for batch_index, batch in enumerate(dataloader):

                epoch_ids.extend(batch['id'])
                epoch_mask_true.extend(to_numpy(batch['mask']))
                epoch_mask_labels.extend(compute_mask_class(batch['mask']))
                epoch_image.extend(to_numpy(batch['image'])[:, 0:1, :, :])

                # Move all data to GPU
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.cuda(non_blocking=True)

                if is_train:
                    with torch.autograd.detect_anomaly():
                        optimizer.zero_grad()
                        predictions = model(batch)
                        losses = dict((key, criterions[key](predictions[key], batch[target_for_dsv(key)])) for key in predictions.keys())
                        total_loss = compute_total_loss(losses, criterion_weights)
                        total_loss.mean().backward()
                        optimizer.step()
                else:
                    predictions = model(batch)
                    losses = dict((key, criterions[key](predictions[key], batch[target_for_dsv(key)])) for key in predictions.keys())
                    total_loss = compute_total_loss(losses, criterion_weights)

                mask_pred_activate = logit_to_prob(predictions['mask'], criterions['mask'])

                epoch_mask_pred.extend(to_numpy(mask_pred_activate))

                # Add losses
                for loss_name in predictions.keys():
                    epoch_losses[loss_name].extend(to_numpy(losses[loss_name]))

                # Log metrics
                for name, metric in metrics.items():
                    metric.update(mask_pred_activate, batch['mask'])

                if is_train and batch_index == 0:
                    # Log gradients at the first batch of epoch
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            summary_writer.add_histogram(f'{tag}/grad/{name}', to_numpy(param.grad), epoch)

                avg_loss.extend(to_numpy(total_loss))
                tq.set_postfix(loss='{:.3f}'.format(avg_loss.avg))
                tq.update()

            del batch, total_loss

    for key, metric in metrics.items():
        metric.log_to_tensorboard(summary_writer, f'{tag}/epoch/' + key, epoch)

    epoch_ids = np.array(epoch_ids)
    epoch_image = np.array(epoch_image)
    epoch_mask_true = np.array(epoch_mask_true)
    epoch_mask_pred = np.array(epoch_mask_pred)

    # End of train epoch

    # Log losses
    for loss_name, epoch_losses in epoch_losses.items():
        if len(epoch_losses):
            summary_writer.add_scalar(f'{tag}/loss/{loss_name}', np.mean(epoch_losses), epoch)
            summary_writer.add_histogram(f'{tag}/loss/{loss_name}/histogram', np.array(epoch_losses), epoch)

    # epoch_mask_labels = np.array(epoch_mask_labels)
    # for cls in np.unique(epoch_mask_labels):
    #     summary_writer.add_scalar(f'{tag}/epoch/seg_loss_class_{cls}', np.mean(epoch_losses[epoch_mask_labels == cls]), epoch)

    # Mine thresholds on val
    if not is_train:
        thresholds, scores = threshold_mining(epoch_mask_pred, epoch_mask_true)
        i = np.argmax(scores)
        optimal_threshold = float(thresholds[i])
        lb_at_optimal_threshold = float(scores[i])
        summary_writer.add_scalar(f'{tag}/epoch/lb/optimal_threshold', optimal_threshold, epoch)
        summary_writer.add_scalar(f'{tag}/epoch/lb/optimal_score', lb_at_optimal_threshold, epoch)

    # Compute LB metric
    precision, result, threshold = do_kaggle_metric(epoch_mask_pred, epoch_mask_true, 0.50)
    lb_50 = np.mean(precision)
    summary_writer.add_scalar(f'{tag}/epoch/lb', lb_50, epoch)

    # Plot negative examples (LB)
    iou_metric_asc = np.argsort(precision)[:64]
    iou_negatives = pd.DataFrame.from_dict({
        'id': epoch_ids[iou_metric_asc],
        'iou_score': precision[iou_metric_asc]
    })

    summary_writer.add_image(f'{tag}/hard_negatives/lb/image', make_grid(torch.from_numpy(epoch_image[iou_metric_asc]), nrow=4, normalize=True), epoch)
    summary_writer.add_image(f'{tag}/hard_negatives/lb/y_true', make_grid(torch.from_numpy(epoch_mask_true[iou_metric_asc]), normalize=False, nrow=4), epoch)
    summary_writer.add_image(f'{tag}/hard_negatives/lb/y_pred', make_grid(torch.from_numpy(epoch_mask_pred[iou_metric_asc]), normalize=False, nrow=4), epoch)
    summary_writer.add_text(f'{tag}/hard_negatives/lb/ids', '```' + iou_negatives.to_csv(index=False) + '```', epoch)

    if is_train:
        # Plot histogram of parameters after each epoch
        for name, param in model.named_parameters():
            if param.grad is not None:
                summary_writer.add_histogram('model/' + name, to_numpy(param.data), epoch)

    metric_scores = {
        f'{tag}_lb': lb_50,
        f'{tag}_loss': avg_loss.avg
    }

    for key, metric in metrics.items():
        metric_scores[f'{tag}_{key}'] = metric.value()

    return metric_scores


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
