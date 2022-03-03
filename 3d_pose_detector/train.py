# from __future__ import print_function, absolute_import, division

import json
import os
import time
import datetime
import argparse
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import _C as config
from evaluate import evaluate
from utils.log import Logger
from utils.utils import AverageMeter, lr_decay, save_ckpt
from utils.data_utils import fetch, read_3d_data, create_2d_data
from utils.generators import PoseGenerator
from utils.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
from models.martinez_model import MartinezModel, init_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config.merge_from_list(args.opts)

    print('==> Loading dataset...')
    if config.dataset == "h36m":
        dataset = Human36mDataset('data/data_3d_h36m.npz')

        dataset = read_3d_data(dataset)
        
        keypoints = create_2d_data(os.path.join('data', 'data_2d_h36m_gt.npz'), dataset)
        
        action_filter = None if config.actions == '*' else config.actions.split(',')
        if action_filter is not None:
            action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
            print('==> Selected actions: {}'.format(action_filter))
    else:
        raise KeyError('Invalid dataset')


    cudnn.benchmark = True
    device = torch.device(config.device)

    # Create model
    print("==> Creating model...")
    num_joints = dataset.skeleton().num_joints()

    model = MartinezModel(num_joints * 2, (num_joints - 1) * 3).to(device) # 1 fewer output than input, since we treat the hip as (0, 0, 0) and predict root-relative coords
    model.apply(init_weights)
    # print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    loss = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Optionally resume from a checkpoint
    if config.train_checkpoint:
        if os.path.isfile(config.train_checkpoint):
            print("==> Loading checkpoint '{}'".format(config.train_checkpoint))
            ckpt = torch.load(config.train_checkpoint, map_location=device)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            ckpt_dir_path = os.path.dirname(config.train_checkpoint)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(config.train_checkpoint))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = config.lr
        time_for_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ckpt_dir_path = os.path.join('checkpoints', time_for_name)
        log_dir_path = os.path.join('logs', time_for_name)

        if not os.path.exists('logs'):
            os.makedirs('logs')

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        if not os.path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
            print('==> Making log dir: {}'.format(log_dir_path))

    json.dump(config, open(os.path.join(ckpt_dir_path, 'config.json'), 'w'))
    logger = Logger(log_dir_path)

    poses_train, poses_train_2d, actions_train = fetch(TRAIN_SUBJECTS, dataset, keypoints, action_filter)
    train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, actions_train), batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True)

    poses_valid, poses_valid_2d, actions_valid = fetch(TEST_SUBJECTS, dataset, keypoints, action_filter)
    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid), batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers, pin_memory=True)

    for epoch in tqdm(range(start_epoch, config.epochs)):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train
        lr_now, glob_step = train(train_loader, model, loss, optimizer, device, logger, config.lr, lr_now,
                                              glob_step, config.lr_decay, config.lr_gamma, config.max_norm)

        # Evaluate
        error_eval_p1 = evaluate(valid_loader, model, device)

        logger.log_testing(error_eval_p1, epoch)

        # Save checkpoint
        if error_best is None or error_eval_p1 < error_best:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, config, suffix='best')

        if (epoch + 1) % config.save_every == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, config, delete_previous=True)


def train(data_loader, model, loss, optimizer, device, logger, lr_init, lr_now, step, decay, gamma, max_norm):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model.train()
    end = time.time()

    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d = targets_3d[:, 1:, :].to(device), inputs_2d.to(device)  # Remove hip joint for 3D poses
        outputs_3d = model(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3)

        optimizer.zero_grad()
        loss_3d_pos = loss(outputs_3d, targets_3d)
        loss_3d_pos.backward()

        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        reduced_loss = loss_3d_pos.item()
        epoch_loss_3d_pos.update(reduced_loss, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 1000 == 0:
            print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Loss: {loss: .4f}' \
                .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                        loss=epoch_loss_3d_pos.avg))
        
        logger.log_training(reduced_loss, lr_now, batch_time.avg, step)

    return lr_now, step


if __name__ == '__main__':
    main()
