# from __future__ import print_function, absolute_import, division

import datetime
import time
import argparse
import json
from matplotlib.pyplot import show
import numpy as np
import os
from scipy.spatial.transform import Rotation
from hypertools.tools import procrustes

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.visualize import show_2D_data, show_3d_prediction
from utils.camera import normalize_screen_coordinates, world_to_camera
from utils.utils import AverageMeter
from utils.data_utils import fetch, read_3d_data, create_2d_data
from utils.generators import PoseGenerator
from utils.loss import mpjpe
from utils.h36m_dataset import Human36mDataset, TEST_SUBJECTS
from config import _C as config
from models.martinez_model import MartinezModel, init_weights
from data.prepare_data_2d_h36m_sh import SH_TO_GT_PERM


def evaluate(data_loader, model, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_error_3d_pos = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model.eval()
    end = time.time()

    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        
        inputs_2d = inputs_2d.to(device)

        outputs_3d = model(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
        outputs_3d = torch.cat([torch.zeros(num_poses, 1, outputs_3d.size(2)), outputs_3d], 1)  # Pad hip joint (0,0,0)

        #al = procrustes(targets_3d.numpy()[0,:,:], outputs_3d.numpy()[0,:,:], scaling=False)

        epoch_error_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)

        #show_2D_data(inputs_2d.cpu().numpy()[0,:,:])
        #show_3d_prediction(outputs_3d.numpy()[0,:,:], targets_3d.numpy()[0,:,:], show=True)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0 or i == len(data_loader) - 1:
            print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | MPJPE: {e1: .4f}' \
                .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                        e1=epoch_error_3d_pos.avg))

    return epoch_error_3d_pos.avg

def predict_on_custom_dataset(keypoints, model, device):
    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model.eval()
    keypoints = torch.Tensor(keypoints)
    
    num_images = keypoints.shape[0]
    keypoints = keypoints.to(device)
    outputs_3d = model(keypoints.reshape(num_images, -1)).reshape(num_images, -1, 3).cpu()
    outputs_3d = torch.cat([torch.zeros(num_images, 1, outputs_3d.size(2)), outputs_3d], 1) # pad hip joint (0,0,0)

    return outputs_3d.numpy()

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
            action_filter = list(map(lambda x: dataset.define_actions(x)[0], action_filter))
            print('==> Selected actions: {}'.format(action_filter))
    
    elif config.dataset == "infiniteform":
        data = np.load(config.train_dataset, allow_pickle=True).tolist()
        filenames = list(data.keys())
        #annot = json.load(open('data/infiniteform/annotations_clean.json', 'r'))
        exercises = []
        keypoints_2d = []
        keypoints_3d = []
        for k,v in data.items():
            keypoints_2d.append(np.array(v['keypoints']))
            keypoints_3d.append(np.array(v['keypoints_3d']))
            #exercises.append(annot[k]['exercise'])

        keypoints_2d = np.array(keypoints_2d)
        keypoints_2d = keypoints_2d[:, SH_TO_GT_PERM, :]
        keypoints_2d = normalize_screen_coordinates(keypoints_2d, w=640, h=480)
        keypoints_3d = np.array(keypoints_3d)
    
    else:
        raise KeyError('Invalid dataset')

    cudnn.benchmark = True
    device = torch.device(config.device)

    # Create model
    print("==> Creating model...")
    num_joints = 16

    model = MartinezModel(num_joints * 2, (num_joints - 1) * 3, linear_size=config.linear_size).to(device) # 1 fewer output than input, since we treat the hip as (0, 0, 0) and predict root-relative coords
    model.apply(init_weights)
    # print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Load in checkpoint to evaluate on
    if os.path.isfile(config.eval_checkpoint):
        print("==> Loading checkpoint '{}'".format(config.eval_checkpoint))
        ckpt = torch.load(config.eval_checkpoint, map_location=device)
        start_epoch, error_best = ckpt['epoch'], ckpt['error']
        model.load_state_dict(ckpt['state_dict'])
        print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
    else:
        raise RuntimeError("==> No checkpoint found at '{}'".format(config.eval_checkpoint))

    # Run evaluation
    print('==> Starting evaluation...')

    if config.dataset == 'infiniteform':
        errors_p1 = []
        end_train = int(keypoints_2d.shape[0] * 0.85)
        _, poses_valid_2d = keypoints_2d[:end_train], keypoints_2d[end_train:]
        _, poses_valid_3d = keypoints_3d[:end_train], keypoints_3d[end_train:]
        #_, exercises_valid = exercises[:end_train], exercises[end_train:]

        '''unique, counts = np.unique(np.array(exercises_valid), return_counts=True)
        for ex in unique:
            ind = np.where(np.array(exercises_valid) == ex)[0]
            pos_ex_3d, pos_ex_2d = poses_valid_3d[ind], poses_valid_2d[ind]
            valid_loader = DataLoader(PoseGenerator(pos_ex_3d, pos_ex_2d, None), batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers, pin_memory=True)
            errors_p1.append(evaluate(valid_loader, model, device))'''

        valid_loader = DataLoader(PoseGenerator(poses_valid_3d, poses_valid_2d, None), batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers, pin_memory=True)

        errors_p1.append(evaluate(valid_loader, model, device))

    elif config.dataset == 'h36m':

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            poses_valid, poses_valid_2d, actions_valid = fetch(TEST_SUBJECTS, dataset, keypoints, [action])
            
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                        batch_size=config.batch_size, shuffle=False,
                                        num_workers=config.num_workers, pin_memory=True)
            errors_p1[i] = evaluate(valid_loader, model, device)

    print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))


if __name__ == '__main__':
    main()
