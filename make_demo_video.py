#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import cv2
from tqdm import tqdm

from pose_detector_2d.model import hg8
from pose_detector_2d.predictor import HumanPosePredictor
from pose_detector_3d.models.martinez_model import MartinezModel
from pose_detector_3d.data.prepare_data_2d_h36m_sh import SH_TO_GT_PERM
from pose_detector_3d.evaluate import predict_on_custom_dataset
from pose_detector_3d.utils.camera import normalize_screen_coordinates
from pose_detector_3d.utils.angles import get_plank_angle, get_squat_angle
from pose_detector_3d.utils.visualize import show_3D_pose
from torchvision import transforms

DEVICE = torch.device('cuda')
PREDICTOR_2D = HumanPosePredictor(hg8(pretrained=True))
PREDICTOR_3D = MartinezModel(16 * 2, (16 - 1) * 3, linear_size=1024).to(DEVICE)

def get2Dprediction(img):
    convert_tensor = transforms.ToTensor()
    if np.array(img).shape[2] > 3:
        im = Image.new("RGB", img.size, (255, 255, 255))
        im.paste(img, mask = img.split()[3])
        img_tensor = convert_tensor(im)
    else:
        img_tensor = convert_tensor(img)
    keypoints = PREDICTOR_2D.estimate_joints(img_tensor, flip=True)
    return keypoints.numpy()

def get3Dprediction(keypoints_2d, img, exercise_type):
    img = np.array(img)
    keypoints_2d_orig = keypoints_2d
    keypoints_2d = keypoints_2d[None, :]
    keypoints_2d = keypoints_2d[:, SH_TO_GT_PERM, :]
    keypoints_2d = normalize_screen_coordinates(keypoints_2d, w=img.shape[1], h=img.shape[0])
    predictions_3d = predict_on_custom_dataset(keypoints_2d, PREDICTOR_3D, DEVICE)
    predictions_3d = predictions_3d.squeeze()

    r = Rotation.from_euler('zyx', [0,90,0], degrees=True)
    predictions_3d = r.apply(predictions_3d)
    if exercise_type == "Plank":
        angles = [get_plank_angle(predictions_3d), "Plank angle (between thorax, hip, and knees): "]
    elif exercise_type == "Squat":
        angles = [np.mean(get_squat_angle(predictions_3d)), "Squat angle (between hip, knees, and ankles): "]

    img = show_3D_pose(predictions_3d, angles, keypoints_2d_orig, img, show=False, azim=0, elev=-90)
    return img

if __name__ == '__main__':
    # eval_checkpoint = os.path.join(os.path.dirname(__file__),'pose_detector_3d/checkpoints/2022-03-03_20-49-48/ckpt_best.pth.tar')
    eval_checkpoint = os.path.join(os.path.dirname(__file__),'pose_detector_3d/checkpoints/trained_on_infiniteform/ckpt_epoch_0300.pth.tar')

    ckpt = torch.load(eval_checkpoint, map_location=DEVICE)
    PREDICTOR_3D.load_state_dict(ckpt['state_dict'])

    # Read in input video and split into frames
    capture = cv2.VideoCapture('/Users/benalexander/Downloads/input_video.mov')
    success = True
    frames = []
    while success: 	
        success, frame = capture.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)


    # Run on each frame
    pose = 'Squat' # or 'Plank'
    all_plots = []
    for img in tqdm(frames):
        keypoints_2d = get2Dprediction(img)
        plot = get3Dprediction(keypoints_2d, img, pose)
        all_plots.append(plot)

    # Write to video
    height, width = plot.shape[:2]
    out = cv2.VideoWriter("demo_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for plot in all_plots:
        out.write(cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))
    out.release()