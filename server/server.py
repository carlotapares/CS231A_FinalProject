#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import io
import os
import numpy as np
import pathlib
import json
from PIL import Image
from io import BytesIO
from sys import argv
from scipy.spatial.transform import Rotation
from base64 import b64decode, b64encode
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from pose_detector_2d.model import hg8
from pose_detector_2d.predictor import HumanPosePredictor
from pose_detector_3d.models.martinez_model import MartinezModel
from pose_detector_3d.data.prepare_data_2d_h36m_sh import SH_TO_GT_PERM
from pose_detector_3d.evaluate import predict_on_custom_dataset
from pose_detector_3d.utils.camera import normalize_screen_coordinates
from pose_detector_3d.utils.angles import get_plank_angle, get_squat_angle
from pose_detector_3d.utils.visualize import show_3D_pose
from torchvision import transforms

DEVICE = torch.device('cpu')
PREDICTOR_2D = HumanPosePredictor(hg8(pretrained=True))
PREDICTOR_3D = MartinezModel(16 * 2, (16 - 1) * 3, linear_size=1024).to(DEVICE)
# DEVICE = torch.device('cuda')

def get2Dprediction(img):
    convert_tensor = transforms.ToTensor()
    if np.array(img).shape[2] > 3:
        im = Image.new("RGB", img.size, (255, 255, 255))
        im.paste(img, mask = img.split()[3])
        img_tensor = convert_tensor(im)
    else:
        img_tensor = convert_tensor(img)
    keypoints = PREDICTOR_2D.estimate_joints(img_tensor, flip=True)
    return keypoints.numpy(), np.array(img).shape

def get3Dprediction(keypoints_2d, img_shape, exercise_type):
    keypoints_2d = keypoints_2d[None, :]
    keypoints_2d = keypoints_2d[:, SH_TO_GT_PERM, :]
    keypoints_2d = normalize_screen_coordinates(keypoints_2d, w=img_shape[1], h=img_shape[0])
    predictions_3d = predict_on_custom_dataset(keypoints_2d, PREDICTOR_3D, DEVICE)
    predictions_3d = predictions_3d.squeeze()

    r = Rotation.from_euler('zyx', [90,-20,90], degrees=True)
    predictions_3d = r.apply(predictions_3d)
    if exercise_type == "Plank":
        angles = [get_plank_angle(predictions_3d), "Plank angle (between thorax, hip, and knees): "]
    elif exercise_type == "Squat":
        angles = [np.mean(get_squat_angle(predictions_3d)), "Squat angle (between hip, knees, and ankles): "]

    img = show_3D_pose(predictions_3d, angles, show=False, azim=0, elev=-90)

    return img

class S(BaseHTTPRequestHandler):

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = json.loads(self.rfile.read(content_length)) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), 'image')
        im = Image.open(BytesIO(b64decode(post_data['image'].split(',')[1])))
        pose = post_data['pose']

        # im.save(os.path.join(pathlib.Path(__file__).parent.resolve(),"output", "image_to_process.png"))
        keypoints_2d, im_shape = get2Dprediction(im)
        img = get3Dprediction(keypoints_2d, im_shape, pose)

        self._set_response()
        
        # image_file = os.path.join(pathlib.Path(__file__).parent.resolve(),"output", "result.png")
        # with open(image_file, "rb") as f:
        #     im_bytes = f.read()
        pil_im = Image.fromarray(img)
        b = io.BytesIO()
        pil_im.save(b, 'png')
        im_bytes = b.getvalue()
        self.wfile.write(json.dumps({'image': b64encode(im_bytes).decode("utf8")}).encode())

def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    eval_checkpoint = os.path.join(os.path.dirname(__file__),'../pose_detector_3d/checkpoints/2022-03-03_20-49-48/ckpt_best.pth.tar')

    ckpt = torch.load(eval_checkpoint, map_location=DEVICE)
    PREDICTOR_3D.load_state_dict(ckpt['state_dict'])

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()