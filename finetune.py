import sys
sys.path.append('./pose_detector_3d/')

from pose_detector_2d.model import hg8
from pose_detector_2d.predictor import HumanPosePredictor
import json
import numpy as np
import random
from tqdm import tqdm
import os
from PIL import Image as pimg
from torchvision import transforms
from scipy.spatial.transform import Rotation
from pose_detector_3d.utils.camera import world_to_camera
from pose_detector_3d.data.prepare_data_2d_h36m_sh import SH_TO_GT_PERM

INFINITEFORM_MPII_NAMES = [
    'right_ankle', 'right_knee', 'right_hip', 'left_hip',
    'left_knee', 'left_ankle', 'pelvis', 'root',
    'neck', 'head', 'right_wrist', 'right_elbow',
    'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist'
]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def preprocess_infiniteform_annotations(annot_path):
    annotations = json.load(open(annot_path, "r"))

    image_name = {}
    for im in annotations['images']:
        id = im['id']
        name = str(int(im['file_name'].split('.')[1]))
        #name = str(int(im['file_name'].split('.')[0]))
        #image_name[id] = (name, im['camera_location'], im['camera_pitch'], im['avatar_exercise'].lower())
        image_name[id] = (name, im['camera_location'], im['camera_pitch'], im['avatar_yaw'])

    annot = {}
    for im in annotations['annotations']:
        if im['percent_in_fov'] != 100.0:
            continue
        #id, camera_t, camera_p, exercise = image_name[im['image_id']]
        id, camera_t, camera_p, camera_y = image_name[im['image_id']]
        keypoints_2d = np.zeros((len(INFINITEFORM_MPII_NAMES),2))
        keypoints_3d = np.zeros((len(INFINITEFORM_MPII_NAMES),3))
        
        for k,v in im['joint_keypoints'].items():
            if k in INFINITEFORM_MPII_NAMES:
                x,y = v['x'], v['y']
                keypoints_2d[INFINITEFORM_MPII_NAMES.index(k)] = (x,y)

                x,y,z = v['x_global'], v['y_global'], v['z_global']
                keypoints_3d[INFINITEFORM_MPII_NAMES.index(k)] = (x,y,z)

        #annot[id] = {'keypoints': keypoints_2d, 'keypoints_3d': keypoints_3d, 'camera_t': camera_t, 'camera_pitch': camera_p, 'exercise': exercise}
        annot[id] = {'keypoints': keypoints_2d, 'keypoints_3d': keypoints_3d, 'camera_t': camera_t, 'camera_pitch': camera_p, 'camera_yaw': camera_y}
    
    outfile = '.'.join(annot_path.split('.')[:-1]) + '_clean.json'
    with open(outfile, 'w') as f:
        f.write(json.dumps(annot,cls=NumpyEncoder))
    return outfile

def evaluate_inf_validation_accuracy(annot, preds, img_idx):
    threshold = 0.5
    pos_gt_src = np.array(annot[img_idx]['keypoints']).reshape(1,16,2)
    waist_diams = np.linalg.norm(pos_gt_src[0,2,:] - pos_gt_src[0,3,:]).reshape(1, -1)

    preds = np.array(preds).reshape(1, 16, 2)
    preds = np.transpose(preds, [1, 2, 0])
    pos_gt_src = np.transpose(pos_gt_src, [1, 2, 0])

    uv_error = preds - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    
    scale = np.multiply(waist_diams, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    less_than_threshold = (scaled_uv_err < threshold)
    return np.mean(less_than_threshold)

def compute_keypoints_3D(annot, idx):
    camera_t, pitch, yaw = annot[idx]['camera_t'], annot[idx]['camera_pitch'], annot[idx]['camera_yaw']
    camera_r = Rotation.from_euler('yz', [pitch, yaw], degrees=True).as_quat()

    keypoints_3d_gt = np.array(annot[idx]['keypoints_3d'])
    keypoints_3d_gt = keypoints_3d_gt[SH_TO_GT_PERM, :]
    keypoints_3d_gt = world_to_camera(keypoints_3d_gt, camera_r, camera_t)
    keypoints_3d_gt[:, :] -= keypoints_3d_gt[:1, :] # remove global offset

    return keypoints_3d_gt

def compute_keypoints_2D(annot_path, images_path):
    annot = json.load(open(annot_path, 'r'))

    predictor_8 = HumanPosePredictor(hg8(pretrained=True), device='cuda')
    inf_form_filenames = []
    all_preds = []

    random.seed(5)
    files_to_analyze = os.listdir(images_path)
    random.shuffle(files_to_analyze)

    for img_name in tqdm(files_to_analyze):
        if 'json' in img_name:
            continue

        #idx = str(int(img_name.split(".")[0]))
        idx = str(int(img_name.split(".")[1]))

        if idx not in list(annot.keys()):
            continue
        img_path = os.path.join(images_path, img_name)
        img = pimg.open(img_path)
        img.load()
        background = pimg.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask = img.split()[3])
        convert_tensor = transforms.ToTensor()
        img_tensor = convert_tensor(background)

        joints_8 = predictor_8.estimate_joints(img_tensor, flip=True)
        
        pck_8 = evaluate_inf_validation_accuracy(annot, joints_8, idx)
        if pck_8 >= 0.5:
            all_preds.append(joints_8.numpy())
            inf_form_filenames.append(idx)
            #inf_form_filenames.append(str(int(img_name.split(".")[0])))

    d = {}
    for i in range(len(all_preds)):
        idx = inf_form_filenames[i]
        keypoints_3D = compute_keypoints_3D(annot, idx)
        d[idx] = {'keypoints': all_preds[i], 'keypoints_3d': keypoints_3D}

    outfile = '/'.join(annot_path.split('/')[:-1]) + '_keypoints_2d_3d.npy'
    np.save(outfile, d, allow_pickle=True)
    return outfile


if __name__ == '__main__':
    
    annot_path = 'pose_detector_3d/data/infiniteform2/annotations.json'

    annot_clean_path = preprocess_infiniteform_annotations(annot_path)

    images_path = 'pose_detector_3d/data/infiniteform2/'
    keypoints_2D_3D_path = compute_keypoints_2D(annot_clean_path, images_path)

    keypoints_2D_3D_path = 'pose_detector_3d/data/infiniteform2_keypoints_2d_3d.npy'

    cmd = ['python', 'pose_detector_3d/train.py', 'dataset', 'infiniteform', 'train_checkpoint', \
        'pose_detector_3d/checkpoints/2022-03-03_20-49-48/ckpt_best.pth.tar', 'train_dataset', keypoints_2D_3D_path, \
            'epochs', '200']
    
    stream = os.popen(' '.join(cmd))
    print(stream.read())