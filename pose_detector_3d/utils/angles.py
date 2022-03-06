import numpy as np
# import json
# from scipy.spatial.transform import Rotation

# from data.prepare_data_2d_h36m_sh import SH_TO_GT_PERM
# from utils.camera import world_to_camera
# from utils.visualize import show_3D_pose

def get_angle_from_joints(joint1, joint2, joint3):
    j2_j1 = joint1 - joint2
    j2_j3 = joint3 - joint2

    cosine_angle = np.dot(j2_j1, j2_j3) / (np.linalg.norm(j2_j1) * np.linalg.norm(j2_j3))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    return angle

def get_squat_angle(predictions_3d):
    rhip, rknee, rankle = predictions_3d[1, :], predictions_3d[2, :], predictions_3d[3, :]
    lhip, lknee, lankle = predictions_3d[4, :], predictions_3d[5, :], predictions_3d[6, :]

    l_angle = get_angle_from_joints(lhip, lknee, lankle)
    r_angle = get_angle_from_joints(rhip, rknee, rankle)
    return l_angle, r_angle

def get_plank_angle(predictions_3d):
    rknee, lknee = predictions_3d[2, :], predictions_3d[5, :]
    thorax, hip = predictions_3d[8, :], predictions_3d[0, :]
    knee_middle = (rknee - lknee) / 2 + lknee
    angle = get_angle_from_joints(thorax, hip, knee_middle)
    return angle


# annot = json.load(open("data/clean_annotations_0503.json", "r"))
# filenames = list(annot.keys())
# for i in range(len(filenames)):
#     if filenames[i] == "106":
#         camera_t, pitch = annot[filenames[i]]['camera_t'], annot[filenames[i]]['camera_pitch']
#         camera_r = Rotation.from_euler('y', pitch, degrees=True).as_quat()

#         keypoints_3d_gt = np.array(annot[filenames[i]]['keypoints_3d'])
#         keypoints_3d_gt = keypoints_3d_gt[SH_TO_GT_PERM, :]
#         keypoints_3d_gt = world_to_camera(keypoints_3d_gt, camera_r, camera_t)
#         keypoints_3d_gt[:, :] -= keypoints_3d_gt[:1, :] # remove global offset

#         # Angle for squat (RHip, RKnee, RAnkle)
#         rhip, rknee, rankle = keypoints_3d_gt[1, :], keypoints_3d_gt[2, :], keypoints_3d_gt[3, :]
#         lhip, lknee, lankle = keypoints_3d_gt[4, :], keypoints_3d_gt[5, :], keypoints_3d_gt[6, :]
#         thorax, hip = keypoints_3d_gt[8, :], keypoints_3d_gt[0, :]
#         knee_middle = (rknee - lknee) / 2 + lknee

#         # temp = keypoints_3d_gt
#         # temp[3,:] = knee_middle
#         # show_3D_pose(temp, show=True)
#         # exit(0)

#         def get_angle_from_joints(joint1, joint2, joint3):
#             j2_j1 = joint1 - joint2
#             j2_j3 = joint3 - joint2

#             cosine_angle = np.dot(j2_j1, j2_j3) / (np.linalg.norm(j2_j1) * np.linalg.norm(j2_j3))
#             angle = np.arccos(cosine_angle)
#             angle = np.degrees(angle)
#             return angle

#         print(get_angle_from_joints(rhip, rknee, rankle))
#         print(get_angle_from_joints(lhip, lknee, lankle))
#         print(get_angle_from_joints(thorax, hip, knee_middle))