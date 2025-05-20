import numpy as np
import torch
import cv2
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import os.path as osp
import pickle as pkl

def fov_to_intrinsics(fovx, fovy, H, W):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = W / 2
    cy = H / 2
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ])
    return K

def draw_3d_boxes_on_batch(
    nusc,
    bin_token,
    output_imgs,     # (1, V, 3, H, W)
    c2ws,             # (V, 4, 4) camera-to-lidar
    fovxs, fovys,     # (V,)
    pkl_dir,
    color=(0, 255, 0),
    thickness=2
):
    B, V, C, H, W = output_imgs.shape
    assert B == 1
    imgs_np = (output_imgs[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # (V, H, W, 3)

    # read box
    with open(osp.join(pkl_dir, f"{bin_token[0]}.pkl"), "rb") as f:
        bin_info = pkl.load(f)

    box_records = []
    for cam in bin_info["sensor_info"]:
        for frame in bin_info["sensor_info"][cam]:
            if "sample_data_token" not in frame:
                continue
            try:
                boxes = nusc.get_boxes(frame["sample_data_token"])
                sd = nusc.get('sample_data', frame['sample_data_token'])
                pose = nusc.get('ego_pose', sd['ego_pose_token'])
                T_ego2world = np.eye(4)
                T_ego2world[:3, :3] = Quaternion(pose['rotation']).rotation_matrix
                T_ego2world[:3, 3] = pose['translation']
                for box in boxes:
                    box_records.append((box, T_ego2world))
            except:
                continue

    rendered = []
    for i in range(V):
        img = imgs_np[i].copy()
        c2w = c2ws[i].cpu().numpy()
        w2c = np.linalg.inv(c2w)
        fovx = fovxs[i].item()
        fovy = fovys[i].item()
        K = fov_to_intrinsics(fovx, fovy, H, W)

        for box, T_ego2world in box_records:
            corners = box.corners()  # (3, 8)
            corners_homo = np.vstack([corners, np.ones((1, 8))])  # (4, 8)

            # world -> ego -> lidar -> camera
            corners_cam = w2c @ T_ego2world @ corners_homo

            if np.all(corners_cam[2, :] <= 0.1):
                continue

            corners_2d = view_points(corners_cam[:3], K, normalize=True)[:2].T.astype(int)

            for j in range(4):
                cv2.line(img, tuple(corners_2d[j]), tuple(corners_2d[(j+1)%4]), color, thickness)
                cv2.line(img, tuple(corners_2d[j+4]), tuple(corners_2d[(j+1)%4+4]), color, thickness)
                cv2.line(img, tuple(corners_2d[j]), tuple(corners_2d[j+4]), color, thickness)

        rendered.append(img)

    rendered = np.stack(rendered).astype(np.float32) / 255.0  # (V, H, W, 3)
    rendered = torch.from_numpy(rendered).permute(0, 3, 1, 2).unsqueeze(0)  # (1, V, 3, H, W)
    return rendered
