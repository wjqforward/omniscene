import numpy as np
import cv2
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import torch
import os.path as osp
import pickle as pkl
from nuscenes.utils.data_classes import Box

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

def draw_nuscenes_boxes(
    nusc,
    bin_token: str,
    output_imgs: torch.Tensor,   # shape: (1, N, 3, H, W)
    c2w_interp: torch.Tensor,    # shape: (1, N, 4, 4)
    fovxs_interp: torch.Tensor,  # shape: (1, N)
    fovys_interp: torch.Tensor,  # shape: (1, N)
    pkl_dir: str,
    data_root: str,
    color=(0, 255, 0),
    thickness=2
):
    B, N, _, H, W = output_imgs.shape    B, N, _, H, W = output_imgs.shape
    assert B == 1
    imgs_np = (output_imgs[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # Load ego2world (lidar pose in global world) from .pkl frame 0
    with open(osp.join(pkl_dir, f"{bin_token[0]}.pkl"), 'rb') as f:
        bin_info = pkl.load(f)
    frame_info = bin_info['sensor_info']['LIDAR_TOP'][0]
    ego2world = np.eye(4)
    ego2world[:3, :3] = Quaternion(frame_info['ego2global_rotation']).rotation_matrix
    ego2world[:3, 3] = np.array(frame_info['ego2global_translation'])

    # Collect boxes (in global world)
    all_boxes = []
    for cam in bin_info['sensor_info']:
        if not cam.startswith('CAM_'):
            continue
        for frame_info in bin_info['sensor_info'][cam]:
            token = frame_info.get('sample_data_token', None)
            if token is None:
                continue
            try:
                boxes = nusc.get_boxes(token)
                all_boxes.extend(boxes)
            except:
                continue

    imgs_out = []
    for i in range(N):
        img = imgs_np[i].copy()
        c2w = c2w_interp[0, i].cpu().numpy()
        w2c = np.linalg.inv(c2w)

        fovx, fovy = fovxs_interp[0, i].item(), fovys_interp[0, i].item()
        fx = 0.5 * W / np.tan(0.5 * fovx)
        fy = 0.5 * H / np.tan(0.5 * fovy)
        cx = W / 2
        cy = H / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        for box in all_boxes:
            corners_world = box.corners()
            corners_homo = np.vstack([corners_world, np.ones((1, 8))])
            corners_lidar = np.linalg.inv(ego2world) @ corners_homo
            corners_cam = w2c @ corners_lidar

            if np.all(corners_cam[2, :] <= 0.1):
                continue

            corners_2d = view_points(corners_cam[:3], K, normalize=True)[:2].T.astype(int)
            for j in range(4):
                cv2.line(img, tuple(corners_2d[j]), tuple(corners_2d[(j + 1) % 4]), color, thickness)
                cv2.line(img, tuple(corners_2d[j + 4]), tuple(corners_2d[(j + 1) % 4 + 4]), color, thickness)
                cv2.line(img, tuple(corners_2d[j]), tuple(corners_2d[j + 4]), color, thickness)

        imgs_out.append(img)

    imgs_out = np.stack(imgs_out).astype(np.float32) / 255.0
    imgs_out = torch.from_numpy(imgs_out).permute(0, 3, 1, 2).unsqueeze(0)
    return imgs_out

