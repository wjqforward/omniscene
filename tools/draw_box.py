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

    B, N, _, H, W = output_imgs.shape
    assert B == 1
    output_imgs_np = (output_imgs[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # (N, H, W, 3)

    # load boxes in the world frame
    with open(osp.join(pkl_dir, f"{bin_token}.pkl"), "rb") as f:
        bin_info = pkl.load(f)

    all_boxes_world = []
    for cam in bin_info["sensor_info"]:
        if not cam.startswith("CAM_"):
            continue
        for frame_info in bin_info["sensor_info"][cam]:
            sample_data_token = frame_info["sample_data_token"]
            boxes = nusc.get_boxes(sample_data_token)
            all_boxes_world.extend(boxes)

    # draw box on each img
    rendered_with_boxes = []
    for i in range(N):
        img = output_imgs_np[i].copy()
        c2w = c2w_interp[0, i].cpu().numpy()  # (4, 4)
        w2c = np.linalg.inv(c2w)
        fovx = fovxs_interp[0, i].item()
        fovy = fovys_interp[0, i].item()
        K = fov_to_intrinsics(fovx, fovy, H, W)

        for box in all_boxes_world:
            box_c = Box(box.center.copy(), box.wlh.copy(), box.orientation)

            # world to cam
            box_c.translate(-c2w[:3, 3])
            box_c.rotate(Quaternion(matrix=w2c[:3, :3]))

            # filter box behind cam
            if box_c.center[2] <= 0.1:
                continue

            corners_3d = box_c.corners()  # (3, 8)
            corners_2d = view_points(corners_3d, K, normalize=True)  # (3, 8)

            corners_2d = corners_2d[:2].T.astype(int)  # (8, 2)
            for j in range(4):
                cv2.line(img, tuple(corners_2d[j]), tuple(corners_2d[(j + 1) % 4]), color, thickness)
                cv2.line(img, tuple(corners_2d[j + 4]), tuple(corners_2d[(j + 1) % 4 + 4]), color, thickness)
                cv2.line(img, tuple(corners_2d[j]), tuple(corners_2d[j + 4]), color, thickness)

        rendered_with_boxes.append(img)

    # Convert back to torch tensor (b, v, 3, h, w)
    rendered_with_boxes = np.stack(rendered_with_boxes, axis=0)  # (v, H, W, 3)
    rendered_with_boxes = rendered_with_boxes.astype(np.float32) / 255.0
    rendered_with_boxes = torch.from_numpy(rendered_with_boxes).permute(0, 3, 1, 2)  # (v, 3, h, w)
    rendered_with_boxes = rendered_with_boxes.unsqueeze(0)  # (1, v, 3, h, w)

    return rendered_with_boxes  # List[np.ndarray], len=N
