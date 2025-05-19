import numpy as np
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from model.utils.image import resize_image, HWC3
from mmdet3d.structures.points import LiDARPoints, BasePoints, get_points_type
from typing import List, Optional, Union
import cv2
import os.path as osp
import json
import copy
import os

def load_info(info):
    img_path = info["data_path"]
    # use lidar coordinate of the key frame as the world coordinate
    c2w = info["sensor2lidar_transform"]
    # opencv cam -> opengl cam, maybe not necessary!
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = c2w@flip_yz

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    
    return img_path, c2w, w2c

def load_conditions(img_paths, reso):
    
    def maybe_resize(img, tgt_reso, ck):
        if not isinstance(img, PIL.Image.Image):
            img = Image.fromarray(img)
        resize_flag = False
        if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
            # img.resize((w, h))
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
            fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h
            ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
            img = img.resize((tgt_reso[1], tgt_reso[0]))
            resize_flag = True
        return np.array(img), ck, resize_flag
    
    imgs, cks = [], []
    depths = []
    depths_m = []
    confs_m = []

    param_root="/home/Wjq99_/data_wjq/nuscenes_depth/nuscenes_param"
    conf_root="/home/Wjq99_/data_wjq/nuscenes_depth/nuscenes_conf"
    depth_root = "/home/Wjq99_/data_wjq/nuscenes_depth/nuscenes_depthlab_norm"
    img_root= "/home/Wjq99_/data_wjq/nuscenes"

    for img_path in img_paths:      
        # param
        if "samples/" in img_path:
            split = "samples"
        elif "sweeps/" in img_path:
            split = "sweeps"
        else:
            raise ValueError(f"must include samples or sweeps: {img_path}")
        
        subpath = img_path.split(f"{split}/", 1)[-1]
        cam = subpath.split('/')[0]
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]

        # param
        param_path = os.path.join(param_root, f"{split}_param", cam, f"{stem}.json")
        param = json.load(open(param_path))
        ck = np.array(param["camera_intrinsic"])
        ck_scale = ck.copy()
    
        orig_h, orig_w = 900, 1600
        target_h, target_w = 224, 400
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        
        ck_scale[0, 0] *= scale_w  # fx
        ck_scale[0, 2] *= scale_w  # cx
        ck_scale[1, 1] *= scale_h  # fy
        ck_scale[1, 2] *= scale_h  # cy

        cks.append(ck_scale)


        # img (900*1600 -> 224*400)
        img_path = os.path.join(img_root, f"{split}", cam, f"{stem}.jpg")
        img = Image.open(img_path)
        h, w = img.height, img.width
        img, ck, resize_flag = maybe_resize(img, reso, ck)
        img = HWC3(img)
        imgs.append(img)

        # metric depth from depthlab (224*400)
        depthm_path = os.path.join(depth_root, f"{split}_depthlab_norm", cam, f"{stem}.png")
        dptm = np.array(Image.open(depthm_path)).astype(np.float32) / 256.0
        depths_m.append(dptm)

        # depth conf (224*400)
        conf_path = os.path.join(conf_root, f"{split}_conf", cam, f"{stem}.npz")
        conf = np.load(conf_path)["conf"].astype(np.float32)
        confs_m.append(conf)

        # same as depths_m (224*400)
        depths.append(dptm)

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # [v c h w]
    depths = torch.from_numpy(np.stack(depths, axis=0)).float()  # [v h w]
    depths_m = torch.from_numpy(np.stack(depths_m, axis=0)).float()  # [v h w]
    confs_m = torch.from_numpy(np.stack(confs_m, axis=0)).float()  # [v h w]
    cks = torch.as_tensor(cks, dtype=torch.float32)

    return imgs, depths, depths_m, confs_m, cks

def load_lidar_info(info):
    pcd_path = info["data_path"]
    lidar2sensor = np.eye(4)
    rot = info["sensor2lidar_rotation"]
    trans = info["sensor2lidar_translation"]
    lidar2sensor[:3, :3] = copy.deepcopy(rot.T)
    lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
    return pcd_path, lidar2sensor

from nuscenes.utils.geometry_utils import view_points
def load_sparse_depths(points, lidar2sensor, w2c, ck):
    # transform lidar points to camera coordinate
    points[:, :3] = points[:, :3] @ lidar2sensor[:3, :3]
    points[:, :3] -= lidar2sensor[:3, 3]
    points_new = np.ones((points.shape[0], 4))
    points_new[:, :3] = points
    points_new = np.matmul(np.repeat(w2c.T[None], points.shape[0], axis=0), points_new[..., None])[:, :, 0]
    points_new /= points_new[:, -1:]
    points_new = points_new[:, :3]
    depths = points_new[:, 2]
    # transform 3d points to 2d images
    points_2d = view_points(points_new.transpose(), ck, normalize=True).transpose()
    width, height = int(ck[0, 2] * 2), int(ck[1, 2] * 2)
    # filter invalid points in 2d
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points_2d[:, 0] > 1)
    mask = np.logical_and(mask, points_2d[:, 0] < width - 1)
    mask = np.logical_and(mask, points_2d[:, 1] > 1)
    mask = np.logical_and(mask, points_2d[:, 1] < height - 1)
    points_2d = points_2d[mask]
    points_2d_int = np.array(points_2d, dtype=int)
    depths = depths[mask]
    depth_map = np.zeros((width, height))
    depth_map[points_2d_int[:, 0], points_2d_int[:, 1]] = depths
    return np.transpose(depth_map, (1, 0))
