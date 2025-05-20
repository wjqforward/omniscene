import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange, einsum
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from .gaussian import GaussianRenderer
from .losses import LPIPS, LossDepthTV
from .utils.image import maybe_resize
from .utils.benchmarker import Benchmarker
from torchmetrics import PearsonCorrCoef
from .utils.interpolation import interpolate_extrinsics

import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import cv2
from piqa import SSIM

from tools.draw_box import draw_nuscenes_boxes
from nuscenes.nuscenes import NuScenes

def interpolate_c2w(c2w_start: torch.Tensor, 
                   c2w_end: torch.Tensor, 
                   t: float = 0.5) -> torch.Tensor:

    c2w_start_np = c2w_start.cpu().numpy()
    c2w_end_np = c2w_end.cpu().numpy()

    R_start = c2w_start_np[:3, :3]
    R_end = c2w_end_np[:3, :3]

    quat_start = Rotation.from_matrix(R_start).as_quat()
    quat_end = Rotation.from_matrix(R_end).as_quat()
    slerp = Slerp([0, 1], Rotation.from_quat([quat_start, quat_end]))
    R_interp = slerp(t).as_matrix()

    t_interp = (1 - t) * c2w_start_np[:3, 3] + t * c2w_end_np[:3, 3]

    c2w_interp = np.eye(4)
    c2w_interp[:3, :3] = R_interp
    c2w_interp[:3, 3] = t_interp

    return torch.from_numpy(c2w_interp).to(device=c2w_start.device, dtype=c2w_start.dtype)

@MODELS.register_module()
class OmniGaussian(BaseModule):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 pixel_gs=None,
                 volume_gs=None,
                 camera_args=None,
                 loss_args=None,
                 dataset_params=None,
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        assert pixel_gs is not None and volume_gs is not None
        self.use_checkpoint = use_checkpoint
        if backbone:
            self.backbone = MODELS.build(backbone)
        if neck:
            self.neck = MODELS.build(neck)
        self.pixel_gs = MODELS.build(pixel_gs)
        self.volume_gs = MODELS.build(volume_gs)

        self.dataset_params = dataset_params
        self.camera_args = camera_args
        self.loss_args = loss_args

        self.renderer = GaussianRenderer(self.device, **camera_args)

        # Perceptual loss
        if self.loss_args.weight_perceptual > 0:
            # self.perceptual_loss = LPIPS(net="vgg")
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None

        # record runtime
        self.benchmarker = Benchmarker()

    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, img, use_reentrant=False)
        else:
            img_feats = self.backbone(img)
        img_feats = self.neck(img_feats) # BV, C, H, W
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    def get_data(self, batch):

        # ================== batch data process ================== #
        device_id = self.device
        data_dict = {}
        # for img feature extraction
        data_dict["imgs"] = batch["inputs"]["rgb"].to(device_id, dtype=self.dtype)
        # for pixel-gs
        rays_o = batch["inputs_pix"]["rays_o"].to(device_id, dtype=self.dtype)
        rays_d = batch["inputs_pix"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["rays_o"] = rays_o
        data_dict["rays_d"] = rays_d
        data_dict["pluckers"] = self.plucker_embedder(rays_o, rays_d)
        data_dict["fxs"] = batch["inputs_pix"]["fx"].to(device_id, dtype=self.dtype)
        data_dict["fys"] = batch["inputs_pix"]["fy"].to(device_id, dtype=self.dtype)
        data_dict["cxs"] = batch["inputs_pix"]["cx"].to(device_id, dtype=self.dtype)
        data_dict["cys"] = batch["inputs_pix"]["cy"].to(device_id, dtype=self.dtype)
        data_dict["c2ws"] = batch["inputs_pix"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["cks"] = batch["inputs_pix"]["ck"].to(device_id, dtype=self.dtype)
        data_dict["depths"] = batch["inputs_pix"]["depth_m"].to(device_id, dtype=self.dtype)
        data_dict["confs"] = batch["inputs_pix"]["conf_m"].to(device_id, dtype=self.dtype)
        # for volume-gs
        img_metas = []
        bs, v, c, h, w = batch["inputs"]["rgb"].shape
        for w2i in batch["inputs_vol"]["w2i"]:
            img_metas.append({"lidar2img": w2i, "img_shape": [[h, w]] * v})
        data_dict["img_metas"] = img_metas
        # for render and loss and eval
        data_dict["output_imgs"] = batch["outputs"]["rgb"].to(device_id, dtype=self.dtype)
        data_dict["output_depths"] = batch["outputs"]["depth"].to(device_id, dtype=self.dtype)
        data_dict["output_depths_m"] = batch["outputs"]["depth_m"].to(device_id, dtype=self.dtype)
        data_dict["output_confs_m"] = batch["outputs"]["conf_m"].to(device_id, dtype=self.dtype)
        data_dict["output_positions"] = (batch["outputs"]["rays_o"] + batch["outputs"]["rays_d"] * \
                            batch["outputs"]["depth_m"].unsqueeze(-1)).to(device_id, dtype=self.dtype)
        data_dict["output_rays_o"] = batch["outputs"]["rays_o"].to(device_id, dtype=self.dtype)
        data_dict["output_rays_d"] = batch["outputs"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["output_c2ws"] = batch["outputs"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["output_fovxs"] = batch["outputs"]["fovx"].to(device_id, dtype=self.dtype)
        data_dict["output_fovys"] = batch["outputs"]["fovy"].to(device_id, dtype=self.dtype)

        data_dict["bin_token"] = batch["bin_token"]

        return data_dict
    
    def configure_optimizers(self, lr):
        backbone_layers = torch.nn.ModuleList([self.backbone])
        backbone_layers_params = list(map(id, backbone_layers.parameters()))
        base_params = list(filter(lambda p: id(p) not in backbone_layers_params, self.parameters()))
        
        opt = torch.optim.AdamW(
            [{'params': base_params}, {'params': backbone_layers.parameters(), 'lr': lr*0.1}],
            lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
        return [opt]
    
    def forward(self, batch, split="train", iter=0, iter_end=100000):
        """Forward training function.
        """
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img)

        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)

        bs = gaussians_all.shape[0]
        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=render_c2w,
            fovx=render_fovxs,
            fovy=render_fovys,
            rays_o=None,
            rays_d=None
        )
        if split == "train" or split == "val":
            render_pkg_pixel = self.renderer.render(
                gaussians=gaussians_pixel,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
            render_pkg_volume = self.renderer.render(
                gaussians=gaussians_volume,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )
        else:
            render_pkg_pixel, render_pkg_volume = None, None
        
        # ======================== losses ======================== #
        loss = 0.0
        loss_terms = {}
        def set_loss(key, split, loss_value, loss_weight=1.0):
            loss_terms[f"{split}/loss_{key}"] = loss_value.item()
            loss_terms[f"{split}/loss_{key}_w"] = loss_value.item() * loss_weight

        # =================== Data preparation =================== #        
        rgb_gt = data_dict["output_imgs"]
        data_dict["rgb_gt"] = rgb_gt
        depth_m_gt = data_dict["output_depths_m"]
        conf_m_gt = data_dict["output_confs_m"]
        data_dict["depth_m_gt"] = depth_m_gt
        data_dict["conf_m_gt"] = conf_m_gt
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        if self.loss_args.mask_dptm and self.loss_args.recon_loss_vol_type == "l2_mask_self":
            depth_for_mask = render_pkg_pixel["depth"].squeeze(2).detach().unsqueeze(-1)
            output_positions = data_dict["output_rays_o"] + data_dict["output_rays_d"] * depth_for_mask
            mask_dptm = (output_positions[..., 0] >= x_start) & (output_positions[..., 0] <= x_end) & \
                        (output_positions[..., 1] >= y_start) & (output_positions[..., 1] <= y_end) & \
                        (output_positions[..., 2] >= z_start) & (output_positions[..., 2] <= z_end)
            mask_dptm = mask_dptm & (depth_for_mask[..., 0] > 0.1)
            mask_dptm = mask_dptm.float()
        elif self.loss_args.mask_dptm and self.loss_args.recon_loss_vol_type == "l2_mask":
            output_positions = data_dict["output_positions"]
            mask_dptm = (output_positions[..., 0] >= x_start) & (output_positions[..., 0] <= x_end) & \
                        (output_positions[..., 1] >= y_start) & (output_positions[..., 1] <= y_end) & \
                        (output_positions[..., 2] >= z_start) & (output_positions[..., 2] <= z_end)
            mask_dptm = mask_dptm.float()
        data_dict["mask_dptm"] = mask_dptm
            
        # ======================== RGB loss ======================== #
        if self.loss_args.weight_recon > 0:
            # RGB loss for omni-gs
            if self.loss_args.recon_loss_type == "l1":
                rec_loss = torch.abs(rgb_gt - render_pkg_fuse["image"])
            elif self.loss_args.recon_loss_type == "l2":
                rec_loss = (rgb_gt - render_pkg_fuse["image"]) ** 2
            loss = loss + (rec_loss.mean() * self.loss_args.weight_recon)
            set_loss("recon", split, rec_loss.mean(), self.loss_args.weight_recon)
        if self.loss_args.weight_recon_vol > 0 and iter < iter_end - 1000:
            # RGB loss for volume-gs
            if self.loss_args.recon_loss_vol_type == "l1":
                rec_loss_vol = torch.abs(rgb_gt - render_pkg_volume["image"])
            elif self.loss_args.recon_loss_vol_type == "l2":
                rec_loss_vol = (rgb_gt - render_pkg_volume["image"]) ** 2
            elif self.loss_args.recon_loss_vol_type == "l2_mask" or self.loss_args.recon_loss_vol_type == "l2_mask_self":
                rec_loss_vol = (rgb_gt * mask_dptm.unsqueeze(2) - render_pkg_volume["image"] * mask_dptm.unsqueeze(2)) ** 2
            loss = loss + (rec_loss_vol.mean() * self.loss_args.weight_recon_vol)
            set_loss("recon_vol", split, rec_loss_vol.mean(), self.loss_args.weight_recon_vol)

        # ==================== Perceptual loss ===================== #
        if self.loss_args.weight_perceptual > 0:
            # Perceptual loss for omni-gs
            ## resize images to smaller size to save memory
            p_inp_pred = maybe_resize(
                render_pkg_fuse["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_gt = maybe_resize(
                rgb_gt.reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_loss = self.perceptual_loss(p_inp_pred, p_inp_gt)
            p_loss = rearrange(p_loss, "(b v) c h w -> b v c h w", b=bs)
            p_loss = p_loss.mean()
            loss = loss + (p_loss * self.loss_args.weight_perceptual)
            set_loss("perceptual", split, p_loss, self.loss_args.weight_perceptual)
        if self.loss_args.weight_perceptual_vol > 0 and iter < iter_end - 1000:
            # Perceptual loss for volume-gs
            p_inp_pred_vol = maybe_resize(
                render_pkg_volume["image"].reshape(-1, 3, self.camera_args.resolution[0], self.camera_args.resolution[1]),
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_inp_mask_vol = maybe_resize(
                mask_dptm.reshape(-1, 1, self.camera_args.resolution[0], self.camera_args.resolution[1]), 
                tgt_reso=self.loss_args.perceptual_resolution
            )
            p_loss_vol = self.perceptual_loss(p_inp_pred_vol * p_inp_mask_vol, p_inp_gt * p_inp_mask_vol)
            p_loss_vol = rearrange(p_loss_vol, "(b v) c h w -> b v c h w", b=bs)
            p_loss_vol = p_loss_vol.mean()
            loss = loss + (p_loss_vol * self.loss_args.weight_perceptual_vol)
            set_loss("perceptual_vol", split, p_loss_vol, self.loss_args.weight_perceptual_vol)

        # ==================== Depth loss ===================== #
        ### Depth loss for omni-gs. For regularization use.
        if self.loss_args.weight_depth_abs > 0:
            depth_abs_loss = torch.abs(render_pkg_fuse["depth"].squeeze(2) - depth_m_gt)
            depth_abs_loss = depth_abs_loss * conf_m_gt
            depth_abs_loss = depth_abs_loss.mean()
            loss = loss + self.loss_args.weight_depth_abs * depth_abs_loss
            set_loss("depth_abs", split, depth_abs_loss, self.loss_args.weight_depth_abs)
        ### Depth loss for volume-gs
        if self.loss_args.weight_depth_abs_vol > 0 and iter < iter_end - 1000:
            if self.loss_args.depth_abs_loss_vol_type == "mask":
                depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"].squeeze(2) * mask_dptm - depth_m_gt * mask_dptm)
                depth_abs_loss_vol = depth_abs_loss_vol * conf_m_gt
            elif self.loss_args.depth_abs_loss_vol_type == "mask_self":
                depth_m_gt_pseudo = render_pkg_pixel["depth"].squeeze(2).detach()
                depth_abs_loss_vol = torch.abs(render_pkg_volume["depth"].squeeze(2) * mask_dptm - depth_m_gt_pseudo * mask_dptm)
            depth_abs_loss_vol = depth_abs_loss_vol.mean()
            loss = loss + self.loss_args.weight_depth_abs_vol * depth_abs_loss_vol
            set_loss("depth_abs_vol", split, depth_abs_loss_vol, self.loss_args.weight_depth_abs_vol)
        
        return loss, loss_terms, render_pkg_fuse, render_pkg_pixel, render_pkg_volume, gaussians_all, gaussians_pixel, gaussians_volume, data_dict
    
    def validation_step(self, batch, val_result_savedir):
        (loss_val, loss_term_val, render_pkg_fuse,
         render_pkg_pixel, render_pkg_volume, gaussians_all,
         gaussians_pixel, gaussians_volume, batch_data) = \
            self.forward(batch, "val")
        self.save_val_results(batch_data, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                                gaussians_all, gaussians_pixel, gaussians_volume, val_result_savedir)
        return loss_term_val
    
    def forward_test(self, batch):
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img, status="test")

        # pixel-gs prediction
        with self.benchmarker.time("pixel_gs"):
            gaussians_pixel, gaussians_feat = self.pixel_gs(
                    rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                    data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                    data_dict["rays_o"], data_dict["rays_d"], status="test")

        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        with self.benchmarker.time("volume_gs"):
            gaussians_volume = self.volume_gs(
                    [img_feats[0]],
                    gaussians_pixel_mask,
                    gaussians_feat_mask,
                    data_dict["img_metas"], status="test")
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)
        bs = gaussians_all.shape[0]
        render_c2w = data_dict["output_c2ws"]
        render_fovxs = data_dict["output_fovxs"]
        render_fovys = data_dict["output_fovys"]
        
        with self.benchmarker.time("render", num_calls=render_c2w.shape[1]):
            render_pkg_fuse = self.renderer.render(
                gaussians=gaussians_all,
                c2w=render_c2w,
                fovx=render_fovxs,
                fovy=render_fovys,
                rays_o=None,
                rays_d=None
            )

        output_imgs = render_pkg_fuse["image"] # b v 3 h w
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w

        target_imgs = data_dict["output_imgs"] # b v 3 h w
        target_depths = data_dict["output_depths"] # b v h w
        target_depths_m = data_dict["output_depths_m"] # b v h w

        preds = {"img": output_imgs, "depth": output_depths, "gaussian": gaussians_all}
        gts = {"img": target_imgs, "depth": target_depths, "depth_m": target_depths_m}

        return preds, gts, data_dict["bin_token"]
    
    def forward_demo(self, batch):
        
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]
        img_feats = self.extract_img_feat(img=img, status="test")

        print(data_dict.keys())
        print(data_dict['bin_token'])
        # pixel-gs prediction
        gaussians_pixel, gaussians_feat = self.pixel_gs(
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),
                data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
                data_dict["rays_o"], data_dict["rays_d"])
        
        # volume-gs prediction
        pc_range = self.dataset_params.pc_range
        x_start, y_start, z_start, x_end, y_end, z_end = pc_range
        gaussians_pixel_mask, gaussians_feat_mask = [], []
        for b in range(bs):
            mask_pixel_i = (gaussians_pixel[b, :, 0] >= x_start) & (gaussians_pixel[b, :, 0] <= x_end) & \
                        (gaussians_pixel[b, :, 1] >= y_start) & (gaussians_pixel[b, :, 1] <= y_end) & \
                        (gaussians_pixel[b, :, 2] >= z_start) & (gaussians_pixel[b, :, 2] <= z_end)
            gaussians_pixel_mask_i = gaussians_pixel[b][mask_pixel_i]
            gaussians_feat_mask_i = gaussians_feat[b][mask_pixel_i]
            gaussians_pixel_mask.append(gaussians_pixel_mask_i)
            gaussians_feat_mask.append(gaussians_feat_mask_i)
        gaussians_volume = self.volume_gs(
                [img_feats[0]],
                gaussians_pixel_mask,
                gaussians_feat_mask,
                data_dict["img_metas"])
        
        gaussians_all = torch.cat([gaussians_pixel, gaussians_volume], dim=1)

        # optimize
        gaussians_all = self.optimize_gaussians(data_dict, gaussians_all, num_iters=0)

        bs = gaussians_all.shape[0]     
        # forward 3 meters, return, and then rotate. backward 3 meters, return, and then rotate.
        c2w_cf = data_dict["output_c2ws"][:, -6]
        c2w_cf_forward = c2w_cf.clone()
        c2w_cf_forward[..., 1, 3] = c2w_cf_forward[..., 1, 3] + 3
        c2w_cfr = data_dict["output_c2ws"][:, -5]
        c2w_cfl = data_dict["output_c2ws"][:, -4]
        c2w_cb = data_dict["output_c2ws"][:, -3]
        c2w_cb[..., 1, 3] = c2w_cb[..., 1, 3] + 1.5
        c2w_cb_backward = c2w_cb.clone()
        c2w_cb_backward[..., 1, 3] = c2w_cb_backward[..., 1, 3] - 3
        c2w_cbl = data_dict["output_c2ws"][:, -2]
        c2w_cbr = data_dict["output_c2ws"][:, -1]
        # cf -> cfr -> cbr -> cb -> cbl -> cfl -> cf
        # TODO: set as parameters
        num_frames_short = 60
        num_frames_long = 120
        num_frames_all = 60 * 4 + 120 * 6
        t_short = torch.linspace(0, 1, num_frames_short, dtype=torch.float32, device=self.device)
        t_long = torch.linspace(0, 1 - 1 / (num_frames_long + 1), num_frames_long, dtype=torch.float32, device=self.device)
        # obtain camera trajectories for each clip
        c2w_interp_forward0 = interpolate_extrinsics(c2w_cf, c2w_cf_forward, t_short)
        c2w_interp_forward1 = interpolate_extrinsics(c2w_cf_forward, c2w_cf, t_short)
        c2w_interp_0 = interpolate_extrinsics(c2w_cf, c2w_cfr, t_long)
        c2w_interp_1 = interpolate_extrinsics(c2w_cfr, c2w_cbr, t_long)
        c2w_interp_2 = interpolate_extrinsics(c2w_cbr, c2w_cb, t_long)
        c2w_interp_backward0 = interpolate_extrinsics(c2w_cb, c2w_cb_backward, t_short)
        c2w_interp_backward1 = interpolate_extrinsics(c2w_cb_backward, c2w_cb, t_short)
        c2w_interp_3 = interpolate_extrinsics(c2w_cb, c2w_cbl, t_long)
        c2w_interp_4 = interpolate_extrinsics(c2w_cbl, c2w_cfl, t_long)
        c2w_interp_5 = interpolate_extrinsics(c2w_cfl, c2w_cf, t_long)
        c2w_interp = torch.cat([c2w_interp_forward0, c2w_interp_forward1,
                                c2w_interp_0, c2w_interp_1, c2w_interp_2,
                                c2w_interp_backward0, c2w_interp_backward1,
                                c2w_interp_3, c2w_interp_4, c2w_interp_5], dim=1)
        fovxs_interp = data_dict["output_fovxs"][:, -6:-5].repeat(1, num_frames_all)
        fovys_interp = data_dict["output_fovys"][:, -6:-5].repeat(1, num_frames_all)
        
        render_pkg_fuse = self.renderer.render(
            gaussians=gaussians_all,
            c2w=c2w_interp,
            fovx=fovxs_interp,
            fovy=fovys_interp,
            rays_o=None,
            rays_d=None
        )

        output_imgs = render_pkg_fuse["image"] # b v 3 h w
        output_depths = render_pkg_fuse["depth"].squeeze(2) # b v h w

        nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuScenes")
        output_imgs = draw_nuscenes_boxes(
            nusc=nusc,
            bin_token=data_dict["bin_token"],
            output_imgs=output_imgs,
            c2w_interp=c2w_interp,
            fovxs_interp=fovxs_interp,
            fovys_interp=fovys_interp,
            pkl_dir="data/nuScenes/v1.0-trainval/bin_infos_3.2m",
            data_root="data/nuScenes"
        )

        preds = {"img": output_imgs, "depth": output_depths}

        return preds, data_dict["bin_token"]

    def save_val_results(self, batch_gt, render_pkg_fuse, render_pkg_pixel, render_pkg_volume,
                         gaussians_all, gaussians_pixel, gaussians_volume, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        batch_size = render_pkg_fuse["image"].shape[0]
        n_rand_view = render_pkg_fuse["image"].shape[1]

        rgbs_gt = batch_gt["output_imgs"].cpu()
        depths_gt = batch_gt["output_depths"]
        depths_gt = (depths_gt / depths_gt.max()).unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        depths_m_gt = batch_gt["output_depths_m"]
        depths_m_gt = (depths_m_gt / 255.0).unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        confs_m_gt = batch_gt["output_confs_m"]
        confs_m_gt = confs_m_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        mask_dptm = batch_gt["mask_dptm"].unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()

        def save_vis(prefix, i, save_dir, n_rand_view, render_pkg, gaussians, rgbs_gt, depths_m_gt, mask_dptm, renderer):
            sample_save_dir = os.path.join(save_dir, f"sample-{i}-{prefix}")
            os.makedirs(sample_save_dir, exist_ok=True)

            for v in range(n_rand_view):
                rgb = render_pkg["image"][i, v].cpu()
                depth = render_pkg["depth"][i, v]
                h, w = depth.shape[1:]
                depth_abs = depth.repeat(3, 1, 1).cpu() / 255.0
                cat_gt = torch.cat(
                        [rgbs_gt[i, v], depths_m_gt[i, v], mask_dptm[i, v]],
                        dim=-1
                    )
                cat_pred = torch.cat(
                        [rgb, depth_abs, mask_dptm[i, v]], dim=-1
                    )
                grid = torch.cat(
                    [cat_gt, cat_pred], dim=1
                )
                grid = (grid.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join(sample_save_dir, f"{v}.png"), grid)
            if gaussians is not None:
                gs_save_path = os.path.join(sample_save_dir, f"sample-{i}-{prefix}.ply")
                gaussians_reformat = torch.cat([gaussians[i:i+1, :, 0:3],
                                                gaussians[i:i+1, :, 6:7],
                                                gaussians[i:i+1, :, 11:14],
                                                gaussians[i:i+1, :, 7:11],
                                                gaussians[i:i+1, :, 3:6]], dim=-1)
                renderer.save_ply(gaussians_reformat, gs_save_path)
        for i in range(batch_size):
            save_vis("omni", i, save_dir, n_rand_view, render_pkg_fuse, gaussians_all, rgbs_gt, depths_m_gt, mask_dptm, self.renderer)
        
        if render_pkg_pixel is not None:
            for i in range(batch_size):
                save_vis("pixel", i, save_dir, n_rand_view, render_pkg_pixel, None, rgbs_gt, depths_m_gt, mask_dptm, self.renderer)
        
        if render_pkg_volume is not None:
            for i in range(batch_size):
                save_vis("volume", i, save_dir, n_rand_view, render_pkg_volume, None, rgbs_gt, depths_m_gt, mask_dptm, self.renderer)

    def optimize_gaussians(self, data_dict, gaussians_all, num_iters=30, lr=2e-3):
        
        assert gaussians_all.dim() == 3, f"should be [B,N,14], and the actual size: {gaussians_all.shape}"
        
        position = gaussians_all[..., 0:3].detach().clone().to('cuda').requires_grad_(True)
        color = gaussians_all[..., 3:6].detach().clone().to('cuda').requires_grad_(True)
        opacity = gaussians_all[..., 6:7].detach().clone().to('cuda').requires_grad_(True)
        rotation = gaussians_all[..., 7:11].detach().clone().to('cuda').requires_grad_(True)
        scale = gaussians_all[..., 11:14].detach().clone().to('cuda').requires_grad_(True)
        
        # param
        params_group = [
            # {'params': [position], 'lr': lr},
            {'params': [color], 'lr': lr},
            {'params': [opacity], 'lr': lr},
            # {'params': [rotation], 'lr': lr},
            # {'params': [scale], 'lr': lr}
        ]

        ssim_criterion = SSIM().to('cuda')
        optimizer = torch.optim.Adam(params_group)
        
        # gt
        gt_imgs = data_dict["imgs"].squeeze(0).float().to('cuda')
        gt_np = gt_imgs.detach().cpu().numpy()
        
        # gt_imgs = gt_imgs.unsqueeze(0)
        for view_idx in range(6):  # 6 views
            view_data = gt_np[view_idx] # (3,224,400)
            view_data = np.transpose(view_data, (1, 2, 0))  # (224,400,3)

            view_data = (view_data * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(f"gt_{view_idx}.png"), 
                view_data
            )

        for iter in range(num_iters):
            optimizer.zero_grad()
            loss = 0.0

            optimized_gaussians = torch.cat([position, color, opacity, rotation, scale], dim=-1)
            
            # render
            render_pkg = self.renderer.render(
                gaussians=optimized_gaussians,
                c2w=data_dict["output_c2ws"][:, -6:],
                fovx=data_dict["output_fovxs"][:, -6:],
                fovy=data_dict["output_fovys"][:, -6:]
            )
            rendered_imgs = render_pkg["image"]
            
            # l1 loss
            # print(rendered_imgs.shape)
            # print(gt_imgs.shape)
            l1_loss = torch.abs(rendered_imgs - gt_imgs.unsqueeze(0)).sum()
            
            rendered_imgs = rendered_imgs.squeeze(0)
            ssim_loss = 0.0
            for v in range(6):
                rendered_view = rendered_imgs[v].unsqueeze(0)  # [1,3,224,400]
                gt_view = gt_imgs[v].unsqueeze(0)              # [1,3,224,400]
                # print(rendered_view.shape)
                # print(gt_view.shape)
                ssim_loss += 1 - ssim_criterion(rendered_view, gt_view)
            
            ssim_loss = ssim_loss * 1e6
            
            loss += l1_loss + ssim_loss
            loss.backward()
            
            # update
            optimizer.step()
            
            print(f"iter {iter+1}/{num_iters} | loss: {loss.item():.4f}")

            # if iter == 0:
            #     rendered_np = rendered_imgs.detach().cpu().numpy()
            #     print(f"[DEBUG] render shape: {rendered_np.shape}")  # (6,3,224,400)

            #     for v in range(6):
                    
            #         view_data = rendered_np[v]  # (3,224,400)
            #         view_data = view_data.transpose(1, 2, 0)  # (224,400,3)

            #         view_data = (view_data * 255).astype(np.uint8)
            #         imageio.imwrite(
            #             os.path.join(f"iter0_v{v}.png"),
            #             view_data
            #         )

        return torch.cat([position, color, opacity, rotation, scale], dim=-1).detach()