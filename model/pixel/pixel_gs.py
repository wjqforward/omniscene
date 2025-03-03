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
from ..utils.ops import get_ray_directions, get_rays
from torch.nn.init import normal_


@MODELS.register_module()
class PixelGaussian(BaseModule):

    def __init__(self,
                 down_block=None,
                 mid_block=None,
                 up_block=None,
                 patch_sizes=None,
                 in_embed_dim=128,
                 out_embed_dims=[128, 256, 512, 512],
                 num_cams=6,
                 near=0.1,
                 far=1000.0,
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.plucker_to_embed = nn.Linear(6, out_embed_dims[0])
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, out_embed_dims[0]))
        
        self.down_blocks = nn.ModuleList([])
        in_channels = out_embed_dims[0] + 1 + 1 # concat pseudo depth and conf
        for i, out_embed_dim in enumerate(out_embed_dims):
            is_final_block = i == len(out_embed_dims) - 1
            patch_size = patch_sizes[i] if patch_sizes is not None else None
            down_block.update(kv_compress_ratio=patch_size)
            down_block.update(attention_head_dim=out_embed_dim // down_block["num_attention_heads"])
            down_block.update(in_channels=in_channels)
            down_block.update(out_channels=out_embed_dim)
            down_block.update(add_downsample=not is_final_block)
            if i == 0:
                down_block.update(resnet_groups=1)
            else:
                down_block.update(resnet_groups=32)
            in_channels = out_embed_dim
            down_block_module = MODELS.build(down_block)
            self.down_blocks.append(down_block_module)
        
        # build middle block
        mid_block.update(in_channels=out_embed_dims[-1])
        mid_block.update(out_channels=out_embed_dims[-1])
        mid_block.update(attention_head_dim=out_embed_dims[-1] // mid_block["num_attention_heads"])
        self.mid_block = MODELS.build(mid_block)

        # build upsample blocks
        reversed_out_embed_dims = out_embed_dims[::-1]
        reversed_patch_sizes = patch_sizes[::-1] if patch_sizes is not None else [None] * len(out_embed_dims)
        out_channels = reversed_out_embed_dims[0]
        self.up_blocks = nn.ModuleList([])
        prev_output_channel = out_channels
        for i, (out_embed_dim, patch_size) in enumerate(zip(reversed_out_embed_dims, reversed_patch_sizes)):
            out_channels = reversed_out_embed_dims[i]
            in_channels = reversed_out_embed_dims[i]
            is_final_block = i == len(reversed_out_embed_dims) - 1
            up_block.update(attention_head_dim=out_embed_dim // up_block["num_attention_heads"])
            up_block.update(kv_compress_ratio=patch_size)
            up_block.update(in_channels=in_channels)
            up_block.update(prev_output_channel=prev_output_channel)
            up_block.update(out_channels=out_channels)
            up_block.update(add_upsample=not is_final_block)
            up_block_module = MODELS.build(up_block)
            self.up_blocks.append(up_block_module)
            prev_output_channel = out_channels
        
        # output & post-process
        self.num_cams = num_cams
        self.near = near
        self.far = far
        self.num_surfaces = 1

        self.upsampler = nn.Sequential(
            nn.Conv2d(in_embed_dim, out_embed_dims[0], 3, 1, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )

        gs_channels = 3 + 1 + 3 + 4 + 3 # offset, opacity, scale, rotation, rgb
        self.gs_channels = gs_channels
        self.feature_norm = nn.GroupNorm(num_channels=out_embed_dims[0], num_groups=32, eps=1e-6)
        self.to_gaussians = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(out_embed_dims[0], gs_channels, 1),
        )
        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: torch.exp(x) * 0.01
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        
        self.delta_clamp = lambda x: x.clamp(-10.0, 6.0)
        self.delta_act = torch.exp
        
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
    
    def forward(self, img_feats, depths_in, confs_in, pluckers, origins, directions, status="train"):
        """Forward training function."""
        # upsample 4x downsampled img features to original size
        img_feats = self.upsampler(img_feats)
        bs = origins.shape[0]
        img_feats = rearrange(img_feats, "(b v) c h w -> b v h w c", b=bs, v=self.num_cams)

        pluckers = rearrange(pluckers, "b v c h w -> b v h w c")
        plucker_embeds = self.plucker_to_embed(pluckers)
        img_feats = img_feats + self.cams_embeds[None, :, None, None] + plucker_embeds
        img_feats = rearrange(img_feats, "b v h w c -> (b v) c h w")

        # rearrange pseudo depths and confs
        depths_in = rearrange(depths_in, "b v h w -> (b v) () h w")
        confs_in = rearrange(confs_in, "b v h w -> (b v) () h w")
        img_feats = torch.cat([img_feats, depths_in / 20.0, confs_in], dim=1)

        # downsample
        sample = img_feats
        down_block_res_samples = (sample,)
        for block_id, down_block in enumerate(self.down_blocks):
            if self.use_checkpoint and status != "test":
                sample, res_samples = torch.utils.checkpoint.checkpoint(
                    down_block, sample, use_reentrant=False)
            else:
                sample, res_samples = down_block(sample)
            down_block_res_samples += res_samples
        
        # middile
        sample = self.mid_block(sample)

        # upsample
        for block_id, up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            if self.use_checkpoint and status != "test":
                input_vars = (sample, res_samples)
                sample = torch.utils.checkpoint.checkpoint(
                    up_block, *input_vars, use_reentrant=False
                )
            else:
                sample = up_block(sample, res_samples)
       
        # rearrange features
        features = self.feature_norm(sample)
        bs = origins.shape[0]

        # post-process
        _, _, h, w = features.shape

        gaussians = self.to_gaussians(features)
        gaussians = rearrange(gaussians, "(b v) (n c) h w -> b (v h w n) c",
                              b=bs, v=self.num_cams, n=1, c=self.gs_channels)
        offsets = gaussians[..., :3]
        opacities = self.opt_act(gaussians[..., 3:4])
        scales = self.scale_act(gaussians[..., 4:7])
        rotations = self.rot_act(gaussians[..., 7:11])
        rgbs = self.rgb_act(gaussians[..., 11:14])

        depths_in = rearrange(depths_in, "(b v) c h w-> b (v h w) c", b=bs, v=self.num_cams)

        origins = rearrange(origins, "b v h w c -> b (v h w) c")
        origins = origins.unsqueeze(-2)
        directions = rearrange(directions, "b v h w c -> b (v h w) c")
        directions = directions.unsqueeze(-2)
        means = origins + directions * depths_in[..., None]
        means = rearrange(means, "b r n c -> b (r n) c")
        means = means + offsets

        gaussians = torch.cat([means, rgbs, opacities, rotations, scales], dim=-1)
        features = rearrange(features, "(b v) c h w -> b (v h w) c", b=bs, v=self.num_cams)
        features = features.unsqueeze(2) # b v*h*w n c
        features = rearrange(features, "b r n c -> b (r n) c")

        return gaussians, features