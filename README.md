<h1 align=center font-weight:100> [CVPR2025] <strong><i>Omni-Scene</i></strong>⚡️: Omni-Gaussian Representation for Ego-Centric Sparse-View Scene Reconstruction</h1>

<p align="center">
    <a href="weidongxu@westlake.edu.cn">Dongxu Wei</a>
    &nbsp;·&nbsp;
    <a href="lizhiqi49@westlake.edu.cn">Zhiqi Li</a>
    &nbsp;·&nbsp;
    <a href="https://ethliup.github.io/">Peidong Liu*</a>
    <h3 align="center"><a href="https://arxiv.org/abs/2412.06273">Paper</a> | <a href="https://wswdx.github.io/omniscene">Project Page</a> </h3>
  </p>

This as an official implementation of our CVPR 2025 paper 
[**Omni-Scene**: Omni-Gaussian Representation for Ego-Centric Sparse-View Scene Reconstruction](https://wswdx.github.io/omniscene), based on the [mmdet3d](https://github.com/open-mmlab/mmdetection3d) framework and [accelerate](https://github.com/huggingface/accelerate) library.

<div align="center">
  <img src="assets/figures/omniscene.jpg" alt=""  width="1100" />
</div>

## News
- 2025/2/27: Our paper is accepted by CVPR 2025!
- 2025/2/27: Code and data will be released in one week. Stay tuned.

## TODO
- [ ] Upload code and data

## Demo (3D Reconstruction)

Reconstruct scenes with 3D Gaussians given 6-view images from [nuScenes](https://www.nuscenes.org/) in a feed-forward manner:

<video src="assets/videos/recon_examples_normal.mp4" alt=""  width="1100" autoplay="true"></video>

## Demo (3D Generation)

Generate scenes with 3D Gaussians by combining our method with 2D multi-view diffusion model [MagicDrive](https://github.com/cure-lab/MagicDrive):

<video src="assets/videos/synth_examples.m4v" alt=""  width="1100" autoplay="true"></video>


## Citation

If you find this useful, please consider citing:

```bibtex
@inproceedings{wei2024omniscene,
    author = {Wei, Dongxu and Li, Zhiqi and Liu, Peidong},
    title = {Omni-scene: omni-gaussian representation for ego-centric sparse-view scene reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2025}
}
```

## Acknowledgments

Omni-Scene is based on [mmdetection3d]((https://github.com/open-mmlab/mmdetection3d)). It is also greatly inspired by the following outstanding contributions to the open-source community: [TPVFormer](https://github.com/wzzheng/TPVFormer), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [MVSplat](https://github.com/donydchen/mvsplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [Metric3D-v2](https://github.com/YvanYin/Metric3D) and [MagicDrive](https://github.com/cure-lab/MagicDrive).
