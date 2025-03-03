from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from torchmetrics import PearsonCorrCoef


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@cache
def get_pcc(device: torch.device):
    return PearsonCorrCoef().to(device)

@torch.no_grad()
def compute_pcc(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height width"],
) -> Float[Tensor, " batch"]:
    value = get_pcc(predicted.device).forward(ground_truth.view(-1), predicted.view(-1))
    return value.mean()

@torch.no_grad()
def compute_absrel(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height width"],
) -> Float[Tensor, " batch"]:
    results_absrel = []
    results_rmse = []
    for i in range(ground_truth.shape[0]):
        gt_depth = ground_truth[i]
        pred_depth = predicted[i]
        mask = gt_depth > 0
        gt_depth = gt_depth[mask]
        pred_depth = pred_depth[mask]
        gt_depth[pred_depth < 1e-3] = 1e-3
        gt_depth[pred_depth > 80] = 80
        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80] = 80
        abs_rel = torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth).unsqueeze(0)
        rmse = (gt_depth - pred_depth) ** 2
        rmse = torch.sqrt(rmse.mean()).unsqueeze(0)
        if torch.isnan(abs_rel).sum() != 0 or torch.isnan(rmse).sum() != 0:
            abs_rel[:] = 0.
            rmse[:] = 0.
        results_absrel.append(abs_rel)
        results_rmse.append(rmse)
    results_absrel = torch.cat(results_absrel, dim=0)
    results_rmse = torch.cat(results_rmse, dim=0)
    return results_absrel, results_rmse

@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
