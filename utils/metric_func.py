import torch
import torch.nn.functional as F
from typing import Literal
import numpy as np

depth_meta_data = {
    'sintel': {"max_depth": 70, "post_clip_max": 70},
    "bonn": {"max_depth": 70},
    "kitti": {"max_depth": None},
    "hammer": {"max_depth": None},
}

def estimate_depth_scale(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute a robust scale factor `s` such that `s * depth_pred ≈ depth_gt`.

    Uses an IRLS (Iteratively Reweighted Least Squares) scheme to approximate
    L1-optimal scaling, initialized by the ratio of mean depths.

    Args:
        depth_pred: Predicted depth map (any shape, may contain NaNs).
        depth_gt: Ground truth depth map (same shape as depth_pred).

    Returns:
        Scalar scale factor `s` (detached from computation graph, clamped ≥ 1e-3).
    """
    s = torch.nanmean(depth_gt) / torch.nanmean(depth_pred)

    for _ in range(10):
        residuals = s * depth_pred - depth_gt
        weights = 1.0 / (residuals.abs() + 1e-8)

        numerator = torch.sum(weights * depth_pred * depth_gt)
        denominator = torch.sum(weights * depth_pred * depth_pred)
        s = numerator / denominator

    return s.clamp(min=1e-3).detach()

def estimate_depth_affine(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate scale (s) and shift (t) to align depth_pred with depth_gt using L1 loss.
    """
    mask = torch.isfinite(depth_pred) & torch.isfinite(depth_gt)
    p, g = depth_pred[mask], depth_gt[mask]

    if p.numel() == 0:
        return torch.tensor(1.0, device=depth_pred.device), torch.tensor(0.0, device=depth_pred.device)

    s_init = (torch.median(g) / (torch.median(p) + 1e-8)).item()

    s = torch.tensor([s_init], requires_grad=True, device=p.device, dtype=p.dtype)
    t = torch.zeros(1, requires_grad=True, device=p.device, dtype=p.dtype)

    optimizer = torch.optim.Adam([s, t], lr=1e-4)
    prev_loss = float('inf')

    for _ in range(1000):
        optimizer.zero_grad()

        loss = (s * p + t - g).abs().sum()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if abs(prev_loss - loss_val) < 1e-6:
            break
        prev_loss = loss_val

    return s.detach().squeeze().clamp(min=1e-3), t.detach().squeeze()

def compute_depth_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    max_depth: float | None = 80.0,
    post_clip_min: float | None = None,
    post_clip_max: float | None = None,
    align_method: Literal['scale', 'affine', 'metric'] = 'scale',
    metrics: list[str] = ['absrel', 'd1'],
):
    if max_depth is not None:
        mask = (depth_gt > 0) & (depth_gt < max_depth)
    else:
        mask = depth_gt > 0

    if torch.sum(mask) == 0:
        return {key: 0.0 for key in metrics}
    depth_pred = depth_pred.to(depth_gt.device)
    if depth_pred.shape != depth_gt.shape:
        if depth_pred.dim() == 2:
            depth_pred = depth_pred.unsqueeze(0).unsqueeze(0)
        elif depth_pred.dim() == 3:
            depth_pred = depth_pred.unsqueeze(1)

        depth_pred = F.interpolate(
            depth_pred,
            size=depth_gt.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        depth_pred = depth_pred.view(depth_gt.shape)

    depth_pred = depth_pred[mask]
    depth_gt = depth_gt[mask]


    if align_method == 'scale':
        scale = estimate_depth_scale(depth_pred, depth_gt)
        depth_pred = depth_pred * scale
    elif align_method == 'affine':
        scale, shift = estimate_depth_affine(depth_pred, depth_gt)
        depth_pred = depth_pred * scale + shift
    elif align_method == 'metric':
        depth_pred = depth_pred

    if post_clip_min is not None:
        depth_pred = torch.clamp(depth_pred, min=post_clip_min)
    if post_clip_max is not None:
        depth_pred = torch.clamp(depth_pred, max=post_clip_max)

    depth_gt = torch.clamp(depth_gt, min=1e-6)
    depth_pred = torch.clamp(depth_pred, min=1e-6)

    abs_rel = torch.mean(torch.abs(depth_pred - depth_gt) / depth_gt).item()
    sq_rel = torch.mean(((depth_pred - depth_gt) ** 2) / depth_gt).item()
    rmse = torch.sqrt(torch.mean((depth_pred - depth_gt) ** 2)).item()
    log_rmse = torch.sqrt(torch.mean((torch.log(depth_pred) - torch.log(depth_gt)) ** 2)).item()

    ratio = torch.maximum(depth_pred / depth_gt, depth_gt / depth_pred)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()
    perfect = (ratio < 1.01).float().mean().item()

    total_metrics = {
        "absrel": abs_rel,
        "sqrel": sq_rel,
        "rmse": rmse,
        "logrmse": log_rmse,
        "d1": delta1,
        "d2": delta2,
        "d3": delta3,
        "perfect": perfect,
    }

    return {
        key: total_metrics[key] for key in metrics}

def estimate_point_scale(
    points: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None
) -> torch.Tensor:
    ndim = points.shape[-1]

    if mask is not None:
        points = points[mask]
        points_gt = points_gt[mask]
        if weight is not None:
            weight = weight[mask]
    else:
        points = points.reshape(-1, ndim)
        points_gt = points_gt.reshape(-1, ndim)
        if weight is not None:
            weight = weight.reshape(-1)

    if points.numel() == 0:
        return torch.tensor(1.0, device=points.device, dtype=points.dtype)

    if weight is not None:
        weight = weight.unsqueeze(-1)
        numerator = torch.sum(weight * points * points_gt)
        denominator = torch.sum(weight * points * points)
    else:
        numerator = torch.sum(points * points_gt)
        denominator = torch.sum(points * points)

    scale = numerator / denominator.clamp(min=1e-8)
    return scale

def compute_points_metrics(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    max_depth: float | None = 80.0,
    post_clip_max: float | None = None,
    align_method: Literal['scale', 'metric'] = 'scale',
    use_weight: bool = True,
    metrics: list[str] = ['absrel', 'd1', 'rmse'],
):
    """
    Compute metrics for point maps (H, W, 3) with automatic masking based on GT depth.
    """
    # 1. Generate mask from GT depth (Z-channel)
    depth_gt = points_gt[..., 2]
    if max_depth is not None:
        mask = (depth_gt > 1e-3) & (depth_gt < max_depth)
    else:
        mask = depth_gt > 1e-3
    points_pred = points_pred.to(points_gt.device)
    # 2. Resizing if necessary
    if points_pred.shape[:-1] != points_gt.shape[:-1]:
        # Handle (B, T, H, W, 3) or (H, W, 3)
        orig_shape = points_pred.shape
        p_flat = points_pred.reshape(-1, *orig_shape[-3:]).permute(0, 3, 1, 2)
        p_flat = F.interpolate(p_flat, size=points_gt.shape[-3:-1], mode='bilinear', align_corners=False)
        points_pred = p_flat.permute(0, 2, 3, 1).reshape(*points_gt.shape)

    # 3. Alignment
    if align_method == 'scale':
        # Use 1/z as weight for alignment if use_weight is True
        weight = 1.0 / (depth_gt + 1e-6) if use_weight else None
        scale = estimate_point_scale(points_pred, points_gt, mask=mask, weight=weight)
        points_pred = points_pred * scale
    elif align_method == 'metric':
        pass

    if post_clip_max is not None:
        pred_z = points_pred[..., 2:3]
        rays = points_pred / (pred_z + 1e-8)
        clipped_z = torch.clamp(pred_z, max=post_clip_max)
        points_pred = rays * clipped_z

    p = points_pred[mask]
    g = points_gt[mask]

    if p.numel() == 0:
        return {key: 0.0 for key in metrics}

    # 4. Core Calculations
    # Euclidean distance error: sqrt((x-x_gt)^2 + (y-y_gt)^2 + (z-z_gt)^2)
    error_3d = torch.norm(p - g, p=2, dim=-1)

    # Magnitude of GT points: sqrt(x_gt^2 + y_gt^2 + z_gt^2)
    gt_mag = torch.norm(g, p=2, dim=-1).clamp(min=1e-2)

    # Point-wise Relative Error (AbsRel)
    rel_error = error_3d / gt_mag
    abs_rel = torch.mean(rel_error).item()

    # 3D RMSE
    rmse = torch.sqrt(torch.mean(error_3d ** 2)).item()

    # Inlier Percentage (Accuracy)
    # Using 0.25 threshold from your script (rel_error < 25%)
    d1 = (rel_error < 0.25).float().mean().item()

    total_metrics = {
        "absrel": abs_rel,
        "rmse": rmse,
        "d1": d1,
    }

    return {key: total_metrics[key] for key in metrics}

def compute_normal_metrics(
    normal_pred,
    normal_gt,
    metrics: list = ['mean', 'median', 'rmse', 'a1', 'a2', 'a3', 'a4', 'a5']
):
    if isinstance(normal_pred, np.ndarray):
        normal_pred = torch.from_numpy(normal_pred)
    if isinstance(normal_gt, np.ndarray):
        normal_gt = torch.from_numpy(normal_gt)

    normal_pred = normal_pred.float().to(normal_gt.device)
    normal_gt = normal_gt.float()

    def standardize_shape(x):
        if x.dim() == 3:
            if x.shape[-1] == 3:
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
        return x

    normal_pred = standardize_shape(normal_pred)
    normal_gt = standardize_shape(normal_gt)

    if normal_pred.shape != normal_gt.shape:
        normal_pred = F.interpolate(
            normal_pred,
            size=normal_gt.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    gt_mag = torch.norm(normal_gt, p=2, dim=1, keepdim=True)
    mask = (gt_mag > 0.5).squeeze(1)

    mask = mask & ~torch.isnan(normal_gt).any(dim=1) & ~torch.isnan(normal_pred).any(dim=1)

    normal_pred = F.normalize(normal_pred, p=2, dim=1)
    normal_gt = F.normalize(normal_gt, p=2, dim=1)

    pred_valid = normal_pred.permute(0, 2, 3, 1)[mask]
    gt_valid = normal_gt.permute(0, 2, 3, 1)[mask]

    if pred_valid.numel() == 0:
        return {key: float('nan') for key in metrics}

    cos_sim = torch.cosine_similarity(pred_valid, gt_valid, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.acos(cos_sim) * (180.0 / torch.pi)

    total_metrics = {
        'mean': angles.mean().item(),
        'median': angles.median().item(),
        'rmse': torch.sqrt(torch.mean(angles ** 2)).item(),
        'a1': (angles < 5.0).float().mean().item() * 100.0,
        'a2': (angles < 7.5).float().mean().item() * 100.0,
        'a3': (angles < 11.25).float().mean().item() * 100.0,
        'a4': (angles < 22.5).float().mean().item() * 100.0,
        'a5': (angles < 30.0).float().mean().item() * 100.0
    }

    return {key: total_metrics[key] for key in metrics}
