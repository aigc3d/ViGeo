from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


depth_meta_data = {
    'sintel': {"max_depth": 70, "post_clip_max": 70},
    "bonn": {"max_depth": 70},
    "kitti": {"max_depth": None},
    "hammer": {"max_depth": None},
}


def _to_tensor(value):
    return torch.from_numpy(value) if isinstance(value, np.ndarray) else value


def as_depth_stack(depth, name='depth') -> torch.Tensor:
    """
    Canonical depth shape: [N, 1, H, W].

    Accepted explicit inputs:
    - [H, W]
    - [N, H, W]
    - [N, 1, H, W]
    - [N, H, W, 1]
    - [1, N, 1, H, W]
    - [1, N, H, W, 1]
    """
    depth = _to_tensor(depth)
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray.")

    if depth.dim() == 2:
        return depth.unsqueeze(0).unsqueeze(0)
    if depth.dim() == 3:
        if depth.shape[-1] == 1:
            return depth.permute(2, 0, 1).unsqueeze(0)
        return depth.unsqueeze(1)
    if depth.dim() == 4:
        if depth.shape[1] == 1:
            return depth
        if depth.shape[-1] == 1:
            return depth.permute(0, 3, 1, 2)
        if depth.shape[0] == 1:
            return depth.permute(1, 0, 2, 3)
    if depth.dim() == 5 and depth.shape[0] == 1:
        if depth.shape[2] == 1:
            return depth.squeeze(0)
        if depth.shape[-1] == 1:
            return depth.squeeze(0).permute(0, 3, 1, 2)

    raise ValueError(f"{name} must be convertible to [N, 1, H, W], got {tuple(depth.shape)}.")


def as_points_stack(points, name='points') -> torch.Tensor:
    """
    Canonical point-map shape: [N, H, W, 3].
    """
    points = _to_tensor(points)
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray.")

    if points.dim() == 3 and points.shape[-1] == 3:
        return points.unsqueeze(0)
    if points.dim() == 4:
        if points.shape[-1] == 3:
            return points
        if points.shape[1] == 3:
            return points.permute(0, 2, 3, 1)
    if points.dim() == 5 and points.shape[0] == 1:
        points = points.squeeze(0)
        if points.shape[-1] == 3:
            return points
        if points.shape[1] == 3:
            return points.permute(0, 2, 3, 1)

    raise ValueError(f"{name} must be convertible to [N, H, W, 3], got {tuple(points.shape)}.")


def as_normal_stack(normal, name='normal') -> torch.Tensor:
    """
    Canonical normal shape: [N, 3, H, W].
    """
    normal = _to_tensor(normal)
    if not isinstance(normal, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray.")

    if normal.dim() == 3:
        if normal.shape[0] == 3:
            return normal.unsqueeze(0)
        if normal.shape[-1] == 3:
            return normal.permute(2, 0, 1).unsqueeze(0)
    if normal.dim() == 4:
        if normal.shape[1] == 3:
            return normal
        if normal.shape[-1] == 3:
            return normal.permute(0, 3, 1, 2)
    if normal.dim() == 5 and normal.shape[0] == 1:
        normal = normal.squeeze(0)
        if normal.shape[1] == 3:
            return normal
        if normal.shape[-1] == 3:
            return normal.permute(0, 3, 1, 2)

    raise ValueError(f"{name} must be convertible to [N, 3, H, W], got {tuple(normal.shape)}.")


def _resize_depth_pred(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
    if depth_pred.shape[0] != depth_gt.shape[0]:
        raise ValueError(
            f"Depth frame count mismatch: pred {tuple(depth_pred.shape)}, gt {tuple(depth_gt.shape)}."
        )
    if depth_pred.shape[-2:] == depth_gt.shape[-2:]:
        return depth_pred
    return F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode='bilinear', align_corners=False)


def _resize_points_pred(points_pred: torch.Tensor, points_gt: torch.Tensor) -> torch.Tensor:
    if points_pred.shape[0] != points_gt.shape[0]:
        raise ValueError(
            f"Point-map frame count mismatch: pred {tuple(points_pred.shape)}, gt {tuple(points_gt.shape)}."
        )
    if points_pred.shape[1:3] == points_gt.shape[1:3]:
        return points_pred
    points_chw = points_pred.permute(0, 3, 1, 2)
    points_chw = F.interpolate(points_chw, size=points_gt.shape[1:3], mode='bilinear', align_corners=False)
    return points_chw.permute(0, 2, 3, 1)


def _resize_normal_pred(normal_pred: torch.Tensor, normal_gt: torch.Tensor) -> torch.Tensor:
    if normal_pred.shape[0] != normal_gt.shape[0]:
        raise ValueError(
            f"Normal frame count mismatch: pred {tuple(normal_pred.shape)}, gt {tuple(normal_gt.shape)}."
        )
    if normal_pred.shape[-2:] == normal_gt.shape[-2:]:
        return normal_pred
    return F.interpolate(normal_pred, size=normal_gt.shape[-2:], mode='bilinear', align_corners=False)


def estimate_depth_scale(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
    s = torch.nanmean(depth_gt) / torch.nanmean(depth_pred)

    for _ in range(10):
        residuals = s * depth_pred - depth_gt
        weights = 1.0 / (residuals.abs() + 1e-8)
        numerator = torch.sum(weights * depth_pred * depth_gt)
        denominator = torch.sum(weights * depth_pred * depth_pred)
        s = numerator / denominator

    return s.clamp(min=1e-3).detach()


def estimate_depth_affine(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isfinite(depth_pred) & torch.isfinite(depth_gt)
    pred_values, gt_values = depth_pred[mask], depth_gt[mask]

    if pred_values.numel() == 0:
        return torch.tensor(1.0, device=depth_pred.device), torch.tensor(0.0, device=depth_pred.device)

    scale_init = (torch.median(gt_values) / (torch.median(pred_values) + 1e-8)).item()
    scale = torch.tensor([scale_init], requires_grad=True, device=pred_values.device, dtype=pred_values.dtype)
    shift = torch.zeros(1, requires_grad=True, device=pred_values.device, dtype=pred_values.dtype)
    optimizer = torch.optim.Adam([scale, shift], lr=1e-4)
    prev_loss = float('inf')

    for _ in range(1000):
        optimizer.zero_grad()
        loss = (scale * pred_values + shift - gt_values).abs().sum()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if abs(prev_loss - loss_val) < 1e-6:
            break
        prev_loss = loss_val

    return scale.detach().squeeze().clamp(min=1e-3), shift.detach().squeeze()


def compute_depth_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    max_depth: float | None = 80.0,
    post_clip_min: float | None = None,
    post_clip_max: float | None = None,
    align_method: Literal['scale', 'affine', 'metric'] = 'scale',
    metrics: list[str] = ['absrel', 'd1'],
):
    depth_gt = as_depth_stack(depth_gt, 'depth_gt').float()
    depth_pred = as_depth_stack(depth_pred, 'depth_pred').to(depth_gt.device).float()
    depth_pred = _resize_depth_pred(depth_pred, depth_gt)

    if max_depth is not None:
        mask = (depth_gt > 0) & (depth_gt < max_depth)
    else:
        mask = depth_gt > 0

    if torch.sum(mask) == 0:
        return None

    depth_pred = depth_pred[mask]
    depth_gt = depth_gt[mask]

    if align_method == 'scale':
        depth_pred = depth_pred * estimate_depth_scale(depth_pred, depth_gt)
    elif align_method == 'affine':
        scale, shift = estimate_depth_affine(depth_pred, depth_gt)
        depth_pred = depth_pred * scale + shift
    elif align_method == 'metric':
        pass
    else:
        raise ValueError(f"Unsupported depth align_method: {align_method}")

    if post_clip_min is not None:
        depth_pred = torch.clamp(depth_pred, min=post_clip_min)
    if post_clip_max is not None:
        depth_pred = torch.clamp(depth_pred, max=torch.max(depth_gt))

    depth_gt = torch.clamp(depth_gt, min=1e-6)
    depth_pred = torch.clamp(depth_pred, min=1e-6)

    abs_rel = torch.mean(torch.abs(depth_pred - depth_gt) / depth_gt).item()
    sq_rel = torch.mean(((depth_pred - depth_gt) ** 2) / depth_gt).item()
    rmse = torch.sqrt(torch.mean((depth_pred - depth_gt) ** 2)).item()
    log_rmse = torch.sqrt(torch.mean((torch.log(depth_pred) - torch.log(depth_gt)) ** 2)).item()

    ratio = torch.maximum(depth_pred / depth_gt, depth_gt / depth_pred)
    total_metrics = {
        "absrel": abs_rel,
        "sqrel": sq_rel,
        "rmse": rmse,
        "logrmse": log_rmse,
        "d1": (ratio < 1.25).float().mean().item(),
        "d2": (ratio < 1.25 ** 2).float().mean().item(),
        "d3": (ratio < 1.25 ** 3).float().mean().item(),
        "perfect": (ratio < 1.0).float().mean().item(),
    }

    return {key: total_metrics[key] for key in metrics}


def estimate_point_scale(
    points: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
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

    return numerator / denominator.clamp(min=1e-8)


def compute_points_metrics(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    max_depth: float | None = 80.0,
    post_clip_max: float | None = None,
    align_method: Literal['scale', 'metric'] = 'scale',
    use_weight: bool = True,
    metrics: list[str] = ['absrel', 'd1'],
):
    points_gt = as_points_stack(points_gt, 'points_gt').float()
    points_pred = as_points_stack(points_pred, 'points_pred').to(points_gt.device).float()
    points_pred = _resize_points_pred(points_pred, points_gt)

    depth_gt = points_gt[..., 2]
    if max_depth is not None:
        mask = (depth_gt > 1e-3) & (depth_gt < max_depth)
    else:
        mask = depth_gt > 1e-3

    if align_method == 'scale':
        weight = 1.0 / (depth_gt + 1e-6) if use_weight else None
        points_pred = points_pred * estimate_point_scale(points_pred, points_gt, mask=mask, weight=weight)
    elif align_method == 'metric':
        pass
    else:
        raise ValueError(f"Unsupported point align_method: {align_method}")

    if post_clip_max is not None:
        pred_z = points_pred[..., 2:3]
        rays = points_pred / (pred_z + 1e-8)
        points_pred = rays * torch.clamp(pred_z, max=post_clip_max)

    pred_values = points_pred[mask]
    gt_values = points_gt[mask]
    if pred_values.numel() == 0:
        return {key: 0.0 for key in metrics}

    error_3d = torch.norm(pred_values - gt_values, p=2, dim=-1)
    gt_mag = torch.norm(gt_values, p=2, dim=-1).clamp(min=1e-2)
    rel_error = error_3d / gt_mag

    total_metrics = {
        "absrel": torch.mean(rel_error).item(),
        "rmse": torch.sqrt(torch.mean(error_3d ** 2)).item(),
        "d1": (rel_error < 0.25).float().mean().item(),
    }

    return {key: total_metrics[key] for key in metrics}


def compute_normal_metrics(
    normal_pred,
    normal_gt,
    metrics: list = ['mean', 'median', 'rmse', 'a1', 'a2', 'a3', 'a4', 'a5'],
):
    normal_gt = as_normal_stack(normal_gt, 'normal_gt').float()
    normal_pred = as_normal_stack(normal_pred, 'normal_pred').to(normal_gt.device).float()
    normal_pred = _resize_normal_pred(normal_pred, normal_gt)

    gt_mag = torch.norm(normal_gt, p=2, dim=1)
    mask = (gt_mag > 0.5) & ~torch.isnan(normal_gt).any(dim=1) & ~torch.isnan(normal_pred).any(dim=1)

    normal_pred = F.normalize(normal_pred, p=2, dim=1)
    normal_gt = F.normalize(normal_gt, p=2, dim=1)

    pred_valid = normal_pred.permute(0, 2, 3, 1)[mask]
    gt_valid = normal_gt.permute(0, 2, 3, 1)[mask]

    if pred_valid.numel() == 0:
        return {key: float('nan') for key in metrics}

    cos_sim = torch.cosine_similarity(pred_valid, gt_valid, dim=-1)
    angles = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * (180.0 / torch.pi)

    total_metrics = {
        'mean': angles.mean().item(),
        'median': angles.median().item(),
        'rmse': torch.sqrt(torch.mean(angles ** 2)).item(),
        'a1': (angles < 5.0).float().mean().item() * 100.0,
        'a2': (angles < 7.5).float().mean().item() * 100.0,
        'a3': (angles < 11.25).float().mean().item() * 100.0,
        'a4': (angles < 22.5).float().mean().item() * 100.0,
        'a5': (angles < 30.0).float().mean().item() * 100.0,
    }

    return {key: total_metrics[key] for key in metrics}
