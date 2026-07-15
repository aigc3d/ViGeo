from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class ReconstructionMetrics:
    acc_mean: float
    acc_median: float
    comp_mean: float
    comp_median: float
    nc_mean: float
    nc_median: float
    nc_accuracy_mean: float
    nc_accuracy_median: float
    nc_completion_mean: float
    nc_completion_median: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _masked_median(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        raise ValueError("No valid points remain for reconstruction alignment.")
    if dim is None:
        return torch.median(selected)
    return torch.median(selected, dim=dim).values


def scale_shift_align_points(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    valid_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match StreamVGGT's Regr3D_t_ScaleShiftInv(..., gt_scale=True)."""
    points_pred = points_pred.float().clone()
    points_gt = points_gt.float().clone()
    valid_masks = valid_masks.bool()

    gt_z_shift = _masked_median(points_gt[..., 2], valid_masks)
    pred_z_shift = _masked_median(points_pred[..., 2], valid_masks)
    points_gt[..., 2] -= gt_z_shift
    points_pred[..., 2] -= pred_z_shift

    pred_valid = points_pred[valid_masks]
    gt_valid = points_gt[valid_masks]
    pred_center = torch.median(pred_valid, dim=0).values
    gt_center = torch.median(gt_valid, dim=0).values
    pred_scale = torch.median(torch.linalg.vector_norm(pred_valid - pred_center, dim=-1)).clamp(1e-3, 1e3)
    gt_scale = torch.median(torch.linalg.vector_norm(gt_valid - gt_center, dim=-1))
    points_pred *= gt_scale / pred_scale
    return points_pred, points_gt


def umeyama_align_points(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    valid_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match Pi3's mv_recon Umeyama Sim(3) coarse alignment."""
    points_pred = points_pred.float().clone()
    points_gt = points_gt.float().clone()
    valid_masks = valid_masks.bool()

    pred = points_pred[valid_masks].cpu().numpy().T
    gt = points_gt[valid_masks].cpu().numpy().T
    if pred.shape[1] == 0:
        raise ValueError("No valid points remain for reconstruction alignment.")

    mu_pred = pred.mean(axis=1, keepdims=True)
    mu_gt = gt.mean(axis=1, keepdims=True)
    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt
    var_pred = np.square(pred_centered).sum(axis=0).mean()
    cov = (gt_centered @ pred_centered.T) / pred.shape[1]
    u, singular_values, vh = np.linalg.svd(cov)
    sign = np.eye(pred.shape[0])
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1
    scale = np.trace(np.diag(singular_values) @ sign) / max(var_pred, 1e-12)
    rotation = u @ sign @ vh
    translation = mu_gt - scale * rotation @ mu_pred

    pred_aligned = scale * np.einsum("nhwj,ij->nhwi", points_pred.cpu().numpy(), rotation) + translation.T
    return torch.from_numpy(pred_aligned).to(points_pred.device, dtype=points_pred.dtype), points_gt


def center_crop_tensor(tensor: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    height, width = tensor.shape[-3:-1] if tensor.ndim >= 4 and tensor.shape[-1] in (1, 3) else tensor.shape[-2:]
    if height < crop_size or width < crop_size:
        raise ValueError(f"Cannot crop {crop_size}x{crop_size} from {(height, width)}")
    top = height // 2 - crop_size // 2
    left = width // 2 - crop_size // 2
    if tensor.ndim >= 4 and tensor.shape[-1] in (1, 3):
        return tensor[..., top : top + crop_size, left : left + crop_size, :]
    return tensor[..., top : top + crop_size, left : left + crop_size]


def _nearest_metrics(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    query_normals: np.ndarray,
    reference_normals: np.ndarray,
) -> tuple[float, float, float, float]:
    tree = cKDTree(reference_points)
    distances, indices = tree.query(query_points, workers=-1)
    normal_dot = np.abs(np.sum(query_normals * reference_normals[indices], axis=-1))
    return (
        float(np.mean(distances)),
        float(np.median(distances)),
        float(np.mean(normal_dot)),
        float(np.median(normal_dot)),
    )


def evaluate_reconstruction(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    valid_masks: torch.Tensor,
    colors: torch.Tensor | None = None,
    icp_threshold: float = 0.1,
    crop_size: int = 224,
    output_dir: str | Path | None = None,
    scene_name: str | None = None,
) -> ReconstructionMetrics:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for ICP and normal-consistency reconstruction metrics.") from exc

    points_pred, points_gt = umeyama_align_points(points_pred, points_gt, valid_masks)
    masks = valid_masks.bool()

    pred = points_pred[masks].cpu().numpy()
    gt = points_gt[masks].cpu().numpy()
    finite = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    pred, gt = pred[finite], gt[finite]
    if colors is not None:
        rgb = colors[masks].cpu().numpy()[finite]
    else:
        rgb = np.full_like(pred, 0.5)

    pred_cloud = o3d.geometry.PointCloud()
    pred_cloud.points = o3d.utility.Vector3dVector(pred)
    pred_cloud.colors = o3d.utility.Vector3dVector(rgb)
    gt_cloud = o3d.geometry.PointCloud()
    gt_cloud.points = o3d.utility.Vector3dVector(gt)
    gt_cloud.colors = o3d.utility.Vector3dVector(rgb)

    registration = o3d.pipelines.registration.registration_icp(
        pred_cloud,
        gt_cloud,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_cloud.transform(registration.transformation)
    pred_cloud.estimate_normals()
    gt_cloud.estimate_normals()

    pred_aligned = np.asarray(pred_cloud.points)
    gt_points = np.asarray(gt_cloud.points)
    pred_normals = np.asarray(pred_cloud.normals)
    gt_normals = np.asarray(gt_cloud.normals)

    acc_mean, acc_median, nc1_mean, nc1_median = _nearest_metrics(
        pred_aligned, gt_points, pred_normals, gt_normals
    )
    comp_mean, comp_median, nc2_mean, nc2_median = _nearest_metrics(
        gt_points, pred_aligned, gt_normals, pred_normals
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = (scene_name or "scene").replace("/", "_")
        o3d.io.write_point_cloud(str(output_dir / f"{safe_name}-pred-aligned.ply"), pred_cloud)
        o3d.io.write_point_cloud(str(output_dir / f"{safe_name}-gt.ply"), gt_cloud)

    return ReconstructionMetrics(
        acc_mean=acc_mean,
        acc_median=acc_median,
        comp_mean=comp_mean,
        comp_median=comp_median,
        nc_mean=(nc1_mean + nc2_mean) / 2,
        nc_median=(nc1_median + nc2_median) / 2,
        nc_accuracy_mean=nc1_mean,
        nc_accuracy_median=nc1_median,
        nc_completion_mean=nc2_mean,
        nc_completion_median=nc2_median,
    )
