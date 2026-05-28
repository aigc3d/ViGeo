"""
Loss and geometry utilities.

This file contains:
    1. Point-map scale and point-map losses
    2. Geometry normalization
    3. Camera, normal, scale, and confidence losses
    4. Depth losses
    5. Global and local point-map losses
    6. Ray utilities and ray loss

Notes:
    - Depth losses are L1-based and can be used with normalized depth.
    - The affine_invariant_* aliases are kept for backward compatibility.
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
import utils3d
from einops import einsum, repeat

from utils.alignment import align_points_scale
from .utils import se3_inverse, weighted_mean, smooth, angle_diff_vec3, harmonic_mean


# -----------------------------------------------------------------------------
# Point-map scale and point-map losses
# -----------------------------------------------------------------------------


@torch.no_grad()
def compute_pointmap_scale(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor,
    align_resolution: int = 4096,
):
    """Compute a scale factor that aligns predicted point maps to GT point maps.

    Args:
        points_pred: [S, H, W, 3]
        points_gt: [S, H, W, 3]
        mask: [S, H, W]
    """
    if mask.sum() < 10:
        raise ValueError('compute_pointmap_scale requires at least 10 valid points.')

    points_pred_lr = F.interpolate(
        points_pred[mask].T[None], size=align_resolution, mode='nearest'
    ).squeeze(0).T

    points_gt_lr = F.interpolate(
        points_gt[mask].T[None], size=align_resolution, mode='nearest'
    ).squeeze(0).T

    scale = align_points_scale(
        points_pred_lr, points_gt_lr, 1 / points_gt_lr[..., 2].clamp_min(1e-2)
    )
    scale[scale <= 0] *= -1

    return scale


def scale_invariant_pointmap_loss(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    beta: float = 0.0,
    ratio: float = 1.0,
):
    """Scale-invariant point-map loss with optional pixel filtering.

    Args:
        points_pred: [S, H, W, 3]
        points_gt: [S, H, W, 3]
        mask: [S, H, W]
        scale: scalar tensor
        ratio: ratio of valid pixels to keep. If < 1, keeps lower-error pixels.
    """
    points_pred = scale * points_pred

    weight = mask.float() / points_gt[..., 2].clamp_min(1e-5)
    avg_weight = weighted_mean(weight, mask, dim=(-2, -1), keepdim=True)
    weight = weight.clamp_max(10.0 * avg_weight)

    diff = (points_pred - points_gt).abs() * weight[..., None]
    pixel_loss = smooth(diff, beta=beta).sum(dim=-1)  # [S, H, W]

    if ratio < 1.0:
        valid_pixel_loss = pixel_loss.view(pixel_loss.size(0), -1)
        valid_mask_flat = mask.view(mask.size(0), -1).bool()

        num_valid = valid_mask_flat.sum(dim=1)
        num_keep = (num_valid * ratio).long()

        final_loss_mask = torch.zeros_like(valid_pixel_loss)

        for i in range(pixel_loss.size(0)):
            if num_keep[i] > 0:
                curr_errors = valid_pixel_loss[i][valid_mask_flat[i]]
                threshold, _ = torch.kthvalue(curr_errors, num_keep[i].item())
                final_loss_mask[i] = (
                    (valid_pixel_loss[i] <= threshold) & valid_mask_flat[i]
                )

        final_loss_mask = final_loss_mask.view_as(pixel_loss)
        loss = (pixel_loss * final_loss_mask).sum() / (final_loss_mask.sum() + 1e-6)
    else:
        loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-6)

    return loss.clamp(max=1.0)


def edge_loss(points_pred: torch.Tensor, points_gt: torch.Tensor, mask: torch.Tensor):
    """Edge-aware normal-like geometric loss.

    Args:
        points_pred: [S, H, W, 3]
        points_gt: [S, H, W, 3]
        mask: [S, H, W]
    """
    edge = ~utils3d.pt.depth_map_edge(points_gt[..., 2], rtol=0.3)
    mask = torch.logical_and(mask, edge)

    dx = points_pred[..., :-1, :, :] - points_pred[..., 1:, :, :]
    dy = points_pred[..., :, :-1, :] - points_pred[..., :, 1:, :]

    gt_dx = points_gt[..., :-1, :, :] - points_gt[..., 1:, :, :]
    gt_dy = points_gt[..., :, :-1, :] - points_gt[..., :, 1:, :]

    mask_dx = mask[..., :-1, :] & mask[..., 1:, :]
    mask_dy = mask[..., :, :-1] & mask[..., :, 1:]

    min_angle, max_angle, beta_rad = math.radians(0.1), math.radians(90), math.radians(3)

    loss_dx = mask_dx * smooth(
        angle_diff_vec3(dx, gt_dx).clamp(min_angle, max_angle), beta=beta_rad
    )
    loss_dy = mask_dy * smooth(
        angle_diff_vec3(dy, gt_dy).clamp(min_angle, max_angle), beta=beta_rad
    )

    return (loss_dx.mean() + loss_dy.mean()) / (2 * max(points_pred.shape[-3:-1]))


# -----------------------------------------------------------------------------
# Geometry normalization
# -----------------------------------------------------------------------------


def normalize_gt(points_gt: torch.Tensor, pose_gt: torch.Tensor, mask: torch.Tensor):
    """Normalize GT geometry into the first-view camera frame.

    Args:
        points_gt: [S, H, W, 3]
        pose_gt: [S, 4, 4]
        mask: [S, H, W]
    """
    points_all = (
        torch.einsum('sik,shwk->shwi', pose_gt[..., :3, :3], points_gt)
        + pose_gt[..., :3, 3].unsqueeze(1).unsqueeze(1)
    )

    w2c_ref = se3_inverse(pose_gt[0])

    points_all = (
        torch.einsum('ij,shwj->shwi', w2c_ref[:3, :3], points_all)
        + w2c_ref[:3, 3][None, None, None]
    )
    pose_gt = torch.einsum('ij,sjk->sik', w2c_ref, pose_gt)

    points_all[~mask] = 0

    norm_factor_gt = (
        points_all.reshape(points_all.shape[0], -1, 3).norm(dim=-1).sum(dim=(-1, -2))
        / (mask.float().sum(dim=(-1, -2, -3)) + 1e-8)
    )

    points_all = points_all / norm_factor_gt
    pose_gt[:, :3, 3] /= norm_factor_gt

    extrinsics = se3_inverse(pose_gt)

    points_gt = (
        torch.einsum('sik,shwk->shwi', extrinsics[..., :3, :3], points_all)
        + extrinsics[..., :3, 3].unsqueeze(1).unsqueeze(1)
    )

    return points_gt, pose_gt, norm_factor_gt


def normalize_pred(points_pred: torch.Tensor, pose_pred: torch.Tensor, mask: torch.Tensor):
    """Normalize predicted point map and camera pose.

    Args:
        points_pred: [S, H, W, 3]
        pose_pred: [S, 4, 4]
        mask: [S, H, W]
    """
    points_all = points_pred.clone()
    points_all[~mask] = 0

    norm_factor_pred = (
        points_all.reshape(points_all.shape[0], -1, 3).norm(dim=-1).sum(dim=(-1, -2))
        / (mask.float().sum(dim=(-1, -2, -3)) + 1e-8)
    )

    points_pred = points_pred / norm_factor_pred

    pose_pred = pose_pred.clone()
    pose_pred[:, :3, 3] /= norm_factor_pred

    return points_pred, pose_pred, norm_factor_pred


# -----------------------------------------------------------------------------
# Camera, normal, scale, and confidence losses
# -----------------------------------------------------------------------------


def camera_loss(pose_pred: torch.Tensor, pose_gt: torch.Tensor):
    """Compute camera pose loss on encoded pose tensors."""
    loss_t = (pose_pred[..., :3] - pose_gt[..., :3]).abs()
    loss_r = (pose_pred[..., 3:7] - pose_gt[..., 3:7]).abs()

    return loss_t.clamp(max=100).mean() + loss_r.mean()

def normal_map_loss(
    normal_pred: torch.Tensor,
    normal_gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    ratio: float = 0.90,
):
    """
    Keep the lowest-error ratio of valid pixels per frame.

    Args:
        normal_pred: [S, H, W, 3] or [..., H, W, 3]
        normal_gt:   [S, H, W, 3] or [..., H, W, 3]
        mask: optional, [S, H, W] or [..., H, W]
        ratio: ratio of valid pixels to keep. 0.90 keeps the lowest-error 90%.
    """

    finite_mask = torch.isfinite(normal_pred).all(dim=-1) & torch.isfinite(normal_gt).all(dim=-1)
    if mask is None:
        valid_mask = torch.linalg.norm(normal_gt, ord=2, dim=-1) > 0
    else:
        valid_mask = mask.bool()
    valid_mask = valid_mask & finite_mask

    if not bool(finite_mask.all()):
        normal_pred = torch.where(finite_mask[..., None], normal_pred, normal_pred.new_zeros(()))
        normal_gt = torch.where(finite_mask[..., None], normal_gt, normal_gt.new_zeros(()))

    pixel_loss = utils3d.pt.angle_between(normal_pred, normal_gt).square()
    H, W = pixel_loss.shape[-2:]
    pixel_loss_flat = pixel_loss.reshape(-1, H * W)
    mask_flat = valid_mask.reshape(-1, H * W)

    if ratio >= 1.0:
        valid_loss = pixel_loss_flat[mask_flat]
        return valid_loss.mean() if valid_loss.numel() > 0 else normal_pred.reshape(-1)[:0].sum()

    keep_mask = torch.zeros_like(mask_flat, dtype=torch.bool)
    for i in range(pixel_loss_flat.size(0)):
        curr_loss = pixel_loss_flat[i][mask_flat[i]]
        if curr_loss.numel() == 0:
            continue

        k = max(1, min(int(curr_loss.numel() * ratio), curr_loss.numel()))
        threshold = torch.kthvalue(curr_loss.detach().float(), k).values.to(pixel_loss.dtype)
        keep_mask[i] = (pixel_loss_flat[i] <= threshold) & mask_flat[i]

    if not keep_mask.any():
        return normal_pred.reshape(-1)[:0].sum()

    return (pixel_loss_flat * keep_mask.to(pixel_loss_flat.dtype)).sum() / (
        keep_mask.sum().to(pixel_loss_flat.dtype) + 1e-6
    )

def mask_loss(
    mask_pred: torch.Tensor,
    sky_mask: torch.Tensor,
    valid_depth_mask: torch.Tensor,
):
    """MoGe v1-style mask L2 loss using sky_mask as invalid supervision."""
    if mask_pred.ndim >= 3 and mask_pred.shape[-3] == 1:
        mask_pred = mask_pred.squeeze(-3)
    if sky_mask.ndim >= 3 and sky_mask.shape[-3] == 1:
        sky_mask = sky_mask.squeeze(-3)
    if valid_depth_mask.ndim >= 3 and valid_depth_mask.shape[-3] == 1:
        valid_depth_mask = valid_depth_mask.squeeze(-3)

    sky = sky_mask > 0.5
    valid = valid_depth_mask.bool()
    supervised = valid | sky

    if not supervised.any():
        return mask_pred.reshape(-1)[:0].sum()

    pixel_loss = sky.to(mask_pred.dtype) * mask_pred.square()
    pixel_loss = pixel_loss + valid.to(mask_pred.dtype) * (1 - mask_pred).square()

    return pixel_loss.mean()

def scale_loss(scale_pred: torch.Tensor, scale_gt: torch.Tensor):
    """Compute scale loss in log space."""
    if scale_pred.dim() == 0:
        scale_gt = scale_gt.log()
    else:
        scale_gt = scale_gt.log().expand_as(scale_pred)

    return F.mse_loss(scale_pred.log(), scale_gt, reduction='mean')


def build_confidence_gt(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    threshold: float = 0.02,
):
    """Build binary confidence labels from aligned point-map error.

    Args:
        points_pred: [S, H, W, 3]
        points_gt: [S, H, W, 3]
        mask: [S, H, W]
        scale: scalar tensor
    """
    aligned_pred = points_pred * scale
    l1_error = torch.abs(aligned_pred - points_gt).sum(dim=-1)

    depth_gt = points_gt[..., 2].clamp_min(1e-5)
    weight = mask.float() / depth_gt

    avg_weight = weighted_mean(weight, mask, dim=(-2, -1), keepdim=True)
    weight = weight.clamp_max(10.0 * avg_weight)

    return (l1_error.detach() * weight < threshold).float()


# -----------------------------------------------------------------------------
# Depth losses
# -----------------------------------------------------------------------------


def image_gradients(tensor: torch.Tensor):
    """Compute finite-difference image gradients.

    Args:
        tensor: [..., H, W]
    """
    grad_x = torch.diff(tensor, dim=-1)  # [..., H, W - 1]
    grad_y = torch.diff(tensor, dim=-2)  # [..., H - 1, W]
    return grad_x, grad_y

def depth_l1_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    ratio: float = 1.0,
):
    weight = mask.float() / depth_gt.clamp_min(1e-5)
    avg_weight = weighted_mean(weight, mask, dim=(-2, -1), keepdim=True)
    weight = weight.clamp_max(10.0 * avg_weight)
    
    pixel_loss = (depth_pred - depth_gt).abs() * weight
    if ratio < 1.0:
        valid_pixel_loss = pixel_loss.view(pixel_loss.size(0), -1)
        valid_mask_flat = mask.view(mask.size(0), -1).bool()

        num_valid = valid_mask_flat.sum(dim=1)
        num_keep = (num_valid * ratio).long()

        final_loss_mask = torch.zeros_like(valid_pixel_loss)

        for i in range(pixel_loss.size(0)):
            if num_keep[i] > 0:
                curr_errors = valid_pixel_loss[i][valid_mask_flat[i]]
                threshold, _ = torch.kthvalue(curr_errors, num_keep[i].item())
                final_loss_mask[i] = (
                    (valid_pixel_loss[i] <= threshold) & valid_mask_flat[i]
                )

        final_loss_mask = final_loss_mask.view_as(pixel_loss)
        loss = (pixel_loss * final_loss_mask).sum() / (final_loss_mask.sum() + 1e-8)
    else:
        loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

    return loss

def depth_gradient_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """Compute L1 loss between predicted and GT depth gradients.

    Args:
        depth_pred: [..., H, W]
        depth_gt: [..., H, W]
        mask: optional, [..., H, W]
    """
    pred_grad_x, pred_grad_y = image_gradients(depth_pred)
    gt_grad_x, gt_grad_y = image_gradients(depth_gt)

    if mask is None:
        return F.l1_loss(pred_grad_x, gt_grad_x) + F.l1_loss(pred_grad_y, gt_grad_y)

    mask_x = mask[..., :, :-1] & mask[..., :, 1:]
    mask_y = mask[..., :-1, :] & mask[..., 1:, :]

    loss_x = F.l1_loss(
        pred_grad_x * mask_x, gt_grad_x * mask_x, reduction='sum'
    ) / (mask_x.sum() + 1e-8)

    loss_y = F.l1_loss(
        pred_grad_y * mask_y, gt_grad_y * mask_y, reduction='sum'
    ) / (mask_y.sum() + 1e-8)

    return loss_x + loss_y


# -----------------------------------------------------------------------------
# Global and local point-map losses
# -----------------------------------------------------------------------------


def global_pointmap_loss(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.0,
):
    """Global weighted point-map loss.

    Args:
        points_pred: [..., H, W, 3]
        points_gt: [..., H, W, 3]
        mask: [..., H, W]
    """
    weight = mask.float() / points_gt[..., 2].clamp_min(1e-5)
    avg_weight = weighted_mean(weight, mask, dim=(-2, -1), keepdim=True)
    weight = weight.clamp_max(10.0 * avg_weight)

    loss = smooth((points_pred - points_gt).abs() * weight[..., None], beta=beta)
    return loss.mean(dim=(-3, -2, -1))


def compute_anchor_sampling_weight(
    points: torch.Tensor,
    mask: torch.Tensor,
    radius_2d: torch.Tensor,
    radius_3d: torch.Tensor,
    num_test: int = 64,
):
    """Compute importance-sampling weights for local patch anchors.

    Args:
        points: [..., H, W, 3]
        mask: [..., H, W]
        radius_2d: 2D patch radius
        radius_3d: 3D patch radius
    """
    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device),
        torch.arange(width, device=points.device),
        indexing='ij',
    )

    test_delta_i = torch.randint(
        -radius_2d, radius_2d + 1, (height, width, num_test), device=points.device
    )
    test_delta_j = torch.randint(
        -radius_2d, radius_2d + 1, (height, width, num_test), device=points.device
    )

    test_i = pixel_i[..., None] + test_delta_i
    test_j = pixel_j[..., None] + test_delta_j

    test_mask = (
        (test_i >= 0)
        & (test_i < height)
        & (test_j >= 0)
        & (test_j < width)
    )

    test_i = test_i.clamp(0, height - 1)
    test_j = test_j.clamp(0, width - 1)

    test_mask = test_mask & mask[..., test_i, test_j]
    test_points = points[..., test_i, test_j, :]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)

    weight = 1 / (
        ((test_dist <= radius_3d[..., None]) & test_mask)
        .float()
        .sum(dim=-1)
        .clamp_min(1)
    )

    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)

    return weight


def local_pointmap_loss(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    mask: torch.Tensor,
    focal: torch.Tensor,
    level: Literal[4, 16, 64],
    num_patches: int = 16,
    beta: float = 0.0,
):
    """Local patch-based weighted point-map loss.

    Args:
        points_pred: [..., H, W, 3]
        points_gt: [..., H, W, 3]
        mask: [..., H, W]
        focal: [...]
        level: local patch scale level
    """
    device, dtype = points_pred.device, points_pred.dtype

    *batch_shape, height, width, _ = points_pred.shape
    batch_size = math.prod(batch_shape)

    points_pred = points_pred.reshape(-1, height, width, 3)
    points_gt = points_gt.reshape(-1, height, width, 3)
    mask = mask.reshape(-1, height, width)
    focal = focal.reshape(-1)

    radius_2d = math.ceil(0.5 / level * (height**2 + width**2) ** 0.5)
    radius_3d = 0.5 / level / focal * points_gt[..., 2]

    anchor_sampling_weights = compute_anchor_sampling_weight(
        points_gt, mask, radius_2d, radius_3d, num_test=64
    )

    where_mask = torch.where(mask)
    random_selection = torch.multinomial(
        anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True
    )

    patch_batch_idx, patch_anchor_i, patch_anchor_j = [
        indices[random_selection] for indices in where_mask
    ]

    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing='ij',
    )

    patch_i = patch_i + patch_anchor_i[:, None, None]
    patch_j = patch_j + patch_anchor_j[:, None, None]

    patch_mask = (
        (patch_i >= 0)
        & (patch_i < height)
        & (patch_j >= 0)
        & (patch_j < width)
    )

    patch_i = patch_i.clamp(0, height - 1)
    patch_j = patch_j.clamp(0, width - 1)

    gt_patch_anchor_points = points_gt[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = (
        0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    )

    gt_patch_points = points_gt[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (
        gt_patch_points - gt_patch_anchor_points[:, None, None, :]
    ).norm(dim=-1)

    patch_mask &= mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= 32)
    num_nonempty_patches = nonempty[0].shape[0]

    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device)

    patch_batch_idx = patch_batch_idx[nonempty]
    patch_i = patch_i[nonempty]
    patch_j = patch_j[nonempty]
    patch_mask = patch_mask[nonempty]
    gt_patch_points = gt_patch_points[nonempty]

    pred_patch_points = points_pred[patch_batch_idx[:, None, None], patch_i, patch_j]

    gt_mean = harmonic_mean(points_gt[..., 2], mask, dim=(-2, -1))

    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(
        0.1 * gt_mean[patch_batch_idx, None, None]
    )

    loss = smooth(
        (pred_patch_points - gt_patch_points).abs() * patch_weight[..., None],
        beta=beta,
    ).mean(dim=(-3, -2, -1))

    loss = torch.scatter_reduce(
        torch.zeros(batch_size, dtype=dtype, device=device),
        dim=0,
        index=patch_batch_idx,
        src=loss,
        reduce='sum',
    ) / num_patches

    return loss.reshape(batch_shape)


# -----------------------------------------------------------------------------
# Ray utilities and ray loss
# -----------------------------------------------------------------------------


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device('cpu'),
):
    """Get normalized coordinates and integer indices for an image grid.

    Args:
        shape: image shape, e.g. (H, W)

    Returns:
        coordinates: [*shape, 2], xy indexing, normalized to [0, 1]
        stacked_indices: [*shape, 2], ij indexing
    """
    indices = [torch.arange(length, device=device) for length in shape]

    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing='ij'), dim=-1)

    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing='xy'), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(points: torch.Tensor):
    """Convert xyz points to homogeneous xyz1 coordinates."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: torch.Tensor):
    """Convert xyz vectors to homogeneous xyz0 coordinates."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: torch.Tensor,
    transformation: torch.Tensor,
):
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(
        transformation,
        homogeneous_coordinates.to(transformation.dtype),
        '... i j, ... j -> ... i',
    )


def transform_cam2world(
    homogeneous_coordinates: torch.Tensor,
    extrinsics: torch.Tensor,
):
    """Transform points or vectors from camera to world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def unproject(
    coordinates: torch.Tensor,
    z: torch.Tensor,
    intrinsics: torch.Tensor,
):
    """Unproject 2D normalized coordinates with the given depth values.

    Args:
        coordinates: [..., 2]
        z: [...]
        intrinsics: [..., 3, 3]
    """
    coordinates = homogenize_points(coordinates)

    ray_directions = einsum(
        intrinsics.float().inverse().to(intrinsics),
        coordinates.to(intrinsics.dtype),
        '... i j, ... j -> ... i',
    )

    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
):
    """Get ray origins and directions in world coordinates.

    Args:
        coordinates: [..., 2]
        extrinsics: [..., 4, 4]
        intrinsics: [..., 3, 3]
    """
    directions = unproject(coordinates, torch.ones_like(coordinates[..., 0]), intrinsics)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def compute_raymap_from_pose(
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    H_ray: int,
    W_ray: int,
):
    """Generate a ray map from camera pose and intrinsics.

    Args:
        pose: [S, 4, 4]
        intrinsics: [S, 3, 3]

    Returns:
        ray: [S, H_ray, W_ray, 6], concatenated as [direction, origin]
    """
    device = pose.device
    num_views = pose.shape[0]

    xy_ray, _ = sample_image_grid((H_ray, W_ray), device)
    xy_ray = xy_ray[None, ...].expand(num_views, -1, -1, -1)

    pose_expanded = repeat(pose, 'v i j -> v h w i j', h=H_ray, w=W_ray)
    intrinsics_expanded = repeat(
        intrinsics, 'v i j -> v h w i j', h=H_ray, w=W_ray
    )

    origins, directions = get_world_rays(
        coordinates=xy_ray,
        extrinsics=pose_expanded,
        intrinsics=intrinsics_expanded,
    )

    return torch.cat([directions, origins], dim=-1)


def ray_loss(
    ray_gt: torch.Tensor,
    ray_pred: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """Compute L1 ray-map loss.

    Args:
        ray_gt: [..., 6]
        ray_pred: [..., 6]
        mask: optional mask
    """
    loss = F.l1_loss(ray_pred, ray_gt, reduction='none')

    if mask is not None:
        mask = mask.squeeze(-1).bool()
        return loss[mask].mean()

    return loss.mean()
