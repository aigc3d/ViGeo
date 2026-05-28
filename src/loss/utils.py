"""
General geometry and math utilities.

This file contains:
    1. Homogeneous coordinate helpers
    2. SE(3) inverse
    3. Weighted / robust math utilities
    4. Pose encoding conversion
    5. Quaternion conversion utilities
"""

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Homogeneous coordinates
# -----------------------------------------------------------------------------


def homogenize_points(points: torch.Tensor):
    """Convert xyz points to homogeneous xyz1 coordinates."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


# -----------------------------------------------------------------------------
# Depth utilities
# -----------------------------------------------------------------------------


def depth_edge(
    depth: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
    kernel_size: int = 3,
    mask: torch.Tensor | None = None,
):
    """Compute depth edge mask from local depth differences.

    Args:
        depth: [..., H, W]
        atol: absolute depth difference threshold
        rtol: relative depth difference threshold
        kernel_size: local window size
        mask: optional, [..., H, W]

    Returns:
        edge: [..., H, W], bool tensor
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])

    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (
            F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2)
            + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
        )
    else:
        diff = (
            F.max_pool2d(
                torch.where(mask, depth, -torch.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            + F.max_pool2d(
                torch.where(mask, -depth, -torch.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )

    edge = torch.zeros_like(depth, dtype=torch.bool)

    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol

    return edge.reshape(*shape)


# -----------------------------------------------------------------------------
# SE(3) utilities
# -----------------------------------------------------------------------------


def se3_inverse(T):
    """Compute the inverse of SE(3) matrices.

    Args:
        T: torch.Tensor or np.ndarray with shape [..., 4, 4]

    Returns:
        Inverse transform with shape [..., 4, 4].
    """
    if torch.is_tensor(T):
        R = T[..., :3, :3]
        t = T[..., :3, 3].unsqueeze(-1)

        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)

        bottom_row = torch.tensor(
            [0, 0, 0, 1], device=T.device, dtype=T.dtype
        ).repeat(*T.shape[:-2], 1, 1)

        return torch.cat([torch.cat([R_inv, t_inv], dim=-1), bottom_row], dim=-2)

    R = T[..., :3, :3]
    t = T[..., :3, 3, np.newaxis]

    R_inv = np.swapaxes(R, -2, -1)
    t_inv = -R_inv @ t

    bottom_row = np.zeros((*T.shape[:-2], 1, 4), dtype=T.dtype)
    bottom_row[..., :, 3] = 1

    return np.concatenate([np.concatenate([R_inv, t_inv], axis=-1), bottom_row], axis=-2)


# -----------------------------------------------------------------------------
# Loss math utilities
# -----------------------------------------------------------------------------


def weighted_mean(
    x: torch.Tensor,
    w: torch.Tensor | None = None,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    eps: float = 1e-7,
):
    """Compute weighted mean."""
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)

    w = w.to(x.dtype)
    return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(
        dim=dim, keepdim=keepdim
    ).add(eps)


def harmonic_mean(
    x: torch.Tensor,
    w: torch.Tensor | None = None,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    eps: float = 1e-7,
):
    """Compute harmonic mean with optional weights."""
    x_inv = x.add(eps).reciprocal()

    if w is None:
        return x_inv.mean(dim=dim, keepdim=keepdim).reciprocal()

    return weighted_mean(x_inv, w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()


def smooth(err: torch.Tensor, beta: float = 0.0):
    """Smooth L1-style robust penalty."""
    if beta == 0:
        return err

    return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    """Compute angular difference between two 3D vectors."""
    cross = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross + eps, dot)


# -----------------------------------------------------------------------------
# Pose encoding utilities
# -----------------------------------------------------------------------------


def extri_to_pose_encoding(extrinsics: torch.Tensor):
    """Convert camera extrinsics to compact pose encoding.

    Args:
        extrinsics: [..., 3, 4] or [..., 4, 4]

    Returns:
        pose_encoding: [..., 7], concatenated as [translation, quaternion_xyzw]
    """
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]

    quat = mat_to_quat(R)
    return torch.cat([T, quat], dim=-1).float()


def pose_encoding_to_extri(pose_encoding: torch.Tensor):
    """Convert compact pose encoding to camera extrinsics.

    Args:
        pose_encoding: [..., 7], concatenated as [translation, quaternion_xyzw]

    Returns:
        extrinsics: [..., 3, 4]
    """
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]

    R = quat_to_mat(quat)
    return torch.cat([R, T[..., None]], dim=-1)


# -----------------------------------------------------------------------------
# Quaternion utilities
# -----------------------------------------------------------------------------


def quat_to_mat(quaternions: torch.Tensor):
    """Convert XYZW quaternions to rotation matrices.

    Args:
        quaternions: [..., 4], scalar-last XYZW

    Returns:
        rotation matrices: [..., 3, 3]
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )

    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor):
    """Convert rotation matrices to XYZW quaternions.

    Args:
        matrix: [..., 3, 3]

    Returns:
        quaternions: [..., 4], scalar-last XYZW
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f'Invalid rotation matrix shape {matrix.shape}.')

    batch_dim = matrix.shape[:-2]

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))

    quat = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))

    quat = quat[..., [1, 2, 3, 0]]
    return standardize_quaternion(quat)


def _sqrt_positive_part(x: torch.Tensor):
    """Return sqrt(max(0, x)) with zero subgradient at x = 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0

    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)

    return ret


def standardize_quaternion(quaternions: torch.Tensor):
    """Standardize unit quaternions to have non-negative real part."""
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)