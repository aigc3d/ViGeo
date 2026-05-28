"""
Training loss module.

This module computes the losses for multi-view geometry prediction, including:
    - scale-invariant point-map loss
    - camera pose loss
    - optional depth L1 and depth gradient losses
    - optional normal / normal-map losses
    - optional scale, confidence, and ray losses

Conventions:
    - points_pred is expected to have shape [B, S, H, W, 3].
    - depth_pred, if provided, is expected to have shape [B, S, 1, H, W].
    - depth_pred is supervised in the same normalized scale as points_pred.
    - GT geometry is normalized per sample before loss computation.
    - Invalid samples with too few valid depth pixels are skipped.
"""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils3d

from .loss_util import (
    compute_pointmap_scale,
    scale_invariant_pointmap_loss,
    depth_l1_loss,
    depth_gradient_loss,
    edge_loss,
    normalize_gt,
    normalize_pred,
    camera_loss,
    normal_map_loss,
    scale_loss,
    build_confidence_gt,
    mask_loss,
    compute_raymap_from_pose,
    ray_loss,
)
from .utils import pose_encoding_to_extri, extri_to_pose_encoding


# Datasets with reliable normal / geometry supervision.
NORMAL_MAP_DATASETS = {
    'tartanair', 'gtasfm', 'pointodyssey', 'bedlam',
    'dynamic_replica', 'lightwheelocc', 'hypersim',
    'omniobject', 'mvssynth', 'matrixcity', 'omniworld',
    'synthia', 'ase', 'spring', 'transphy3d', 'carlaocc', 'tartanground',
}

# Real-world datasets usually have noisier or less complete geometry.
REAL_DATASETS = {
    'blendedmvs', 'wildrgbd', 'waymo',
    'arkitscenes', 'arkitscenes_highres', 'scannetpp', 'dl3dv',
}

# Datasets that provide metric-scale supervision.
METRIC_DATASETS = {
    'tartanair', 'gtasfm', 'pointodyssey', 'bedlam',
    'dynamic_replica', 'lightwheelocc', 'hypersim',
    'mvssynth', 'matrixcity', 'synthia', 'ase',
    'wildrgbd', 'waymo', 'arkitscenes', 'arkitscenes_highres', 'scannetpp',
}


# Datasets where sky_mask is reliable enough for mask-head supervision.
SKY_MASK_DATASETS = NORMAL_MAP_DATASETS | {'dl3dv'}


class VideoDepthLoss(nn.Module):
    def __init__(self, losses: dict | None = None):
        super().__init__()
        self.losses = losses or {}
        self.loss_keys = sorted(self.losses.keys()) + ['total']

    @staticmethod
    def _zero_loss(sample_pred):
        """Create a differentiable zero tensor on the correct device."""
        zero_loss = None

        for value in sample_pred.values():
            if torch.is_tensor(value):
                value_zero = value.sum() * 0
                zero_loss = value_zero if zero_loss is None else zero_loss + value_zero

        if zero_loss is None:
            raise ValueError('sample_pred must contain at least one tensor.')

        return zero_loss

    def _reduce_losses(self, batch_losses, valid_count, zero_loss):
        """Average collected losses and fill missing losses with zero."""
        return {
            key: sum(batch_losses[key]) / valid_count if key in batch_losses else zero_loss
            for key in self.loss_keys
        }

    @staticmethod
    def _build_gt_geometry(depth_gt_i, intrinsic_gt_i, pose_gt_i, mask_i):
        """Build GT point map, normalized GT geometry, and GT ray map.

        Args:
            depth_gt_i: [S, 1, H, W]
            intrinsic_gt_i: [S, 3, 3]
            pose_gt_i: [S, 4, 4]
            mask_i: [S, H, W]
        """
        # Detach GT tensors to avoid tracking gradients through target geometry.
        points_gt_i = utils3d.pt.depth_map_to_point_map(
            depth_gt_i.detach().squeeze(1), intrinsic_gt_i.detach()
        )

        points_gt_i_norm, pose_gt_i_norm, _ = normalize_gt(
            points_gt_i.float(), pose_gt_i.float(), mask_i
        )

        ray_gt_i = compute_raymap_from_pose(
            pose_gt_i_norm.detach(),
            intrinsic_gt_i.detach(),
            depth_gt_i.shape[-2],
            depth_gt_i.shape[-1],
        )

        return points_gt_i, points_gt_i_norm, pose_gt_i_norm, ray_gt_i

    @staticmethod
    def _build_pred_geometry(points_pred_i, pose_pred_i, mask_i):
        """Normalize predicted point map and camera pose.

        Args:
            points_pred_i: [S, H, W, 3]
            pose_pred_i: [S, 4, 4]
            mask_i: [S, H, W]
        """
        points_pred_i_norm, pose_pred_i_norm, norm_factor_pred_i = normalize_pred(
            points_pred_i, pose_pred_i, mask_i
        )
        return points_pred_i_norm, pose_pred_i_norm, norm_factor_pred_i

    @staticmethod
    def _compute_scales(
        points_pred_i,
        points_gt_i,
        points_pred_i_norm,
        points_gt_i_norm,
        mask_i,
    ):
        """Compute both metric and normalized point-map scales."""
        scale_gt_i = compute_pointmap_scale(points_pred_i, points_gt_i, mask=mask_i)
        scale_i = compute_pointmap_scale(points_pred_i_norm, points_gt_i_norm, mask=mask_i)
        return scale_gt_i, scale_i

    def forward(self, sample_pred, sample_target):
        """
        Args:
            sample_pred:
                points_pred: [B, S, H, W, 3]
                pose_pred: encoded pose, converted to [B, S, 4, 4]
                depth_pred: optional, [B, S, 1, H, W]
                normal_pred: optional, [B, S, 3, H, W]
                scale_pred: optional, shape depends on scale head
                conf_pred: optional, [B, S, 1, H, W]
                mask_pred: optional, [B, S, 1, H, W]
                ray_pred: optional, expected to match ray_gt shape

            sample_target:
                depth: [B, S, 1, H, W]
                intrinsic: [B, S, 3, 3]
                pose: [B, S, 4, 4]
                normal: optional, [B, S, 3, H, W]
                sky_mask: optional, [B, S, 1, H, W], 1 marks sky pixels
                img_metas['data_name']: list[str] with length B

        Returns:
            dict[str, Tensor], including configured losses and 'total'.
        """
        depth_gt = sample_target['depth']              # [B, S, 1, H, W]
        intrinsic_gt = sample_target['intrinsic']      # [B, S, 3, 3]
        pose_gt = sample_target['pose']                # [B, S, 4, 4]
        data_names = sample_target['img_metas']['data_name']
        normal_gt = sample_target.get('normal')        # optional, [B, S, 3, H, W]
        sky_mask_gt = sample_target.get('sky_mask')    # optional, [B, S, 1, H, W]

        points_pred = sample_pred['points_pred']       # [B, S, H, W, 3]
        pose_pred = pose_encoding_to_extri(sample_pred['pose_pred'])  # [B, S, 4, 4]

        depth_pred = sample_pred.get('depth_head_pred')     # optional, [B, S, 1, H, W]
        normal_pred = sample_pred.get('normal_pred')   # optional, [B, S, 3, H, W]
        scale_pred = sample_pred.get('scale_pred')     # optional
        conf_pred = sample_pred.get('conf_pred')       # optional, [B, S, 1, H, W]
        mask_pred = sample_pred.get('mask_pred')       # optional, [B, S, 1, H, W]
        ray_pred = sample_pred.get('ray_pred')         # optional

        mask = (depth_gt > 0).squeeze(2)  # [B, S, H, W]
        batch_size = points_pred.shape[0]

        zero_loss = self._zero_loss(sample_pred)
        batch_losses = defaultdict(list)
        valid_count = 0

        for i in range(batch_size):
            mask_i = mask[i]  # [S, H, W]

            if mask_i.sum() < 10:
                continue

            valid_count += 1

            sample_losses = self._forward_single_sample(
                data_name_i=data_names[i],
                depth_gt_i=depth_gt[i],
                intrinsic_gt_i=intrinsic_gt[i],
                pose_gt_i=pose_gt[i],
                points_pred_i=points_pred[i],
                pose_pred_i=pose_pred[i],
                depth_pred_i=None if depth_pred is None else depth_pred[i],
                mask_i=mask_i,
                normal_pred_i=None if normal_pred is None else normal_pred[i],
                normal_gt_i=None if normal_gt is None else normal_gt[i],
                sky_mask_i=None if sky_mask_gt is None else sky_mask_gt[i],
                scale_pred_i=None if scale_pred is None else scale_pred[i],
                conf_pred_i=None if conf_pred is None else conf_pred[i],
                mask_pred_i=None if mask_pred is None else mask_pred[i],
                ray_pred_i=None if ray_pred is None else ray_pred[i],
            )

            for name, value in sample_losses.items():
                batch_losses[name].append(value)

        # If all samples are invalid, return differentiable zero losses.
        valid_count = max(valid_count, 1)

        return self._reduce_losses(batch_losses, valid_count, zero_loss)

    def _forward_single_sample(
        self,
        *,
        data_name_i,
        depth_gt_i,
        intrinsic_gt_i,
        pose_gt_i,
        points_pred_i,
        pose_pred_i,
        depth_pred_i,
        mask_i,
        normal_pred_i,
        normal_gt_i,
        sky_mask_i,
        scale_pred_i,
        conf_pred_i,
        mask_pred_i,
        ray_pred_i,
    ):
        """Compute all configured losses for one batch element.

        Args:
            depth_gt_i: [S, 1, H, W]
            intrinsic_gt_i: [S, 3, 3]
            pose_gt_i: [S, 4, 4]
            points_pred_i: [S, H, W, 3]
            pose_pred_i: [S, 4, 4]
            depth_pred_i: optional, [S, 1, H, W]
            mask_i: [S, H, W]
            normal_pred_i: optional, [S, 3, H, W]
            normal_gt_i: optional, [S, 3, H, W]
            sky_mask_i: optional, [S, 1, H, W]
            conf_pred_i: optional, [S, 1, H, W]
            mask_pred_i: optional, [S, 1, H, W]
        """
        # Force GT geometry construction to float32 for numerical stability.
        with torch.amp.autocast('cuda', dtype=torch.float32):
            points_gt_i, points_gt_i_norm, pose_gt_i_norm, ray_gt_i = self._build_gt_geometry(
                depth_gt_i, intrinsic_gt_i, pose_gt_i, mask_i
            )

        points_pred_i_norm, pose_pred_i_norm, norm_factor_pred_i = self._build_pred_geometry(
            points_pred_i, pose_pred_i, mask_i
        )

        scale_gt_i, scale_i = self._compute_scales(
            points_pred_i, points_gt_i, points_pred_i_norm, points_gt_i_norm, mask_i
        )

        depth_gt_i_norm = points_gt_i_norm[..., 2].detach()  # [S, H, W]
        depth_pred_i_aligned = None

        if depth_pred_i is not None:
            # depth_pred_i is expected to share the same raw scale as points_pred_i.
            depth_pred_i = depth_pred_i.squeeze(1)  # [S, H, W]

            # Do not backpropagate depth loss through the point-map normalization statistic.
            depth_pred_i_norm = depth_pred_i / norm_factor_pred_i.detach()
            depth_pred_i_aligned = depth_pred_i_norm * scale_i

        sample_losses = {}

        for loss_name in self.losses:
            if loss_name == 'scale_invariant_pointmap_loss':
                ratio = 0.8 if data_name_i in REAL_DATASETS else 1.0
                sample_losses[loss_name] = scale_invariant_pointmap_loss(
                    points_pred_i_norm, points_gt_i_norm, mask_i, scale=scale_i, ratio=ratio
                )

            elif loss_name == 'normal_loss':
                if data_name_i not in NORMAL_MAP_DATASETS:
                    continue

                sample_losses[loss_name] = edge_loss(
                    points_pred_i_norm, points_gt_i_norm, mask_i
                )

            elif loss_name == 'normal_map_loss':
                if normal_pred_i is None or normal_gt_i is None:
                    continue
                if data_name_i not in NORMAL_MAP_DATASETS or data_name_i == 'tartanground':
                    continue

                # [S, 3, H, W] -> [S, H, W, 3]
                sample_losses[loss_name] = normal_map_loss(
                    normal_pred_i.permute(0, 2, 3, 1),
                    normal_gt_i.permute(0, 2, 3, 1),
                    ratio=1.0 if data_name_i in ['transphy3d', 'carlaocc'] else 0.9
                )

            elif loss_name == 'camera_loss':
                pose_pred_i_norm_scaled = pose_pred_i_norm.clone()
                pose_pred_i_norm_scaled[:, :3, 3] *= scale_i

                sample_losses[loss_name] = camera_loss(
                    extri_to_pose_encoding(pose_pred_i_norm_scaled),
                    extri_to_pose_encoding(pose_gt_i_norm),
                )

            elif loss_name == 'scale_loss':
                if scale_pred_i is None or data_name_i not in METRIC_DATASETS:
                    continue

                sample_losses[loss_name] = scale_loss(scale_pred_i, scale_gt_i.detach())

            elif loss_name == 'ray_loss':
                if ray_pred_i is None:
                    continue

                sample_losses[loss_name] = ray_loss(ray_gt_i, ray_pred_i)

            elif loss_name == 'depth_loss':
                if depth_pred_i_aligned is None:
                    continue
                ratio = 0.8 if data_name_i in REAL_DATASETS else 1.0
                sample_losses[loss_name] = depth_l1_loss(
                    depth_pred_i_aligned, depth_gt_i_norm, mask_i, ratio
                )

            elif loss_name == 'depth_gradient_loss':
                if data_name_i not in NORMAL_MAP_DATASETS or depth_pred_i_aligned is None:
                    continue

                sample_losses[loss_name] = depth_gradient_loss(
                    depth_pred_i_aligned, depth_gt_i_norm, mask_i
                )

            elif loss_name == 'conf_loss':
                if conf_pred_i is None:
                    continue

                conf_label_i = build_confidence_gt(
                    points_pred_i_norm, points_gt_i_norm, mask_i, scale_i
                )
                conf_logits_i = conf_pred_i.squeeze(1)  # [S, H, W]

                sample_losses[loss_name] = F.binary_cross_entropy_with_logits(
                    conf_logits_i[mask_i], conf_label_i[mask_i]
                )

            elif loss_name == 'mask_loss':
                if (
                    mask_pred_i is None
                    or sky_mask_i is None
                    or data_name_i not in SKY_MASK_DATASETS
                ):
                    continue

                sample_losses[loss_name] = mask_loss(
                    mask_pred_i,
                    sky_mask_i,
                    mask_i,
                )

            else:
                raise KeyError(f'Unsupported loss: {loss_name}')

        if sample_losses:
            sample_losses['total'] = sum(
                sample_losses[name] * self.losses[name] for name in sample_losses
            )

        return sample_losses
