from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils3d

from .loss_util import (
    global_pointmap_loss,
    local_pointmap_loss,
    edge_loss,
    normalize_gt,
    normal_map_loss,
)
from .utils import weighted_mean


NORMAL_MAP_DATASETS = {
    'tartanair', 'gtasfm', 'pointodyssey', 'bedlam',
    'dynamic_replica', 'lightwheelocc', 'hypersim',
    'omniobject', 'mvssynth', 'matrixcity', 'omniworld',
    'synthia', 'ase', 'spring', 'transphy3d',
}


class DepthCompletionLoss(nn.Module):
    def __init__(self, losses: dict | None = None):
        super().__init__()
        self.losses = losses or {}
        self.loss_keys = sorted(self.losses.keys()) + ['total']

    @staticmethod
    def _zero_loss(sample_pred):
        zero_loss = None

        for value in sample_pred.values():
            if torch.is_tensor(value):
                value_zero = value.sum() * 0
                zero_loss = value_zero if zero_loss is None else zero_loss + value_zero

        if zero_loss is None:
            raise ValueError('sample_pred must contain at least one tensor.')

        return zero_loss

    def _confidence_loss(
        self,
        points_pred_ij,
        points_gt_ij_norm,
        norm_factor_gt,
        conf_pred_ij,
    ):
        points_pred_ij_norm = points_pred_ij / norm_factor_gt
        valid_mask = points_gt_ij_norm[..., 2] > 0

        weight = valid_mask.float() / points_gt_ij_norm[..., 2].clamp_min(1e-5)
        weight = weight.clamp_max(
            10.0 * weighted_mean(weight, valid_mask, dim=(-2, -1), keepdim=True)
        )

        l1_error = F.l1_loss(
            points_gt_ij_norm[valid_mask],
            points_pred_ij_norm[valid_mask],
            reduction='none',
        ).detach()
        weighted_error = l1_error.mean(dim=-1) * weight[valid_mask]

        confidence_gt = weighted_error < 0.02
        conf_logits = conf_pred_ij.squeeze()[valid_mask]

        return F.binary_cross_entropy_with_logits(
            conf_logits,
            confidence_gt.float(),
        )

    def forward(self, sample_pred, sample_target):
        depth_gt = sample_target['depth']              # [B, S, 1, H, W]
        intrinsic_gt = sample_target['intrinsic']      # [B, S, 3, 3]
        pose_gt = sample_target['pose']                # [B, S, 4, 4]
        data_names = sample_target['img_metas']['data_name']
        normal_gt = sample_target.get('normal')        # optional, [B, S, 3, H, W]

        points_pred = sample_pred['points_pred']       # [B, S, H, W, 3]
        normal_pred = sample_pred.get('normal_pred')   # optional, [B, S, 3, H, W]
        conf_pred = sample_pred.get('conf_pred')       # optional, [B, S, 1, H, W]

        mask = (depth_gt > 0).squeeze(2)               # [B, S, H, W]
        batch_size, seq_len = points_pred.shape[:2]

        zero_loss = self._zero_loss(sample_pred)
        batch_losses = defaultdict(list)

        for i in range(batch_size):
            sample_losses = defaultdict(list)

            with torch.amp.autocast('cuda', dtype=torch.float32):
                points_gt_i = utils3d.pt.depth_map_to_point_map(
                    depth_gt[i].detach().squeeze(1),
                    intrinsic_gt[i].detach(),
                )
                points_gt_i_norm, _, norm_factor_gt = normalize_gt(
                    points_gt_i.float(),
                    pose_gt[i].float(),
                    mask[i],
                )

            for j in range(seq_len):
                mask_ij = mask[i, j]

                if mask_ij.sum() < 10:
                    continue

                depth_gt_ij = depth_gt[i, j]
                intrinsic_gt_ij = intrinsic_gt[i, j]
                points_pred_ij = points_pred[i, j]

                with torch.amp.autocast('cuda', dtype=torch.float32):
                    points_gt_ij = utils3d.pt.depth_map_to_point_map(
                        depth_gt_ij.detach(),
                        intrinsic_gt_ij.unsqueeze(0).detach(),
                    ).squeeze(0)

                focal_gt_ij = 1 / (
                    1 / intrinsic_gt_ij[0, 0] ** 2
                    + 1 / intrinsic_gt_ij[1, 1] ** 2
                ) ** 0.5

                frame_losses = {}

                for loss_name in self.losses:
                    if loss_name == 'global_loss':
                        frame_losses[loss_name] = global_pointmap_loss(
                            points_pred_ij,
                            points_gt_ij,
                            mask_ij,
                        )

                    elif 'local_loss' in loss_name:
                        level = int(loss_name.split('_')[-1])

                        if level == 4 or (
                            level in [16, 64] and data_names[i] in NORMAL_MAP_DATASETS
                        ):
                            frame_losses[loss_name] = local_pointmap_loss(
                                points_pred_ij,
                                points_gt_ij,
                                mask_ij,
                                focal=focal_gt_ij,
                                level=level,
                                num_patches=level**2,
                            )

                    elif loss_name == 'normal_loss':
                        if data_names[i] in NORMAL_MAP_DATASETS:
                            frame_losses[loss_name] = edge_loss(
                                points_pred_ij,
                                points_gt_ij,
                                mask_ij,
                            )

                    elif loss_name == 'normal_map_loss':
                        if (
                            normal_pred is not None
                            and normal_gt is not None
                            and data_names[i] in NORMAL_MAP_DATASETS
                        ):
                            frame_losses[loss_name] = normal_map_loss(
                                normal_pred[i, j].permute(1, 2, 0),
                                normal_gt[i, j].permute(1, 2, 0),
                            )

                    elif loss_name == 'conf_loss':
                        if conf_pred is not None:
                            frame_losses[loss_name] = self._confidence_loss(
                                points_pred_ij,
                                points_gt_i_norm[j],
                                norm_factor_gt,
                                conf_pred[i, j],
                            )

                    else:
                        raise KeyError(f'Unsupported depth completion loss: {loss_name}')

                if frame_losses:
                    frame_losses['total'] = sum(
                        frame_losses[name] * self.losses[name]
                        for name in frame_losses
                    )

                    for name, value in frame_losses.items():
                        sample_losses[name].append(value)

            for name, values in sample_losses.items():
                batch_losses[name].append(torch.stack(values).mean())

        return {
            key: sum(batch_losses[key]) / batch_size if key in batch_losses else zero_loss
            for key in self.loss_keys
        }
