from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from scipy.spatial.transform import Rotation


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def ensure_4x4(poses: np.ndarray | torch.Tensor) -> np.ndarray:
    poses = _as_numpy(poses).astype(np.float64)
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] != (3, 4):
        raise ValueError(f"Expected poses with shape (...,3,4) or (...,4,4), got {poses.shape}")
    bottom = np.zeros((*poses.shape[:-2], 1, 4), dtype=poses.dtype)
    bottom[..., 0, 3] = 1.0
    return np.concatenate([poses, bottom], axis=-2)


def invert_poses(poses: np.ndarray | torch.Tensor) -> np.ndarray:
    return np.linalg.inv(ensure_4x4(poses))


def c2w_to_pose_rows(poses_c2w: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    poses = ensure_4x4(poses_c2w)
    pose_rows = []
    for pose in poses:
        xyz = pose[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        pose_rows.append([xyz[0], xyz[1], xyz[2], qw, qx, qy, qz])
    timestamps = np.arange(len(pose_rows), dtype=np.float64)
    return np.asarray(pose_rows, dtype=np.float64), timestamps


def make_traj(pose_rows: np.ndarray, timestamps: np.ndarray) -> PoseTrajectory3D:
    return PoseTrajectory3D(
        positions_xyz=pose_rows[:, :3],
        orientations_quat_wxyz=pose_rows[:, 3:],
        timestamps=timestamps.astype(np.float64),
    )


def evaluate_pose(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    timestamps: np.ndarray,
    output_file: str | Path | None = None,
) -> dict[str, float]:
    pred_timestamps = np.arange(len(pred_poses), dtype=np.float64)
    if len(pred_timestamps) == len(timestamps):
        pred_timestamps = timestamps.copy()

    pred_traj = make_traj(pred_poses, pred_timestamps)
    gt_traj = make_traj(gt_poses, timestamps)
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    ate_result = main_ape.ape(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    rpe_rot_result = main_rpe.rpe(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    rpe_trans_result = main_rpe.rpe(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(f"{ate_result}\n{rpe_rot_result}\n{rpe_trans_result}\n")
    return {
        "ate": float(ate_result.stats["rmse"]),
        "rpe_trans": float(rpe_trans_result.stats["rmse"]),
        "rpe_rot": float(rpe_rot_result.stats["rmse"]),
    }
