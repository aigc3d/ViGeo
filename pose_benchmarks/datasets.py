from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


SINTEL_PASS_DEFAULT = "clean"


@dataclass(frozen=True)
class PoseSequence:
    dataset: str
    sequence: str
    image_paths: list[Path]
    gt_poses: np.ndarray
    timestamps: np.ndarray


def numeric_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def discover_sintel_sequences(data_root: str | Path, sintel_pass: str = SINTEL_PASS_DEFAULT) -> list[str]:
    base = Path(data_root) / "sintel"
    image_root = base / sintel_pass
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing Sintel image directory: {image_root}")
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


def _read_sintel_cam(path: Path) -> np.ndarray:
    tag = 202021.25
    with path.open("rb") as handle:
        check = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if check != tag:
            raise ValueError(f"Bad Sintel .cam tag in {path}: {check}")
        _intrinsic = np.fromfile(handle, dtype=np.float64, count=9).reshape(3, 3)
        world_to_camera = np.fromfile(handle, dtype=np.float64, count=12).reshape(3, 4)
    pose = np.eye(4, dtype=np.float64)
    pose[:3] = world_to_camera
    return np.linalg.inv(pose)


def _c2w_to_pose_row(c2w: np.ndarray) -> np.ndarray:
    xyz = c2w[:3, 3]
    qx, qy, qz, qw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    return np.asarray([xyz[0], xyz[1], xyz[2], qw, qx, qy, qz], dtype=np.float64)


def load_sintel_sequence(
    data_root: str | Path,
    sequence: str,
    sintel_pass: str = SINTEL_PASS_DEFAULT,
    stride: int = 1,
) -> PoseSequence:
    base = Path(data_root) / "sintel"
    image_dir = base / sintel_pass / sequence
    cam_dir = base / "camdata_left" / sequence
    image_paths = sorted(image_dir.glob("*.png"), key=numeric_key)[::stride]
    if not image_paths:
        raise RuntimeError(f"No Sintel images found under {image_dir}")

    gt_poses, timestamps = [], []
    for image_path in image_paths:
        cam_path = cam_dir / image_path.with_suffix(".cam").name
        c2w = _read_sintel_cam(cam_path)
        gt_poses.append(_c2w_to_pose_row(c2w))
        timestamps.append(float(image_path.stem.split("_")[-1]))

    gt_poses = np.stack(gt_poses)
    gt_poses[:, :3] -= gt_poses[:, :3].mean(axis=0, keepdims=True)
    return PoseSequence(
        dataset="sintel",
        sequence=sequence,
        image_paths=image_paths,
        gt_poses=gt_poses,
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def discover_sequences(data_root: str | Path, dataset: str, sintel_pass: str = SINTEL_PASS_DEFAULT) -> list[str]:
    dataset = dataset.lower()
    if dataset == "sintel":
        return discover_sintel_sequences(data_root, sintel_pass=sintel_pass)
    raise ValueError(f"Unsupported pose dataset: {dataset}")


def load_sequence(
    data_root: str | Path,
    dataset: str,
    sequence: str,
    stride: int = 1,
    sintel_pass: str = SINTEL_PASS_DEFAULT,
) -> PoseSequence:
    dataset = dataset.lower()
    if dataset == "sintel":
        return load_sintel_sequence(data_root, sequence, sintel_pass=sintel_pass, stride=stride)
    raise ValueError(f"Unsupported pose dataset: {dataset}")
