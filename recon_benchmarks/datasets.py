from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from PIL import Image


DATASET_NAMES = ("7scenes", "nrgbd", "7scenes-sparse", "7scenes-dense", "nrgbd-sparse", "nrgbd-dense")
DEFAULT_PI3_EVAL_DATASETS = ("7scenes-sparse", "nrgbd-sparse")
DATASET_DIRS = {"7scenes": "7scenes", "nrgbd": "neural_rgbd"}
DEFAULT_STRIDES = {"7scenes": 200, "nrgbd": 500}
PI3_DATASET_SETTINGS = {
    "7scenes": ("7scenes", 200),
    "7scenes-sparse": ("7scenes", 200),
    "7scenes-dense": ("7scenes", 40),
    "nrgbd": ("nrgbd", 500),
    "nrgbd-sparse": ("nrgbd", 500),
    "nrgbd-dense": ("nrgbd", 100),
}


@dataclass
class ReconstructionScene:
    dataset: str
    scene_id: str
    image_paths: list[Path]
    images: torch.Tensor
    depths: torch.Tensor
    intrinsics: torch.Tensor
    camera_poses: torch.Tensor
    points_gt: torch.Tensor
    valid_masks: torch.Tensor


def _numeric_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def canonical_dataset(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset not in PI3_DATASET_SETTINGS:
        raise ValueError(f"Unsupported reconstruction dataset: {dataset}")
    return PI3_DATASET_SETTINGS[dataset][0]


def default_stride_for_dataset(dataset: str) -> int:
    dataset = dataset.lower()
    if dataset not in PI3_DATASET_SETTINGS:
        raise ValueError(f"Unsupported reconstruction dataset: {dataset}")
    return PI3_DATASET_SETTINGS[dataset][1]


def _crop_image_depth(
    image: Image.Image,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    left, top, right, bottom = bbox
    image = image.crop(bbox)
    depth = depth[top:bottom, left:right]
    intrinsics = intrinsics.copy()
    intrinsics[0, 2] -= left
    intrinsics[1, 2] -= top
    return image, depth, intrinsics


def _scaled_intrinsics(
    intrinsics: np.ndarray,
    input_resolution: np.ndarray,
    output_resolution: np.ndarray,
    scale: float,
) -> np.ndarray:
    # Match DUSt3R/StreamVGGT's OpenCV <-> COLMAP half-pixel convention.
    output = intrinsics.copy()
    output[0, 2] += 0.5
    output[1, 2] += 0.5
    output[:2] *= scale
    margins = input_resolution * scale - output_resolution
    output[:2, 2] -= 0.5 * margins
    output[0, 2] -= 0.5
    output[1, 2] -= 0.5
    return output


def crop_resize_on_principal_point(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image = Image.fromarray(image_rgb)
    width, height = image.size
    cx, cy = np.rint(intrinsics[:2, 2]).astype(int)
    margin_x = min(cx, width - cx)
    margin_y = min(cy, height - cy)
    if margin_x <= width / 5 or margin_y <= height / 5:
        raise ValueError(f"Principal point is too close to an image boundary: {(cx, cy)} in {(width, height)}")

    image, depth, intrinsics = _crop_image_depth(
        image,
        depth,
        intrinsics,
        (cx - margin_x, cy - margin_y, cx + margin_x, cy + margin_y),
    )

    input_resolution = np.asarray(image.size, dtype=np.float64)
    target_resolution = np.asarray(resolution, dtype=np.float64)
    scale = float(np.max(target_resolution / input_resolution) + 1e-8)
    output_resolution = np.floor(input_resolution * scale).astype(int)
    resample = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC
    image = image.resize(tuple(output_resolution), resample=resample)
    depth = cv2.resize(depth, tuple(output_resolution), interpolation=cv2.INTER_NEAREST)
    intrinsics = _scaled_intrinsics(intrinsics, input_resolution, output_resolution, scale)

    margins = output_resolution - np.asarray(resolution)
    left, top = np.rint(0.5 * margins).astype(int)
    right, bottom = left + resolution[0], top + resolution[1]
    return _crop_image_depth(image, depth, intrinsics, (left, top, right, bottom))


def depth_to_world_points(depth: torch.Tensor, intrinsics: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
    height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype),
        torch.arange(width, dtype=depth.dtype),
        indexing="ij",
    )
    x = (xx - intrinsics[0, 2]) / intrinsics[0, 0] * depth
    y = (yy - intrinsics[1, 2]) / intrinsics[1, 1] * depth
    points_camera = torch.stack((x, y, depth, torch.ones_like(depth)), dim=-1)
    return torch.einsum("ij,hwj->hwi", camera_pose, points_camera)[..., :3]


def _points_in_first_camera(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_poses: torch.Tensor,
) -> torch.Tensor:
    world_points = torch.stack(
        [depth_to_world_points(depth, intrinsic, pose) for depth, intrinsic, pose in zip(depths, intrinsics, camera_poses)]
    )
    world_to_first = torch.linalg.inv(camera_poses[0])
    homogeneous = torch.cat((world_points, torch.ones_like(world_points[..., :1])), dim=-1)
    return torch.einsum("ij,shwj->shwi", world_to_first, homogeneous)[..., :3]


def discover_seven_scenes(root: Path) -> list[str]:
    scenes = []
    for category in sorted(path for path in root.iterdir() if path.is_dir()):
        split_path = category / "TestSplit.txt"
        if not split_path.exists():
            continue
        for line in split_path.read_text().splitlines():
            number = "".join(filter(str.isdigit, line))
            scenes.append(f"{category.name}/seq-{number.zfill(2)}")
    return scenes


def discover_nrgbd_scenes(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "images").is_dir())


def discover_scenes(data_root: str | Path, dataset: str) -> list[str]:
    dataset = dataset.lower()
    base_dataset = canonical_dataset(dataset)
    root = Path(data_root) / DATASET_DIRS[base_dataset]
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {dataset} dataset directory: {root}")
    if base_dataset == "7scenes":
        return discover_seven_scenes(root)
    if base_dataset == "nrgbd":
        return discover_nrgbd_scenes(root)
    raise ValueError(f"Unsupported reconstruction dataset: {dataset}")


def _read_nrgbd_poses(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    poses = []
    for offset in range(0, len(lines), 4):
        block = lines[offset : offset + 4]
        if len(block) < 4:
            break
        if any("nan" in line.lower() for line in block):
            poses.append(np.eye(4, dtype=np.float32))
        else:
            poses.append(np.asarray([[float(value) for value in line.split()] for line in block], dtype=np.float32))
    return np.stack(poses)


def _load_seven_scenes_raw(
    root: Path,
    scene_id: str,
    stride: int,
    max_frames: int | None = None,
    project_missing_depth: bool = False,
):
    scene_dir = root / scene_id
    all_image_paths = sorted(scene_dir.glob("frame-*.color.png"), key=_numeric_key)
    frame_ids = list(range(0, len(all_image_paths), stride))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise RuntimeError(f"No 7-Scenes RGB frames found under {scene_dir}")
    records = []
    intrinsics = np.asarray([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    for frame_id in frame_ids:
        prefix = f"frame-{frame_id:06d}"
        image_path = scene_dir / f"{prefix}.color.png"
        depth_path = scene_dir / f"{prefix}.depth.proj.png"
        pose_path = scene_dir / f"{prefix}.pose.txt"
        if not depth_path.exists():
            if not project_missing_depth:
                raise FileNotFoundError(
                    f"Missing projected depth {depth_path}. Run preprocess/prepare_reconstruction_datasets.py first."
                )
            from preprocess.prepare_reconstruction_datasets import project_depth_to_rgb

            raw_path = scene_dir / f"{prefix}.depth.png"
            raw_depth = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            if raw_depth is None:
                raise FileNotFoundError(f"Missing raw 7-Scenes depth: {raw_path}")
            depth = project_depth_to_rgb(raw_depth)
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0) / 1000.0
        depth[(depth > 10.0) | (depth < 1e-3)] = 0.0
        records.append((image_path, _read_rgb(image_path), depth, intrinsics.copy(), np.loadtxt(pose_path).astype(np.float32)))
    return records


def _load_nrgbd_raw(root: Path, scene_id: str, stride: int, max_frames: int | None = None):
    scene_dir = root / scene_id
    image_paths = sorted((scene_dir / "images").glob("img*.png"), key=_numeric_key)
    frame_ids = list(range(0, len(image_paths), stride))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise RuntimeError(f"No NRGBD RGB frames found under {scene_dir / 'images'}")
    poses = _read_nrgbd_poses(scene_dir / "poses.txt")
    intrinsics = np.asarray(
        [[554.2562584220408, 0.0, 320.0], [0.0, 554.2562584220408, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    records = []
    for index in frame_ids:
        image_path = scene_dir / "images" / f"img{index}.png"
        depth_path = scene_dir / "depth" / f"depth{index}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0) / 1000.0
        depth[(depth > 10.0) | (depth < 1e-3)] = 0.0
        pose = poses[index].copy()
        pose[:, 1:3] *= -1.0
        records.append((image_path, _read_rgb(image_path), depth, intrinsics.copy(), pose))
    return records


def load_scene(
    data_root: str | Path,
    dataset: str,
    scene_id: str,
    resolution: tuple[int, int] = (518, 392),
    stride: int | None = None,
    max_frames: int | None = None,
    project_missing_depth: bool = False,
) -> ReconstructionScene:
    dataset = dataset.lower()
    base_dataset = canonical_dataset(dataset)
    root = Path(data_root) / DATASET_DIRS[base_dataset]
    stride = stride or default_stride_for_dataset(dataset)
    if base_dataset == "7scenes":
        records = _load_seven_scenes_raw(
            root,
            scene_id,
            stride,
            max_frames=max_frames,
            project_missing_depth=project_missing_depth,
        )
    elif base_dataset == "nrgbd":
        records = _load_nrgbd_raw(root, scene_id, stride, max_frames=max_frames)
    else:
        raise ValueError(f"Unsupported reconstruction dataset: {dataset}")

    paths, images, depths, intrinsics, poses = [], [], [], [], []
    for path, image, depth, intrinsic, pose in records:
        image, depth, intrinsic = crop_resize_on_principal_point(image, depth, intrinsic, resolution)
        paths.append(path)
        images.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
        depths.append(torch.from_numpy(depth.astype(np.float32)))
        intrinsics.append(torch.from_numpy(intrinsic.astype(np.float32)))
        poses.append(torch.from_numpy(pose.astype(np.float32)))

    image_tensor = torch.stack(images)
    depth_tensor = torch.stack(depths)
    intrinsic_tensor = torch.stack(intrinsics)
    pose_tensor = torch.stack(poses)
    valid_masks = (depth_tensor > 0) & torch.isfinite(depth_tensor)
    points_gt = _points_in_first_camera(depth_tensor, intrinsic_tensor, pose_tensor)
    valid_masks &= torch.isfinite(points_gt).all(dim=-1)

    return ReconstructionScene(
        dataset=dataset,
        scene_id=scene_id,
        image_paths=paths,
        images=image_tensor,
        depths=depth_tensor,
        intrinsics=intrinsic_tensor,
        camera_poses=pose_tensor,
        points_gt=points_gt,
        valid_masks=valid_masks,
    )
