import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from benchmark_defs import TASK_POINTMAP


BONN_SCENES = [
    'rgbd_bonn_balloon2',
    'rgbd_bonn_crowd2',
    'rgbd_bonn_crowd3',
    'rgbd_bonn_person_tracking2',
    'rgbd_bonn_synchronous',
]

RECONSTRUCTION_DATASETS = ("7scenes", "nrgbd", "7scenes-sparse", "7scenes-dense", "nrgbd-sparse", "nrgbd-dense")
RECONSTRUCTION_DATASET_DIRS = {"7scenes": "7scenes", "nrgbd": "neural_rgbd"}
RECONSTRUCTION_DATASET_SETTINGS = {
    "7scenes": ("7scenes", 200),
    "7scenes-sparse": ("7scenes", 200),
    "7scenes-dense": ("7scenes", 40),
    "nrgbd": ("nrgbd", 500),
    "nrgbd-sparse": ("nrgbd", 500),
    "nrgbd-dense": ("nrgbd", 100),
}


@dataclass(frozen=True)
class PoseSequence:
    dataset: str
    sequence: str
    image_paths: list[Path]
    gt_poses: np.ndarray
    timestamps: np.ndarray


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


def numeric_path_key(path):
    path = Path(path)
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def read_depth_sintel(path):
    with open(path, "rb") as f:
        assert np.fromfile(f, dtype=np.float32, count=1)[0] == 202021.25
        w, h = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32, count=-1).reshape((h, w))


def read_depth_bonn(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 5000.0


def read_depth_kitti(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 256.0


def read_depth_hammer(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return np.zeros((480, 640), dtype=np.float32) if depth is None else depth.astype(np.float32) / 1000.0


def read_normal_map(path, dataset_name=None, valid_threshold=0.1):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dataset_name == 'nyuv2':
        img = img[45:472, 43:608, :]

    img_float = img.astype(np.float32) / 255.0
    valid_mask = np.linalg.norm(img_float, axis=-1, keepdims=True) > valid_threshold
    normal = img_float * 2.0 - 1.0
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, normal_norm, out=np.zeros_like(normal), where=normal_norm > 1e-6)
    return np.where(valid_mask, normal, np.zeros_like(normal))


def read_sintel_calib(path):
    with open(path, 'rb') as f:
        _ = np.fromfile(f, dtype=np.float32, count=1)[0]
        return np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3)).astype(np.float32)


def read_kitti_calib(path):
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                return np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)[:, :3]
    return np.eye(3, dtype=np.float32)


def get_bonn_intrinsic():
    return np.array(
        [[542.822841, 0.0, 315.593520],
         [0.0, 542.576870, 237.756098],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def get_intrinsic(dataset, calib_path):
    if dataset == 'sintel':
        return read_sintel_calib(calib_path)
    if 'kitti' in dataset:
        return read_kitti_calib(calib_path)
    return get_bonn_intrinsic()


def get_sintel_files(base_path, scene):
    scene_dir = os.path.join(base_path, 'clean', scene)
    files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.png')])
    imgs = [os.path.join(scene_dir, f) for f in files]
    depths = [os.path.join(base_path, 'depth', scene, f.replace('.png', '.dpt')) for f in files]
    normals = [os.path.join(base_path, 'normal', scene, f) for f in files]
    calibs = [os.path.join(base_path, 'camdata_left', scene, f.replace('.png', '.cam')) for f in files]
    return imgs, depths, normals, calibs


def get_bonn_files(base_path, scene):
    img_dir = os.path.join(base_path, scene, 'rgb')
    depth_dir = os.path.join(base_path, scene, 'depth')
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
    depths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)])
    return imgs, depths, [None] * len(imgs), [None] * len(imgs)


def get_kitti_files(base_path, scene):
    depth_dir = os.path.join(base_path, 'depth', scene, 'proj_depth/groundtruth/image_02')
    files = sorted(os.listdir(depth_dir))
    imgs = [os.path.join(base_path, 'image', scene, 'image_02/data', f) for f in files]
    depths = [os.path.join(depth_dir, f) for f in files]
    calib = os.path.join(base_path, 'calib', scene[:10], 'calib_cam_to_cam.txt')
    return imgs, depths, [None] * len(imgs), [calib] * len(files)


def get_nyuv2_files(base_path, scene):
    return (
        [os.path.join(base_path, f"{scene}_img.png")],
        [None],
        [os.path.join(base_path, f"{scene}_normal.png")],
        [None],
    )


def get_hammer_files(base_path, scene):
    target_dir = (
        os.path.join(base_path, scene, "polarization")
        if os.path.isdir(os.path.join(base_path, scene, "polarization"))
        else os.path.join(base_path, scene)
    )
    imgs = sorted([os.path.join(target_dir, 'rgb', f) for f in os.listdir(os.path.join(target_dir, 'rgb'))])
    depths = [
        os.path.join(target_dir, '_gt', f"{os.path.splitext(os.path.basename(f))[0]}.png")
        for f in imgs
    ]
    normals = [
        os.path.join(target_dir, 'normal', f"{os.path.splitext(os.path.basename(f))[0]}_normal.png")
        for f in imgs
    ]
    return imgs, depths, normals, [None] * len(imgs)


def strip_depth_fields(get_files):
    def wrapped(base_path, scene):
        imgs, depths, _, _ = get_files(base_path, scene)
        return imgs, depths
    return wrapped


def strip_pointmap_fields(get_files):
    def wrapped(base_path, scene):
        imgs, depths, _, calibs = get_files(base_path, scene)
        return imgs, depths, calibs
    return wrapped


def get_scenes(path, dataset):
    if not os.path.exists(path):
        return []
    if dataset == 'sintel':
        clean_dir = os.path.join(path, 'clean')
        if not os.path.exists(clean_dir):
            return []
        return sorted([d for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))])
    if dataset == 'kitti':
        return sorted(os.listdir(os.path.join(path, 'image')))
    if dataset == 'nyuv2':
        return sorted({os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(path, '*_img.png'))})
    if dataset == 'hammer':
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    return []


def get_normal_benchmark_scenes(path, dataset):
    if dataset == 'hammer':
        return sorted([d for d in os.listdir(path) if d.startswith('scene')])
    return get_scenes(path, dataset)


def depth_to_points(depth_tensor, K_tensor):
    if depth_tensor.dim() == 2:
        depth_tensor = depth_tensor.unsqueeze(0)
    if K_tensor.dim() == 2:
        K_tensor = K_tensor.unsqueeze(0)

    n, h, w = depth_tensor.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=depth_tensor.device),
        torch.arange(w, device=depth_tensor.device),
        indexing='ij',
    )
    x = x.float().unsqueeze(0).expand(n, -1, -1)
    y = y.float().unsqueeze(0).expand(n, -1, -1)

    fx = K_tensor[:, 0, 0].view(n, 1, 1)
    fy = K_tensor[:, 1, 1].view(n, 1, 1)
    cx = K_tensor[:, 0, 2].view(n, 1, 1)
    cy = K_tensor[:, 1, 2].view(n, 1, 1)
    z = depth_tensor
    return torch.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), dim=-1)


def get_vigeo_dataset_config(data_root, task):
    hammer_slice = slice(0, 220, 2) if task == 'normal' else slice(0, 300)
    return {
        'sintel': {
            'base': os.path.join(data_root, 'sintel'),
            'slice': slice(0, 50),
            'get': get_sintel_files,
            'read_d': read_depth_sintel,
            'scenes': lambda p: get_scenes(p, 'sintel'),
        },
        'bonn': {
            'base': os.path.join(data_root, 'bonn'),
            'slice': slice(30, 140),
            'get': get_bonn_files,
            'read_d': read_depth_bonn,
            'scenes': BONN_SCENES,
        },
        'bonn_400': {
            'base': os.path.join(data_root, 'bonn'),
            'slice': slice(0, 400),
            'get': get_bonn_files,
            'read_d': read_depth_bonn,
            'scenes': BONN_SCENES,
        },
        'kitti': {
            'base': os.path.join(data_root, 'kitti'),
            'slice': slice(0, 110),
            'get': get_kitti_files,
            'read_d': read_depth_kitti,
            'scenes': lambda p: get_scenes(p, 'kitti'),
        },
        'kitti_300': {
            'base': os.path.join(data_root, 'kitti'),
            'slice': slice(0, 300),
            'get': get_kitti_files,
            'read_d': read_depth_kitti,
            'scenes': lambda p: get_scenes(p, 'kitti'),
        },
        'nyuv2': {
            'base': os.path.join(data_root, 'nyuv2', 'test'),
            'slice': slice(None),
            'get': get_nyuv2_files,
            'read_d': None,
            'scenes': lambda p: get_scenes(p, 'nyuv2'),
        },
        'hammer': {
            'base': os.path.join(data_root, 'hammer'),
            'slice': hammer_slice,
            'get': get_hammer_files,
            'read_d': read_depth_hammer,
            'scenes': lambda p: get_scenes(p, 'hammer'),
        },
    }


def get_depth_benchmark_dataset_config(data_root, task):
    if task == TASK_POINTMAP:
        return {
            'sintel': {
                'base': os.path.join(data_root, 'sintel'),
                'slice': slice(0, 50),
                'get': strip_pointmap_fields(get_sintel_files),
                'read_d': read_depth_sintel,
                'scenes': lambda p: get_scenes(p, 'sintel'),
            },
            'bonn': {
                'base': os.path.join(data_root, 'bonn'),
                'slice': slice(30, 140),
                'get': strip_pointmap_fields(get_bonn_files),
                'read_d': read_depth_bonn,
                'scenes': BONN_SCENES,
            },
            'kitti': {
                'base': os.path.join(data_root, 'kitti'),
                'slice': slice(0, 110),
                'get': strip_pointmap_fields(get_kitti_files),
                'read_d': read_depth_kitti,
                'scenes': lambda p: get_scenes(p, 'kitti'),
            },
        }

    return {
        'sintel': {
            'base': os.path.join(data_root, 'sintel'),
            'slice': slice(0, 50),
            'get': strip_depth_fields(get_sintel_files),
            'read_d': read_depth_sintel,
            'scenes': lambda p: get_scenes(p, 'sintel'),
        },
        'bonn': {
            'base': os.path.join(data_root, 'bonn'),
            'slice': slice(30, 140),
            'get': strip_depth_fields(get_bonn_files),
            'read_d': read_depth_bonn,
            'scenes': BONN_SCENES,
        },
        'bonn_200': {
            'base': os.path.join(data_root, 'bonn'),
            'slice': slice(0, 200),
            'get': strip_depth_fields(get_bonn_files),
            'read_d': read_depth_bonn,
            'scenes': BONN_SCENES,
        },
        'bonn_400': {
            'base': os.path.join(data_root, 'bonn'),
            'slice': slice(0, 400),
            'get': strip_depth_fields(get_bonn_files),
            'read_d': read_depth_bonn,
            'scenes': BONN_SCENES,
        },
        'kitti': {
            'base': os.path.join(data_root, 'kitti'),
            'slice': slice(0, 110),
            'get': strip_depth_fields(get_kitti_files),
            'read_d': read_depth_kitti,
            'scenes': lambda p: get_scenes(p, 'kitti'),
        },
        'kitti_300': {
            'base': os.path.join(data_root, 'kitti'),
            'slice': slice(0, 300),
            'get': strip_depth_fields(get_kitti_files),
            'read_d': read_depth_kitti,
            'scenes': lambda p: get_scenes(p, 'kitti'),
        },
        'hammer': {
            'base': os.path.join(data_root, 'hammer'),
            'slice': slice(0, 300),
            'get': strip_depth_fields(get_hammer_files),
            'read_d': read_depth_hammer,
            'scenes': lambda p: get_scenes(p, 'hammer'),
        },
    }


def get_normal_benchmark_dataset_config(data_root):
    return {
        'sintel': {
            'base': os.path.join(data_root, 'sintel'),
            'slice': slice(None),
            'get': get_sintel_files,
            'scenes': lambda p: get_normal_benchmark_scenes(p, 'sintel'),
        },
        'nyuv2': {
            'base': os.path.join(data_root, 'nyuv2', 'test'),
            'slice': slice(None),
            'get': get_nyuv2_files,
            'scenes': lambda p: get_normal_benchmark_scenes(p, 'nyuv2'),
        },
        'hammer': {
            'base': os.path.join(data_root, 'hammer'),
            'slice': slice(0, 220, 2),
            'get': get_hammer_files,
            'scenes': lambda p: get_normal_benchmark_scenes(p, 'hammer'),
        },
    }


# ==============================================================================
# ViGeo camera pose estimation data
# ==============================================================================

def read_sintel_camera_pose(path):
    with open(path, "rb") as handle:
        check = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if check != 202021.25:
            raise ValueError(f"Bad Sintel .cam tag in {path}: {check}")
        _intrinsic = np.fromfile(handle, dtype=np.float64, count=9).reshape(3, 3)
        world_to_camera = np.fromfile(handle, dtype=np.float64, count=12).reshape(3, 4)
    pose = np.eye(4, dtype=np.float64)
    pose[:3] = world_to_camera
    return np.linalg.inv(pose)


def c2w_to_pose_row(c2w):
    xyz = c2w[:3, 3]
    qx, qy, qz, qw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    return np.asarray([xyz[0], xyz[1], xyz[2], qw, qx, qy, qz], dtype=np.float64)


def get_pose_sequences(data_root, dataset, sintel_pass="clean"):
    dataset = dataset.lower()
    if dataset == "sintel":
        image_root = Path(data_root) / "sintel" / sintel_pass
        if not image_root.is_dir():
            return []
        return sorted(path.name for path in image_root.iterdir() if path.is_dir())
    raise ValueError(f"Unsupported pose dataset: {dataset}")


def load_pose_sequence(data_root, dataset, sequence, stride=1, sintel_pass="clean"):
    dataset = dataset.lower()
    if dataset == "sintel":
        base = Path(data_root) / "sintel"
        image_dir = base / sintel_pass / sequence
        cam_dir = base / "camdata_left" / sequence
        image_paths = sorted(image_dir.glob("*.png"), key=numeric_path_key)[::stride]
        if not image_paths:
            raise RuntimeError(f"No Sintel images found under {image_dir}")

        gt_poses, timestamps = [], []
        for image_path in image_paths:
            c2w = read_sintel_camera_pose(cam_dir / image_path.with_suffix(".cam").name)
            gt_poses.append(c2w_to_pose_row(c2w))
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

    raise ValueError(f"Unsupported pose dataset: {dataset}")


# ==============================================================================
# ViGeo 3D reconstruction data
# ==============================================================================

def canonical_reconstruction_dataset(dataset):
    dataset = dataset.lower()
    if dataset not in RECONSTRUCTION_DATASET_SETTINGS:
        raise ValueError(f"Unsupported reconstruction dataset: {dataset}")
    return RECONSTRUCTION_DATASET_SETTINGS[dataset][0]


def default_reconstruction_stride(dataset):
    dataset = dataset.lower()
    if dataset not in RECONSTRUCTION_DATASET_SETTINGS:
        raise ValueError(f"Unsupported reconstruction dataset: {dataset}")
    return RECONSTRUCTION_DATASET_SETTINGS[dataset][1]


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_image_depth(image, depth, intrinsics, bbox):
    left, top, right, bottom = bbox
    image = image.crop(bbox)
    depth = depth[top:bottom, left:right]
    intrinsics = intrinsics.copy()
    intrinsics[0, 2] -= left
    intrinsics[1, 2] -= top
    return image, depth, intrinsics


def scale_intrinsics(intrinsics, input_resolution, output_resolution, scale):
    output = intrinsics.copy()
    output[0, 2] += 0.5
    output[1, 2] += 0.5
    output[:2] *= scale
    margins = input_resolution * scale - output_resolution
    output[:2, 2] -= 0.5 * margins
    output[0, 2] -= 0.5
    output[1, 2] -= 0.5
    return output


def crop_resize_on_principal_point(image_rgb, depth, intrinsics, resolution):
    image = Image.fromarray(image_rgb)
    width, height = image.size
    cx, cy = np.rint(intrinsics[:2, 2]).astype(int)
    margin_x = min(cx, width - cx)
    margin_y = min(cy, height - cy)
    if margin_x <= width / 5 or margin_y <= height / 5:
        raise ValueError(f"Principal point is too close to an image boundary: {(cx, cy)} in {(width, height)}")

    image, depth, intrinsics = crop_image_depth(
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
    intrinsics = scale_intrinsics(intrinsics, input_resolution, output_resolution, scale)

    margins = output_resolution - np.asarray(resolution)
    left, top = np.rint(0.5 * margins).astype(int)
    right, bottom = left + resolution[0], top + resolution[1]
    return crop_image_depth(image, depth, intrinsics, (left, top, right, bottom))


def depth_to_world_points(depth, intrinsics, camera_pose):
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


def points_in_first_camera(depths, intrinsics, camera_poses):
    world_points = torch.stack(
        [depth_to_world_points(depth, intrinsic, pose) for depth, intrinsic, pose in zip(depths, intrinsics, camera_poses)]
    )
    world_to_first = torch.linalg.inv(camera_poses[0])
    homogeneous = torch.cat((world_points, torch.ones_like(world_points[..., :1])), dim=-1)
    return torch.einsum("ij,shwj->shwi", world_to_first, homogeneous)[..., :3]


def get_seven_scenes_reconstruction_scenes(root):
    scenes = []
    for category in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        split_path = category / "TestSplit.txt"
        if not split_path.exists():
            continue
        for line in split_path.read_text().splitlines():
            number = "".join(filter(str.isdigit, line))
            scenes.append(f"{category.name}/seq-{number.zfill(2)}")
    return scenes


def get_nrgbd_reconstruction_scenes(root):
    return sorted(path.name for path in Path(root).iterdir() if path.is_dir() and (path / "images").is_dir())


def get_reconstruction_scenes(data_root, dataset):
    base_dataset = canonical_reconstruction_dataset(dataset)
    root = Path(data_root) / RECONSTRUCTION_DATASET_DIRS[base_dataset]
    if not root.is_dir():
        return []
    if base_dataset == "7scenes":
        return get_seven_scenes_reconstruction_scenes(root)
    if base_dataset == "nrgbd":
        return get_nrgbd_reconstruction_scenes(root)
    raise ValueError(f"Unsupported reconstruction dataset: {dataset}")


def read_nrgbd_poses(path):
    lines = Path(path).read_text().splitlines()
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


def load_seven_scenes_reconstruction_raw(root, scene_id, stride, max_frames=None, project_missing_depth=False):
    scene_dir = Path(root) / scene_id
    all_image_paths = sorted(scene_dir.glob("frame-*.color.png"), key=numeric_path_key)
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
        records.append((image_path, read_rgb(image_path), depth, intrinsics.copy(), np.loadtxt(pose_path).astype(np.float32)))
    return records


def load_nrgbd_reconstruction_raw(root, scene_id, stride, max_frames=None):
    scene_dir = Path(root) / scene_id
    image_paths = sorted((scene_dir / "images").glob("img*.png"), key=numeric_path_key)
    frame_ids = list(range(0, len(image_paths), stride))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise RuntimeError(f"No NRGBD RGB frames found under {scene_dir / 'images'}")
    poses = read_nrgbd_poses(scene_dir / "poses.txt")
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
        records.append((image_path, read_rgb(image_path), depth, intrinsics.copy(), pose))
    return records


def load_reconstruction_scene(
    data_root,
    dataset,
    scene_id,
    resolution=(518, 392),
    stride=None,
    max_frames=None,
    project_missing_depth=False,
):
    dataset = dataset.lower()
    base_dataset = canonical_reconstruction_dataset(dataset)
    stride = default_reconstruction_stride(dataset) if stride is None else stride
    root = Path(data_root) / RECONSTRUCTION_DATASET_DIRS[base_dataset]
    if base_dataset == "7scenes":
        records = load_seven_scenes_reconstruction_raw(
            root,
            scene_id,
            stride,
            max_frames=max_frames,
            project_missing_depth=project_missing_depth,
        )
    elif base_dataset == "nrgbd":
        records = load_nrgbd_reconstruction_raw(root, scene_id, stride, max_frames=max_frames)
    else:
        raise ValueError(f"Unsupported reconstruction dataset: {dataset}")

    image_paths, images, depths, intrinsics, poses = [], [], [], [], []
    for image_path, rgb, depth, intrinsic, pose in records:
        image, depth, intrinsic = crop_resize_on_principal_point(rgb, depth, intrinsic, resolution)
        image_paths.append(image_path)
        images.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
        depths.append(torch.from_numpy(depth).float())
        intrinsics.append(torch.from_numpy(intrinsic).float())
        poses.append(torch.from_numpy(pose).float())

    images = torch.stack(images)
    depths = torch.stack(depths)
    intrinsics = torch.stack(intrinsics)
    camera_poses = torch.stack(poses)
    points_gt = points_in_first_camera(depths, intrinsics, camera_poses)
    valid_masks = depths > 1e-3
    return ReconstructionScene(
        dataset=dataset,
        scene_id=scene_id,
        image_paths=image_paths,
        images=images,
        depths=depths,
        intrinsics=intrinsics,
        camera_poses=camera_poses,
        points_gt=points_gt,
        valid_masks=valid_masks,
    )
