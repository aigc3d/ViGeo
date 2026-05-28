import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image

from benchmark_defs import TASK_POINTMAP


BONN_SCENES = [
    'rgbd_bonn_balloon2',
    'rgbd_bonn_crowd2',
    'rgbd_bonn_crowd3',
    'rgbd_bonn_person_tracking2',
    'rgbd_bonn_synchronous',
]


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
