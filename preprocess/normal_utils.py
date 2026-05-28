import cv2
import numpy as np
import torch

from depth2normal import Depth2normal


def build_normal_estimator(min_depth, max_depth, device):
    normal_estimator = Depth2normal(
        d_min=min_depth,
        d_max=max_depth,
        k=5,
        d=1,
        gamma=0.05,
        min_nghbr=4,
    ).to(device)
    normal_estimator.eval()
    return normal_estimator


def visualize_normal(normal_array):
    mag = np.linalg.norm(normal_array, axis=-1, keepdims=True)
    aligned = np.divide(normal_array, mag, out=np.zeros_like(normal_array), where=mag > 1e-6)
    vis = np.clip((aligned + 1.0) / 2.0, 0.0, 1.0)
    return cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def depth_map_to_point_map(depth, intrinsic):
    height, width = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def normal_from_depth(depth, intrinsic, normal_estimator, device):
    pts_cam = depth_map_to_point_map(depth, intrinsic)
    pts_tensor = torch.from_numpy(pts_cam).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        normal_tensor, valid_mask_tensor = normal_estimator(pts_tensor)

    normal = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    valid_mask = valid_mask_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    normal[~valid_mask[..., 0]] = 0.0
    return normal
