import os
import cv2
import json
import utils3d
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class dl3dv_data(base_data):
    def __init__(
        self,
        refine=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.refine = refine
    
    def sample_video(self, scene_id: int, seq_len: int | None = None, seed: int | None = None):
        """
        Samples a video sequence and retrieves associated frames, depths, and masks.
        """
        scene = self.scenes[scene_id]

        image_paths = scene['image']
        depth_paths = scene['depth']
        calib_paths = scene['calib']
        sky_mask_paths = scene['sky_mask']
        outlier_paths = scene['outlier']

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        if seq_len is not None:
            seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)
        else:
            # seq_ids = np.arange(len(image_paths))
            seq_ids = np.arange(100)
        
        views = defaultdict(list)
        
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth, sky_mask = self.load_depth(
                depth_paths[i], sky_mask_paths[i], outlier_paths[i]
            )
            depth = self.filter_depth(depth)

            intrinsic, pose = self.load_calib(calib_paths[i])
            intrinsic = self.normalize_intrinsic(
                intrinsic.copy(), image.shape[0], image.shape[1]
            )
            
            views['image'].append(image)
            views['depth'].append(depth)
            views['sky_mask'].append(sky_mask)
            views['pose'].append(pose)
            views['normal'].append(None)
            views['intrinsic'].append(intrinsic)
            views['image_path'].append(image_paths[i])
            views['depth_path'].append(depth_paths[i])
            views['instance'].append(str(i))
        
        return dict(views)
    
    def load_depth(
        self, depth_path: str, sky_mask_path: str, outlier_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        
        sky_mask_img = cv2.imread(
            os.path.join(self.data_path, sky_mask_path), cv2.IMREAD_UNCHANGED
        )
        sky_mask = sky_mask_img >= 127
        
        if self.refine:
            data = np.load(os.path.join(self.data_path, depth_path.replace('.npy', '.refine.npy')), allow_pickle=True).item()
            depth = data['depth']
            conf = data['confidence']

            depth[sky_mask] = 0.0

            valid_mask = depth > 0
            valid_confs = conf[valid_mask]
            
            if valid_confs.size > 0:
                conf_threshold = np.percentile(valid_confs, 20)
                
                low_conf_mask = valid_mask & (conf < conf_threshold)
                
                depth[low_conf_mask] = 0.0
        else:
            depth = np.load(os.path.join(self.data_path, depth_path))
        
            outlier_mask_img = cv2.imread(
                os.path.join(self.data_path, outlier_path), cv2.IMREAD_UNCHANGED
            )
            outlier_mask = outlier_mask_img >= 127
        
            depth[sky_mask] = 0.0
            depth[outlier_mask] = 0.0
        
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = threshold_depth_map(depth, max_percentile=98, min_percentile=-1)
        
        return depth.astype(np.float32), sky_mask.astype(np.float32)
    
    def load_calib(self, calib_path: str):
        calib_data = np.load(os.path.join(self.data_path, calib_path))
        
        intrinsics = calib_data["intrinsic"].astype(np.float32)
        pose = calib_data["pose"].astype(np.float32)
        
        return intrinsics, pose

def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
    """
    # depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map
