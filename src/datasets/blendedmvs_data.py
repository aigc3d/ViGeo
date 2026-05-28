import os
import cv2
import utils3d
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

@DATASETS.register_module()
class blendedmvs_data(base_data):
    def __init__(
        self,
        refine=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.refine = refine
    
    def sample_video(self, scene_id: int, seq_len: int, seed: int):
        scene = self.scenes[scene_id]

        image_paths = scene['image']
        depth_paths = scene['depth']
        calib_paths = scene['calib']        

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            sky_mask = np.zeros_like(depth)
            depth = self.filter_depth(depth)
            intrinsic, pose = self.load_calib(calib_paths[i])
            intrinsic = self.normalize_intrinsic(intrinsic.copy(), image.shape[0], image.shape[1])
            views['image'].append(image)
            views['depth'].append(depth)
            views['normal'].append(None)
            views['sky_mask'].append(sky_mask)
            views['pose'].append(pose)
            views['intrinsic'].append(intrinsic)
            views['image_path'].append(image_paths[i])
            views['depth_path'].append(depth_paths[i])
            views['instance'].append(str(i))
            views['scene_id'].append(scene['scene'])
        
        return views
    
    def load_depth(self, depth_path):
        if self.refine:
            data = np.load(os.path.join(self.data_path, depth_path.replace('.exr', '.refine.npy')), allow_pickle=True).item()
            depth = data['depth']
            conf = data['confidence']

            edge = utils3d.depth_map_edge(depth, rtol=0.5)
            sky_mask = cv2.imread(os.path.join(self.data_path, depth_path.replace('.exr', '.mask.png')), -1) == 0
            depth[sky_mask] = 0
            depth[edge > 0] = 0

            valid_mask = depth > 0
            valid_confs = conf[valid_mask]
            
            if valid_confs.size > 0:
                conf_threshold = np.percentile(valid_confs, 10)
                
                low_conf_mask = valid_mask & (conf < conf_threshold)
                
                depth[low_conf_mask] = 0.0
        else:
            depth = cv2.imread(
                os.path.join(self.data_path, depth_path), cv2.IMREAD_ANYDEPTH)
        
        depth = threshold_depth_map(depth, max_percentile=99, min_percentile=-1)
        return depth

    def load_calib(self, calib_path):
        data = np.load(os.path.join(self.data_path, calib_path))
        intrinsic = np.float32(data["intrinsics"])
        R_c2w = np.float32(data["R_cam2world"])
        T_c2w = np.float32(data["t_cam2world"])

        pose = np.eye(4, dtype=np.float32)
        
        pose[:3, :3] = R_c2w
        pose[:3, 3] = T_c2w.reshape(3) 

        return intrinsic, pose

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
