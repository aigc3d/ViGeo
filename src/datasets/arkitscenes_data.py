import os
import cv2
import utils3d
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class arkitscenes_data(base_data):
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
        calib_path = scene['calib']

        intrin, poses = self.load_calib(calib_path)
        

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)
        
        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth, sky_mask = self.load_depth(depth_paths[i])
            depth = self.filter_depth(depth)
            pose = poses[i].copy()
            intrinsic = self.normalize_intrinsic(intrin.copy(), image.shape[0], image.shape[1])
            views['image'].append(image)
            views['depth'].append(depth)
            views['normal'].append(None)
            views['sky_mask'].append(sky_mask)
            views['pose'].append(pose)
            views['intrinsic'].append(intrinsic)
            views['image_path'].append(image_paths[i])
            views['depth_path'].append(depth_paths[i])
            views['instance'].append(str(i))
        
        return views
    
    def load_depth(self, depth_path):
        if self.refine:
            data = np.load(os.path.join(self.data_path, depth_path.replace('.png', '.refine.npy')), allow_pickle=True).item()
            depth = data['depth']
            conf = data['confidence']

            valid_mask = depth > 0
            valid_confs = conf[valid_mask]
            
            if valid_confs.size > 0:
                conf_threshold = np.percentile(valid_confs, 10)
                
                low_conf_mask = valid_mask & (conf < conf_threshold)
                
                depth[low_conf_mask] = 0.0
        else:
            depth = cv2.imread(
                os.path.join(self.data_path, depth_path), -1) #(H, W)
            depth = depth.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0  # invalid
            edge = utils3d.depth_map_edge(depth, rtol=0.5)
            depth[edge > 0] = 0

        sky_mask = np.zeros_like(depth)
        return depth.astype(np.float32), sky_mask.astype(np.float32)
    
    def load_calib(self, calib_path):
        data = np.load(os.path.join(self.data_path, calib_path))
        intrinsics = data['intrinsics']
        fx, fy, cx, cy = intrinsics[:, 2:6].mean(axis=0)

        intrinsics = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float32)
        
        poses = data['trajectories']
        return intrinsics, poses.astype(np.float32)

@DATASETS.register_module()
class arkitscenes_highres_data(arkitscenes_data):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)