import os
import cv2
import utils3d
import numpy as np
from PIL import Image
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class scannetpp_data(base_data):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    
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
            depth = self.load_depth(depth_paths[i])
            sky_mask = np.zeros_like(depth)
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
        depth = np.load(os.path.join(self.data_path, depth_path))
        return depth.astype(np.float32)
    
    def load_calib(self, calib_path):
        data = np.load(os.path.join(self.data_path, calib_path))
        intrinsic = data['intrinsic']
        pose = data['pose']

        return intrinsic.astype(np.float32), pose.astype(np.float32)