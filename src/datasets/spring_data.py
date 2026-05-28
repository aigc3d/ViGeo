import os
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class spring_data(base_data):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    
    def sample_video(self, scene_id: int, seq_len: int, seed: int):
        scene = self.scenes[scene_id]

        image_paths = scene['image']
        depth_paths = scene['depth']
        normal_paths = scene['normal']
        calib_paths = scene['calib']

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth, sky_mask = self.load_depth(depth_paths[i])
            normal = self.load_normal(normal_paths[i])
            depth = self.filter_depth(depth)
            intrinsic, pose = self.load_calib(calib_paths[i])
            intrinsic = self.normalize_intrinsic(intrinsic, image.shape[0], image.shape[1])
            views['image'].append(image)
            views['depth'].append(depth)
            views['normal'].append(normal)
            views['sky_mask'].append(sky_mask)
            views['pose'].append(pose)
            views['intrinsic'].append(intrinsic)
            views['image_path'].append(image_paths[i])
            views['depth_path'].append(depth_paths[i])
            views['instance'].append(str(i))
        
        return views
    
    def load_depth(self, depth_path):
        depth = np.load(os.path.join(self.data_path, depth_path))
        sky_mask = (depth == 0).astype(np.float32)
        return depth.astype(np.float32), sky_mask
    
    def load_calib(self, calib_path):
        data = np.load(os.path.join(self.data_path, calib_path))
        pose = data["pose"]
        intrinsic = data["intrinsics"]
        return intrinsic.astype(np.float32), pose.astype(np.float32)