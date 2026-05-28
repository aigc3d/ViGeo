import os
import cv2
import json
import utils3d
import numpy as np
import imageio.v2 as imageio
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class omniobject_data(base_data):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        with open(os.path.join(self.data_path, 'scale.json')) as f:
            self.scales = json.load(f)
    
    def sample_video(self, scene_id: int, seq_len: int, seed: int):
        scene = self.scenes[scene_id]

        scale = self.scales[scene['scene']]
        image_paths = scene['image']
        depth_paths = scene['depth']
        normal_paths = scene['normal']
        calib_path = scene['calib']

        intrin, poses = self.load_calib(calib_path)
        

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth, sky_mask = self.load_depth(depth_paths[i])
            normal = self.load_normal(normal_paths[i])
            depth = depth / scale / 1000.0
            pose = poses[i].copy()
            pose[:3, 3] = pose[:3, 3] / scale / 1000.0
            intrinsic = self.normalize_intrinsic(intrin.copy(), image.shape[0], image.shape[1])
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
    
    def load_image(self, image_path):
        img = imageio.imread(os.path.join(self.data_path, image_path))
        if img.shape[-1] == 4:
            alpha_channel = img[..., 3]
            rgb_channels = img[..., :3]
            white_background = np.full_like(rgb_channels, 255)
            img = np.where(alpha_channel[..., None] == 0, white_background, rgb_channels)
        else:
            img = img[..., :3]
        return img
    
    def load_depth(self, depth_path):
        depth = cv2.imread(os.path.join(self.data_path, depth_path), cv2.IMREAD_UNCHANGED)

        # Use the last channel of the depth data and convert to float32
        depth = depth[..., -1].astype(np.float32)
        sky_mask = (depth >= 65504.0)
        depth[depth >= 65504.0] = 0.0  # Set invalid depth values to 0
        return depth.astype(np.float32), sky_mask.astype(np.float32)
    
    def load_calib(self, calib_path):
        data = np.load(os.path.join(self.data_path, calib_path))
        
        intrinsic = data['intrinsics']
        pose = data['pose']
        return intrinsic.astype(np.float32), pose.astype(np.float32)