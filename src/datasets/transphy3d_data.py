import os
import io
import json
import numpy as np
from PIL import Image
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class transphy3d_data(base_data):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    
    def sample_video(self, scene_id: int, seq_len: int | None = None, seed: int | None = None):
        scene = self.scenes[scene_id]

        image_paths = scene['image']
        depth_paths = scene['depth']
        calib_paths = scene['calib']
        normal_paths = scene['normal']
        depth_json_paths = scene['depth_json']

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        if seq_len is not None:
            seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)
        else:
            seq_ids = np.arange(40, 120)
        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i], depth_json_paths[i])
            sky_mask = np.zeros_like(depth)
            normal = self.load_normal(normal_paths[i])
            intrinsic, pose = self.load_calib(calib_paths[i])
            depth = self.filter_depth(depth)
            views['image'].append(image)
            views['depth'].append(depth)
            views['sky_mask'].append(sky_mask)
            views['normal'].append(normal)
            views['pose'].append(pose)
            views['intrinsic'].append(intrinsic)
            views['image_path'].append(image_paths[i])
            views['depth_path'].append(depth_paths[i])
            views['instance'].append(str(i))
        
        return views
    
    def load_depth(self, depth_path, depth_json_path):
        depth_array = np.array(
            Image.open(os.path.join(self.data_path, depth_path)))
        depth_normalized = depth_array.astype(np.float32) / 65535.0
        with open(os.path.join(self.data_path, depth_json_path), 'r') as f:
            depth_info = json.load(f)
        max_depth = depth_info.get('max_depth', None)
        depth = depth_normalized * max_depth
        return depth.astype(np.float32)
    
    def load_normal(self, normal_path):
        img = np.array(Image.open(os.path.join(self.data_path, normal_path)))
        data = np.array(img).astype(np.float32)

        normal = (data / 127.5) - 1.0
        normal[..., 1] *= -1
        normal[..., 2] *= -1
            
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1e-6)
        return normal
    
    def load_calib(self, calib_path):
        with open(os.path.join(self.data_path, calib_path), 'r') as f:
            data = json.load(f)
        camera_matrices = data['camera_matrices']
        extrinsics = np.array(camera_matrices['extrinsics'])
        intrinsics = np.array(camera_matrices['intrinsics'])
        return intrinsics.astype(np.float32), np.linalg.inv(extrinsics.astype(np.float32))
