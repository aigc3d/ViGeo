import os
import cv2
import json
import utils3d
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class ase_data(base_data):
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
        calib_path = scene['calib']

        intrin, poses = self.load_calib(calib_path)

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            normal = self.load_normal(normal_paths[i])
            
            if normal is None:
                intrinsic2 = self.normalize_intrinsic(intrin.copy(), image.shape[0], image.shape[1])
                normal, normal_mask = utils3d.depth_map_to_normal_map(
                    depth, intrinsic2, mask=(depth > 0)
                )
                normal[..., 1] *= -1
                normal[..., 2] *= -1

                normal[~normal_mask] = 0.0
            
                norm = np.linalg.norm(normal, axis=-1, keepdims=True)
                normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1e-6)
            
            sky_mask = np.zeros_like(depth)
            depth = self.filter_depth(depth)
            key = image_paths[i].split('/')[-1].split('.')[0][4:]
            w2c = np.array(poses[key]['poses_w2c']).reshape(3, 4)
            w2c = np.vstack([w2c, np.array([0, 0, 0, 1])])
            pose = np.linalg.inv(w2c).astype(np.float32)
            intrinsic = self.normalize_intrinsic(intrin.copy(), image.shape[0], image.shape[1])
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
    
    def load_depth(self, depth_path):
        depth = cv2.imread(os.path.join(self.data_path, depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth.reshape(512, 512)
        depth = depth * 0.001
        return depth.astype(np.float32)
    
    def load_calib(self, calib_path):
        with open(os.path.join(self.data_path, calib_path), 'r') as f:
            data = json.load(f)
        intrinsic = np.array(data['intrinsics']).reshape(3, 3)
        pose = data['camera_poses']
        return intrinsic.astype(np.float32), pose