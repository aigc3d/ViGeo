import os
import cv2
import json
import utils3d
import imageio
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

@DATASETS.register_module()
class omniworld_data(base_data):
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
            depth, sky_mask = self.load_depth(depth_paths[i])
            normal = self.load_normal(normal_paths[i])
            pose = poses[i].copy()
            depth = self.filter_depth(depth)
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
    
    def load_depth(self, depth_path):
        depth = imageio.v2.imread(os.path.join(self.data_path, depth_path)) / 65535.0
        near_mask = depth < 0.0015   # 1. too close
        sky_mask = depth > (65500.0 / 65535.0) # 2. filter sky
        far_mask = depth > np.percentile(depth[~sky_mask], 95) # 3. filter far area (optional)
        near, far = 1., 1000.

        depth = depth / (far - depth * (far - near)) / 0.004

        valid = ~(near_mask | far_mask | sky_mask)
        depth[~valid] = 0

        return depth.astype(np.float32), sky_mask.astype(np.float32)
    
    def load_calib(self, calib_path):
        with open(os.path.join(self.data_path, calib_path), 'r') as f:
            cam = json.load(f)
        
        intrinsics = np.eye(3)
        f = np.mean(cam["focals"])
        intrinsics[0, 0] = f          # fx
        intrinsics[1, 1] = f          # fy
        intrinsics[0, 2] = cam["cx"]              # cx
        intrinsics[1, 2] = cam["cy"]              # cy

        quat_wxyz = np.array(cam["quats"])           # (S, 4)  (w,x,y,z)
        quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)

        rotations = R.from_quat(quat_xyzw).as_matrix()      # (S, 3, 3)
        translations = np.array(cam["trans"])               # (S, 3)
        S = rotations.shape[0]
        extrinsics = np.repeat(np.eye(4)[None, ...], S, axis=0)
        extrinsics[:, :3, :3] = rotations
        extrinsics[:, :3, 3] = translations
        return intrinsics.astype(np.float32), np.linalg.inv(extrinsics).astype(np.float32)