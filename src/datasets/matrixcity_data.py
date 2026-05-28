import os
import cv2
import utils3d
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class matrixcity_data(base_data):
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
            depth = self.filter_depth(depth)
            pose = poses[i].copy()
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
        depth = cv2.imread(
            os.path.join(self.data_path, depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
        sky_mask = (depth==65504).astype(np.float32)
        edge_mask = utils3d.depth_map_edge(depth, rtol=0.4)
        depth = depth / 100.0
        depth[edge_mask > 0] = 0
        return depth, sky_mask
    
    def load_calib(self, calib_path):
        data = np.load(os.path.join(self.data_path, calib_path))
        camera_angle_x = data['camera_angle_x']

        w = 1000.0
        f = float(.5 * w / np.tan(.5 * camera_angle_x))

        intrinsic = np.zeros((3, 3), dtype=np.float32)
        intrinsic[0, 0] = f
        intrinsic[1, 1] = f
        intrinsic[0, 2] = 500
        intrinsic[1, 2] = 500
        intrinsic[2, 2] = 1

        pose = data['pose']
        return intrinsic.astype(np.float32), pose.astype(np.float32)