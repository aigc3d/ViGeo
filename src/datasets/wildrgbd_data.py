import os
import cv2
import numpy as np
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class wildrgbd_data(base_data):
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
            depth = self.load_depth(depth_paths[i], rng)
            sky_mask = np.zeros_like(depth)
            depth = self.filter_depth(depth)
            intrinsic, pose = self.load_calib(calib_paths[i])
            intrinsic = self.normalize_intrinsic(intrinsic, image.shape[0], image.shape[1])
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
    
    def load_depth(self, depth_path, rng):
        if self.refine:
            data = np.load(os.path.join(self.data_path, depth_path.replace('.png', '.refine.npy')), allow_pickle=True).item()
            depth = data['depth']
            conf = data['confidence']

            valid_mask = depth > 0
            valid_confs = conf[valid_mask]
            
            if valid_confs.size > 0:
                conf_threshold = np.percentile(valid_confs, 20)
                
                low_conf_mask = valid_mask & (conf < conf_threshold)
                
                depth[low_conf_mask] = 0.0
        else:
            depth = cv2.imread(os.path.join(self.data_path, depth_path), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000.0

        # if rng.choice(2, p=[0.9, 0.1]):
        #     mask = cv2.imread(os.path.join(self.data_path, 
        #         depth_path.replace('depth', 'masks')), cv2.IMREAD_UNCHANGED).astype(np.float32)
        #     mask = (mask / 255.0) > 0.1
        #     depth *= mask
        return depth
    
    def load_calib(self, calib_path):
        data = np.load(os.path.join(self.data_path, calib_path))
        intrinsic = data['camera_intrinsics']
        pose = data['camera_pose']
        return intrinsic.astype(np.float32), pose.astype(np.float32)