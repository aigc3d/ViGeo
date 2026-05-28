import os
import OpenEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class bedlam_data(base_data):
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
        calib_data = self.load_calib(calib_path)
        cam_ints = calib_data["intrinsics"].astype(np.float32)
        poses = calib_data["pose"].astype(np.float32)

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            normal = self.load_normal(normal_paths[i])
            sky_mask = np.zeros_like(depth).astype(np.float32)
            depth = self.filter_depth(depth)
            intrinsic = cam_ints[i].copy()
            pose = poses[i].copy()
            intrinsic = self.normalize_intrinsic(intrinsic, image.shape[0], image.shape[1])
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

    def load_calib(self, calib_path):
        calib_data = np.load(os.path.join(self.data_path, calib_path))
        return calib_data
    
    def load_depth(self, depth_path):
        depth= OpenEXR.File(os.path.join(self.data_path, depth_path)).parts[0].channels['Depth'].pixels
        depth = depth.astype(np.float32)/100.0
        return depth