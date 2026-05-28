import os
import cv2
import numpy as np
from PIL import Image
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class carlaocc_data(base_data):
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
        segmentation_paths = scene['semantics']

        calib_path = scene['pose']
        calib_data = self.load_calib(calib_path)
        cam_ints = calib_data["intrinsics"].astype(np.float32)
        poses = calib_data["poses"].astype(np.float32)

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            normal = self.load_normal(normal_paths[i])
            sky_mask = self.load_segmentation(segmentation_paths[i])
            depth[sky_mask > 0] = 0
            normal[sky_mask > 0] = 0.0
            pose = poses[i].copy()
            depth = self.filter_depth(depth)
            intrinsic = self.normalize_intrinsic(cam_ints.copy(), image.shape[0], image.shape[1])
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

    def load_calib(self, calib_path):
        calib_data = np.load(os.path.join(self.data_path, calib_path))
        return calib_data
    
    def load_segmentation(self, seg_path):
        seg_img = cv2.imread(os.path.join(self.data_path, seg_path), -1)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
        sky_mask = (seg_img[:, :, 0] == 70) & (seg_img[:, :, 1] == 130) & (seg_img[:, :, 2] == 180)
        return sky_mask.astype(np.float32)
    
    def load_depth(self, depth_path):
        depth = cv2.imread(
            os.path.join(self.data_path, depth_path), -1).astype(np.float32) / 65535.0 * 80.0
        return depth
    
    def load_normal(self, normal_path):
        img = np.array(Image.open(os.path.join(self.data_path, normal_path)))
        data = np.array(img).astype(np.float32)

        normal = (data / 127.5) - 1.0
        normal *= -1
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1e-6)
        return normal