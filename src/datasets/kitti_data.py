import os
import json
import numpy as np
from PIL import Image
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class kitti_data(base_data):
    def __init__(
        self,
        start: int = 0,
        end: int | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.start = start
        self.end = end if end is not None else 1000
    
    def load_annotations(self, train_test_split):
        with open(train_test_split, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
        return scenes
    
    def sample_video(self, index):
        scene = self.scenes[index]

        image_paths = scene['image'][self.start:self.end]
        depth_paths = scene['depth'][self.start:self.end]
        calib_path = scene['calib']
        
        views = defaultdict(list)
        for i in range(len(image_paths)):
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            sky_mask = None
            pose = None
            intrinsic = self.load_intrinsic(calib_path)
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
    
    def load_depth(self, depth_path):
        depth = Image.open(os.path.join(self.data_path, depth_path))
        depth = np.asarray(depth, dtype=np.float32) / 256.0

        return depth
    
    def load_intrinsic(self, calib_path):
        with open(os.path.join(self.data_path, calib_path)) as f:
            for line in f:
                if line.startswith('P_rect_02:'):
                    intrinsic = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)[:, :3]
                    return intrinsic