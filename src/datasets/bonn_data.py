import os
import json
import numpy as np
from PIL import Image
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict

@DATASETS.register_module()
class bonn_data(base_data):
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
        
        views = defaultdict(list)
        for i in range(len(image_paths)):
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            depth = self.filter_depth(depth)
            sky_mask = None
            pose = None
            intrinsic = self.load_intrinsic()
            intrinsic = self.normalize_intrinsic(intrinsic, image.shape[0], image.shape[1])
            views['image'].append(image)
            views['depth'].append(depth)
            views['sky_mask'].append(sky_mask)
            views['pose'].append(pose)
            views['normal'].append(None)
            views['intrinsic'].append(intrinsic)
            views['image_path'].append(image_paths[i])
            views['depth_path'].append(depth_paths[i])
            views['instance'].append(str(i))
        
        return views
    
    def load_depth(self, depth_path):
        depth = Image.open(os.path.join(self.data_path, depth_path))
        depth = np.asarray(depth, dtype=np.float32) / 5000.0
        return depth
    
    @staticmethod
    def load_intrinsic():
        K = np.array([
            [542.822841, 0.0,        315.593520],
            [0.0,        542.576870, 237.756098],
            [0.0,        0.0,        1.0       ]
        ])

        return K.astype(np.float32)