import os
import json
import numpy as np
from PIL import Image
from .base_data import base_data
from mmengine import DATASETS
from collections import defaultdict
TAG_FLOAT = 202021.25

@DATASETS.register_module()
class sintel_data(base_data):
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
        calib_paths = scene['calib']
        
        views = defaultdict(list)
        for i in range(len(image_paths)):
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            sky_mask = None
            pose = None
            calib_data = self.load_calib(calib_paths[i])
            
            intrinsic = calib_data['intrinsics']
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
        f = open(os.path.join(self.data_path, depth_path),'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
        width = np.fromfile(f,dtype=np.int32,count=1)[0]
        height = np.fromfile(f,dtype=np.int32,count=1)[0]
        size = width*height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
        depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
        return depth.astype(np.float32)
    
    def load_calib(self, calib_path):
        f = open(os.path.join(self.data_path, calib_path),'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
        intrinsics = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
        extrinsics = np.fromfile(f,dtype='float64',count=12).reshape((3,4))

        return {
            'intrinsics': intrinsics.astype(np.float32), 
            'extrinsics': extrinsics.astype(np.float32)
        }