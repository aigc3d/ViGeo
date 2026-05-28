import os
import numpy as np
from mmengine import DATASETS
from .base_data import base_data
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


def xyzqxqyqxqw_to_c2w(pose):
    x, y, z, qx, qy, qz, qw = pose
    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = [x, y, z]
    w2c = np.linalg.inv(c2w)
    w2c = w2c[[1, 2, 0, 3]]
    c2w = np.linalg.inv(w2c)
    return c2w

@DATASETS.register_module()
class tartanair_data(base_data):
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
        pose_path = scene['pose']
        poses = self.load_pose(pose_path)

        rng = np.random.default_rng(seed)
        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            depth = self.load_depth(depth_paths[i])
            normal = self.load_normal(normal_paths[i])
            sky_mask = (depth > 1000).astype(np.float32)
            depth = self.filter_depth(depth)
            pose = np.array(xyzqxqyqxqw_to_c2w(poses[i].copy()), dtype=np.float32)
            intrinsic = self.normalize_intrinsic(self.load_intrinsic(), image.shape[0], image.shape[1])
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
    
    def load_pose(self, pose_path):
        poses = np.loadtxt(os.path.join(self.data_path, pose_path))
        return poses

    def load_depth(self, depth_path):
        depth = np.load(os.path.join(self.data_path, depth_path))
        return depth.astype(np.float32) # [1, 720, 1440] 

    @staticmethod
    def load_intrinsic():
        fx = 320.0  # focal length x
        fy = 320.0  # focal length y
        cx = 320.0  # optical center x
        cy = 240.0  # optical center y

        intrinsic = np.zeros((3, 3))
        intrinsic[0, 0] = fx
        intrinsic[0, 2] = cx
        intrinsic[1, 1] = fy
        intrinsic[1, 2] = cy
        intrinsic[2, 2] = 1
        return intrinsic.astype(np.float32)