import os
import cv2
import json
import numpy as np
import pandas as pd
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
class tartanground_data(base_data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_annotations(self, train_test_split):
        scenes = pd.read_parquet(train_test_split)
        return scenes
    
    def sample_video(self, scene_id: int, seq_len: int, seed: int):
        rng = np.random.default_rng(seed)
        scene = self.scenes.iloc[scene_id]

        image_paths = scene['image']
        depth_paths = scene['depth']
        seg_paths = scene['seg']
        seg_label_map = scene['seg_label_map']

        with open(os.path.join(self.data_path, seg_label_map), 'r') as f:
            json_data = json.load(f)
            sky_label = int(json_data['name_map']['sky']) if 'sky' in json_data['name_map'] else None

        pose_path = scene['calib']
        poses = self.load_pose(pose_path)

        seq_ids = self.sample_sequence(len(image_paths), seq_len, rng)

        views = defaultdict(list)
        for i in seq_ids:
            image = self.load_image(image_paths[i])
            pose = np.array(xyzqxqyqxqw_to_c2w(poses[i].copy()), dtype=np.float32)
            depth = self.load_depth(depth_paths[i])
            sky_mask = self.load_segmentation(seg_paths[i], sky_label)

            depth = self.filter_depth(depth)
            depth[sky_mask > 0] = 0

            intrinsic_matrix = self.load_intrinsic()

            views['image'].append(image)
            views['depth'].append(depth)
            views['normal'].append(None)
            views['sky_mask'].append(sky_mask)
            views['pose'].append(pose)
            views['intrinsic'].append(intrinsic_matrix)
            views['image_path'].append(image_paths[i])
            views['depth_path'].append(depth_paths[i])
            views['instance'].append(str(i))

        return views
    
    def load_pose(self, pose_path):
        full_path = os.path.join(self.data_path, pose_path)
        return np.loadtxt(full_path)

    def load_depth(self, depth_path):
        full_path = os.path.join(self.data_path, depth_path)
        depth_rgba = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1).astype(np.float32)
    
    def load_segmentation(self, seg_path, sky_label):
        full_path = os.path.join(self.data_path, seg_path)
        segmentation = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if sky_label is not None:
            sky_mask = segmentation == sky_label
        else:
            sky_mask = np.zeros_like(segmentation)
        return sky_mask.astype(np.float32)

    @staticmethod
    def load_intrinsic():
        intrinsic_matrix = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], dtype=np.float32)
        return intrinsic_matrix
