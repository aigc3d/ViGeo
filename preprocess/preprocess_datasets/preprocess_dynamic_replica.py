# References:
# https://github.com/CUT3R/CUT3R/blob/main/datasets_preprocess/preprocess_dynamic_replica.py

import os
import cv2
import h5py
import gzip
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from argparse import ArgumentParser

from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass,
)

def _get_pytorch3d_camera(entry_viewpoint, image_size, scale: float):
    """
    Convert the camera parameters stored in an annotation to PyTorch3D convention.

    Returns:
        R, tvec, focal, principal_point
    """
    assert entry_viewpoint is not None
    principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
    focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )

    fmt = entry_viewpoint.intrinsics_format
    if fmt.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig
    elif fmt.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {fmt}")

    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    # Prepare rotation and translation for PyTorch3D
    R = torch.tensor(entry_viewpoint.R, dtype=torch.float)
    T = torch.tensor(entry_viewpoint.T, dtype=torch.float)
    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    T_pytorch3d[..., :2] *= -1
    R_pytorch3d[..., :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.T

    return R, tvec, focal_length_px, principal_point_px


@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from .json for Dynamic Replica."""

    camera_name: Optional[str] = None
    instance_id_map_path: Optional[str] = None
    flow_forward: Optional[str] = None
    flow_forward_mask: Optional[str] = None
    flow_backward: Optional[str] = None
    flow_backward_mask: Optional[str] = None
    trajectories: Optional[str] = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dynamic_replica')
    parser.add_argument('--output_path', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path

    for split in ["train", "test", "valid"]:
        frame_annotations_file = os.path.join(data_path, f"frame_annotations_{split}.jgz")
        with gzip.open(frame_annotations_file, "rt", encoding="utf8") as zipfile:
            frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])
        
        # Group frames by sequence and camera.
        seq_annot = defaultdict(lambda: defaultdict(list))
        for frame_annot in frame_annots_list:
            seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(
                frame_annot
            )
        
        for seq_name in tqdm(seq_annot.keys(), desc=f"Processing split '{split}'"):
            for cam in ["left", "right"]:
                tgt_dir = os.path.join(data_path, seq_name + '_source_' + cam, 'calib.npz')
                poses = []
                intrins = []
                for framedata in tqdm(
                seq_annot[seq_name][cam], desc=f"Seq {seq_name} [{cam}]", leave=False):
                    timestamp = framedata.frame_timestamp
                    viewpoint = framedata.viewpoint
                    R, t, focal, pp = _get_pytorch3d_camera(
                    viewpoint, framedata.image.size, scale=1.0
                    )
                    intrinsics = np.eye(3)
                    intrinsics[0, 0] = focal[0].item()
                    intrinsics[1, 1] = focal[1].item()
                    intrinsics[0, 2] = pp[0].item()
                    intrinsics[1, 2] = pp[1].item()
                    
                    pose = np.eye(4)
                    # Invert the camera pose.
                    pose[:3, :3] = R.numpy().T
                    pose[:3, 3] = -R.numpy().T @ t.numpy()
                    poses.append(pose)
                    intrins.append(intrinsics)
                poses = np.stack(poses)
                intrins = np.stack(intrins)

                np.savez(tgt_dir, intrinsics=intrins, pose=poses)

if __name__ == "__main__":
    main()
