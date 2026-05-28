import os
import re
import sys
import yaml
import json
import subprocess

import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from munch import Munch
from scipy.spatial.transform import Rotation

class ScannetppScene_Release:
    def __init__(
        self,
        scene_id,
        data_root=None,
        dslr_folder_name=None,
        iphone_folder_name=None,
        scan_folder_name=None,
    ):
        self._scene_id = scene_id
        self.data_root = self.path_or_none(data_root)

        if dslr_folder_name is None:
            dslr_folder_name = "dslr"
        if iphone_folder_name is None:
            iphone_folder_name = "iphone"
        if scan_folder_name is None:
            scan_folder_name = "scans"

        self.dslr_folder_name = dslr_folder_name
        self.iphone_folder_name = iphone_folder_name
        self.scan_folder_name = scan_folder_name

    @staticmethod
    def path_or_none(path):
        if path is not None:
            path = Path(path)
        return path

    @property
    def scene_id(self):
        return self._scene_id


    @property
    def scene_root_dir(self):
        return self.data_root / self._scene_id

    @property
    def scans_dir(self):
        '''
        dir containing all scan-related data
        '''
        return self.data_root / self._scene_id / self.scan_folder_name

    @property
    def pc_dir(self):
        '''
        dir containing 1mm point cloud
        '''
        return self.scans_dir

    @property
    def scan_pc_path(self):
        '''
        path to point cloud
        '''
        return self.pc_dir / 'pc_aligned.ply'

    @property
    def scan_pc_mask_path(self):
        '''
        path to the point cloud mask
        '''
        return self.pc_dir / 'pc_aligned_mask.txt'

    @property
    def scan_transformed_poses_path(self):
        '''
        path containing all scanner poses transformed to aligned coordinates
        in a single file
        '''
        return self.pc_dir / 'scanner_poses.json'

    @property
    def mesh_dir(self):
        '''
        dir containing all the meshes and related data
        put meshes in the same dir as 1mm PCs
        '''
        return self.scans_dir

    @property
    def scan_mesh_path(self):
        '''
        path to the mesh
        '''
        return self.mesh_dir / 'mesh_aligned_0.05.ply'

    @property
    def scan_mesh_mask_path(self):
        '''
        path to the mesh mask
        '''
        return self.mesh_dir / 'mesh_aligned_0.05_mask.txt'

    @property
    def scan_mesh_segs_path(self):
        return self.mesh_dir / f'segments.json'

    @property
    def scan_anno_json_path(self):
        return self.mesh_dir / f'segments_anno.json'

    @property
    def scan_sem_mesh_path(self):
        return self.mesh_dir / f'mesh_aligned_0.05_semantic.ply'

    ##########################################################
    # Panocam assets
    ##########################################################
    @property
    def pano_dir(self):
        return self.data_root / self._scene_id / 'panocam'

    @property
    def pano_rgb_dir(self):
        return self.pano_dir / 'images'

    @property
    def pano_anon_mask_dir(self):
        return self.pano_dir / 'anon_mask'

    @property
    def pano_depth_dir(self):
        return self.pano_dir / 'depth'

    @property
    def pano_azim_dir(self):
        return self.pano_dir / 'azim'

    @property
    def pano_elev_dir(self):
        return self.pano_dir / 'elev'

    @property
    def pano_resized_rgb_dir(self):
        return self.pano_dir / 'resized_images'

    @property
    def pano_resized_depth_dir(self):
        return self.pano_dir / 'resized_depth'

    @property
    def pano_resized_mask_dir(self):
        return self.pano_dir / 'resized_anon_mask'

    @property
    def pano_resized_azim_dir(self):
        return self.pano_dir / 'resized_azim'

    @property
    def pano_resized_elev_dir(self):
        return self.pano_dir / 'resized_elev'
    ##########################################################
    # DSLR assets
    ##########################################################
    @property
    def dslr_dir(self):
        return self.data_root / self._scene_id / self.dslr_folder_name

    @property
    def dslr_resized_dir(self):
        return self.dslr_dir / 'resized_images'

    @property
    def dslr_resized_mask_dir(self):
        return self.dslr_dir / 'resized_anon_masks'

    @property
    def dslr_original_dir(self):
        return self.dslr_dir / 'original_images'

    @property
    def dslr_original_mask_dir(self):
        return self.dslr_dir / 'original_anon_masks'

    @property
    def dslr_resized_undistorted_dir(self):
        return self.dslr_dir / 'resized_undistorted_images'

    @property
    def dslr_resized_undistorted_mask_dir(self):
        return self.dslr_dir / 'resized_undistorted_masks'

    @property
    def dslr_colmap_dir(self):
        return self.dslr_dir / 'colmap'

    @property
    def dslr_nerfstudio_transform_path(self):
        return self.dslr_dir / 'nerfstudio' / 'transforms.json'

    @property
    def dslr_nerfstudio_transform_undistorted_path(self):
        return self.dslr_dir / 'nerfstudio' / 'transforms_undistorted.json'

    @property
    def dslr_train_test_lists_path(self):
        return self.dslr_dir / 'train_test_lists.json'

    ##########################################################
    # iPhone assets
    ##########################################################
    @property
    def iphone_data_dir(self):
        return self.data_root / self._scene_id / self.iphone_folder_name

    @property
    def iphone_video_path(self):
        return self.iphone_data_dir / 'rgb.mkv'

    @property
    def iphone_rgb_dir(self):
        return self.iphone_data_dir / 'rgb'

    @property
    def iphone_video_mask_path(self):
        return self.iphone_data_dir / 'rgb_mask.mkv'

    @property
    def iphone_video_mask_dir(self):
        return self.iphone_data_dir / 'rgb_masks'

    @property
    def iphone_depth_path(self):
        return self.iphone_data_dir / 'depth.bin'

    @property
    def iphone_depth_dir(self):
        return self.iphone_data_dir / 'depth'

    @property
    def iphone_pose_intrinsic_imu_path(self):
        return self.iphone_data_dir / 'pose_intrinsic_imu.json'

    @property
    def iphone_colmap_dir(self):
        return self.iphone_data_dir / 'colmap'

    @property
    def iphone_nerfstudio_transform_path(self):
        return self.iphone_data_dir / 'nerfstudio' / 'transforms.json'

    @property
    def iphone_exif_path(self):
        return self.iphone_data_dir / 'exif.json'

def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)

def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j

def run_command(cmd: str, verbose=False, exit_on_error=True):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        if out.stderr is not None:
            print(out.stderr.decode("utf-8"))
        if exit_on_error:
            sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out

def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world