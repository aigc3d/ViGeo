import os
import cv2
import yaml
import numpy as np
import trimesh
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process CarlaOcc Camera Poses")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    calib_file = os.path.join(args.data_root, 'calib', 'calib.yaml')
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    with open(calib_file, 'r') as f:
        calib_data = yaml.safe_load(f)

    swap_matrix = np.array([
        [ 0.0,  0.0,  1.0,  0.0 ],
        [-1.0,  0.0,  0.0,  0.0 ],
        [ 0.0, -1.0,  0.0,  0.0 ],
        [ 0.0,  0.0,  0.0,  1.0 ]
    ])

    sensor_mapping = {
        'cam_00': 'image_00',
        'cam_01': 'image_01',
        'cam_02': 'image_02',
        'cam_03': 'image_03',
        'cam_04': 'image_04',
        'cam_05': 'image_05',
        'cam_bev': 'image_bev'
    }

    default_intrinsic = np.array(calib_data['cam_settings']['intrinsics'])

    cam_configs = {}
    for yaml_key, out_name in sensor_mapping.items():
        if yaml_key in calib_data['sensors']:
            c2l_matrix = np.array(calib_data['sensors'][yaml_key]['transform'])
            
            if 'intrinsics' in calib_data['sensors'][yaml_key]:
                cam_intrinsic = np.array(calib_data['sensors'][yaml_key]['intrinsics'])
            else:
                cam_intrinsic = default_intrinsic
                
            cam_configs[out_name] = {
                'c2l': c2l_matrix,
                'intrinsic': cam_intrinsic
            }

    scenes = sorted([d for d in os.listdir(args.data_root) 
                     if d.startswith('Town') and os.path.isdir(os.path.join(args.data_root, d))])

    for scene in tqdm(scenes, desc="Processing Scenes"):
        pose_dir = os.path.join(args.data_root, scene, 'poses')
        lidar_pose_file = os.path.join(pose_dir, 'lidar.txt')

        if not os.path.exists(lidar_pose_file):
            continue

        lidar_poses_raw = np.loadtxt(lidar_pose_file)
        
        if lidar_poses_raw.size == 0:
            continue
            
        if lidar_poses_raw.ndim == 1:
            lidar_poses_matrix = lidar_poses_raw[1:].reshape(1, 4, 4)
        else:
            lidar_poses_matrix = lidar_poses_raw[:, 1:].reshape(-1, 4, 4)

        for cam_name, config in cam_configs.items():
            c2l = config['c2l']
            intrinsic = config['intrinsic']
            
            camera_poses_world = lidar_poses_matrix @ c2l @ swap_matrix
            
            out_file_path = os.path.join(pose_dir, f"{cam_name}.npz")
            
            np.savez_compressed(
                out_file_path, 
                poses=camera_poses_world, 
                intrinsics=intrinsic
            )

if __name__ == "__main__":
    main()
