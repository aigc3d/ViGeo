import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
# from pyquaternion import Quaternion # Commented out as pose generation is skipped
from argparse import ArgumentParser

def split_videos(data):
    token_to_node = {item['token']: item for item in data}
    start_tokens = [item['token'] for item in data if item['prev'] is None]
    
    videos = []
    for token in start_tokens:
        sequence = []
        current_token = token
        while current_token is not None:
            sequence.append(current_token)
            current_token = token_to_node[current_token]['next']
        videos.append(sequence)
    return videos, token_to_node

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./lightwheelocc')
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    data = []
    for split in ['train', 'val']:
        pkl_file = os.path.join(data_path, f'lightwheel_occ_infos_{split}.pkl')
        if not os.path.exists(pkl_file):
            continue

        with open(pkl_file, 'rb') as f:
            data_infos = pickle.load(f)
        
        data_infos = data_infos['infos']
        videos, token_to_node = split_videos(data_infos)
        
        cam_views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        for cam_view in cam_views:
            for video in tqdm(videos, desc=f"Updating {split} {cam_view}"):
                image_files = []
                depth_files = []
                normal_files = []
                calib_files = []

                for token in video:
                    timestamp_data = token_to_node[token]
                    if cam_view not in timestamp_data['cams']:
                        continue
                        
                    cam_info = timestamp_data['cams'][cam_view]
                    
                    # --- [Commented Out] Pose Calculation ---
                    """
                    ego2global_rot = Quaternion(timestamp_data['ego2global_rotation']).rotation_matrix
                    ego2global_trans = timestamp_data['ego2global_translation']
                    T_ego2global = np.eye(4)
                    T_ego2global[:3, :3] = ego2global_rot
                    T_ego2global[:3, 3] = ego2global_trans

                    sensor2ego_rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                    sensor2ego_trans = cam_info['sensor2ego_translation']
                    T_cam2ego = np.eye(4)
                    T_cam2ego[:3, :3] = sensor2ego_rot
                    T_cam2ego[:3, 3] = sensor2ego_trans
                    
                    T_cam2global = T_ego2global @ T_cam2ego
                    cam_intrinsic = np.array(cam_info['cam_intrinsic'])
                    """

                    # --- Path Mapping ---
                    cam_path = cam_info['cam_path']
                    depth_path = cam_info['depth_path']
                    
                    # Normal path mapping logic
                    normal_path = depth_path.replace('samples', 'normal').replace('depths', 'normal').replace('depth', 'normal')
                    normal_path = os.path.splitext(normal_path)[0] + ".npy"
                    
                    # Pose/Calib path mapping logic (targeting already existing .npz)
                    calib_path = cam_path.replace('samples', 'poses').replace('.jpeg', '.npz')
                    
                    # --- [Commented Out] Save NPZ ---
                    """
                    full_calib_path = os.path.join(data_path, calib_path)
                    os.makedirs(os.path.dirname(full_calib_path), exist_ok=True)
                    np.savez(
                        full_calib_path,
                        pose=T_cam2global.astype(np.float32),
                        intrinsics=cam_intrinsic.astype(np.float32),
                    )
                    """

                    image_files.append(cam_path)
                    depth_files.append(depth_path)
                    normal_files.append(normal_path)
                    calib_files.append(calib_path)
            
                if len(image_files) < 32:
                    continue

                data.append({
                    'scene': f"{split}_{cam_view}_{video[0][:8]}",
                    'image': image_files,
                    'depth': depth_files,
                    'normal': normal_files,
                    'calib': calib_files,
                })
    
    print(f"Update finished: {len(data)} sequences recorded in lightwheelocc_train.json")
    with open(os.path.join(output_path, 'lightwheelocc_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
