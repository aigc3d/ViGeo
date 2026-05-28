import os
import cv2
import json
import utils3d
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial

def parse_args():
    parser = ArgumentParser(description="Parallel Depth to Normal Map Converter")
    parser.add_argument('--data_path', type=str, default='./dynamic_replica')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def read_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # Ensure depth is reshaped to the expected 512x512 resolution
    depth = depth.reshape(512, 512)
    depth = depth * 0.001
    return depth.astype(np.float32) 

def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        data = json.load(f)
    intrinsic = np.array(data['intrinsics']).reshape(3, 3)
    pose = data['camera_poses']
    return intrinsic.astype(np.float32)

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] = intrinsic_norm[0, :] / width
    intrinsic_norm[1, :] = intrinsic_norm[1, :] / height
    return intrinsic_norm

def process_scene(scene_path):
    scene_path = Path(scene_path)
    rgb_dir = scene_path / 'rgb'
    depth_dir = scene_path / 'depth'
    normal_dir = scene_path / 'normal'
    calib_file = scene_path / 'meta_info.json'
    if not calib_file.exists():
        return

    normal_dir.mkdir(parents=True, exist_ok=True)

    intrinsic_raw = read_calib(calib_file)

    rgb_files = sorted([f.name for f in rgb_dir.glob('*.jpg')])
    
    for file_name in rgb_files:
        depth_name = file_name.replace('rgb_', 'depth_').replace('.jpg', '.png')
        save_name = file_name.replace('rgb_', 'normal_').replace('.jpg', '.npy')
        
        depth_path = depth_dir / depth_name
        save_path = normal_dir / save_name

        if save_path.exists():
            continue
        depth = read_depth(depth_path)

        h, w = depth.shape
        intrinsic = normalize_intrinsic(intrinsic_raw, h, w)

        normal, normal_mask = utils3d.np.depth_map_to_normal_map(
            depth, 
            intrinsic, 
            mask=(depth > 0)
        )

        np.save(save_path, {
            'normal': normal.astype(np.float32),
            'normal_mask': normal_mask.astype(np.bool_)
        })

def main():
    args = parse_args()
    data_root = Path(args.data_path)

    all_scenes = sorted([str(d) for d in data_root.iterdir() if d.is_dir()])
    
    if not all_scenes:
        print(f"No scenes found in {args.data_path}")
        exit()

    print(f"Total scenes: {len(all_scenes)}")
    print(f"Using {args.num_workers} workers for parallel processing...")

    worker_func = partial(process_scene, data_root=data_root)
    
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_scene, all_scenes), 
                     total=len(all_scenes), 
                     desc="Processing Scenes"):
            pass

    print("\nAll normal maps generated successfully!")

if __name__ == "__main__":
    main()
