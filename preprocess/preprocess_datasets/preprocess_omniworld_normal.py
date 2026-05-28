import os
import json
import utils3d
import imageio
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='omniworld')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def read_omniworld_depth(depth_path):
    depth = imageio.v2.imread(str(depth_path)) / 65535.0
    
    near_mask = depth < 0.0015
    sky_mask = depth > (65500.0 / 65535.0)
    
    valid_sky = ~sky_mask
    if np.any(valid_sky):
        far_mask = depth > np.percentile(depth[valid_sky], 95)
    else:
        far_mask = np.zeros_like(depth, dtype=bool)

    near, far = 1.0, 1000.0
    depth = depth / (far - depth * (far - near)) / 0.004

    valid = ~(near_mask | far_mask | sky_mask)
    depth[~valid] = 0.0

    return depth.astype(np.float32), valid

def process_split(task_info):
    data_path, scene_name, split_idx, idxs = task_info
    
    scene_dir = Path(data_path) / 'annotations' / scene_name
    calib_file = scene_dir / 'camera' / f'split_{split_idx}.json'
    depth_dir = scene_dir / 'depth'
    normal_dir = scene_dir / 'normal'

    normal_dir.mkdir(parents=True, exist_ok=True)

    with open(calib_file, 'r') as f:
        cam = json.load(f)
        
    intrinsic_raw = np.eye(3, dtype=np.float32)
    f_val = np.mean(cam["focals"])
    intrinsic_raw[0, 0] = f_val
    intrinsic_raw[1, 1] = f_val
    intrinsic_raw[0, 2] = cam["cx"]
    intrinsic_raw[1, 2] = cam["cy"]

    for idx in idxs:
        basename = f"{idx:06d}"
        depth_path = depth_dir / f"{basename}.png"
        save_npy_path = normal_dir / f"{basename}.npy"
        
        if save_npy_path.exists():
            continue

        depth, valid_mask = read_omniworld_depth(depth_path)
        h, w = depth.shape
            
        intrinsic = normalize_intrinsic(intrinsic_raw, h, w)

        normal, normal_mask = utils3d.np.depth_map_to_normal_map(
            depth, 
            intrinsic, 
            mask=valid_mask
        )

        np.save(save_npy_path, {
            'normal': normal.astype(np.float32),
            'normal_mask': normal_mask.astype(np.bool_)
        })

def main():
    args = parse_args()
    data_root = Path(args.data_path)

    anno_dir = data_root / 'annotations'

    tasks = []
    scenes = sorted([d.name for d in anno_dir.iterdir() if d.is_dir()])
    
    for scene in scenes:
        scene_dir = anno_dir / scene
        split_info_file = scene_dir / "split_info.json"
            
        with open(split_info_file, "r", encoding="utf-8") as f:
            split_info = json.load(f)
            
        split_num = split_info["split_num"]
        for split_idx in range(split_num):
            idxs = split_info["split"][split_idx]
            tasks.append((str(data_root), scene, split_idx, idxs))

    print(f"Total split tasks: {len(tasks)}")
    print(f"Using {args.num_workers} workers...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_split, tasks), 
                      total=len(tasks), 
                      desc="Generating OmniWorld Normals"):
            pass
            
    print("OmniWorld normal maps generated successfully!")

if __name__ == "__main__":
    main()
