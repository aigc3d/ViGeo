import os
import utils3d
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='预处理后的 Spring 数据集根目录')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def process_scene(scene_path):
    scene_dir = Path(scene_path)
    
    depth_dir = scene_dir / 'depth'
    cam_dir = scene_dir / 'cam'
    normal_dir = scene_dir / 'normal'

    if not depth_dir.exists() or not cam_dir.exists():
        return f"Skip {scene_dir.name}: missing depth or cam dir"

    normal_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(list(depth_dir.glob("*.npy")))
    
    for depth_path in depth_files:
        base_id = depth_path.stem
        cam_path = cam_dir / f"{base_id}.npz"
        save_npy_path = normal_dir / f"{base_id}_normal.npy"
        
        if save_npy_path.exists():
            continue

        depth = np.load(depth_path).astype(np.float32)
        valid_mask = (depth > 0)
                
        h, w = depth.shape
        
        cam_data = np.load(cam_path)
        intrinsic_raw = cam_data['intrinsics'].astype(np.float32)
            
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

    if not data_root.exists():
        print(f"Error: 找不到目录 {data_root}")
        exit()

    scenes = sorted([str(d) for d in data_root.iterdir() if d.is_dir()])
    
    print(f"Total scenes: {len(scenes)}")
    print(f"Using {args.num_workers} workers...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_scene, scenes), 
                      total=len(scenes), 
                      desc="Generating Spring Normals"):
            pass

    print("Spring normal maps generated successfully!")

if __name__ == "__main__":
    main()
