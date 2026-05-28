import os
import utils3d
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Generate normal maps for processed Hypersim dataset")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the processed Hypersim directory')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Number of CPU processes')
    return parser.parse_args()

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] = intrinsic_norm[0, :] / width
    intrinsic_norm[1, :] = intrinsic_norm[1, :] / height
    return intrinsic_norm

def process_subscene(subscene_path):
    subscene_dir = Path(subscene_path)
    
    depth_files = sorted(list(subscene_dir.glob("*_depth.npy")))
    if not depth_files:
        return f"Skip {subscene_dir.name}: No depth files."

    for depth_path in depth_files:
        frame_id = depth_path.name.split('_')[0]
        
        cam_path = subscene_dir / f"{frame_id}_cam.npz"
        save_path = subscene_dir / f"{frame_id}_normal.npy"

        if save_path.exists():
            continue
            
        depth = np.load(depth_path)
        h, w = depth.shape
            
        cam_data = np.load(cam_path)
        intrinsic_raw = cam_data['intrinsics'].astype(np.float32)
            
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

    if not data_root.exists():
        print(f"Error: Directory {data_root} does not exist.")
        exit()

    subscenes = []
    for scene_dir in data_root.iterdir():
        if scene_dir.is_dir():
            for cam_dir in scene_dir.iterdir():
                if cam_dir.is_dir():
                    subscenes.append(str(cam_dir))

    if not subscenes:
        print("No valid subscenes found. Please check your data_path.")
        exit()

    print(f"Total camera trajectories to process: {len(subscenes)}")
    print(f"Using {args.num_workers} processes...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_subscene, subscenes), 
                      total=len(subscenes), 
                      desc="Generating Normals"):
            pass

    print("\nAll Hypersim normal maps generated successfully!")

if __name__ == "__main__":
    main()
