import os
import utils3d
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Generate normal maps for processed MVS Synth dataset")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def process_sequence(seq_path):
    seq_dir = Path(seq_path)
    depth_dir = seq_dir / 'depth'
    cam_dir = seq_dir / 'cam'
    normal_dir = seq_dir / 'normal'

    if not depth_dir.exists() or not cam_dir.exists():
        return f"Skip {seq_dir.name}: missing depth or cam directory."

    normal_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(list(depth_dir.glob("*.npy")))
    

    for depth_path in depth_files:
        basename = depth_path.stem
        cam_path = cam_dir / f"{basename}.npz"
        save_path = normal_dir / f"{basename}.npy"
        
        if save_path.exists():
            continue
            
        if not cam_path.exists():
            continue
            
        depth = np.load(depth_path).astype(np.float32)
        depth = depth * 0.1
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

    return f"Finished sequence {seq_dir.name}"


def main():
    args = parse_args()
    data_root = Path(args.data_path)

    sequences = sorted([str(d) for d in data_root.iterdir() if d.is_dir()])

    if not sequences:
        print("No valid sequence directories found.")
        exit()

    print(f"Total sequences to process: {len(sequences)}")
    print(f"Using {args.num_workers} processes...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_sequence, sequences), 
                      total=len(sequences), 
                      desc="Generating MVS Synth Normals"):
            pass

    print("\nAll normal maps generated successfully!")

if __name__ == "__main__":
    main()
