import os
import utils3d
import numpy as np
import multiprocessing
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='预处理后数据的根目录 (包含 000000 等序列)')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def read_synthia_depth(depth_path):
    depth_img = np.array(Image.open(depth_path).convert('RGB')).astype(np.float32)

    R = depth_img[:, :, 0]
    G = depth_img[:, :, 1]
    B = depth_img[:, :, 2]
    
    depth = 5000 * (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    
    sky_mask = (depth > 4999)
    depth[sky_mask] = 0.0
    
    valid_mask = (depth > 0)
    return depth.astype(np.float32), valid_mask

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def process_sequence(seq_path):
    seq_dir = Path(seq_path)
    
    depth_dir = seq_dir / 'depth'
    cam_file = seq_dir / 'cam.npz'
    normal_dir = seq_dir / 'normal'

    normal_dir.mkdir(parents=True, exist_ok=True)

    cam_data = np.load(cam_file)
    intrinsic_raw = cam_data['intrinsics'].astype(np.float32)

    depth_files = sorted(list(depth_dir.glob("*.png")))
    
    for depth_path in depth_files:
        base_id = depth_path.stem
        save_npy_path = normal_dir / f"{base_id}_normal.npy"
        
        if save_npy_path.exists():
            continue
            
        depth, valid_mask = read_synthia_depth(depth_path)
            
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

    sequences = sorted([str(d) for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Using {args.num_workers} workers...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_sequence, sequences), 
                      total=len(sequences), 
                      desc="Generating SYNTHIA Normals"):
            pass
    
    print("SYNTHIA normal maps generated successfully!")

if __name__ == "__main__":
    main()
