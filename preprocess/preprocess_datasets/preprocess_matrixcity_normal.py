import os
import cv2
import utils3d
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def parse_args():
    parser = ArgumentParser(description="Generate normal maps for processed MatrixCity dataset")
    parser.add_argument('--data_path', type=str, required=True, help='预处理后数据的根目录 (包含 00000, 00001 等文件夹)')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def read_depth(depth_path):
    depth = cv2.imread(
            depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    sky_mask = (depth==65504).astype(np.float32)
    edge_mask = utils3d.depth_map_edge(depth, rtol=0.4)
    depth = depth / 100.0
    depth[edge_mask > 0] = 0
    depth[sky_mask > 0] = 0
    return depth.astype(np.float32)

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def process_sequence(seq_path):
    seq_dir = Path(seq_path)
    pose_file = seq_dir / 'pose.npz'
    
    if not pose_file.exists():
        return f"Skip {seq_dir.name}: pose.npz not found"

    pose_data = np.load(pose_file)
    camera_angle_x = pose_data['camera_angle_x'].item()
        
    w = 1000.0
    f = float(.5 * w / np.tan(.5 * camera_angle_x))

    intrinsic_raw = np.zeros((3, 3), dtype=np.float32)
    intrinsic_raw[0, 0] = f
    intrinsic_raw[1, 1] = f
    intrinsic_raw[0, 2] = 500.0
    intrinsic_raw[1, 2] = 500.0
    intrinsic_raw[2, 2] = 1.0

    intrinsic = normalize_intrinsic(intrinsic_raw, height=1000, width=1000)
    depth_files = sorted(list(seq_dir.glob("*.exr")))
    
    for depth_path in depth_files:
        frame_id = depth_path.stem 
        save_path = seq_dir / f"{frame_id}_normal.npy"
        
        if save_path.exists():
            continue
            
        depth = read_depth(depth_path)    
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

    if not data_root.exists():
        print(f"Error: Directory {data_root} does not exist.")
        exit()

    sequences = sorted([str(d) for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])

    if not sequences:
        print("No valid sequence directories found.")
        exit()

    print(f"Total sequences to process: {len(sequences)}")
    print(f"Using {args.num_workers} processes...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_sequence, sequences), 
                      total=len(sequences), 
                      desc="Generating MatrixCity Normals"):
            pass

    print("\nAll normal maps generated successfully!")

if __name__ == "__main__":
    main()
