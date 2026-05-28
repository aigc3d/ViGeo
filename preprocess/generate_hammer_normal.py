import os
import glob
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from normal_utils import build_normal_estimator, normal_from_depth, visualize_normal

def read_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None: 
        return None
    return depth.astype(np.float32) / 1000.0

def process_scene(base_dir, args, normal_estimator, device):
    intrinsic_path = os.path.join(base_dir, "intrinsics.txt")
    if not os.path.exists(intrinsic_path):
        print(f"Skipping {base_dir}, intrinsics.txt not found.")
        return

    intrinsic = np.loadtxt(intrinsic_path)
    rgb_dir = os.path.join(base_dir, "rgb")
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.*"))) 

    if not rgb_paths:
        return

    output_vis_dir = os.path.join(base_dir, "normal")
    os.makedirs(output_vis_dir, exist_ok=True)
    
    progress_bar = tqdm(rgb_paths, desc=f"Processing {os.path.basename(os.path.dirname(base_dir))}")
    for rgb_p in progress_bar:
        frame_name = os.path.splitext(os.path.basename(rgb_p))[0]
        
        depth_p = os.path.join(base_dir, "_gt", f"{frame_name}.png")
        pose_p = os.path.join(base_dir, "_pose", f"{frame_name}.txt")
        
        if not (os.path.exists(depth_p) and os.path.exists(pose_p)): 
            continue 
            
        depth = read_depth(depth_p)
        if depth is None: 
            continue

        nml_cam = normal_from_depth(depth, intrinsic, normal_estimator, device)
        normal_vis = visualize_normal(nml_cam)
        normal_save_path = os.path.join(output_vis_dir, f"{frame_name}_normal.png")
        cv2.imwrite(normal_save_path, normal_vis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate normal maps from depth maps for multiple scenes.")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory containing scene folders (e.g., .../benchmark_datasets/hammer)")
    parser.add_argument('--max_depth', type=float, default=10.0, help="Maximum valid depth")
    parser.add_argument('--min_depth', type=float, default=0.1, help="Minimum valid depth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for Normal Estimation")
    
    normal_estimator = build_normal_estimator(args.min_depth, args.max_depth, device)

    # Find all scene directories inside the root directory
    scene_dirs = sorted(glob.glob(os.path.join(args.root_dir, "scene*")))
    
    if not scene_dirs:
        print(f"No scene directories found in {args.root_dir}")
    else:
        for scene_dir in scene_dirs:
            # According to your screenshot, the actual data is inside 'polarization'
            polarization_dir = os.path.join(scene_dir, "polarization")
            if os.path.isdir(polarization_dir):
                process_scene(polarization_dir, args, normal_estimator, device)
            else:
                # Fallback just in case some scenes don't have the 'polarization' subfolder
                process_scene(scene_dir, args, normal_estimator, device)

    print("All scenes processed successfully!")
