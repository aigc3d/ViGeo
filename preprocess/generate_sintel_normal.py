import glob
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset_io import read_depth_sintel, read_sintel_calib
from normal_utils import build_normal_estimator, normal_from_depth, visualize_normal

# ================= Main Processing Loop =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--max_depth', type=float, default=100.0)
    parser.add_argument('--min_depth', type=float, default=0.001)
    args = parser.parse_args()

    final_dir = os.path.join(args.data_root, 'final')
    if not os.path.exists(final_dir):
        raise FileNotFoundError(f"Cannot find 'final' directory in {args.data_root}")

    scenes = sorted(os.listdir(final_dir))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for Normal Estimation")
    

    for scene in scenes:
        if scene != 'market_6':
            continue
        if scene in ['mountain_1']:
            max_depth = 600
        elif scene in ['ambush_4']:
            max_depth = 200
        elif scene in ['market_6']:
            max_depth = 30
        else:
            max_depth = args.max_depth
        normal_estimator = build_normal_estimator(args.min_depth, max_depth, device)
        
        
        scene_dir = os.path.join(final_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
            
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, '*.png')))
        if not rgb_files:
            continue

        normal_out_dir = os.path.join(args.data_root, 'normal', scene)
        os.makedirs(normal_out_dir, exist_ok=True)
        
        progress_bar = tqdm(rgb_files, desc=f"Processing {scene}")
        
        for rgb_p in progress_bar:
            frame_name = os.path.basename(rgb_p)
            
            depth_p = os.path.join(args.data_root, 'depth', scene, frame_name.replace('.png', '.dpt'))
            calib_p = os.path.join(args.data_root, 'camdata_left', scene, frame_name.replace('.png', '.cam'))
            
            if not (os.path.exists(depth_p) and os.path.exists(calib_p)):
                continue
                
            depth = read_depth_sintel(depth_p)
            intrinsic = read_sintel_calib(calib_p)

            nml_cam = normal_from_depth(depth, intrinsic, normal_estimator, device)
            normal_vis = visualize_normal(nml_cam)
            normal_save_path = os.path.join(normal_out_dir, frame_name)
            cv2.imwrite(normal_save_path, normal_vis)

    print("All Sintel normals calculated and saved successfully.")
