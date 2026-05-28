import os
import cv2
import pickle
import multiprocessing
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import utils3d

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./lightwheelocc')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def process_normal(task):
    depth_path, img_path, intrinsic_raw, save_npy_path = task
    
    # Skip if already generated
    if os.path.exists(save_npy_path):
        return True

    os.makedirs(os.path.dirname(save_npy_path), exist_ok=True)
    
    try:
        # Read original image size for accurate intrinsic normalization
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size

        # Decode LightwheelOCC specific depth format using cv2
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            return f"Failed to read depth image: {depth_path}"
            
        depth_img = depth_img.astype(np.float32)
        depth = depth_img[:, :, 0] + (depth_img[:, :, 1] * 256.0)
        depth = depth * 0.01
        depth = cv2.bilateralFilter(depth, d=5, sigmaColor=0.1, sigmaSpace=5.0)
        
        valid_mask = (depth > 0)
        
        intrinsic = normalize_intrinsic(intrinsic_raw, orig_h, orig_w)
        normal, normal_mask = utils3d.np.depth_map_to_normal_map(
            depth, 
            intrinsic, 
            mask=valid_mask
        )

        np.save(save_npy_path, {
            'normal': normal.astype(np.float32),
            'normal_mask': normal_mask.astype(np.bool_)
        })
        return True
        
    except Exception as e:
        return f"Error processing {depth_path}: {str(e)}"

def main():
    args = parse_args()
    data_path = args.data_path

    tasks = []
    
    print("Parsing lightwheelocc infos to build task list...")
    for split in ['train', 'val']:
        pkl_path = os.path.join(data_path, f'lightwheel_occ_infos_{split}.pkl')
        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping {split} split...")
            continue
            
        with open(pkl_path, 'rb') as f:
            data_infos = pickle.load(f)['infos']
        
        for timestamp_data in data_infos:
            for cam_view in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                if cam_view not in timestamp_data['cams']:
                    continue
                    
                cam_info = timestamp_data['cams'][cam_view]
                
                depth_path = os.path.join(data_path, cam_info['depth_path'])
                img_path = os.path.join(data_path, cam_info['cam_path'])
                cam_intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
                
                rel_depth_path = cam_info['depth_path']
                
                # Unified directory replacement to 'normal'
                rel_normal_path = rel_depth_path.replace('samples', 'normal').replace('depths', 'normal').replace('depth', 'normal')
                
                # Use original filename with .npy extension (no _normal suffix)
                base_name = os.path.splitext(rel_normal_path)[0]
                save_npy_path = os.path.join(data_path, f"{base_name}.npy")
                
                tasks.append((depth_path, img_path, cam_intrinsic, save_npy_path))
                
    print(f"Total frames to process: {len(tasks)}")
    print(f"Using {args.num_workers} CPU workers...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_normal, tasks), 
                           total=len(tasks), 
                           desc="Generating Surface Normals"):
            if result is not True:
                print(result) 

    print("Surface normals generation successfully finished!")

if __name__ == "__main__":
    main()
