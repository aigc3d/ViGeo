import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./tartanair')
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def collect_camera_data(data_path, seq, camera_side):
    img_dir = f"image_{camera_side}"
    depth_dir = f"depth_{camera_side}"
    normal_dir = f"normal_{camera_side}"
    pose_file = f"pose_{camera_side}.txt"

    # Full paths for directory scanning
    img_path = os.path.join(data_path, seq, img_dir)
    dep_path = os.path.join(data_path, seq, depth_dir)
    norm_path = os.path.join(data_path, seq, normal_dir)

    # Check if directories exist once
    if not all(os.path.exists(p) for p in [img_path, dep_path, norm_path]):
        return None

    # Use sets for O(1) lookup instead of disk I/O
    img_files_set = set(os.listdir(img_path))
    dep_files_set = set(os.listdir(dep_path))
    norm_files_set = set(os.listdir(norm_path))

    valid_image_paths = []
    valid_depth_paths = []
    valid_normal_paths = []

    # Sort the image list to maintain sequence order
    sorted_imgs = sorted(list(img_files_set))

    for img_file in sorted_imgs:
        if not img_file.endswith('.png'):
            continue
            
        stem = img_file.replace(".png", "")
        depth_name = f"{stem}_depth.npy"
        normal_name = f"{stem}_normal.npy"
        
        # Memory-based lookup is significantly faster than os.path.exists
        if depth_name in dep_files_set and normal_name in norm_files_set:
            valid_image_paths.append(os.path.join(seq, img_dir, img_file))
            valid_depth_paths.append(os.path.join(seq, depth_dir, depth_name))
            valid_normal_paths.append(os.path.join(seq, normal_dir, normal_name))

    if not valid_image_paths:
        return None

    return {
        'image': valid_image_paths,
        'depth': valid_depth_paths,
        'normal': valid_normal_paths,
        'pose': os.path.join(seq, pose_file)
    }

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    sequences = []
    for name in sorted(os.listdir(data_path)):
        env_path = os.path.join(data_path, name)
        if not os.path.isdir(env_path):
            continue
        easy_path = os.path.join(env_path, 'Easy')
        if os.path.isdir(easy_path):
            for s in os.listdir(easy_path):
                if os.path.isdir(os.path.join(easy_path, s)):
                    sequences.append(os.path.join(name, 'Easy', s))
    
    data = []
    for seq in tqdm(sequences):
        for side in ['left', 'right']:
            result = collect_camera_data(data_path, seq, side)
            if result:
                data.append(result)
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'tartanair_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
