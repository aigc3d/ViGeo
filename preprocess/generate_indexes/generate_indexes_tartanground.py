import os
import json
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="TartanGround root directory")
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def collect_camera_data(data_path, rel_seq_path, camera_view):
    """
    rel_seq_path: e.g., "Sewerage/Data_diff/P1003"
    """
    if rel_seq_path == 'ModularNeighborhoodIntExt/Data_omni/P0005' and camera_view == "":
        print(rel_seq_path, camera_view)
        return None
    img_dir = f"image_{camera_view}"
    depth_dir = f"depth_{camera_view}"
    seg_dir = f"seg_{camera_view}"      # <-- Added seg directory
    pose_file = f"pose_{camera_view}.txt"

    # Full absolute paths for scanning
    base_dir = os.path.join(data_path, rel_seq_path)
    img_path = os.path.join(base_dir, img_dir)
    dep_path = os.path.join(base_dir, depth_dir)
    seg_path_full = os.path.join(base_dir, seg_dir)  # <-- Added seg path
    pose_path = os.path.join(base_dir, pose_file)

    # 1. Validate pose line count to prevent DataLoader out-of-bounds
    try:
        with open(pose_path, 'r') as f:
            num_poses = sum(1 for line in f if line.strip())
    except Exception:
        return None

    # 2. O(1) set lookup
    img_files_set = set(os.listdir(img_path)) if os.path.exists(img_path) else set()
    dep_files_set = set(os.listdir(dep_path)) if os.path.exists(dep_path) else set()
    seg_files_set = set(os.listdir(seg_path_full)) if os.path.exists(seg_path_full) else set() # <-- Lookup for seg

    valid_image_paths = []
    valid_depth_paths = []
    valid_seg_paths = []  # <-- Added list for seg

    sorted_imgs = sorted(list(img_files_set))

    for img_file in sorted_imgs:
        if not img_file.endswith(('.png', '.jpg')):
            continue
            
        stem = os.path.splitext(img_file)[0]
        depth_name = f"{stem}_depth.png"
        seg_name = f"{stem}_seg.png"    # <-- Seg file naming rule
        
        # Strictly require image, depth, and seg to be present
        if depth_name in dep_files_set and seg_name in seg_files_set:
            valid_image_paths.append(os.path.join(rel_seq_path, img_dir, img_file))
            valid_depth_paths.append(os.path.join(rel_seq_path, depth_dir, depth_name))
            valid_seg_paths.append(os.path.join(rel_seq_path, seg_dir, seg_name))

    # 3. Strict filtering
    if len(valid_image_paths) != num_poses or len(valid_image_paths) < 32:
        print(f"[Debug] Skipped {rel_seq_path}/{camera_view}: length mismatch or too short.")
        return None

    # Replace / with _ to generate a unique scene name
    scene_id = rel_seq_path.replace(os.sep, '_') + f"_{camera_view}"
    
    # Extract root scene name (e.g., "Sewerage") to locate the global json map
    scene_root = rel_seq_path.split(os.sep)[0]
    seg_json_path = os.path.join(scene_root, 'seg_label_map.json')

    return {
        'scene': scene_id,
        'image': valid_image_paths,
        'depth': valid_depth_paths,
        'seg': valid_seg_paths,         # <-- Added seg list to dataset
        'calib': os.path.join(rel_seq_path, pose_file),
        'seg_label_map': seg_json_path  # <-- Added mapping json path
    }

def main():
    args = parse_args()
    root_path = os.path.abspath(args.data_path)
    output_path = args.output_path

    # Directory scanning logic
    scenes = [s for s in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, s))]
    
    all_p_tasks = []
    for scene_name in scenes:
        scene_dir = os.path.join(root_path, scene_name)
        for dtype in ['Data_diff', 'Data_omni', 'Data_anymal']:
            dtype_path = os.path.join(scene_dir, dtype)
            if not os.path.isdir(dtype_path): 
                continue
            ps = [p for p in os.listdir(dtype_path) if p.startswith('P') and os.path.isdir(os.path.join(dtype_path, p))]
            for p_name in ps:
                rel_path = os.path.join(scene_name, dtype, p_name)
                all_p_tasks.append(rel_path)

    print(f"[*] Found {len(all_p_tasks)} valid P sequences, starting camera view scan...")

    cameras = [
        'lcam_front', 'lcam_back', 'lcam_left', 'lcam_right', 'lcam_bottom',
        'rcam_front', 'rcam_back', 'rcam_left', 'rcam_right', 'rcam_bottom'
    ]
    
    data = []
    total_num = 0
    for rel_seq_path in tqdm(all_p_tasks, desc="Progress (P folders)"):
        for cam in cameras:
            result = collect_camera_data(root_path, rel_seq_path, cam)
            if result:
                total_num += len(result['image'])
                data.append(result)
                
    print(f"[*] Total valid images collected: {total_num}")
    
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, 'tartanground_train.parquet')
    
    df = pd.DataFrame(data)
    df.to_parquet(out_file, engine='pyarrow', index=False)
        
    print(f"\n[+] Processing complete! Extracted {len(data)} valid view sequences. Saved to: {out_file}")

if __name__ == "__main__":
    main()
