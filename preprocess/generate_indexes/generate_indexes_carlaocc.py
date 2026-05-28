import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Fast Flattened CarlaOcc JSON index")
    parser.add_argument('--data_path', type=str, default='CarlaOccV1')
    parser.add_argument('--output_path', type=str, default='./train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = []
    
    # Get all sequence directories
    all_dirs = os.listdir(data_path)
    sequences = sorted([d for d in all_dirs if d.startswith('Town') and os.path.isdir(os.path.join(data_path, d))])
    
    cameras = ['image_00', 'image_01', 'image_02', 'image_03', 'image_04', 'image_05']
    
    for seq in tqdm(sequences, desc="Processing Sequences"):
        seq_dir = os.path.join(data_path, seq)
        
        for cam in cameras:
            # Full paths for directory checking
            cam_rgb_dir = os.path.join(seq_dir, 'rgb', cam)
            cam_depth_dir = os.path.join(seq_dir, 'depth', cam)
            cam_normal_dir = os.path.join(seq_dir, 'normal', cam)
            cam_sem_dir = os.path.join(seq_dir, 'semantics', cam)
            pose_path_full = os.path.join(seq_dir, 'poses', f'{cam}.npz')

            # 1. Directory & Pose Level Check: Ensure all required folders and the pose file exist
            if not all(os.path.exists(p) for p in [cam_rgb_dir, cam_depth_dir, cam_normal_dir, cam_sem_dir, pose_path_full]):
                continue

            try:
                calib_data = np.load(pose_path_full)
                num_poses = calib_data['poses'].shape[0]
            except Exception as e:
                print(f"[Error] Failed to load pose {pose_path_full}: {e}")
                continue
            # -----------------------------------------------

            # 2. O(1) Set Lookup: Load directory contents into memory sets for ultra-fast checking
            rgb_set = set(os.listdir(cam_rgb_dir))
            depth_set = set(os.listdir(cam_depth_dir))
            normal_set = set(os.listdir(cam_normal_dir))
            sem_set = set(os.listdir(cam_sem_dir))
            
            if not depth_set:
                continue

            image_files = []
            depth_files = []
            normal_files = []
            sem_files = []

            # Pre-define path templates to speed up string joining
            rgb_base = os.path.join(seq, 'rgb', cam)
            depth_base = os.path.join(seq, 'depth', cam)
            normal_base = os.path.join(seq, 'normal', cam)
            sem_base = os.path.join(seq, 'semantics', cam)

            # Sort the depth list to maintain chronological order
            all_depths = sorted(list(depth_set))

            for d_file in all_depths:
                if not d_file.endswith('.png'):
                    continue

                frame_id = d_file.replace('.png', '')
                
                # Assume standard naming convention (.png for all based on your original logic)
                img_file = f"{frame_id}.png"
                norm_file = f"{frame_id}.png"
                sem_file = f"{frame_id}.png"
                
                # 3. Frame Level Check: Only add to list if ALL modalities exist for this specific frame
                if img_file in rgb_set and norm_file in normal_set and sem_file in sem_set:
                    image_files.append(os.path.join(rgb_base, img_file))
                    depth_files.append(os.path.join(depth_base, d_file))
                    normal_files.append(os.path.join(normal_base, norm_file))
                    sem_files.append(os.path.join(sem_base, sem_file))

            # 4. Final Sanity Check: Ensure we actually collected valid frames
            if not image_files:
                continue

            if len(image_files) != num_poses:
                print(f"[Warning] Skipped {seq}/{cam}: Frames ({len(image_files)}) do not match Poses ({num_poses})")
                continue
            # ---------------------------------------

            # Construct relative pose path for JSON
            pose_path = os.path.join(seq, 'poses', f'{cam}.npz')
            
            data.append({
                "scene": f"{seq}_{cam}",
                "image": image_files,
                "depth": depth_files,
                "normal": normal_files,
                "semantics": sem_files,
                "pose": pose_path
            })

    print(f"Processing complete: {len(data)} total entries found.")
    
    out_file = os.path.join(output_path, 'carlaocc_train.json')
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"Fast JSON index saved to: {out_file}")

if __name__ == "__main__":
    main()
