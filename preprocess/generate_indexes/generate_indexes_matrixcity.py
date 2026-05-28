import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/video')
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = []
    scenes = sorted(os.listdir(data_path))
    
    for scene in tqdm(scenes):
        scene_dir = os.path.join(data_path, scene)
        if not os.path.isdir(scene_dir):
            continue
            
        # List all files and use a set for O(1) lookup
        all_files = os.listdir(scene_dir)
        all_files_set = set(all_files)
        
        # Filter PNG files as the base frames
        png_files = sorted([f for f in all_files if f.endswith('.png')])
        
        image_files = []
        depth_files = []
        normal_files = []
        # Pose file is per scene
        calib_file = os.path.join(scene, 'pose.npz')

        for f in png_files:
            # ID is the stem of the filename, e.g., '0004'
            frame_id = f.replace('.png', '')
            
            # MatrixCity naming convention:
            # Depth:  0004.exr
            # Normal: 0004_normal.npy
            target_depth = f"{frame_id}.exr"
            target_normal = f"{frame_id}_normal.npy"

            # Check if associated depth and normal exist
            if target_depth in all_files_set and target_normal in all_files_set:
                image_files.append(os.path.join(scene, f))
                depth_files.append(os.path.join(scene, target_depth))
                normal_files.append(os.path.join(scene, target_normal))

        if len(image_files) < 32:
            # Skipping scene if frame count is below threshold
            continue

        data.append({
            "scene": scene,
            "image": image_files,
            "depth": depth_files,
            "normal": normal_files,
            "calib": calib_file
        })

    print(f"MatrixCity processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'matrixcity_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
