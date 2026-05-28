import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
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
        scene_full_path = os.path.join(data_path, scene)
        if not os.path.isdir(scene_full_path):
            continue
            
        sub_scenes = sorted(os.listdir(scene_full_path))
        for sub_scene in sub_scenes:
            curr_dir = os.path.join(scene_full_path, sub_scene)
            if not os.path.isdir(curr_dir):
                continue
            
            # List all files once for fast lookup
            all_files = os.listdir(curr_dir)
            all_files_set = set(all_files)
            
            # Filter RGB files to get unique IDs
            rgb_files = sorted([f for f in all_files if f.endswith('_rgb.png')])
            
            image_files = []
            depth_files = []
            normal_files = []
            calib_files = []
            
            for file in rgb_files:
                # Extract ID: 000006_rgb.png -> 000006
                frame_id = file.split('_')[0]
                
                target_depth = f"{frame_id}_depth.npy"
                target_normal = f"{frame_id}_normal.npy"
                target_cam = f"{frame_id}_cam.npz"
                
                # Check if all matching files exist in memory
                if target_depth in all_files_set and \
                   target_normal in all_files_set and \
                   target_cam in all_files_set:
                    
                    image_files.append(os.path.join(scene, sub_scene, file))
                    depth_files.append(os.path.join(scene, sub_scene, target_depth))
                    normal_files.append(os.path.join(scene, sub_scene, target_normal))
                    calib_files.append(os.path.join(scene, sub_scene, target_cam))
            
            if len(image_files) < 32:
                continue
                
            data.append({
                'scene': f"{scene}/{sub_scene}",
                'image': image_files,
                'depth': depth_files,
                'normal': normal_files,
                'calib': calib_files
            })

    print(f"Hypersim processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'hypersim_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
