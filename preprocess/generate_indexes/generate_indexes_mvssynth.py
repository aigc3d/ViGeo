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

    scenes = sorted(os.listdir(data_path))
    data = []
    
    for scene in tqdm(scenes):
        scene_path = os.path.join(data_path, scene)
        rgb_dir = os.path.join(scene_path, 'rgb')
        depth_dir = os.path.join(scene_path, 'depth')
        cam_dir = os.path.join(scene_path, 'cam')
        normal_dir = os.path.join(scene_path, 'normal')

        # Check if all required directories exist
        if not all(os.path.isdir(d) for d in [rgb_dir, depth_dir, cam_dir, normal_dir]):
            continue

        # Get file lists and use sets for fast lookup
        files = sorted(os.listdir(rgb_dir))
        depth_set = set(os.listdir(depth_dir))
        cam_set = set(os.listdir(cam_dir))
        normal_set = set(os.listdir(normal_dir))

        image_files = []
        depth_files = []
        normal_files = []
        calib_files = []

        for file in files:
            if not file.endswith('.jpg'):
                continue
            
            # Extract ID: 0000.jpg -> 0000
            frame_id = file.replace('.jpg', '')
            
            target_depth = f"{frame_id}.npy"
            target_cam = f"{frame_id}.npz"
            target_normal = f"{frame_id}.npy"

            # Verify all modalities exist
            if target_depth in depth_set and target_cam in cam_set and target_normal in normal_set:
                image_files.append(os.path.join(scene, 'rgb', file))
                depth_files.append(os.path.join(scene, 'depth', target_depth))
                normal_files.append(os.path.join(scene, 'normal', target_normal))
                calib_files.append(os.path.join(scene, 'cam', target_cam))

        if len(image_files) < 32:
            continue

        data.append({
            'scene': scene,
            'image': image_files,
            'depth': depth_files,
            'normal': normal_files,
            'calib': calib_files
        })
        
    print(f"MVS-Synth processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'mvssynth_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
