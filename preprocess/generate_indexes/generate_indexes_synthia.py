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
        scene_dir = os.path.join(data_path, scene)
        img_dir = os.path.join(scene_dir, "image")
        dep_dir = os.path.join(scene_dir, "depth")
        norm_dir = os.path.join(scene_dir, "normal")
        
        # Check basic directories
        if not all(os.path.isdir(d) for d in [img_dir, dep_dir, norm_dir]):
            continue

        # Use sets for fast O(1) matching
        img_files = sorted(os.listdir(img_dir))
        depth_set = set(os.listdir(dep_dir))
        normal_set = set(os.listdir(norm_dir))
        
        image_list = []
        depth_list = []
        normal_list = []
        calib_file = os.path.join(scene, 'cam.npz')

        for f in img_files:
            # e.g., 000119.png -> 000119
            frame_id = f.split('.')[0]
            
            # Synthia naming pattern:
            # Image: 000119.png
            # Depth: 000119.png (based on your script)
            # Normal: 000119_normal.npy
            target_depth = f
            target_normal = f"{frame_id}_normal.npy"

            if target_depth in depth_set and target_normal in normal_set:
                image_list.append(os.path.join(scene, "image", f))
                depth_list.append(os.path.join(scene, "depth", target_depth))
                normal_list.append(os.path.join(scene, "normal", target_normal))

        if len(image_list) < 32:
            continue

        data.append({
            'scene': scene,
            'calib': calib_file,
            'image': image_list,
            'depth': depth_list,
            'normal': normal_list
        })

    print(f"Synthia processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'synthia_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
