import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='spring')
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
        rgb_dir = os.path.join(scene_dir, "rgb")
        depth_dir = os.path.join(scene_dir, "depth")
        cam_dir = os.path.join(scene_dir, "cam")
        normal_dir = os.path.join(scene_dir, "normal")

        # Basic directory check
        if not all(os.path.isdir(d) for d in [rgb_dir, depth_dir, cam_dir, normal_dir]):
            continue

        # Use sets for fast lookup
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        depth_set = set(os.listdir(depth_dir))
        cam_set = set(os.listdir(cam_dir))
        normal_set = set(os.listdir(normal_dir))

        image_list = []
        depth_list = []
        normal_list = []
        calib_list = []

        for f in rgb_files:
            # 0002.png -> 0002
            frame_id = f.replace('.png', '')
            
            target_depth = f"{frame_id}.npy"
            target_cam = f"{frame_id}.npz"
            target_normal = f"{frame_id}_normal.npy" # Pattern: ID_normal.npy

            if target_depth in depth_set and target_cam in cam_set and target_normal in normal_set:
                image_list.append(os.path.join(scene, "rgb", f))
                depth_list.append(os.path.join(scene, "depth", target_depth))
                normal_list.append(os.path.join(scene, "normal", target_normal))
                calib_list.append(os.path.join(scene, "cam", target_cam))

        if len(image_list) < 24:
            continue

        data.append({
            'scene': scene,
            'image': image_list,
            'depth': depth_list,
            'normal': normal_list,
            'calib': calib_list
        })

    print(f"Spring processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'spring_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
