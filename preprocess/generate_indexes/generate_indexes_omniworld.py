import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def load_split_info(scene_dir):
    with open(os.path.join(scene_dir, "split_info.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_fps(scene_dir):
    fps_file = os.path.join(scene_dir, 'fps.txt')
    if not os.path.exists(fps_file):
        return 30.0
    with open(fps_file, 'r') as f:
        for line in f:
            if line.strip().startswith("FPS:"):
                try:
                    return float(line.split(":")[1].strip().split()[0])
                except:
                    break
    return 30.0

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="omniworld")
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    anno_root = os.path.join(data_path, 'annotations')
    scenes = sorted(os.listdir(anno_root))
    
    data = []

    for scene in tqdm(scenes):
        scene_path = os.path.join(anno_root, scene)
        if not os.path.isdir(scene_path):
            continue

        split_info = load_split_info(scene_path)
        split_num = split_info["split_num"]
        
        # Directories for lookup
        depth_dir = os.path.join(scene_path, "depth")
        normal_dir = os.path.join(scene_path, "normal")
        
        # Use sets for fast O(1) matching
        depth_set = set(os.listdir(depth_dir)) if os.path.exists(depth_dir) else set()
        normal_set = set(os.listdir(normal_dir)) if os.path.exists(normal_dir) else set()

        for split_idx in range(split_num):
            idxs = split_info["split"][split_idx]
            calib_rel_path = os.path.join("annotations", scene, "camera", f"split_{split_idx}.json")
            
            image_files = []
            depth_files = []
            normal_files = []

            # Process basenames (frame IDs)
            for idx in idxs:
                basename = f'{idx:06d}'
                
                img_name = f"{basename}.png"
                dep_name = f"{basename}.png"
                norm_name = f"{basename}.npy"

                # Check if all modality files exist
                # Images are in 'videos' folder, Depth/Normal in 'annotations'
                if dep_name in depth_set and norm_name in normal_set:
                    image_files.append(os.path.join("videos", scene, "color", img_name))
                    depth_files.append(os.path.join("annotations", scene, "depth", dep_name))
                    normal_files.append(os.path.join("annotations", scene, "normal", norm_name))

            if len(image_files) < 24:
                print(f"OmniWorld (scene {scene}) (idx {split_idx}): only {len(image_files)} valid frames, skipping.")
                continue

            data.append({
                "scene": f"{scene}_{split_idx}",
                "calib": calib_rel_path,
                "image": image_files,
                "depth": depth_files,
                "normal": normal_files
            })
    
    print(f"OmniWorld processing complete: {len(data)} sub-scenes found.")
    with open(os.path.join(output_path, 'omniworld_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
