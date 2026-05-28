import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./pointodyssey')
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

# https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/src/dust3r/datasets/pointodyssey.py#L22
scenes_to_use = [
    "cnb_dlab_0215_3rd", "cnb_dlab_0215_ego1", "cnb_dlab_0225_3rd", "cnb_dlab_0225_ego1",
    "dancing", "dancingroom0_3rd", "footlab_3rd", "footlab_ego1", "footlab_ego2",
    "girl", "girl_egocentric", "human_egocentric", "human_in_scene", "human_in_scene1",
    "kg", "kg_ego1", "kg_ego2", "kitchen_gfloor", "kitchen_gfloor_ego1", "kitchen_gfloor_ego2",
    "scene_carb_h_tables", "scene_carb_h_tables_ego1", "scene_carb_h_tables_ego2",
    "scene_j716_3rd", "scene_j716_ego1", "scene_j716_ego2",
    "scene_recording_20210910_S05_S06_0_3rd", "scene_recording_20210910_S05_S06_0_ego2",
    "scene1_0129", "scene1_0129_ego", "seminar_h52_3rd", "seminar_h52_ego1", "seminar_h52_ego2",
]

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = []
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(data_path, split)
        if not os.path.exists(split_dir):
            continue
            
        scenes = os.listdir(split_dir)
        for scene in tqdm(scenes):
            if scene not in scenes_to_use:
                continue
            
            scene_dir = os.path.join(split_dir, scene)
            rgb_dir = os.path.join(scene_dir, "rgbs")
            depth_dir = os.path.join(scene_dir, "depths")
            normal_dir = os.path.join(scene_dir, "normals")
            
            # Use sets for O(1) memory lookup to avoid slow disk IO
            depth_set = set(os.listdir(depth_dir)) if os.path.exists(depth_dir) else set()
            normal_set = set(os.listdir(normal_dir)) if os.path.exists(normal_dir) else set()
            
            rgbs = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
            
            image_files = []
            depth_files = []
            normal_files = []
            calib_file = os.path.join(split, scene, 'anno.npz')

            for file in rgbs:
                # rgb_00108.jpg -> 00108
                frame_id = file.split('.')[0].split('_')[1]
                
                target_depth = f"depth_{frame_id}.png"
                target_normal = f"normal_{frame_id}.npy"

                if target_depth in depth_set and target_normal in normal_set:
                    image_files.append(os.path.join(split, scene, 'rgbs', file))
                    depth_files.append(os.path.join(split, scene, 'depths', target_depth))
                    normal_files.append(os.path.join(split, scene, 'normals', target_normal))

            if len(image_files) < 32:
                continue

            data.append({
                'scene': f"{split}/{scene}",
                'calib': calib_file,
                'image': image_files,
                'depth': depth_files,
                'normal': normal_files
            })
    
    print(f"PointOdyssey processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'pointodyssey_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
