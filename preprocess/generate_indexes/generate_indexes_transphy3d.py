import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./transphy3d', help='Root directory of the dataset')
    parser.add_argument('--output_path', type=str, default='train_test_split', help='Directory to save the JSON')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    split_configs = [
        ('train', 'train'),
        ('test', 'test'),
        ('parametric_train/training', 'para_train'),
        ('parametric_train/validation', 'para_val'),
        ('parametric_train/test', 'para_test')
    ]

    all_data = []

    for sub_dir, split_tag in split_configs:
        full_sub_path = os.path.join(data_path, sub_dir)
        if not os.path.exists(full_sub_path):
            continue
            
        scenes = sorted(os.listdir(full_sub_path))
        print(f"Processing split: {split_tag}")
        
        for scene in tqdm(scenes):
            scene_abs_path = os.path.join(full_sub_path, scene)
            if not os.path.isdir(scene_abs_path):
                continue

            files = os.listdir(scene_abs_path)
            
            stems = sorted([f.replace('.image.png', '') for f in files if f.endswith('.image.png')])

            image_list = []
            depth_list = []
            normal_list = []
            camera_list = []
            depth_json_list = []

            for stem in stems:
                img_rel = os.path.join(sub_dir, scene, f"{stem}.image.png")
                dep_rel = os.path.join(sub_dir, scene, f"{stem}.depth.png")
                norm_rel = os.path.join(sub_dir, scene, f"{stem}.normal.png")
                cam_rel = os.path.join(sub_dir, scene, f"{stem}.metadata.json")
                djson_rel = os.path.join(sub_dir, scene, f"{stem}.depth.json")

                requirements = [
                    os.path.join(data_path, dep_rel),
                    os.path.join(data_path, norm_rel),
                    os.path.join(data_path, cam_rel),
                    os.path.join(data_path, djson_rel)
                ]

                if all(os.path.exists(p) for p in requirements):
                    image_list.append(img_rel)
                    depth_list.append(dep_rel)
                    normal_list.append(norm_rel)
                    camera_list.append(cam_rel)
                    depth_json_list.append(djson_rel)

            if len(image_list) >= 32:
                all_data.append({
                    'scene': f"{split_tag}/{scene}",
                    'image': image_list,
                    'depth': depth_list,
                    'normal': normal_list,
                    'calib': camera_list,      # metadata.json
                    'depth_json': depth_json_list # depth.json
                })

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, 'transphy3d_train.json')
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
