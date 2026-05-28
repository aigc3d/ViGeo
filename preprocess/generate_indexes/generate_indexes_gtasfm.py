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
    for split in ["train", "test"]:
        image_root = os.path.join(data_path, split, "image")
        if not os.path.exists(image_root):
            continue
            
        scenes = sorted(os.listdir(image_root))
        for scene in tqdm(scenes):
            scene_img_dir = os.path.join(image_root, scene)
            scene_depth_dir = os.path.join(data_path, split, "depth", scene)
            scene_calib_dir = os.path.join(data_path, split, "calib", scene)
            scene_normal_dir = os.path.join(data_path, split, "normal", scene)

            # Check if all directories exist
            if not all(os.path.isdir(d) for d in [scene_img_dir, scene_depth_dir, scene_calib_dir, scene_normal_dir]):
                continue

            files = sorted(os.listdir(scene_img_dir))
            
            # Use sets for faster lookup
            depth_set = set(os.listdir(scene_depth_dir))
            calib_set = set(os.listdir(scene_calib_dir))
            normal_set = set(os.listdir(scene_normal_dir))

            image_files = []
            depth_files = []
            calib_files = []
            normal_files = []

            for file in files:
                if not file.endswith('.png') and not file.endswith('.jpg'):
                    continue
                
                # Based on your script: file.split('.')[0][6:] 
                # e.g., 'image_0000.png' -> '0000'
                frame_id = file.split('.')[0][6:]
                
                target_depth = f'depth_{frame_id}.npy'
                target_calib = f'calib_{frame_id}.npy'
                target_normal = f'normal_{frame_id}.npy'

                if target_depth in depth_set and target_calib in calib_set and target_normal in normal_set:
                    image_files.append(os.path.join(split, "image", scene, file))
                    depth_files.append(os.path.join(split, "depth", scene, target_depth))
                    calib_files.append(os.path.join(split, "calib", scene, target_calib))
                    normal_files.append(os.path.join(split, "normal", scene, target_normal))

            data.append({
                'scene': f"{split}/{scene}",
                'image': image_files,
                'depth': depth_files,
                'normal': normal_files,
                'calib': calib_files
            })

    print(f"GTA-Sfm processing complete: {len(data)} scenes.")
    with open(os.path.join(output_path, 'gtasfm_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
