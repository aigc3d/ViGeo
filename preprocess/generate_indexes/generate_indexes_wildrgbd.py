import os
import json
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    objects = os.listdir(data_path)
    data = []
    for obj in tqdm(objects):
        obj_path = os.path.join(obj, 'scenes')
        scenes_full_dir = os.path.join(data_path, obj_path)
        
        if not os.path.isdir(scenes_full_dir):
            continue
            
        scenes = os.listdir(scenes_full_dir)
        for scene in scenes:
            scene_path = os.path.join(obj_path, scene)
            rgb_dir = os.path.join(data_path, scene_path, 'rgb')
            
            if not os.path.isdir(rgb_dir):
                continue
                
            files = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
            files.sort()

            image_files = [os.path.join(scene_path, 'rgb', file) for file in files]
            depth_files = [os.path.join(scene_path, 'depth', file.replace('.jpg', '.png')) for file in files]
            calib_files = [os.path.join(scene_path, 'metadata', file.replace('.jpg', '.npz')) for file in files]

            filtered_images = []
            filtered_depths = []
            filtered_calibs = []
            for img, dep, cal in zip(image_files, depth_files, calib_files):
                if os.path.exists(os.path.join(data_path, img)) and \
                   os.path.exists(os.path.join(data_path, dep)) and \
                   os.path.exists(os.path.join(data_path, cal)):
                    filtered_images.append(img)
                    filtered_depths.append(dep)
                    filtered_calibs.append(cal)

            image_files = filtered_images
            depth_files = filtered_depths
            calib_files = filtered_calibs
            # -------------------------------------------------------------------------

            if len(image_files) < 32:
                print(f'Skipping due to insufficient number of frames')
                continue

            data.append({
                'calib': calib_files,
                'image': image_files,
                'depth': depth_files,
                })
    
    print(f"wildrgbd has {len(data)} scenes")
    with open(os.path.join(output_path, 'wildrgbd_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
