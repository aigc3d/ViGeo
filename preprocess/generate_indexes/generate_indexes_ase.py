import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dynamic_replica')
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
        scene_base = os.path.join(data_path, scene)
        rgb_dir = os.path.join(scene_base, 'rgb')
        depth_dir = os.path.join(scene_base, 'depth')
        normal_dir = os.path.join(scene_base, 'normal')
        calib_file = os.path.join(scene, 'meta_info.json')

        # Check only for the directories and the specific calib file you need
        if not all(os.path.isdir(d) for d in [rgb_dir, depth_dir, normal_dir]):
            continue
        
        if not os.path.exists(os.path.join(data_path, calib_file)):
            continue

        # Use sets for fast memory-based lookup
        rgb_files = sorted(os.listdir(rgb_dir))
        depth_set = set(os.listdir(depth_dir))
        normal_set = set(os.listdir(normal_dir))

        image_list = []
        depth_list = []
        normal_list = []

        for f in rgb_files:
            if not (f.startswith('rgb_') and f.endswith('.jpg')):
                continue
            
            # Extract the frame ID (e.g., 000000)
            frame_id = f.split('_')[1].split('.')[0]
            target_depth = f"depth_{frame_id}.png"
            target_normal = f"normal_{frame_id}.npy"

            # Match depth and normal using set lookup
            if target_depth in depth_set and target_normal in normal_set:
                image_list.append(os.path.join(scene, 'rgb', f))
                depth_list.append(os.path.join(scene, 'depth', target_depth))
                normal_list.append(os.path.join(scene, 'normal', target_normal))

        # Filtering based on frame count
        if len(image_list) < 100:
            continue

        data.append({
            'image': image_list,
            'depth': depth_list,
            'normal': normal_list,
            'calib': calib_file
        })
    
    print(f"Done. Processed {len(data)} ASE scenes.")
    with open(os.path.join(output_path, 'ase_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
