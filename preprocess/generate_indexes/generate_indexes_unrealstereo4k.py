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

    data = []
    scenes = os.listdir(data_path)
    for scene in tqdm(scenes):
        files = os.listdir(os.path.join(data_path, scene, 'Image0'))
        files.sort()
        image_files = [os.path.join(scene, 'image_low', f) for f in files]
        depth_files = [os.path.join(scene, 'depth_low', f.replace('.png', '.npy')) for f in files]
        calib_file = os.path.join(scene, 'pose.npz')

        if len(image_files) < 32:
            print(f'Skipping {scene} due to insufficient number of frames')
            continue

        data.append({
            "scene": scene,
            "image": image_files,
            "depth": depth_files,
            "calib": calib_file
        })

    
    print(f"unrealstereo4k has {len(data)} scenes")
    with open(os.path.join(output_path, 'unrealstereo4k_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
