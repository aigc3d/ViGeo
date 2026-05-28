import os
import json
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./irs')
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    data = []
    for scene in ['Home', 'Office', 'Restaurant', 'Store']:
        sub_scenes = os.listdir(os.path.join(data_path, scene))
        for sub_scene in tqdm(sub_scenes):
            files = os.listdir(os.path.join(data_path, scene, sub_scene, 'rgb'))
            image_files = []
            depth_files = []
            for i in range(len(files)):
                id = i + 1
                image_file = os.path.join(scene, sub_scene, 'rgb', f'l_{id}.png')
                depth_file = os.path.join(scene, sub_scene, 'depth', f'd_{id}.npy')
                if not os.path.exists(os.path.join(data_path, image_file)):
                    print(f"Skipping {image_file} because it does not exist.")
                    continue
                if not os.path.exists(os.path.join(data_path, depth_file)):
                    print(f"Skipping {depth_file} because it does not exist.")
                    continue
                image_files.append(image_file)
                depth_files.append(depth_file)
            
            if len(image_files) < 32:
                print(f"Skipping {scene} {sub_scene} because it has only {len(image_files)} frames, less than 32.")
                continue

            data.append(
                {
                    'scene': scene + '/' + sub_scene,
                    'image': image_files,
                    'depth': depth_files,
                }
            )
    
    print(f"irs has {len(data)} scenes")
    with open(os.path.join(output_path, 'irs_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
