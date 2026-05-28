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

    
    data = []

    for split in ['Training', 'Validation']:
        scenes = os.listdir(os.path.join(data_path, split))
        for scene in tqdm(scenes):
            files = [f[:-4] for f in os.listdir(os.path.join(data_path, split, scene, 'vga_wide')) if not f.startswith('.ipynb')]
            files.sort()
            image_files = [os.path.join(
                split, scene, 'vga_wide', file + '.jpg') for file in files]
            depth_files = [os.path.join(
                split, scene, 'highres_depth', file + '.png') for file in files]
            calib_file = os.path.join(split, scene, 'scene_metadata.npz')

            data.append(
                {
                    'image': image_files,
                    'depth': depth_files,
                    'calib': calib_file,
                }
            )
    
    print(f"arkitscenes_highres has {len(data)} scenes")
    with open(os.path.join(output_path, 'arkitscenes_highres_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
