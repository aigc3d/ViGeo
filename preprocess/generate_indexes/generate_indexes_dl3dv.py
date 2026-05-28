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

    scenes = os.listdir(data_path)
    data = []
    for scene in tqdm(scenes):
        files = [f[:-4] for f in os.listdir(os.path.join(data_path, scene, 'dense', 'rgb'))]
        files.sort()

        files = [f for f in files if 'ipynb_checkpo' not in f]

        image_files = [
            os.path.join(scene, 'dense/rgb', file + '.png') for file in files
        ]
        depth_files = [
            os.path.join(scene, 'dense/depth', file + '.npy') for file in files
        ]
        sky_mask_files = [
            os.path.join(scene, 'dense/sky_mask', file + '.png') for file in files
        ]
        outlier_files = [
            os.path.join(scene, 'dense/outlier_mask', file + '.png') for file in files
        ]
        calib_files = [
            os.path.join(scene, 'dense/cam', file + '.npz') for file in files
        ]
        
        if len(image_files) < 32:
            print(f"Skipping {scene} because it has only {len(image_files)} frames, less than 32.")
            continue

        data.append(
            {
                'image': image_files,
                'depth': depth_files,
                'calib': calib_files,
                'sky_mask': sky_mask_files,
                'outlier': outlier_files
            }
        )
    
    print(f"dl3dv has {len(data)} scenes")
    with open(os.path.join(output_path, 'dl3dv_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
