import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
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

    scenes = os.listdir(os.path.join(data_path, 'image'))
    data = []
    for scene in tqdm(scenes):
        image_files = []
        depth_files = []
        calib_file = os.path.join('calib', scene[:10], 'calib_cam_to_cam.txt')
        files = os.listdir(os.path.join(data_path, 'depth', scene, 'proj_depth/groundtruth/image_02'))
        files.sort()
        for file in files:
            depth_file = os.path.join(
                'depth', scene, 'proj_depth/groundtruth/image_02', file)
            image_file = os.path.join(
                'image', scene, 'image_02/data', file)
            if not os.path.exists(os.path.join(data_path, image_file)):
                print(f'{image_file} does not exist')
            image_files.append(image_file)
            depth_files.append(depth_file)
        data.append(
            {
                'scene': scene,
                'calib': calib_file,
                'image': image_files,
                'depth': depth_files
            }
        )
    
    with open(os.path.join(output_path, 'kitti_eval.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
