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

    # tgt_scenes = ['rgbd_bonn_crowd']

    tgt_scenes = ['rgbd_bonn_balloon2', 'rgbd_bonn_crowd2', 'rgbd_bonn_crowd3',
    'rgbd_bonn_person_tracking2', 'rgbd_bonn_synchronous']

    scenes = os.listdir(os.path.join(data_path))
    scenes.sort()
    data = []
    for scene in tqdm(scenes):
        if scene not in tgt_scenes:
            continue
        image_files = os.listdir(os.path.join(data_path, scene, 'rgb'))
        image_files.sort()
        image_files = [os.path.join(scene, 'rgb', f) for f in image_files]
        depth_files = os.listdir(os.path.join(data_path, scene, 'depth'))
        depth_files.sort()
        depth_files = [os.path.join(scene, 'depth', f) for f in depth_files]

        data.append(
                {
                    'scene': scene,
                    'image': image_files,
                    'depth': depth_files,
                }
            )

    with open(os.path.join(output_path, 'bonn_eval.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
