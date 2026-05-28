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

    tgt_scenes = os.listdir(os.path.join(data_path, 'final'))

    scenes = os.listdir(os.path.join(data_path, 'final'))
    data = []
    for scene in tqdm(scenes):
        if scene not in tgt_scenes:
            continue
        files = os.listdir(os.path.join(data_path, 'final', scene))
        files.sort()
        image_files = [os.path.join('final', scene, f) for f in files]
        depth_files = [os.path.join('depth', scene, f.replace('.png', '.dpt')) for f in files]
        calib_files = [os.path.join('camdata_left', scene, f.replace('.png', '.cam')) for f in files]

        data.append(
                {
                    'scene': scene,
                    'calib': calib_files,
                    'image': image_files,
                    'depth': depth_files,
                }
            )

    with open(os.path.join(output_path, 'sintel_eval.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
