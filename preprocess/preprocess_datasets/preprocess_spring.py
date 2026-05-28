import os
import json
import shutil
import numpy as np
import imageio.v2 as imageio
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    scenes = [f for f in os.listdir(data_path) if not f.endswith('.pkl')]
    for scene in tqdm(scenes):
        os.makedirs(os.path.join(output_path, scene), exist_ok=True)
        shutil.copytree(os.path.join(data_path, scene, 'rgb'), os.path.join(output_path, scene, 'rgb'))
        shutil.copytree(os.path.join(data_path, scene, 'depth'), os.path.join(output_path, scene, 'depth'))
        os.makedirs(os.path.join(output_path, scene, 'cam'), exist_ok=True)
        files = os.listdir(os.path.join(data_path, scene, 'cam'))
        files = [f for f in files if f.endswith('.npz')]
        for file in files:
            shutil.copy(os.path.join(data_path, scene, 'cam', file), os.path.join(output_path, scene, 'cam', file))

if __name__ == "__main__":
    main()
