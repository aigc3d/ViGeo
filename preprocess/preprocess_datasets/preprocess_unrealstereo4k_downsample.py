import os
import cv2
import json
import shutil
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import dataset
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='unrealstereo4k')
    parser.add_argument('--output_path', type=str, default='unrealstereo4k')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    scenes = os.listdir(data_path)
    for scene in tqdm(scenes):
        files = os.listdir(os.path.join(data_path, scene, 'Image0'))
        image_dir = os.path.join(output_path, scene, 'image_low')
        depth_dir = os.path.join(output_path, scene, 'depth_low')

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        for file in tqdm(files):
            # image = Image.open(os.path.join(data_path, scene, 'Image0', file))
            # new_size = tuple(dim // 4 for dim in image.size) 
            # resized_image = image.resize(new_size, Image.LANCZOS)
            # resized_image.save(os.path.join(image_dir, file))

            depth = np.load(os.path.join(data_path, scene, 'depth', file.replace('.png', '.npy')))
            h, w = depth.shape[:2]
            new_h, new_w = int(h * 0.25), int(w * 0.25)
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            np.save(os.path.join(depth_dir, file.replace('.png', '.npy')), depth)

if __name__ == "__main__":
    main()
