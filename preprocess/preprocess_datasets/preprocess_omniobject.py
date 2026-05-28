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
    parser.add_argument('--data_path', type=str, default='omniobject')
    parser.add_argument('--output_path', type=str, default='omniobject')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    scenes = os.listdir(data_path)
    for scene in tqdm(scenes):
        files = os.listdir(
            os.path.join(data_path, scene, 'render/images')
        )
        files.sort()
        transform_path = os.path.join(data_path, scene, 'render', 'transforms.json')
        with open(transform_path, "r") as fp:
            transforms = json.load(fp)
        
        img = imageio.imread(os.path.join(
            data_path, scene, 'render/images', files[0]
        ))

        if img.shape[-1] == 4:
            alpha_channel = img[..., 3]
            rgb_channels = img[..., :3]
            white_background = np.full_like(rgb_channels, 255)
            img = np.where(alpha_channel[..., None] == 0, white_background, rgb_channels)
        else:
            img = img[..., :3]

        H, W, _ = img.shape
        camera_angle_x = float(transforms["camera_angle_x"])
        frames = transforms["frames"]
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        intrinsics = np.array(
            [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32
        )

        poses = []
        for frame in frames:
            pose = np.array(frame["transform_matrix"], dtype=np.float32)
            pose[:, 1:3] *= -1  # Invert Y and Z axes if necessary
            poses.append(pose)
        
        np.savez(os.path.join(output_path, scene, 'render', 'cam.npz'), intrinsics=intrinsics, pose=poses)

if __name__ == "__main__":
    main()
