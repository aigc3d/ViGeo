import os
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
        files.sort()
        pose = []
        for file in tqdm(files):
            disp = np.load(
                os.path.join(data_path, scene, 'Disp0', file.replace('.png', '.npy'))
            )

            with open(os.path.join(data_path, scene, 'Extrinsics0', file.replace('.png', '.txt')), 'r') as f:
                lines = f.readlines()
            intrinsic = np.array([float(x) for x in lines[0].strip().split()]).reshape(3, 3)
            extrinsic1 = np.array([float(x) for x in lines[1].strip().split()]).reshape(3, 4)

            with open(os.path.join(data_path, scene, 'Extrinsics1', file.replace('.png', '.txt')), 'r') as f:
                lines = f.readlines()
            extrinsic2 = np.array([float(x) for x in lines[1].split()]).reshape(3, 4)
            baseline = abs(extrinsic1[0, 3] - extrinsic2[0, 3])
            focal_length = intrinsic[0, 0]
            depth = baseline * focal_length / disp

            tgt_path = os.path.join(output_path, scene, 'depth')
            os.makedirs(tgt_path, exist_ok=True)
            np.save(os.path.join(tgt_path, file.replace('.png', '.npy')), depth)
            extrinsic1 = np.vstack([extrinsic1, [0, 0, 0, 1]])
        
            pose.append(extrinsic1)
        pose = np.stack(pose)
        intrin = intrinsic
        np.savez(os.path.join(output_path, scene, 'pose.npz'), pose=pose, intrinsic=intrin)

if __name__ == "__main__":
    main()
