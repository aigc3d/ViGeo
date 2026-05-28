import os
import json
import shutil
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='matrixcity')
    parser.add_argument('--output_path', type=str, default='matrixcity')
    return parser.parse_args()

def split_list_into_chunks(lst, chunk_size=100):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    camera = 'street'
    # big city
    image_paths = []
    depth_paths = []
    intrinsic = []
    seq_id = 0
    for split in ['train', 'test']:
        scenes = os.listdir(os.path.join(data_path, 'big_city', camera, split))
        for scene in scenes:
            transform_path = os.path.join(data_path, 'big_city', camera, split, scene, 'transforms.json')
            with open(transform_path, 'r') as f:
                data = json.load(f)
            angle_x = data['camera_angle_x']
            w = 1000.0
            f = float(.5 * w / np.tan(.5 * angle_x))

            chunks = split_list_into_chunks(data['frames'], chunk_size=100)

            for chunk in tqdm(chunks):
                tgt_path = os.path.join(output_path, '{:05d}'.format(seq_id))
                os.makedirs(tgt_path, exist_ok=True)
                pose = []
                for index in range(len(chunk)):
                    frame_index = chunk[index]['frame_index']
                    rot_mat = np.array(chunk[index]['rot_mat'])

                    rot_mat[:3,:3] *= 100
                    rot_mat[:3,3] /= 100

                    formatted = f"{frame_index:04d}" if frame_index < 1000 else str(frame_index)
                    shutil.copy(os.path.join(data_path, 'big_city', camera, split, scene, formatted + '.png'), 
                    os.path.join(tgt_path, '{:04}.png'.format(index)))
                    shutil.copy(os.path.join(data_path, 'big_city_depth', camera, split, scene + '_depth', formatted + '.exr'),
                    os.path.join(tgt_path, '{:04}.exr'.format(index)))
                    pose.append(rot_mat)
                pose = np.stack(pose)
                np.savez(os.path.join(tgt_path, 'pose.npz'), camera_angle_x=angle_x, pose=pose)
                seq_id += 1
    
    
    # small city
    for split in ['train_dense', 'test']:
        scenes = os.listdir(os.path.join(data_path, 'small_city', camera, split))
        for scene in scenes:
            transform_path = os.path.join(data_path, 'small_city', camera, split, scene, 'transforms.json')
            with open(transform_path, 'r') as f:
                data = json.load(f)
            angle_x = data['camera_angle_x']
            w = 1000.0
            f = float(.5 * w / np.tan(.5 * angle_x))

            chunks = split_list_into_chunks(data['frames'], chunk_size=100)

            for chunk in tqdm(chunks):
                tgt_path = os.path.join(output_path, '{:05d}'.format(seq_id))
                os.makedirs(tgt_path, exist_ok=True)
                pose = []
                for index in range(len(chunk)):
                    frame_index = chunk[index]['frame_index']
                    rot_mat = np.array(chunk[index]['rot_mat'])

                    rot_mat[:3,:3] *= 100
                    rot_mat[:3,3] /= 100

                    formatted = f"{frame_index:04d}" if frame_index < 1000 else str(frame_index)
                    shutil.copy(os.path.join(data_path, 'small_city', camera, split, scene, formatted + '.png'), 
                    os.path.join(tgt_path, '{:04}.png'.format(index)))
                    shutil.copy(os.path.join(data_path, 'small_city_depth', camera, split, scene + '_depth', formatted + '.exr'),
                    os.path.join(tgt_path, '{:04}.exr'.format(index)))
                    pose.append(rot_mat)
                pose = np.stack(pose)
                np.savez(os.path.join(tgt_path, 'pose.npz'), camera_angle_x=angle_x, pose=pose)
                seq_id += 1

if __name__ == "__main__":
    main()
