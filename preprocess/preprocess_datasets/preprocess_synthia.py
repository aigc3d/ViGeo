import os
import json
import shutil
import numpy as np
from random import shuffle
from tqdm import tqdm
from argparse import ArgumentParser
invalid = [
'test5_23segs_weather_2_spawn_0_roadTexture_2_P_None_C_None_B_None_WC_None',

'test5_25segs_weather_4_spawn_0_roadTexture_2_P_None_C_None_B_None_WC_None',

'test5_26segs_weather_0_spawn_0_roadTexture_2_P_None_C_None_B_None_WC_None',

'test5_29segs_weather_3_spawn_0_roadTexture_0_P_None_C_None_B_None_WC_None',

'test5_18segs_weather_4_spawn_0_roadTexture_1_P_None_C_None_B_None_WC_None',

'test5_26segs_weather_2_spawn_0_roadTexture_1_P_None_C_None_B_None_WC_None',

'test5_24segs_weather_4_spawn_0_roadTexture_2_P_None_C_None_B_None_WC_None',
]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='synthia')
    parser.add_argument('--output_path', type=str, default='video_synthia')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    seq_id = 0
    for split in ['test', 'train']:
        scenes = os.listdir(os.path.join(data_path, split))
        for scene in tqdm(scenes):
            if scene in invalid:
                continue
            sub_scenes = os.listdir(os.path.join(data_path, split, scene))
            for sub_scene in sub_scenes:

                if not os.path.exists(os.path.join(data_path, split, scene, sub_scene, 'calib_kitti')):
                    print(f'skipping {scene}: {sub_scene}')
                    continue

                tgt_dir = f'{output_path}/{str(seq_id).zfill(6)}'
                os.makedirs(tgt_dir, exist_ok=True)
                image_dir = os.path.join(tgt_dir, 'image')
                depth_dir = os.path.join(tgt_dir, 'depth')
                os.makedirs(image_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                image_files = os.listdir(os.path.join(data_path, split, scene, sub_scene, 'RGB'))
                image_files.sort()
                poses = []
                for image_file in image_files:
                    pose_file = os.path.join(data_path, split, scene, sub_scene, 'Information', image_file.replace('.png', '.json'))
                    image_path = os.path.join(data_path, split, scene, sub_scene, 'RGB', image_file)
                    tgt_image_path = os.path.join(image_dir, image_file)
                    depth_path = os.path.join(data_path, split, scene, sub_scene, 'Depth', image_file)
                    tgt_depth_path = os.path.join(depth_dir, image_file)
                    shutil.copy(image_path, tgt_image_path)
                    shutil.copy(depth_path, tgt_depth_path)
                    
                    with open(pose_file, 'r') as f:
                        data = json.load(f)
                    pose = np.array(data['extrinsic']['matrix']).reshape(4, 4)
                    poses.append(pose)
                
                calib = {}
                with open(os.path.join(data_path, split, scene, sub_scene, 'calib_kitti', image_file.replace('.png', '.txt')), 'r') as f:
                    for line in f:
                        key, value = line.split(':', 1)
                        calib[key.strip()] = value.strip().split()
                intrinsic = np.asarray(calib['P0'], dtype=np.float32)
                intrinsic = np.reshape(intrinsic, [3, 4])[:, :3]
                np.savez(os.path.join(tgt_dir, 'cam.npz'), intrinsics=intrinsic, poses=np.stack(poses))
                seq_id += 1

if __name__ == "__main__":
    main()
