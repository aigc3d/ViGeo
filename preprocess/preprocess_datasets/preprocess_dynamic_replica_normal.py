import os
import json
import utils3d
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial
from argparse import ArgumentParser

def read_depth(depth_path):
    with Image.open(depth_path) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] = intrinsic_norm[0, :] / width
    intrinsic_norm[1, :] = intrinsic_norm[1, :] / height
    return intrinsic_norm

def process_scene(scene, data_path):
    scene_path = os.path.join(data_path, scene)
    img_dir = os.path.join(scene_path, 'images')
    depth_dir = os.path.join(scene_path, 'depths')
    normal_dir = os.path.join(scene_path, 'normals')

    calib_file = os.path.join(scene_path, 'calib.npz')

    calib_data = np.load(calib_path := calib_file)
    intrinsics_all = calib_data["intrinsics"].astype(np.float32)
    
    os.makedirs(normal_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    for i in range(len(files)):
        img_id = files[i].split('.')[0][-4:]
        depth_filename = f"{scene}_{img_id}.geometric.png"
        depth_path = os.path.join(depth_dir, depth_filename)
        save_path = os.path.join(normal_dir, f"{scene}_{img_id}.normal.npy")

        if os.path.exists(save_path):
            continue

        depth = read_depth(depth_path)
        h, w = depth.shape
            
        intrinsic = normalize_intrinsic(intrinsics_all[i].copy(), height=h, width=w)
            
        normal, normal_mask = utils3d.np.depth_map_to_normal_map(
            depth, 
            intrinsic, 
            mask=(depth > 0)
        )

        np.save(save_path, {
                'normal': normal.astype(np.float32),
                'normal_mask': normal_mask.astype(np.bool_)
            })

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dynamic_replica')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count()) 
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path

    scenes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

    print(f"Total scenes to process: {len(scenes)}")
    print(f"Using {args.num_workers} CPU workers")

    worker_func = partial(process_scene, data_path=data_path)
    
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(worker_func, scenes), total=len(scenes), desc="Overall Progress"))

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
