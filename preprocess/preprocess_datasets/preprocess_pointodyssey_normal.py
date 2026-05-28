import os
import utils3d
import numpy as np
import multiprocessing
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='PointOdyssey 数据集根目录')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def read_pointodyssey_depth(depth_path):
    depth_img = Image.open(depth_path)
    depth = np.asarray(depth_img, dtype=np.float32) / 65535.0 * 1000.0
    
    valid_mask = (depth > 0)
    return depth, valid_mask

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def process_scene(scene_path):
    scene_dir = Path(scene_path)
    
    depth_dir = scene_dir / 'depths'
    cam_file = scene_dir / 'anno.npz'
    normal_dir = scene_dir / 'normals'

    if not depth_dir.exists() or not cam_file.exists():
        return f"Skip {scene_dir.name}: Missing depth or anno.npz"

    normal_dir.mkdir(parents=True, exist_ok=True)

    calib_data = np.load(cam_file)
    cam_ints = calib_data["intrinsics"].astype(np.float32)

    depth_files = sorted(list(depth_dir.glob("depth_*.png")))
    
    for i, depth_path in enumerate(depth_files):
        base_id = depth_path.name.replace('depth_', '').replace('.png', '')
        save_npy_path = normal_dir / f"normal_{base_id}.npy"
        
        if save_npy_path.exists():
            continue

        depth, valid_mask = read_pointodyssey_depth(depth_path)
        h, w = depth.shape
            
        intrinsic_raw = cam_ints[i]
        intrinsic = normalize_intrinsic(intrinsic_raw, h, w)

        normal, normal_mask = utils3d.np.depth_map_to_normal_map(
            depth, 
            intrinsic, 
            mask=valid_mask
        )

        np.save(save_npy_path, {
            'normal': normal.astype(np.float32),
            'normal_mask': normal_mask.astype(np.bool_)
        })


def main():
    args = parse_args()
    data_root = Path(args.data_path)

    tasks = []
    for split in ["train", "test", "val"]:
        split_dir = data_root / split
        
        for scene_dir in split_dir.iterdir():
            if scene_dir.is_dir():
                tasks.append(str(scene_dir))

    print(f"Total scenes to process: {len(tasks)}")
    print(f"Using {args.num_workers} workers...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_scene, tasks), 
                      total=len(tasks), 
                      desc="Generating PointOdyssey Normals"):
            pass
    
    print("PointOdyssey normal maps generated successfully!")

if __name__ == "__main__":
    main()
