import os
import utils3d
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='tartanair')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def get_tartanair_intrinsic():
    intrinsic = np.zeros((3, 3), dtype=np.float32)
    intrinsic[0, 0] = 320.0
    intrinsic[1, 1] = 320.0
    intrinsic[0, 2] = 320.0
    intrinsic[1, 2] = 240.0
    intrinsic[2, 2] = 1.0
    return intrinsic

def process_camera(task_info):
    seq_path, camera_side = task_info
    seq_dir = Path(seq_path)
    
    depth_dir = seq_dir / f'depth_{camera_side}'
    normal_dir = seq_dir / f'normal_{camera_side}'

    if not depth_dir.exists():
        return f"Skip {seq_dir.name}/{camera_side}"

    normal_dir.mkdir(parents=True, exist_ok=True)

    intrinsic_raw = get_tartanair_intrinsic()

    depth_files = sorted(list(depth_dir.glob("*.npy")))
    
    for depth_path in depth_files:
        base_id = depth_path.name.replace('_depth.npy', '')
        save_npy_path = normal_dir / f"{base_id}_normal.npy"
        
        if save_npy_path.exists():
            continue

        depth = np.load(depth_path).astype(np.float32)
        depth = np.squeeze(depth)
            
        invalid_mask = depth > 1000.0
        depth[invalid_mask] = 0.0
        valid_mask = depth > 0.0
                
        h, w = depth.shape
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
    for env_dir in data_root.iterdir():
        if not env_dir.is_dir():
            continue
        for level in ['Easy', 'Hard']:
            level_dir = env_dir / level
            if not level_dir.exists():
                continue
            for seq_dir in level_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                tasks.append((str(seq_dir), 'left'))
                tasks.append((str(seq_dir), 'right'))

    print(f"Total camera tasks (left & right): {len(tasks)}")
    print(f"Using {args.num_workers} workers...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_camera, tasks), 
                      total=len(tasks), 
                      desc="Generating TartanAir Normals"):
            pass

    print("TartanAir normal maps generated successfully!")

if __name__ == "__main__":
    main()
