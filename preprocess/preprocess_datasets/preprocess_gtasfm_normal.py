import os
import utils3d
import numpy as np
import multiprocessing
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='解包后数据的根目录')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] = intrinsic_norm[0, :] / width
    intrinsic_norm[1, :] = intrinsic_norm[1, :] / height
    return intrinsic_norm

def process_scene(task_info):
    data_path, split, scene = task_info
    
    depth_dir = os.path.join(data_path, split, 'depth', scene)
    calib_dir = os.path.join(data_path, split, 'calib', scene)
    normal_dir = os.path.join(data_path, split, 'normal', scene)
    
    os.makedirs(normal_dir, exist_ok=True)

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    
    for depth_file in depth_files:
        frame_id = depth_file.split('_')[1].split('.')[0]
        
        calib_file = f"calib_{frame_id}.npy"
        normal_file = f"normal_{frame_id}.npy"
        
        depth_path = os.path.join(depth_dir, depth_file)
        calib_path = os.path.join(calib_dir, calib_file)
        save_path = os.path.join(normal_dir, normal_file)

        # if os.path.exists(save_path):
        #     continue

        depth = np.load(depth_path)
        sky_mask = (depth == 10000).astype(np.float32)
        depth[sky_mask > 0] = 0
        h, w = depth.shape
            
        calib_dict = np.load(calib_path, allow_pickle=True).item()
        intrinsic_raw = calib_dict['intrinsic'].astype(np.float32)
            
        intrinsic = normalize_intrinsic(intrinsic_raw, h, w)

        normal, normal_mask = utils3d.np.depth_map_to_normal_map(
            depth, 
            intrinsic, 
            mask=(depth > 0)
        )

        np.save(save_path, {
            'normal': normal.astype(np.float32),
            'normal_mask': normal_mask.astype(np.bool_)
        })


def main():
    args = parse_args()
    
    tasks = []
    
    for split in ['train', 'test']:
        split_path = os.path.join(args.data_path, split)
        depth_base = os.path.join(split_path, 'depth')
            
        scenes = sorted(os.listdir(depth_base))
        for scene in scenes:
            tasks.append((args.data_path, split, scene))

    print(f"Total scenes to process: {len(tasks)}")
    print(f"Using {args.num_workers} processes...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_scene, tasks), 
                      total=len(tasks), 
                      desc="Generating Normals"):
            pass

    print("\nAll normal maps generated successfully!")

if __name__ == "__main__":
    main()
