import os
import cv2
import json
import utils3d
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

# The coordinate of the normal provided by datasets is unknown
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='omniobject')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()

def read_exr_depth(depth_path):
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None, None

    if len(depth_raw.shape) == 3:
        depth = depth_raw[..., -1]
    else:
        depth = depth_raw

    depth = depth.astype(np.float32)
    invalid_mask = (depth >= 65500.0) | np.isinf(depth)
    depth[invalid_mask] = 0.0
    
    valid_mask = (depth > 0)
    return depth, valid_mask

def normalize_intrinsic(intrinsic: np.ndarray, height: int, width: int):
    intrinsic_norm = intrinsic.copy()
    intrinsic_norm[0, :] /= width
    intrinsic_norm[1, :] /= height
    return intrinsic_norm

def process_scene(task_info):
    scene_path, scale = task_info
    
    scene_dir = Path(scene_path)
    render_dir = scene_dir / 'render'
    
    depth_dir = render_dir / 'depths'
    cam_file = render_dir / 'cam.npz'
    normal_dir = render_dir / 'normal'

    normal_dir.mkdir(parents=True, exist_ok=True)

    try:
        cam_data = np.load(cam_file)
        intrinsic_raw = cam_data['intrinsics'].astype(np.float32)
    except Exception as e:
        return f"Error reading cam.npz in {scene_dir.name}: {e}"

    depth_files = sorted(list(depth_dir.glob("*_depth.exr")))
    
    for depth_path in depth_files:
        base_id = depth_path.name.replace('_depth.exr', '')
        save_npy_path = normal_dir / f"{base_id}_normal.npy"
        
        if save_npy_path.exists():
            continue

        depth, valid_mask = read_exr_depth(depth_path)
        depth = depth / scale / 1000.0
            
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

    scale_file = data_root / 'scale.json'
    with open(scale_file, 'r') as f:
        scales_dict = json.load(f)
    
    scene_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    tasks = []
    for d in scene_dirs:
        scene_name = d.name
        if scene_name in scales_dict.keys():
            tasks.append((str(d), scales_dict[scene_name]))
    
    print(f"Total scenes to process: {len(tasks)}")
    print(f"Using {args.num_workers} workers...")
    
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_scene, tasks), 
                      total=len(tasks), 
                      desc="Generating Normals"):
            pass
    
    print("OmniObject normal maps generated successfully!")

if __name__ == "__main__":
    main()
