import os
import cv2
import yaml
import json
import renderpy
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scannetpp import (
    ScannetppScene_Release,
    read_txt_list,
    load_yaml_munch,
    load_json,
    run_command,
    pose_from_qwxyz_txyz
)
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_rgb(scene):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {scene.iphone_video_path} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    run_command(cmd, verbose=True)

def undistort_frames(frames, image_dir, depth_dir, output_image_dir, output_depth_dir, map1, map2):
    for frame in tqdm(frames):
        image_path = Path(image_dir) / frame
        out_image_path = Path(output_image_dir) / frame
        out_depth_path = Path(output_depth_dir) / frame.replace(".jpg", ".npy")

        if os.path.exists(out_image_path):
            continue
        if os.path.exists(out_depth_path):
            continue
        print(image_path)
        image = cv2.imread(str(image_path))
        undistorted_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image_path), undistorted_image)

        depth_path = Path(depth_dir) / frame.replace(".jpg", ".npy")
        depth = np.load(str(depth_path))
        undistorted_depth = cv2.remap(
            depth,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST, # Use nearest neighbor interpolation
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0 # Assuming 20.0 is an appropriate border value for missing depth information
        )
        
        out_depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_depth_path), undistorted_depth)

def process_scene(scene_id, data_path):
    # rgb.mkv error
    if scene_id in ['08bd80ce2a']:
        return
    
    scene = ScannetppScene_Release(scene_id, data_root=Path(data_path) / "data")
    extract_rgb(scene)

    colmap_dir = scene.iphone_colmap_dir

    with open(osp.join(colmap_dir, "cameras.txt"), "r") as f:
        raw = f.read().splitlines()[3:]  # skip header

    camera = raw[0].split(" ")
    intrinsics = [camera[1]] + [float(cam) for cam in camera[2:]]
    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, distortion, (width, height), 1, (width, height), True
        )
    map1, map2 = cv2.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )
    
    with open(osp.join(colmap_dir, "images.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]  # skip header
        raw = raw[0::2]
        raw.sort(key=lambda x: x.split()[-1])
    
    poses = []

    frames = []
    for item in raw:
        frames.append(item.split(" ")[-1].split('/')[-1])
        poses.append(pose_from_qwxyz_txyz(item.split(" ")[1:-2]))
    poses = np.stack(poses)
    np.savez(str(scene.iphone_data_dir / 'cam.npz'), pose=poses, intrinsic=new_K)
    
    image_dir = scene.iphone_rgb_dir
    depth_dir = scene.iphone_data_dir / 'render_depth'

    frames = [frame for frame in frames if os.path.exists(
        depth_dir / frame.replace(".jpg", ".npy")
    )]

    output_image_dir = scene.iphone_data_dir / 'undistort_rgb'
    output_depth_dir = scene.iphone_data_dir / 'undistort_depth'
    undistort_frames(
        frames,
        image_dir,
        depth_dir,
        output_image_dir,
        output_depth_dir,
        map1,
        map2)

    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='matrixcity')
    parser.add_argument('--output_path', type=str, default='matrixcity')
    parser.add_argument('--num_workers', type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    num_workers = args.num_workers
    scene_ids = []

    scene_ids += read_txt_list(Path(data_path) / 'splits' / 'nvs_sem_train.txt')
    scene_ids += read_txt_list(Path(data_path) / 'splits' / 'nvs_sem_val.txt')
    # for sid in tqdm(scene_ids):
    # #     process_scene(sid, data_path)
    print(f"Processing {len(scene_ids)} scenes with {args.num_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_scene, sid, data_path) for sid in scene_ids]

        for future in tqdm(as_completed(futures), total=len(scene_ids), desc="Total Scenes"):
            result = future.result()

if __name__ == "__main__":
    main()
