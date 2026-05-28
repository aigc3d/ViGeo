import os
import cv2
import yaml
import json
import renderpy
import os.path as osp
import numpy as np
from tqdm import tqdm
from pathlib import Path
from munch import Munch
from copy import deepcopy
from argparse import ArgumentParser
from colmap import read_model
import multiprocessing
from scannetpp import (
    ScannetppScene_Release,
    read_txt_list,
    load_yaml_munch,
    load_json,
    pose_from_qwxyz_txyz
)

def compute_undistort_intrinsic(K, height, width, distortion_params):
    assert len(distortion_params.shape) == 1
    assert distortion_params.shape[0] == 4  # OPENCV_FISHEYE has k1, k2, k3, k4

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        distortion_params,
        (width, height),
        np.eye(3),
        balance=0.0,
    )
    # Make the cx and cy to be the center of the image
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0
    return new_K


def undistort_frames(
    frames,
    K,
    height,
    width,
    distortion_params,
    input_image_dir,
    input_mask_dir,
    input_depth_dir,
    out_image_dir,
    out_mask_dir,
    out_depth_dir,
):
    new_K = compute_undistort_intrinsic(K, height, width, distortion_params)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, distortion_params, np.eye(3), new_K, (width, height), cv2.CV_32FC1
    )

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame["file_path"]
        image = cv2.imread(str(image_path))
        undistorted_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out_image_path = Path(out_image_dir) / frame["file_path"]
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image_path), undistorted_image)

    #     [Optional]: Mask
    #     mask_path = Path(input_mask_dir) / frame["mask_path"]
    #     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    #     if np.all(mask > 0):
    #         # No invalid pixels. Just use empty mask
    #         undistorted_mask = np.zeros((height, width), dtype=np.uint8) + 255
    #     else:
    #         undistorted_mask = cv2.remap(
    #             mask,
    #             map1,
    #             map2,
    #             interpolation=cv2.INTER_LINEAR,
    #             borderMode=cv2.BORDER_CONSTANT,
    #             borderValue=255,
    #         )
    #         # Filter the mask valid: 255, invalid: 0
    #         undistorted_mask[undistorted_mask < 255] = 0
        
    #     out_mask_path = Path(out_mask_dir) / frame["mask_path"]
    #     out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    #     cv2.imwrite(str(out_mask_path), undistorted_mask)

        # Depth

        file_name = frame["file_path"].replace('.JPG', '.npy') # Depth map is png format
        depth_path = Path(input_depth_dir) / file_name
        depth = np.load(str(depth_path))
        undistorted_depth = cv2.remap(
            depth,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST, # Use nearest neighbor interpolation
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0 # Assuming 20.0 is an appropriate border value for missing depth information
        )
        out_depth_path = Path(out_depth_dir) / file_name
        out_depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_depth_path), undistorted_depth)
        
    return new_K

def update_transforms_json(transforms, new_K, new_height, new_width):
    new_transforms = deepcopy(transforms)
    new_transforms["h"] = new_height
    new_transforms["w"] = new_width
    new_transforms["fl_x"] = new_K[0, 0]
    new_transforms["fl_y"] = new_K[1, 1]
    new_transforms["cx"] = new_K[0, 2]
    new_transforms["cy"] = new_K[1, 2]
    # The undistortion will be PINHOLE and have no distortion paramaters
    new_transforms["camera_model"] = "PINHOLE"
    for key in ("k1", "k2", "k3", "k4"):
        if key in new_transforms:
            new_transforms[key] = 0.0
    return new_transforms

def process_scene(scene_id, data_path):
    scene = ScannetppScene_Release(scene_id, data_root=Path(data_path) / "data")
    input_image_dir = 'resized_images'
    input_image_dir = scene.dslr_dir / input_image_dir

    input_mask_dir = 'resized_anon_masks'
    input_mask_dir = scene.dslr_dir / input_mask_dir

    input_depth_dir = 'render_depth'
    input_depth_dir = scene.dslr_dir / input_depth_dir

    input_transforms_path = 'nerfstudio/transforms.json'
    input_transforms_path = scene.dslr_dir / input_transforms_path

    out_image_dir = 'undistorted_images'
    out_mask_dir = 'undistorted_anon_masks'
    out_depth_dir = 'undistorted_depth'
    out_transforms_path = 'nerfstudio/transforms_undistorted.json'

    out_image_dir = scene.dslr_dir / out_image_dir
    out_mask_dir = scene.dslr_dir / out_mask_dir
    out_depth_dir = scene.dslr_dir / out_depth_dir
    out_transforms_path = scene.dslr_dir / out_transforms_path

    transforms = load_json(input_transforms_path)
    assert len(transforms["frames"]) > 0
    frames = deepcopy(transforms["frames"])
    if "test_frames" not in transforms:
        print(f"{scene_id} has no test split")
    else:
        assert len(transforms["test_frames"]) > 0
        frames += transforms["test_frames"]
        
    height = int(transforms["h"])
    width = int(transforms["w"])
    distortion_params = np.array(
        [
            float(transforms["k1"]),
            float(transforms["k2"]),
            float(transforms["k3"]),
            float(transforms["k4"]),
        ]
    )
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )

    new_K = undistort_frames(
        frames,
        K,
        height,
        width,
        distortion_params,
        input_image_dir,
        input_mask_dir,
        input_depth_dir,
        out_image_dir,
        out_mask_dir,
        out_depth_dir)
    
    colmap_dir = scene.dslr_dir / 'colmap'
    with open(osp.join(colmap_dir, "images.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]  # skip header
        raw = raw[0::2]
        raw.sort(key=lambda x: x.split()[-1])
    
    poses = []

    for item in raw:
        poses.append(pose_from_qwxyz_txyz(item.split(" ")[1:-2]))
    poses = np.stack(poses)
    np.savez(str(scene.dslr_dir / 'cam.npz'), pose=poses, intrinsic=new_K)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='matrixcity')
    parser.add_argument('--output_path', type=str, default='matrixcity')
    parser.add_argument('--num_workers', type=int, default=12)
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    num_workers = args.num_workers
    scene_ids = []

    scene_ids += read_txt_list(Path(data_path) / 'splits' / 'nvs_sem_train.txt')
    scene_ids += read_txt_list(Path(data_path) / 'splits' / 'nvs_sem_val.txt')

    for scene_id in tqdm(scene_ids, desc="scene"):
        process_scene(scene_id, data_path)

if __name__ == "__main__":
    main()
