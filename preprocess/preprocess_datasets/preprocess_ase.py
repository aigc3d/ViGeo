import os
import cv2
import json
import trimesh
from PIL import Image, ImageOps
from tqdm import tqdm
import imageio.v3 as iio
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from projectaria_tools.projects import ase
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core import calibration
from multiprocessing import Pool, cpu_count
import traceback

tgt_res = 512
tgt_foc = 300
fps = 10

def devignetting_official(src_img, vignetting_mask):
    vignetting_mask = vignetting_mask.astype(np.float32) / 255.0
    zero_mask = vignetting_mask == 0
    vignetting_mask[zero_mask] = 1
    vignetting_mask = pow(vignetting_mask, 1/2.2)
    corrected_img = src_img / vignetting_mask
    corrected_img[zero_mask] = 0
    corrected_img[corrected_img > 255] = 255
    corrected_img = corrected_img.astype(np.uint8)
    return corrected_img

def devignette_image(rgb_origin, vignette_mask, mode=1):
    if mode == 0:
        vignette_mask = np.array(vignette_mask)[:, :, [2, 1, 0]].astype(np.uint8)
        alpha = 1.0; beta = 1.0; gamma = 1.0
        rgb_devignette = cv2.addWeighted(rgb_origin, alpha, vignette_mask, beta, gamma)
    elif mode == 1:
        vignette_mask = vignette_mask.convert('L')
        inverted_mask = ImageOps.invert(vignette_mask)
        inverted_mask_array = np.array(inverted_mask).astype(np.float32) / 255.0
        zero_mask = inverted_mask_array == 0
        inverted_mask_array[zero_mask] = 1.0
        rgb_devignette = rgb_origin.astype(np.float32) / inverted_mask_array[:, :, None]
        rgb_devignette[zero_mask] = 0
        rgb_devignette = np.clip(rgb_devignette, 0, 255)
    elif mode == 2:
        vignette_mask = np.array(vignette_mask)[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        rgb_devignette = np.clip(rgb_origin.astype(np.float32) * vignette_mask, 0, 255.0)
    elif mode == 3:
        vignette_mask = 255.0 - np.array(vignette_mask)[:, :, 0:3]
        rgb_devignette = devignetting_official(rgb_origin.astype(np.float32), vignette_mask)
    return rgb_devignette.astype(np.uint8)

def read_language_file(filepath):
    assert os.path.exists(filepath), f"Could not find language file: {filepath}"
    with open(filepath, "r") as f:
        entities = []
        for line in f.readlines():
            line = line.rstrip()
            entries = line.split(", ")
            command = entries[0]
            entity_parameters = {}
            for parameter_def in entries[1:]:
                key, value = parameter_def.split("=")
                entity_parameters[key] = float(value)
            entities.append((command, entity_parameters))
    print(f"Loaded scene commands with a total of {len(entities)} entities.")
    return entities

def _transform_from_Rt(R, t):
    M = np.identity(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def _read_trajectory_line(line):
    line = line.rstrip().split(",")
    pose = {}
    pose["timestamp"] = int(line[1])
    translation = np.array([float(p) for p in line[3:6]])
    quat_xyzw = np.array([float(o) for o in line[6:10]])
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()
    pose["position"] = translation
    pose["rotation"] = rot_matrix
    pose["transform"] = _transform_from_Rt(rot_matrix, translation)
    return pose

def distance_to_depth(K, dist, uv=None):
    if uv is None and len(dist.shape) >= 2:
        uv = np.stack(np.meshgrid(np.arange(dist.shape[1]), np.arange(dist.shape[0])), -1)
        uv = uv.reshape(-1, 2)
        dist = dist.reshape(-1)
    if isinstance(dist, np.ndarray):
        uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1).T
        temp_point = np.linalg.inv(K) @ uvh
        z = dist / np.linalg.norm(temp_point.T, axis=1)
        return z.reshape(dist.shape)
    else:
        import torch
        uvh = torch.cat([uv, torch.ones(len(uv), 1).to(uv)], -1)
        temp_point = torch.inverse(K) @ uvh.T
        z = dist / torch.linalg.norm(temp_point.T, dim=1)
        return z.reshape(dist.shape)

def read_trajectory_file(filepath):
    assert os.path.exists(filepath), f"Could not find trajectory file: {filepath}"
    with open(filepath, "r") as f:
        _ = f.readline()
        positions, rotations, transforms, timestamps = [], [], [], []
        for line in f.readlines():
            pose = _read_trajectory_line(line)
            positions.append(pose["position"])
            rotations.append(pose["rotation"])
            transforms.append(pose["transform"])
            timestamps.append(pose["timestamp"])
        positions = np.stack(positions)
        rotations = np.stack(rotations)
        transforms = np.stack(transforms)
        timestamps = np.array(timestamps)
    print(f"Loaded trajectory with {len(timestamps)} device poses.")
    return {
        "ts": positions,
        "Rs": rotations,
        "Ts_world_from_device": transforms,
        "timestamps": timestamps,
    }

def read_points_file(filepath):
    assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
    df = pd.read_csv(filepath, compression="gzip")
    point_cloud = df[["px_world", "py_world", "pz_world"]]
    print(f"Loaded point cloud with {len(point_cloud)} points.")
    return point_cloud.to_numpy()


def process_scene(scene_info):
    scene, data_path, output_path, vignette_mask_path = scene_info
    try:
        image_path = os.path.join(data_path, scene, 'rgb')
        depth_path = os.path.join(data_path, scene, 'depth')
        trajectory_path = os.path.join(data_path, scene, 'trajectory.csv')

        if not os.path.exists(os.path.join(output_path, scene)):
            print(f"[SKIP] Scene {scene}")
            return False

        if not (os.path.exists(image_path) and os.path.exists(depth_path) and os.path.exists(trajectory_path)):
            print(f"[SKIP] Scene {scene}: missing input data")
            return False

        trajectory = read_trajectory_file(trajectory_path)

        rgb_files = sorted([f for f in os.listdir(image_path) if f.endswith(".jpg")])
        num_frames = len(rgb_files)
        if num_frames < 100:
            print(f"[SKIP] Scene {scene}: only {num_frames} frames (<100)")
            return False
        if len(trajectory["Ts_world_from_device"]) != num_frames:
            print(f"[SKIP] Scene {scene}: trajectory length mismatch")
            return False

        # Calibration setup
        camera_label = "camera-rgb"
        src_calib = ase.get_ase_rgb_calibration()
        tgt_calib = calibration.get_linear_camera_calibration(
            tgt_res, tgt_res, tgt_foc, camera_label,
            src_calib.get_transform_device_camera()
        )
        pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(tgt_calib)
        principal = pinhole_cw90.get_principal_point()
        focal_length = pinhole_cw90.get_focal_lengths()
        cx, cy = principal[0], principal[1]
        fx, fy = focal_length
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

        vignette_mask = Image.open(vignette_mask_path)

        out_dir = os.path.join(output_path, scene)
        out_depth_path = os.path.join(out_dir, "depth")
        out_color_path = os.path.join(out_dir, "rgb")
        os.makedirs(out_depth_path, exist_ok=True)
        os.makedirs(out_color_path, exist_ok=True)

        all_cameras = {}
        poses = []
        for frame_idx in range(num_frames):
            frame_id = str(frame_idx).zfill(7)
            rgb_file = os.path.join(image_path, f"vignette{frame_id}.jpg")
            depth_file = os.path.join(depth_path, f"depth{frame_id}.png")

            # Read depth
            depth = np.array(Image.open(depth_file))
            if len(depth.shape) >= 3:
                depth = depth[:, :, 0]
            max_val = np.iinfo(depth.dtype).max
            depth_mask = (depth < max_val) & (~np.isnan(depth))
            if depth_mask.sum() <= 0.1 * tgt_res * tgt_res:
                print(f"[WARN] Scene {scene}, frame {frame_id}: too few valid depth pixels")
            depth = depth.astype(np.float32)
            depth[~depth_mask] = 0.0

            # Read RGB
            bgr = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)[:, :, :3]
            bgr_devignette = devignette_image(bgr, vignette_mask, mode=1)

            # Undistort
            rectified_bgr = calibration.distort_by_calibration(
                bgr_devignette.copy(order='C'), tgt_calib, src_calib, InterpolationMethod.BILINEAR
            )
            rectified_depth = calibration.distort_by_calibration(
                depth.copy(order='C'), tgt_calib, src_calib, InterpolationMethod.NEAREST_NEIGHBOR
            )

            # Rotate CW90 (k=3 => 270° CCW = 90° CW)
            rotated_image = np.rot90(rectified_bgr, k=3).astype(np.uint8)
            rotated_depth = np.rot90(rectified_depth, k=3)

            # Convert distance to depth
            rotated_depth_Z = distance_to_depth(K, rotated_depth).reshape((rotated_depth.shape[0], rotated_depth.shape[1]))

            # Save
            cv2.imwrite(os.path.join(out_color_path, f"rgb_{frame_id}.jpg"), rotated_image)
            cv2.imwrite(os.path.join(out_depth_path, f"depth_{frame_id}.png"), rotated_depth_Z.astype(np.uint16))

            # Camera pose
            T_world_from_device = trajectory["Ts_world_from_device"][frame_idx]
            T_Device_Camera = pinhole_cw90.get_transform_device_camera().to_matrix()
            pose_cam2world = T_world_from_device @ T_Device_Camera
            poses.append(pose_cam2world)
            pose_world2cam = np.linalg.inv(pose_cam2world)
            frame_extrinsics = pose_world2cam[:3, :4].reshape(-1).tolist()
            all_cameras[frame_id] = {'poses_w2c': frame_extrinsics}

        # Save meta
        meta_info_dict = {
            'H': tgt_res,
            'W': tgt_res,
            'fps': fps,
            'num_frames': num_frames,
            'num_depths': num_frames,
            'intrinsics': K.reshape(-1).tolist(),
            'camera_poses': all_cameras,
        }
        with open(os.path.join(out_dir, "meta_info.json"), 'w') as fp:
            json.dump(meta_info_dict, fp, indent=4)   
        # poses = np.stack(poses)
        # np.savez(os.path.join(out_dir, "poses.npz"), poses=poses, intrinsics=K)


        print(f"[DONE] Scene {scene}")
        return True

    except Exception as e:
        print(f"[ERROR] Scene {scene}: {e}")
        traceback.print_exc()
        return False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='ase')
    parser.add_argument('--output_path', type=str, default='ase')
    parser.add_argument('--vignette_mask', type=str, default='./preprocess/vignette.png')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path
    vignette_mask_path = args.vignette_mask
    num_workers = min(args.num_workers, cpu_count())

    scenes = sorted([s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))])

    # Prepare arguments for each process
    scene_args = [(scene, data_path, output_path, vignette_mask_path) for scene in scenes]

    print(f"Processing {len(scenes)} scenes with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_scene, scene_args), total=len(scenes)))

    success_count = sum(results)
    print(f"\nCompleted: {success_count}/{len(scenes)} scenes processed successfully.")

if __name__ == "__main__":
    main()
