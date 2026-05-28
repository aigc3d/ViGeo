import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path
    scenes = os.listdir(os.path.join(data_path))
    scenes.sort()
    data = []
    camera = "dslr"
    
    for scene in tqdm(scenes):
        files = [f[:-4] for f in os.listdir(os.path.join(data_path, scene, camera, "undistorted_images"))]
        files.sort()
        image_files = [os.path.join(scene, camera, "undistorted_images", f + ".JPG") for f in files]
        depth_files = [os.path.join(scene, camera, "undistorted_depth", f + ".npy") for f in files]
        calib_file = os.path.join(scene, camera, 'cam.npz')
        assert os.path.exists(os.path.join(data_path, calib_file))
        if len(image_files) < 32:
            print(f"Skipping {scene} because it has only {len(image_files)} frames, less than 32.")
            continue

        data.append(
            {'scene': scene + '-' + camera,
            'calib': calib_file,
            'image': image_files,
            'depth': depth_files})
    
    camera = "iphone"
    for scene in tqdm(scenes):
        if scene in ['08bd80ce2a']:
            continue

        files = [f[:-4] for f in os.listdir(os.path.join(data_path, scene, camera, "undistort_rgb"))]
        files.sort()
        image_files = [os.path.join(scene, camera, "undistort_rgb", f + ".jpg") for f in files]
        depth_files = [os.path.join(scene, camera, "undistort_depth", f + ".npy") for f in files]
        calib_file = os.path.join(scene, camera, 'cam.npz')
        assert os.path.exists(os.path.join(data_path, calib_file))
        cam_data = np.load(os.path.join(data_path, calib_file))
        pose = cam_data['pose']
        assert pose.shape[0] == len(image_files)
        if len(image_files) < 32:
            print(f"Skipping {scene} because it has only {len(image_files)} frames, less than 32.")
            continue

        data.append(
            {'scene': scene + '-' + camera,
            'calib': calib_file,
            'image': image_files,
            'depth': depth_files})

    
    print(f"scannetpp has {len(data)} scenes")
    with open(os.path.join(output_path, 'scannetpp_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
