import os
import h5py
import json
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

    
    scenes = [f for f in os.listdir(data_path) if not f.endswith('.h5') and not f.endswith('.pkl')]

    
    invalid_dict = {}
    with h5py.File(os.path.join(data_path, 'invalid_files.h5'), "r") as h5f:
        for scene in h5f:
            h5_data = h5f[scene]["invalid_pairs"][:]
            invalid_pairs = set(
                tuple(pair.decode("utf-8").split("_")) for pair in h5_data
            )
            invalid_dict[scene] = invalid_pairs
    
    data = []
    for scene in tqdm(scenes):
        if len(invalid_dict[scene]) > 0:
            continue

        if scene in ['segment-4898453812993984151_199_000_219_000_with_camera_labels.tfrecord']:
            continue
        
        for camera in ['1', '2', '3', '4', '5']:
            files = [f[:-4] for f in os.listdir(os.path.join(data_path, scene)) if f.endswith(camera + '.jpg')]
            files.sort()

            image_files = [os.path.join(scene, f + '.jpg') for f in files]
            depth_files = [os.path.join(scene, f + '.exr') for f in files]
            calib_files = [os.path.join(scene, f + '.npz') for f in files]

            data.append({
                "scene": scene + '-' + camera,
                "image": image_files,
                "depth": depth_files,
                "calib": calib_files
            })

    
    print(f"waymo has {len(data)} scenes")
    with open(os.path.join(output_path, 'waymo_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
