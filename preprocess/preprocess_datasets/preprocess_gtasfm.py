import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    for split in ['train', 'test']:
        if not os.path.exists(os.path.join(args.output_path, split)):
            os.makedirs(os.path.join(args.output_path, split))
        
        img_dir = os.path.join(args.output_path, split, 'image')
        os.makedirs(img_dir, exist_ok=True)
        depth_dir = os.path.join(args.output_path, split, 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        calib_dir = os.path.join(args.output_path, split, 'calib')
        os.makedirs(calib_dir, exist_ok=True)

        files = os.listdir(os.path.join(data_path, split))
        for file in tqdm(files):
            h5file = h5py.File(os.path.join(data_path, split, file),"r")
            img_num = len(h5file.keys()) // 4
            for i in tqdm(range(img_num)):
                img_name = "image_%d"%i
                K_name = "K_%d"%i
                pose_name = "pose_%d"%i
                depth_name = "depth_%d"%i

                img = cv2.imdecode(h5file[img_name][:],cv2.IMREAD_COLOR)
                K = h5file[K_name][:]
                pose = h5file[pose_name][:]
                depth = h5file[depth_name][:]

                calib_dict = {'pose': pose, 'intrinsic': K}

                os.makedirs(os.path.join(img_dir, file.replace('.hdf5', '')), exist_ok=True)
                os.makedirs(os.path.join(depth_dir, file.replace('.hdf5', '')), exist_ok=True)
                os.makedirs(os.path.join(calib_dir, file.replace('.hdf5', '')), exist_ok=True)
                cv2.imwrite(os.path.join(img_dir, file.replace('.hdf5', ''), "image_%04d"%i + '.png'), img)
                np.save(os.path.join(depth_dir, file.replace('.hdf5', ''), "depth_%04d"%i + '.npy'), depth)
                np.save(os.path.join(calib_dir, file.replace('.hdf5', ''), "calib_%04d"%i + '.npy'), calib_dict)

if __name__ == "__main__":
    main()
