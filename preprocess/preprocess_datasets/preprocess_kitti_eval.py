import os
import shutil
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    tgt_image_path = os.path.join(output_path, 'image')
    os.makedirs(tgt_image_path, exist_ok=True)
    scenes = os.listdir(os.path.join(data_path, 'val'))
    for scene in scenes:
        image_folder = os.path.join(data_path, scene[:10], scene)
        shutil.copytree(image_folder, os.path.join(tgt_image_path, scene))
    tgt_depth_path = os.path.join(output_path, 'depth')
    os.makedirs(tgt_depth_path, exist_ok=True)
    shutil.copytree(os.path.join(data_path, 'val'), os.path.join(tgt_depth_path))

if __name__ == "__main__":
    main()
