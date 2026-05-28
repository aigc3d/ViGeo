import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='omniobject')
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = []
    # Using scale.json to get the list of scenes
    scale_json_path = os.path.join(data_path, 'scale.json')
    if not os.path.exists(scale_json_path):
        print(f"Error: {scale_json_path} not found.")
        exit()

    with open(scale_json_path, 'r') as f:
        scenes = json.load(f)

    for scene in tqdm(scenes):
        render_dir = os.path.join(data_path, scene, 'render')
        img_dir = os.path.join(render_dir, 'images')
        dep_dir = os.path.join(render_dir, 'depths')
        norm_dir = os.path.join(render_dir, 'normal') # Using 'normal' as requested
        calib_file = os.path.join(scene, 'render/cam.npz')

        if not all(os.path.isdir(os.path.join(data_path, d)) for d in [img_dir, dep_dir, norm_dir]):
            continue

        # Fast lookup using sets
        img_files_raw = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        depth_set = set(os.listdir(dep_dir))
        normal_set = set(os.listdir(norm_dir))
        
        image_files = []
        depth_files = []
        normal_files = []

        # OmniObject3D uses 'r_{i}.png' naming convention
        for i in range(len(img_files_raw)):
            target_img = f"r_{i}.png"
            target_depth = f"r_{i}_depth.exr"
            target_normal = f"r_{i}_normal.npy" # Changed from .png to .npy based on image

            if target_img in img_files_raw and \
               target_depth in depth_set and \
               target_normal in normal_set:
                
                image_files.append(os.path.join(scene, 'render/images', target_img))
                depth_files.append(os.path.join(scene, 'render/depths', target_depth))
                normal_files.append(os.path.join(scene, 'render/normal', target_normal))

        if len(image_files) < 24:
            continue

        data.append({
            "scene": scene,
            "image": image_files,
            "depth": depth_files,
            "normal": normal_files,
            "calib": calib_file
        })

    print(f"OmniObject3D processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'omniobject_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
