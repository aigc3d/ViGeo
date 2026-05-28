import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dynamic_replica')
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    scenes = sorted(os.listdir(data_path))
    data = []

    for scene in tqdm(scenes):
        if scene.endswith('.jgz'):
            continue
            
        scene_dir = os.path.join(data_path, scene)
        img_dir = os.path.join(scene_dir, 'images')
        dep_dir = os.path.join(scene_dir, 'depths')
        norm_dir = os.path.join(scene_dir, 'normals')

        if not all(os.path.isdir(d) for d in [img_dir, dep_dir, norm_dir]):
            continue

        # Use sets for faster lookup
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        depth_set = set(os.listdir(dep_dir))
        normal_set = set(os.listdir(norm_dir))
        
        image_list = []
        depth_list = []
        normal_list = []
        calib_file = os.path.join(scene, 'calib.npz')

        for f in img_files:
            # Extract last 4 digits as frame ID, e.g., '0077'
            frame_id = f.split('.')[0][-4:]
            
            # Dynamic Replica naming pattern:
            # Image:  scene-0077.png
            # Depth:  scene_0077.geometric.png
            # Normal: scene_0077.normal.npy
            target_img = f"{scene}-{frame_id}.png"
            target_depth = f"{scene}_{frame_id}.geometric.png"
            target_normal = f"{scene}_{frame_id}.normal.npy"

            # Check existence in memory sets
            if target_depth in depth_set and target_normal in normal_set:
                image_list.append(os.path.join(scene, 'images', target_img))
                depth_list.append(os.path.join(scene, 'depths', target_depth))
                normal_list.append(os.path.join(scene, 'normals', target_normal))

        if len(image_list) < 32:
            continue

        data.append({
            'image': image_list,
            'depth': depth_list,
            'normal': normal_list,
            'calib': calib_file,
        })
    
    print(f"Dynamic Replica complete: {len(data)} scenes processed.")
    with open(os.path.join(output_path, 'dynamic_replica_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
