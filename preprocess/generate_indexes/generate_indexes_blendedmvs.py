import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='train_test_split')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)
    
    scenes = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    scenes.sort()
    
    data = []
    for scene in tqdm(scenes, desc="处理场景中"):
        scene_path = os.path.join(data_path, scene)
        
        files = [f[:-4] for f in os.listdir(scene_path) if f.endswith('.jpg')]
        files.sort()

        if len(files) == 0:
            continue

        if len(files) < 24:
            continue

        image_files = [os.path.join(scene, f + '.jpg') for f in files]
        depth_files = [os.path.join(scene, f + '.exr') for f in files]
        calib_files = [os.path.join(scene, f + '.npz') for f in files]
    
        data.append({
            "scene": scene,
            "image": image_files,
            "depth": depth_files,
            "calib": calib_files
        })

    print(f"BlendedMVS processing complete. Found {len(data)} valid scenes.")
    
    # Export to JSON
    out_file = os.path.join(output_path, 'blendedmvs_train.json')
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Configuration saved to: {out_file}")

if __name__ == "__main__":
    main()
