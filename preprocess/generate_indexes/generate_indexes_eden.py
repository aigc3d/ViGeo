import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

def group_sequences(filenames):
    import re
    from collections import defaultdict

    groups = defaultdict(list)
    pattern = re.compile(r'^([AENSW])_(\d+)_([L])\.png$')

    for f in filenames:
        match = pattern.match(f)
        if match:
            prefix, num_str, side = match.groups()
            groups[(prefix, side)].append((int(num_str), f))
    
    result = []
    for (prefix, side), items in sorted(groups.items()):
        sorted_files = [f for _, f in sorted(items)]
        result.append(sorted_files)
    
    return result
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./eden')
    parser.add_argument('--output_path', type=str, default='train_test_split')

    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    output_path = args.output_path

    scenes = os.listdir(os.path.join(data_path, 'RGB'))
    data = []
    for scene in tqdm(scenes):
        weathers = os.listdir(
            os.path.join(data_path, 'RGB', scene)
        )
        for weather in weathers:
            files = os.listdir(os.path.join(data_path, 'RGB', scene, weather))
            group_files = group_sequences(files)
            for group in group_files:
                image_files = [os.path.join('RGB', scene, weather, file) for file in group]
                depth_files = [os.path.join('Depth', scene, weather, file.replace('.png', '.mat')) for file in group]
                print(group[0])
                camera_files = [os.path.join('cam_matrix', scene, weather, file[:-6] + '.mat') for file in group]
                segmentation_files = [os.path.join('Segmentation', scene, weather, file.replace('.png', '_Inst.png')) for file in group]
            
                if len(image_files) < 32:
                    print(f"Skipping {scene} {weather} because it has only {len(image_files)} frames, less than 32.")
                    continue
                data.append(
                    {
                        'image': image_files,
                        'depth': depth_files,
                        'calib': camera_files,
                        'segmentation': segmentation_files
                    }
                )
    print(f"eden has {len(data)} scenes")
    with open(os.path.join(output_path, 'eden_train.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
