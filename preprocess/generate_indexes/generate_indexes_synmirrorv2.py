import os
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

    tgt_folders = ['abo_v4', 'abo_v4_multiple', 'hf-objaverse-v4']

    data = []
    
    for folder in tgt_folders:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            continue

        # All target folders now share the same two-level directory structure
        intermediate_dirs = os.listdir(folder_path)
        intermediate_dirs.sort()
        
        # Added desc=folder to tqdm to show which dataset is currently processing
        for intermediate_dir in tqdm(intermediate_dirs, desc=folder):
            intermediate_path = os.path.join(folder_path, intermediate_dir)
            if not os.path.isdir(intermediate_path):
                continue

            scenes = os.listdir(intermediate_path)
            scenes.sort()
            
            for scene in scenes:
                scene_path = os.path.join(intermediate_path, scene)
                if not os.path.isdir(scene_path):
                    continue

                hdf5_files = [f for f in os.listdir(scene_path) if f.endswith('.hdf5')]
                
                # Skip if no hdf5 files are found in the directory
                if len(hdf5_files) == 0:
                    continue

                # Sort by the number in the filename to prevent 10.hdf5 from appearing before 2.hdf5
                hdf5_files.sort(key=lambda x: int(x.split('.')[0]))
                hdf5_files = [os.path.join(folder, intermediate_dir, scene, f) for f in hdf5_files]

                data.append(
                    {
                        'dataset': folder,
                        'scene': f"{intermediate_dir}/{scene}",
                        'hdf5_files': hdf5_files,
                    }
                )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Synmirrorv2 processing complete: {len(data)} scenes found.")
    with open(os.path.join(output_path, 'dataset_eval.json'), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
