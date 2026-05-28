import os
import yaml
import renderpy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from munch import Munch
from argparse import ArgumentParser
from colmap import read_model
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scannetpp import (
    ScannetppScene_Release,
    read_txt_list,
    load_yaml_munch
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='matrixcity')
    parser.add_argument('--output_path', type=str, default='matrixcity')
    parser.add_argument('--device', type=str, choices=["iphone", "dslr"])
    return parser.parse_args()

def render_scene(scene_id, data_path, device):
    scene = ScannetppScene_Release(scene_id, data_root=Path(data_path) / "data")
    render_engine = renderpy.Render()
    render_engine.setupMesh(str(scene.scan_mesh_path))

    if device == "dslr":
        cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
    elif device == "iphone":
        cameras, images, points3D = read_model(scene.iphone_colmap_dir, ".txt")
    else:
        raise NotImplementedError
    assert len(cameras) == 1, "Multiple cameras not supported"
    camera = next(iter(cameras.values()))

    fx, fy, cx, cy = camera.params[:4]
    params = camera.params[4:]
    camera_model = camera.model
    render_engine.setupCamera(
        camera.height, camera.width,
        fx, fy, cx, cy,
        camera_model,
        params,
    )

    near = 0.05
    far = 20.0

    depth_dir = Path(data_path) / "data" / scene_id / device / "render_depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image in tqdm(images.items()):
        depth_name = image.name.split('/')[-1].split(".")[0] + ".npy"
        if os.path.exists(depth_dir / depth_name):
            continue
        world_to_camera = image.world_to_camera
        rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
        rgb = rgb.astype(np.uint8)
        
        np.save(depth_dir / depth_name, depth)

def main():
    args = parse_args()
    num_workers = int(multiprocessing.cpu_count() * 0.75)
    print(f'Using {num_workers} workers')
    data_path = args.data_path
    output_path = args.output_path
    scene_ids = []

    scene_ids += read_txt_list(Path(data_path) / 'splits' / 'nvs_sem_train.txt')
    scene_ids += read_txt_list(Path(data_path) / 'splits' / 'nvs_sem_val.txt')

    # pool = multiprocessing.Pool(processes=num_workers)

    print(f"Processing {len(scene_ids)} scenes with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(render_scene, sid, data_path, args.device) for sid in scene_ids]

        for future in tqdm(as_completed(futures), total=len(scene_ids), desc="Total Scenes"):
            result = future.result()
    
    # for scene_id in tqdm(scene_ids, desc="scene"):
    #     render_scene(scene_id, data_path, args.device)
        # pool.apply_async(render_scene, args=(scene_id, data_path, args.device))
    
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    main()
