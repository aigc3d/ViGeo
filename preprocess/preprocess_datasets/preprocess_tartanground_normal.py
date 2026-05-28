import os
import cv2
import json
import numpy as np
import multiprocessing
from tqdm import tqdm
from argparse import ArgumentParser
import utils3d

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def process_task_direct_to_nas(task_args):
    depth_path, nas_save_path, seg_path, intrinsic, sky_label = task_args
    try:
        depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None: return False
        depth = depth_rgba.view("<f4")
        depth = np.squeeze(depth, axis=-1).copy()
        
        sky_mask = np.isnan(depth)
        if sky_label is not None and os.path.exists(seg_path):
            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if seg_img is not None:
                if len(seg_img.shape) >= 3: seg_img = seg_img[:, ..., 0]
                sky_mask |= (seg_img == sky_label)
        else:
            sky_mask |= (depth > 1000.0)
            
        depth[sky_mask] = 0
        valid_mask = (depth > 0)
        normal, _ = utils3d.np.depth_map_to_normal_map(depth, intrinsic, mask=valid_mask)
        normal_rgb = ((normal + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(nas_save_path, normal_bgr)
        return True
    except Exception as e:
        return f"Err: {str(e)}"

def run_pipeline(args):
    root_path = os.path.abspath(args.data_path)
    intrinsic_mtx = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], dtype=np.float32)

    scenes = [s for s in (args.scenes or os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, s))]
    
    print(f"[*] 正在快速扫描目录结构...")
    all_p_tasks = []
    for scene_name in scenes:
        scene_dir = os.path.join(root_path, scene_name)
        for dtype in ['Data_diff', 'Data_omni', 'Data_anymal']:
            dtype_path = os.path.join(scene_dir, dtype)
            if not os.path.isdir(dtype_path): continue
            ps = [p for p in os.listdir(dtype_path) if p.startswith('P') and os.path.isdir(os.path.join(dtype_path, p))]
            for p_name in ps:
                all_p_tasks.append((scene_name, dtype, p_name))

    print(f"[*] 发现 {len(all_p_tasks)} 个 P 文件夹，开始处理...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for scene_name, dtype, p_name in tqdm(all_p_tasks, desc="总进度 (P 文件夹)"):
            
            scene_dir = os.path.join(root_path, scene_name)
            dtype_path = os.path.join(scene_dir, dtype)
            p_dir = os.path.join(dtype_path, p_name)

            sky_label = None
            json_p = os.path.join(scene_dir, 'seg_label_map.json')
            if os.path.exists(json_p):
                try:
                    with open(json_p, 'r') as f: 
                        sky_label = int(json.load(f)['name_map']['sky'])
                except: pass

            tasks_in_group = []
            depth_items = [d for d in os.listdir(p_dir) if d.startswith('depth_') and os.path.isdir(os.path.join(p_dir, d))]

            for d_name in depth_items:
                d_dir = os.path.join(p_dir, d_name)
                suffix = d_name.replace('depth_', '')
                nas_n_dir = os.path.join(args.nas_out, scene_name, dtype, p_name, f"png_normal_{suffix}")
                
                os.makedirs(nas_n_dir, exist_ok=True)
                nas_exists = set(os.listdir(nas_n_dir)) if os.path.exists(nas_n_dir) else set()
                
                for img_name in os.listdir(d_dir):
                    if not img_name.endswith('_depth.png'): continue
                    save_name = img_name.replace('_depth.png', '_normal.png')
                    
                    if save_name not in nas_exists:
                        tasks_in_group.append((
                            os.path.join(d_dir, img_name),
                            os.path.join(nas_n_dir, save_name),
                            os.path.join(p_dir, f"seg_{suffix}", img_name.replace('_depth.png', '_seg.png')),
                            intrinsic_mtx,
                            sky_label
                        ))

            if tasks_in_group:
                list(tqdm(pool.imap_unordered(process_task_direct_to_nas, tasks_in_group, chunksize=16), 
                          total=len(tasks_in_group), 
                          desc=f"  └─ {scene_name[:10]}.../{p_name}", 
                          leave=False))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--nas_out', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--scenes', type=str, nargs='+', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    run_pipeline(args)

if __name__ == "__main__":
    main()
