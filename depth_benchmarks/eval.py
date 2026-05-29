import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(PROJECT_ROOT), str(CURRENT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_defs import (
    DEPTH_BENCHMARK_DEFAULT_DATASETS,
    DEPTH_BENCHMARK_SUPPORTED_DATASETS,
    DEPTH_BENCHMARK_TASKS,
    DEPTH_METRICS,
    DEPTH_SUMMARY_COLUMNS,
    TASK_MONO_DEPTH,
    TASK_POINTMAP,
    TASK_VIDEO_DEPTH,
)
from dataset_io import depth_to_points, get_depth_benchmark_dataset_config, get_intrinsic
from eval_utils import format_metrics, make_summary_row, print_summary, write_summary_table
try:
    from .model_registry import (
        ALL_MODEL_NAMES,
        MONO_DEPTH_MODELS,
        POINTMAP_MODELS,
        VIDEO_DEPTH_MODELS,
        get_model,
    )
except ImportError:
    from model_registry import (
        ALL_MODEL_NAMES,
        MONO_DEPTH_MODELS,
        POINTMAP_MODELS,
        VIDEO_DEPTH_MODELS,
        get_model,
    )
from metric_func import (
    as_depth_stack,
    as_points_stack,
    compute_depth_metrics,
    compute_points_metrics,
    depth_meta_data,
)


TASKS = DEPTH_BENCHMARK_TASKS
TASK_MODELS = {
    TASK_VIDEO_DEPTH: VIDEO_DEPTH_MODELS,
    TASK_MONO_DEPTH: MONO_DEPTH_MODELS,
    TASK_POINTMAP: POINTMAP_MODELS,
}
SUMMARY_COLUMNS = DEPTH_SUMMARY_COLUMNS
OUTPUT_DIR = PROJECT_ROOT


def depth2disparity(depth):
    disp = torch.zeros_like(depth) if isinstance(depth, torch.Tensor) else np.zeros_like(depth)
    mask = depth > 0
    disp[mask] = 1.0 / depth[mask]
    return disp


# ==============================================================================
# 1. Geometry Helpers
# ==============================================================================


def fov_to_intrinsic(fov_x, fov_y, w, h):
    n = fov_x.shape[0]
    fx = w / (2 * torch.tan(fov_x / 2))
    fy = h / (2 * torch.tan(fov_y / 2))
    cx = torch.full_like(fx, w / 2)
    cy = torch.full_like(fy, h / 2)

    K = torch.zeros((n, 3, 3), device=fov_x.device, dtype=fov_x.dtype)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0
    return K


# ==============================================================================
# 3. Model Inference Adapters
# ==============================================================================

def run_video_depth_inference(model, model_name, img_files, depth_gt=None):
    with torch.no_grad():
        if model_name == 'da3':
            output = model.inference(img_files, ref_view_strategy="first")
            depth_pred = torch.from_numpy(output.depth).unsqueeze(1)

        elif model_name in ['flashdepth', 'pi3']:
            images_np = np.stack([np.array(Image.open(f), dtype=np.float32) / 255.0 for f in img_files])
            images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).cuda()
            if model_name == 'flashdepth':
                depth_pred = model.infer(images_tensor.unsqueeze(0)).squeeze(0)
            else:
                depth_pred = model(images_tensor)['depth']

        elif model_name == 'stream3r':
            from stream3r.models.components.utils.load_fn import load_and_preprocess_images
            images = load_and_preprocess_images(img_files).cuda()
            for i in range(images.shape[0]):
                predictions = model.forward_stream(images[i : i + 1])
            model.clear()
            depth_pred = predictions['depth'].squeeze(0).squeeze(-1)

        elif model_name == 'vggt':
            from vggt.utils.load_fn import load_and_preprocess_images
            images = load_and_preprocess_images(img_files).cuda()
            depth_pred = model(images)['depth'].squeeze(0).squeeze(-1)

        elif model_name == 'vggt_omega':
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            images = load_and_preprocess_images(img_files, image_resolution=512).cuda()
            depth_pred = model(images)['depth']
            if depth_pred.dim() > 3:
                depth_pred = depth_pred.squeeze(0).squeeze(-1)

        elif model_name in ['streamvggt', 'infinitevggt']:
            from streamvggt.utils.load_fn import load_and_preprocess_images
            images = load_and_preprocess_images(img_files).cuda()
            output = model.inference([images[i : i + 1] for i in range(images.shape[0])])
            depth_pred = output['depth'].squeeze(0).squeeze(-1)

        elif model_name == 'vda':
            image_np = np.stack([np.array(Image.open(f), dtype=np.float32) / 255.0 for f in img_files])
            depth_pred_np = model.infer_video_depth(image_np)
            depth_pred = torch.clamp(
                torch.from_numpy(depth_pred_np).unsqueeze(1),
                max=torch.max(depth_gt).item() if depth_gt is not None else None,
            )

        elif model_name == 'geometrycrafter':
            frames_tensor = torch.from_numpy(
                np.stack([np.array(Image.open(f), dtype=np.float32) / 255.0 for f in img_files])
            ).to(device='cuda', dtype=torch.float32).permute(0, 3, 1, 2)
            orig_h, orig_w = frames_tensor.shape[2], frames_tensor.shape[3]
            h_64, w_64 = (orig_h // 64) * 64, (orig_w // 64) * 64

            if orig_h != h_64 or orig_w != w_64:
                frames_tensor = F.interpolate(
                    frames_tensor, size=(h_64, w_64), mode='bicubic', antialias=True
                ).clamp(0, 1)

            rec_point_map, _ = model.pipe(
                frames_tensor,
                model.vae,
                model.prior,
                height=h_64,
                width=w_64,
                num_inference_steps=5,
                guidance_scale=1.0,
                window_size=110,
                decode_chunk_size=8,
                overlap=25,
                force_projection=True,
                force_fixed_focal=True,
                use_extract_interp=False,
                track_time=False,
                low_memory_usage=False,
            )

            if orig_h != h_64 or orig_w != w_64:
                rec_point_map = F.interpolate(
                    rec_point_map.permute(0, 3, 1, 2), size=(orig_h, orig_w), mode='bilinear'
                ).permute(0, 2, 3, 1)
            depth_pred = rec_point_map[..., 2].unsqueeze(1).float()

        elif model_name == 'depthcrafter':
            frames_tensor = torch.tensor(
                np.stack([np.array(Image.open(f)).astype(np.float32) / 255.0 for f in img_files])
            ).float().permute(0, 3, 1, 2)
            orig_h, orig_w = frames_tensor.shape[2], frames_tensor.shape[3]
            h_64, w_64 = round(orig_h / 64) * 64, round(orig_w / 64) * 64

            if orig_h != h_64 or orig_w != w_64:
                frames_tensor = F.interpolate(
                    frames_tensor, size=(h_64, w_64), mode='bicubic', antialias=True
                ).clamp(0, 1)

            res = model(
                frames_tensor.permute(0, 2, 3, 1).numpy(),
                height=h_64,
                width=w_64,
                output_type="np",
                guidance_scale=1.0,
                num_inference_steps=5,
                window_size=110,
                overlap=25,
                track_time=False,
            ).frames[0]
            depth_np = res.sum(-1) / res.shape[-1]
            depth_pred = torch.from_numpy(depth_np).unsqueeze(1).float().cuda()

            if orig_h != h_64 or orig_w != w_64:
                depth_pred = F.interpolate(depth2disparity(depth_pred), size=(orig_h, orig_w), mode='bilinear')

        else:
            raise ValueError(f"Unsupported video depth model: {model_name}")

    return as_depth_stack(depth_pred)


def run_mono_depth_inference(model, model_name, img_path, gt_for_clamp=None):
    with torch.no_grad():
        if model_name == 'da3':
            output = model.inference([img_path], ref_view_strategy="first")
            depth_pred = torch.from_numpy(output.depth).unsqueeze(1)

        elif model_name in ['flashdepth', 'pi3']:
            image = np.array(Image.open(img_path), dtype=np.float32) / 255.0
            images_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
            if model_name == 'flashdepth':
                depth_pred = model.infer(images_tensor.unsqueeze(0)).squeeze(0)
            else:
                depth_pred = model(images_tensor)['depth']

        elif model_name == 'stream3r':
            from stream3r.models.components.utils.load_fn import load_and_preprocess_images
            image = load_and_preprocess_images([img_path]).cuda()
            predictions = model.forward_stream(image)
            model.clear()
            depth_pred = predictions['depth'].squeeze(0).squeeze(-1)

        elif model_name == 'vggt':
            from vggt.utils.load_fn import load_and_preprocess_images
            image = load_and_preprocess_images([img_path]).cuda()
            depth_pred = model(image)['depth'].squeeze(0).squeeze(-1)

        elif model_name == 'vggt_omega':
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            image = load_and_preprocess_images([img_path], image_resolution=512).cuda()
            depth_pred = model(image)['depth']
            if depth_pred.dim() > 3:
                depth_pred = depth_pred.squeeze(0).squeeze(-1)

        elif model_name in ['streamvggt', 'infinitevggt']:
            from streamvggt.utils.load_fn import load_and_preprocess_images
            image = load_and_preprocess_images([img_path]).cuda()
            depth_pred = model.inference([image])['depth'].squeeze(0).squeeze(-1)

        elif model_name == 'vda':
            image_np = (np.array(Image.open(img_path), dtype=np.float32) / 255.0)[np.newaxis, ...]
            depth_pred = torch.from_numpy(model.infer_video_depth(image_np)).unsqueeze(1)
            if gt_for_clamp is not None:
                depth_pred = torch.clamp(depth_pred, max=torch.max(gt_for_clamp).item())

        elif model_name in ['geometrycrafter', 'depthcrafter']:
            img_raw = Image.open(img_path)
            w_orig, h_orig = img_raw.size
            img_t = torch.from_numpy(
                np.array(img_raw, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).cuda()
            h_64, w_64 = (h_orig // 64) * 64, (w_orig // 64) * 64
            img_t_resized = F.interpolate(
                img_t, size=(h_64, w_64), mode='bicubic', antialias=True
            ).clamp(0, 1)

            if model_name == 'geometrycrafter':
                res, _ = model.pipe(
                    img_t_resized, model.vae, model.prior,
                    height=h_64, width=w_64, num_inference_steps=5, window_size=1
                )
                depth_pred = res[..., 2].unsqueeze(1).float()
            else:
                frames_in = img_t_resized.permute(0, 2, 3, 1).cpu().numpy()
                res = model(
                    frames_in,
                    height=h_64,
                    width=w_64,
                    output_type="np",
                    num_inference_steps=5,
                    window_size=1,
                ).frames[0]
                depth_pred = depth2disparity(
                    torch.from_numpy(res.sum(-1) / res.shape[-1]).unsqueeze(1).float().cuda()
                )

            depth_pred = F.interpolate(depth_pred, size=(h_orig, w_orig), mode='bilinear', align_corners=True)

        else:
            raise ValueError(f"Unsupported mono depth model: {model_name}")

    return as_depth_stack(depth_pred)


def run_pointmap_inference(model, model_name, img_files):
    with torch.no_grad():
        if model_name == 'da3':
            output = model.inference(img_files, ref_view_strategy="first")
            depth_pred = torch.from_numpy(output.depth).cuda()
            intrinsics_pred = torch.from_numpy(output.intrinsics).cuda()
            points_pred = depth_to_points(depth_pred, intrinsics_pred)

        elif model_name == 'pi3':
            images_np = np.stack([np.array(Image.open(f), dtype=np.float32) / 255.0 for f in img_files])
            images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).cuda()
            points_pred = model(images_tensor)['local_points']
            if points_pred.shape[-1] != 3:
                points_pred = points_pred.permute(0, 2, 3, 1)

        elif model_name in ['vggt', 'vggt_omega', 'streamvggt', 'stream3r']:
            if model_name == 'stream3r':
                from stream3r.models.components.utils.load_fn import load_and_preprocess_images
                images = load_and_preprocess_images(img_files).cuda()
                for i in range(images.shape[0]):
                    output = model.forward_stream(images[i : i + 1])
                model.clear()
            elif model_name == 'vggt':
                from vggt.utils.load_fn import load_and_preprocess_images
                images = load_and_preprocess_images(img_files).cuda()
                output = model(images)
            elif model_name == 'vggt_omega':
                from vggt_omega.utils.load_fn import load_and_preprocess_images
                images = load_and_preprocess_images(img_files, image_resolution=512).cuda()
                output = model(images)
            else:
                from streamvggt.utils.load_fn import load_and_preprocess_images
                images = load_and_preprocess_images(img_files).cuda()
                output = model.inference([images[i : i + 1] for i in range(images.shape[0])])

            depth_pred = output['depth'].squeeze(-1)
            if depth_pred.ndim == 4:
                depth_pred = depth_pred.squeeze(0)

            if 'pose_enc' not in output:
                raise KeyError(f"{model_name} pointmap output must include pose_enc.")
            pose_enc = output['pose_enc']
            if pose_enc.ndim == 3:
                pose_enc = pose_enc.squeeze(0)

            fov_y, fov_x = pose_enc[..., -2], pose_enc[..., -1]
            n, h, w = depth_pred.shape
            K_pred = fov_to_intrinsic(fov_x, fov_y, w, h)
            points_pred = depth_to_points(depth_pred, K_pred)

        elif model_name == 'geometrycrafter':
            frames_tensor = torch.from_numpy(
                np.stack([np.array(Image.open(f), dtype=np.float32) / 255.0 for f in img_files])
            ).to(device='cuda', dtype=torch.float32).permute(0, 3, 1, 2)
            orig_h, orig_w = frames_tensor.shape[2], frames_tensor.shape[3]
            h_64, w_64 = (orig_h // 64) * 64, (orig_w // 64) * 64

            if orig_h != h_64 or orig_w != w_64:
                frames_tensor = F.interpolate(
                    frames_tensor, size=(h_64, w_64), mode='bicubic', antialias=True
                ).clamp(0, 1)

            rec_point_map, _ = model.pipe(
                frames_tensor,
                model.vae,
                model.prior,
                height=h_64,
                width=w_64,
                num_inference_steps=5,
                guidance_scale=1.0,
                window_size=110,
                decode_chunk_size=8,
                overlap=25,
                force_projection=True,
                force_fixed_focal=True,
                use_extract_interp=False,
                track_time=False,
                low_memory_usage=False,
            )

            if orig_h != h_64 or orig_w != w_64:
                rec_point_map = F.interpolate(
                    rec_point_map.permute(0, 3, 1, 2), size=(orig_h, orig_w), mode='bilinear'
                ).permute(0, 2, 3, 1)
            points_pred = rec_point_map

        else:
            raise ValueError(f"Unsupported pointmap model: {model_name}")

    return as_points_stack(points_pred)


# ==============================================================================
# 4. Scene Evaluation
# ==============================================================================

def evaluate_video_depth_scene(model, args, cfg, scene, ds_name, device):
    all_imgs, all_depths = cfg['get'](cfg['base'], scene)
    imgs = all_imgs[cfg['slice']]
    depth_files = all_depths[cfg['slice']]

    depth_gt = as_depth_stack(torch.stack([
        torch.from_numpy(cfg['read_d'](f))
        for f in depth_files
    ]))
    depth_pred = run_video_depth_inference(model, args.model_name, imgs, depth_gt)

    return compute_depth_metrics(
        depth_pred.to(device).float(),
        depth_gt.to(device).float(),
        metrics=DEPTH_METRICS,
        align_method=args.align_method,
        **depth_meta_data.get(ds_name.split('_')[0], {}),
    )


def evaluate_mono_depth_scene(model, args, cfg, scene, ds_name, device):
    all_imgs, all_depths = cfg['get'](cfg['base'], scene)
    imgs = all_imgs[cfg['slice']]
    depth_files = all_depths[cfg['slice']]

    scene_metrics = defaultdict(list)
    for img_path, depth_path in tqdm(list(zip(imgs, depth_files)), leave=False):
        gt_tensor = as_depth_stack(torch.from_numpy(cfg['read_d'](depth_path))).to(device).float()
        pred_tensor = run_mono_depth_inference(model, args.model_name, img_path, gt_tensor).to(device).float()

        metrics = compute_depth_metrics(
            pred_tensor,
            gt_tensor,
            metrics=DEPTH_METRICS,
            align_method=args.align_method,
            **depth_meta_data.get(ds_name.split('_')[0], {}),
        )
        if metrics:
            for key, value in metrics.items():
                scene_metrics[key].append(float(value))

    if not scene_metrics:
        return None
    return scene_metrics


def evaluate_pointmap_scene(model, args, cfg, scene, ds_name, device):
    all_imgs, all_depths, all_calibs = cfg['get'](cfg['base'], scene)
    img_files = all_imgs[cfg['slice']]
    depth_files = all_depths[cfg['slice']]
    calib_files = all_calibs[cfg['slice']]

    depths_gt, intrinsics_gt = [], []
    for idx, depth_path in enumerate(depth_files):
        depths_gt.append(torch.from_numpy(cfg['read_d'](depth_path)))
        intrinsics_gt.append(torch.from_numpy(get_intrinsic(ds_name, calib_files[idx])))

    depth_gt_tensor = torch.stack(depths_gt).to(device)
    K_gt_tensor = torch.stack(intrinsics_gt).to(device)
    points_gt = as_points_stack(depth_to_points(depth_gt_tensor, K_gt_tensor))
    points_pred = run_pointmap_inference(model, args.model_name, img_files)

    return compute_points_metrics(
        points_pred.float().to(device),
        points_gt.float().to(device),
        align_method=args.align_method,
        use_weight=True,
        metrics=DEPTH_METRICS,
        **depth_meta_data.get(ds_name.split('_')[0], {}),
    )


# ==============================================================================
# 5. Evaluation Loop
# ==============================================================================

def validate_args(args):
    if args.model_name not in TASK_MODELS[args.task]:
        supported = ', '.join(TASK_MODELS[args.task])
        raise ValueError(f"Model '{args.model_name}' does not support task '{args.task}'. Supported: {supported}")

    supported_datasets = DEPTH_BENCHMARK_SUPPORTED_DATASETS[args.task]
    for dataset in args.datasets:
        if dataset not in supported_datasets:
            supported = ', '.join(supported_datasets)
            raise ValueError(f"Dataset '{dataset}' does not support task '{args.task}'. Supported: {supported}")

    if args.task == TASK_POINTMAP and args.align_method == 'affine':
        raise ValueError("Pointmap evaluation supports align_method scale or metric.")


def evaluate(args):
    validate_args(args)
    model = get_model(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_dict = get_depth_benchmark_dataset_config(args.data_root, args.task)
    rows_written = 0

    print("=" * 60)
    print(f"Task: {args.task.upper()} | Benchmark: {args.model_name.upper()} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Align Method: {args.align_method}")
    print("=" * 60)

    for ds_name in args.datasets:
        if ds_name not in config_dict:
            print(f"[SKIP] Dataset is not configured for {args.task}: {ds_name}")
            continue

        cfg = config_dict[ds_name]
        if not os.path.exists(cfg['base']):
            print(f"[SKIP] Dataset path does not exist for {ds_name}: {cfg['base']}")
            continue

        scenes = cfg['scenes'](cfg['base']) if callable(cfg['scenes']) else cfg['scenes']
        val_metrics = defaultdict(list)

        for scene in tqdm(scenes, desc=ds_name):
            torch.cuda.empty_cache()

            if args.task == TASK_VIDEO_DEPTH:
                metrics = evaluate_video_depth_scene(model, args, cfg, scene, ds_name, device)
                if metrics:
                    tqdm.write(f"[{ds_name.upper()} - {scene}] {format_metrics(metrics)}")
                    for key, value in metrics.items():
                        val_metrics[key].append(float(value))
            elif args.task == TASK_MONO_DEPTH:
                metrics = evaluate_mono_depth_scene(model, args, cfg, scene, ds_name, device)
                if metrics:
                    display_metrics = {key: float(np.mean(values)) for key, values in metrics.items()}
                    tqdm.write(f"[{ds_name.upper()} - {scene}] {format_metrics(display_metrics)}")
                    for key, values in metrics.items():
                        val_metrics[key].extend([float(value) for value in values])
            elif args.task == TASK_POINTMAP:
                metrics = evaluate_pointmap_scene(model, args, cfg, scene, ds_name, device)
                if metrics:
                    tqdm.write(f"[{ds_name.upper()} - {scene}] {format_metrics(metrics)}")
                    for key, value in metrics.items():
                        val_metrics[key].append(float(value))
            else:
                raise ValueError(f"Unsupported task: {args.task}")

        if val_metrics:
            summary_metrics = {key: float(np.mean(values)) for key, values in val_metrics.items()}
            print_summary(
                ds_name,
                summary_metrics,
                line_formatter=lambda key, value: (
                    f"[Model: {args.model_name}][{ds_name.upper()}] Final Average {key}: {value:.4f}"
                ),
            )
            row = make_summary_row(args.task, ds_name, args.model_name, summary_metrics, SUMMARY_COLUMNS)
            write_summary_table(OUTPUT_DIR, f"eval_results_{args.task}_summary", SUMMARY_COLUMNS, [row])
            rows_written += 1

    if rows_written == 0:
        raise RuntimeError(f"No {args.task} summary rows were produced; no CSV data was written.")

    print(f"\nWrote {rows_written} {args.task} summary rows.")


# ==============================================================================
# 7. CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline depth and pointmap models.")
    parser.add_argument('--task', type=str, required=True, choices=TASKS)
    parser.add_argument('--model_name', type=str, required=True, choices=ALL_MODEL_NAMES)
    parser.add_argument('--data_root', type=str, default='./benchmark_datasets')
    parser.add_argument('--align_method', type=str, default='scale', choices=['scale', 'affine', 'metric'])
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = DEPTH_BENCHMARK_DEFAULT_DATASETS[args.task]
    return args


if __name__ == '__main__':
    evaluate(parse_args())
