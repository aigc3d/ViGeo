import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image
from tqdm import tqdm

from benchmark_defs import (
    DEPTH_METRICS,
    NORMAL_METRICS,
    POSE_METRICS,
    RECONSTRUCTION_METRICS,
    TASK_MONO_DEPTH,
    TASK_NORMAL,
    TASK_POINTMAP,
    TASK_POSE_ESTIMATION,
    TASK_RECONSTRUCTION,
    TASK_VIDEO_DEPTH,
    VIGEO_DEFAULT_DATASETS,
    VIGEO_SUPPORTED_DATASETS,
    VIGEO_SUMMARY_COLUMNS,
    VIGEO_TASKS,
)
from dataset_io import depth_to_points, get_intrinsic, get_vigeo_dataset_config, read_normal_map
from dataset_io import (
    default_reconstruction_stride,
    get_pose_sequences,
    get_reconstruction_scenes,
    load_pose_sequence,
    load_reconstruction_scene,
)
from eval_utils import format_metrics, has_valid_metrics, make_summary_row, print_summary, write_summary_table
from metric_func import (
    as_depth_stack,
    as_normal_stack,
    as_points_stack,
    compute_depth_metrics,
    compute_normal_metrics,
    compute_points_metrics,
    depth_meta_data,
)
from pose_benchmarks.metrics import c2w_to_pose_rows, evaluate_pose
from recon_benchmarks.metrics import evaluate_reconstruction


TASKS = VIGEO_TASKS
SUMMARY_COLUMNS = VIGEO_SUMMARY_COLUMNS
OUTPUT_DIR = Path(__file__).resolve().parent


# ==============================================================================
# 1. Model Loading and Inference
# ==============================================================================

def load_vigeo_model(checkpoint_path, device, use_fp16=False):
    from vigeo import ViGeoModel

    model = ViGeoModel().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    if use_fp16:
        model.half()
    return model


def make_benchmark_name(args):
    return f"vigeo_{args.mode}"


def make_infer_kwargs(args):
    infer_kwargs = {'mode': args.mode, 'total_budget': args.total_budget}
    if args.mode in ['chunk', 'online']:
        infer_kwargs['chunk_size'] = args.chunk_size
    return infer_kwargs


def run_inference(model, args, images_tensor):
    infer_kwargs = make_infer_kwargs(args)
    with torch.no_grad():
        if args.task == TASK_MONO_DEPTH:
            mono_kwargs = {**infer_kwargs, 'chunk_size': 1} if 'chunk_size' in infer_kwargs else infer_kwargs
            return torch.cat(
                [
                    as_depth_stack(model.infer(images_tensor[:, t:t + 1], **mono_kwargs)['depth_pred'])
                    for t in range(images_tensor.shape[1])
                ],
                dim=0,
            )

        output = model.infer(images_tensor, **infer_kwargs)
        if args.task == TASK_VIDEO_DEPTH:
            return as_depth_stack(output['depth_pred'])
        if args.task == TASK_POINTMAP:
            return as_points_stack(output['points_pred'])
        if args.task == TASK_NORMAL:
            if output['normal_pred'] is None:
                raise ValueError("ViGeo output does not include normal_pred.")
            return F.normalize(as_normal_stack(-output['normal_pred']), p=2, dim=1)

    raise ValueError(f"Unsupported task: {args.task}")


def load_pose_image_tensor(image_paths, width, device):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("No pose images provided.")
    original_width, original_height = images[0].size
    target_width = width
    target_height = round(original_height * (target_width / original_width) / 14) * 14
    to_tensor = tvf.ToTensor()
    tensors = [
        to_tensor(image.resize((target_width, target_height), Image.Resampling.LANCZOS))
        for image in images
    ]
    stacked = torch.stack(tensors).to(device)
    patch_h, patch_w = stacked.shape[-2] // 14, stacked.shape[-1] // 14
    return F.interpolate(
        stacked,
        (patch_h * 14, patch_w * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).unsqueeze(0)


@torch.no_grad()
def run_pose_inference(model, args, image_paths, device):
    images_tensor = load_pose_image_tensor(image_paths, args.pose_load_img_size, device)
    infer_kwargs = make_infer_kwargs(args)
    output = model.infer(images_tensor[0], num_tokens=None, resize_output=False, **infer_kwargs)
    return c2w_to_pose_rows(output['pose_pred'].float().cpu())


def transform_local_points(points: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    if poses.shape[-2:] == (3, 4):
        bottom = torch.zeros((*poses.shape[:-2], 1, 4), dtype=poses.dtype, device=poses.device)
        bottom[..., 0, 3] = 1
        poses = torch.cat((poses, bottom), dim=-2)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    world = torch.einsum("sij,shwj->shwi", poses, homogeneous)
    first_inverse = torch.linalg.inv(poses[0])
    return torch.einsum("ij,shwj->shwi", first_inverse, world)[..., :3]


@torch.no_grad()
def run_reconstruction_inference(model, args, images_tensor):
    infer_kwargs = make_infer_kwargs(args)
    output = model.infer(images_tensor, num_tokens=None, resize_output=True, **infer_kwargs)
    points = output['points_pred'].float()
    poses = output['pose_pred'].float()
    return transform_local_points(points, poses).cpu()


# ==============================================================================
# 4. Scene Evaluation
# ==============================================================================

def make_image_tensor(img_path, dataset):
    img_pil = Image.open(img_path).convert("RGB")
    if dataset == 'nyuv2':
        img_pil = img_pil.crop((43, 45, 608, 472))
    return torch.from_numpy(np.asarray(img_pil, dtype=np.float32) / 255.0).permute(2, 0, 1)


def build_scene_tensors(args, cfg, scene, dataset, device):
    all_imgs, all_depths, all_normals, all_calibs = cfg['get'](cfg['base'], scene)
    sample_slice = cfg['slice']
    img_files = all_imgs[sample_slice]
    depth_files = all_depths[sample_slice]
    normal_files = all_normals[sample_slice]
    calib_files = all_calibs[sample_slice]

    images, targets = [], []
    for idx, img_path in enumerate(img_files):
        target = None

        if args.task in [TASK_MONO_DEPTH, TASK_VIDEO_DEPTH]:
            target = as_depth_stack(torch.from_numpy(cfg['read_d'](depth_files[idx])))
        elif args.task == TASK_POINTMAP:
            depth = torch.from_numpy(cfg['read_d'](depth_files[idx])).float()
            intrinsic = torch.from_numpy(get_intrinsic(dataset, calib_files[idx])).float()
            target = as_points_stack(depth_to_points(depth, intrinsic))
        elif args.task == TASK_NORMAL:
            normal_path = normal_files[idx]
            if normal_path is None or not os.path.exists(normal_path):
                continue
            normal_np = read_normal_map(normal_path, dataset)
            if normal_np is None:
                continue
            target = as_normal_stack(normal_np)

        images.append(make_image_tensor(img_path, dataset))
        targets.append(target)

    if not images:
        return None, None

    images_tensor = torch.stack(images).unsqueeze(0).to(device)
    target_tensor = torch.cat(targets, dim=0).to(device)
    return images_tensor, target_tensor


def compute_scene_metrics(args, pred, target, dataset):
    pred = pred.float()
    target = target.float()
    ds_meta = depth_meta_data.get(dataset.split('_')[0], {})

    if args.task == TASK_MONO_DEPTH:
        scene_metrics = defaultdict(list)
        for pred_frame, target_frame in zip(pred, target):
            frame_metrics = compute_depth_metrics(
                pred_frame,
                target_frame,
                metrics=DEPTH_METRICS,
                align_method=args.align_method,
                **ds_meta,
            )
            if frame_metrics:
                for key, value in frame_metrics.items():
                    scene_metrics[key].append(value)
        return {key: float(np.mean(values)) for key, values in scene_metrics.items()}

    if args.task == TASK_VIDEO_DEPTH:
        return compute_depth_metrics(
            pred,
            target,
            metrics=DEPTH_METRICS,
            align_method=args.align_method,
            **ds_meta,
        )

    if args.task == TASK_POINTMAP:
        return compute_points_metrics(
            pred,
            target,
            metrics=DEPTH_METRICS,
            align_method=args.align_method,
            use_weight=True,
            **ds_meta,
        )

    if args.task == TASK_NORMAL:
        return compute_normal_metrics(pred, target, metrics=NORMAL_METRICS)

    raise ValueError(f"Unsupported task: {args.task}")


# ==============================================================================
# 5. Evaluation Loop
# ==============================================================================

def evaluate_pose_estimation(args, model, device):
    rows_written = 0
    for dataset in args.datasets:
        sequences = get_pose_sequences(args.data_root, dataset, sintel_pass=args.sintel_pass)
        if args.limit_sequences is not None:
            sequences = sequences[:args.limit_sequences]
        if not sequences:
            print(f"[SKIP] No pose sequences found for {dataset}.")
            continue

        val_metrics = defaultdict(list)
        failed = 0
        for sequence_name in tqdm(sequences, desc=dataset):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sequence = load_pose_sequence(
                args.data_root,
                dataset,
                sequence_name,
                stride=args.pose_eval_stride,
                sintel_pass=args.sintel_pass,
            )
            pred_poses, _ = run_pose_inference(model, args, sequence.image_paths, device)
            try:
                metrics = evaluate_pose(pred_poses, sequence.gt_poses, sequence.timestamps)
            except Exception as exc:
                failed += 1
                tqdm.write(f"[{dataset.upper()} - {sequence_name}] failed: {type(exc).__name__}: {exc}")
                continue

            if not has_valid_metrics(metrics, "ate"):
                failed += 1
                tqdm.write(f"[{dataset.upper()} - {sequence_name}] No valid pose metrics found.")
                continue

            tqdm.write(f"[{dataset.upper()} - {sequence_name}] {format_metrics(metrics)}")
            for key, value in metrics.items():
                val_metrics[key].append(float(value))

        if val_metrics:
            summary_metrics = {key: float(np.mean(values)) for key, values in val_metrics.items()}
            summary_metrics["num_sequences"] = len(next(iter(val_metrics.values())))
            summary_metrics["num_failed"] = failed
            print_summary(dataset, {key: summary_metrics[key] for key in POSE_METRICS})
            row = make_summary_row(args.task, dataset, make_benchmark_name(args), summary_metrics, SUMMARY_COLUMNS)
            write_summary_table(OUTPUT_DIR, "eval_results_vigeo_summary", SUMMARY_COLUMNS, [row])
            rows_written += 1

    if rows_written == 0:
        raise RuntimeError("No ViGeo pose summary rows were produced; no CSV data was written.")

    print(f"\nWrote {rows_written} ViGeo pose summary rows.")


def evaluate_reconstruction_task(args, model, device):
    rows_written = 0
    for dataset in args.datasets:
        scenes = get_reconstruction_scenes(args.data_root, dataset)
        if args.limit_scenes is not None:
            scenes = scenes[:args.limit_scenes]
        if not scenes:
            print(f"[SKIP] No reconstruction scenes found for {dataset}.")
            continue

        val_metrics = defaultdict(list)
        stride = default_reconstruction_stride(dataset)
        if dataset.startswith("7scenes"):
            stride = args.seven_scenes_stride
        elif dataset.startswith("nrgbd"):
            stride = args.nrgbd_stride

        for scene_id in tqdm(scenes, desc=dataset):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            scene = load_reconstruction_scene(
                args.data_root,
                dataset,
                scene_id,
                resolution=(args.recon_width, args.recon_height),
                stride=stride,
            )
            pred_points = run_reconstruction_inference(model, args, scene.images)
            metrics = evaluate_reconstruction(
                pred_points,
                scene.points_gt,
                scene.valid_masks,
                colors=scene.images.permute(0, 2, 3, 1),
                icp_threshold=args.icp_threshold,
                crop_size=args.crop_size,
            ).as_dict()

            if not has_valid_metrics(metrics, "acc_mean"):
                tqdm.write(f"[{dataset.upper()} - {scene_id}] No valid reconstruction metrics found.")
                continue

            tqdm.write(f"[{dataset.upper()} - {scene_id}] {format_metrics({k: metrics[k] for k in RECONSTRUCTION_METRICS})}")
            for key in RECONSTRUCTION_METRICS:
                val_metrics[key].append(float(metrics[key]))

        if val_metrics:
            summary_metrics = {key: float(np.mean(values)) for key, values in val_metrics.items()}
            summary_metrics["num_scenes"] = len(next(iter(val_metrics.values())))
            print_summary(dataset, {key: summary_metrics[key] for key in RECONSTRUCTION_METRICS})
            row = make_summary_row(args.task, dataset, make_benchmark_name(args), summary_metrics, SUMMARY_COLUMNS)
            write_summary_table(OUTPUT_DIR, "eval_results_vigeo_summary", SUMMARY_COLUMNS, [row])
            rows_written += 1

    if rows_written == 0:
        raise RuntimeError("No ViGeo reconstruction summary rows were produced; no CSV data was written.")

    print(f"\nWrote {rows_written} ViGeo reconstruction summary rows.")


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_vigeo_model(args.checkpoint_path, device, use_fp16=args.use_fp16)

    print("=" * 60)
    print(f"Task: {args.task.upper()} | Benchmark: {make_benchmark_name(args).upper()}")
    if args.task not in [TASK_NORMAL, TASK_POSE_ESTIMATION, TASK_RECONSTRUCTION]:
        print(f"Align Method: {args.align_method}")
    print("=" * 60)

    if args.task == TASK_POSE_ESTIMATION:
        return evaluate_pose_estimation(args, model, device)
    if args.task == TASK_RECONSTRUCTION:
        return evaluate_reconstruction_task(args, model, device)

    rows_written = 0
    config_dict = get_vigeo_dataset_config(args.data_root, args.task)

    for dataset in args.datasets:
        cfg = config_dict[dataset]
        if not os.path.exists(cfg['base']):
            print(f"[SKIP] Dataset path does not exist for {dataset}: {cfg['base']}")
            continue

        scenes = cfg['scenes'](cfg['base']) if callable(cfg['scenes']) else cfg['scenes']
        val_metrics = defaultdict(list)

        for scene in tqdm(scenes, desc=dataset):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images_tensor, target_tensor = build_scene_tensors(args, cfg, scene, dataset, device)
            if images_tensor is None:
                continue

            pred = run_inference(model, args, images_tensor).to(device)
            metrics = compute_scene_metrics(args, pred, target_tensor, dataset)

            if not has_valid_metrics(metrics):
                tqdm.write(f"[{dataset.upper()} - {scene}] No valid pixels found.")
                continue

            tqdm.write(f"[{dataset.upper()} - {scene}] {format_metrics(metrics)}")
            for key, value in metrics.items():
                val_metrics[key].append(float(value))

        if val_metrics:
            summary_metrics = {key: float(np.mean(values)) for key, values in val_metrics.items()}
            print_summary(dataset, summary_metrics)
            row = make_summary_row(args.task, dataset, make_benchmark_name(args), summary_metrics, SUMMARY_COLUMNS)
            write_summary_table(OUTPUT_DIR, "eval_results_vigeo_summary", SUMMARY_COLUMNS, [row])
            rows_written += 1

    if rows_written == 0:
        raise RuntimeError("No ViGeo summary rows were produced; no CSV data was written.")

    print(f"\nWrote {rows_written} ViGeo summary rows.")


def validate_args(args):
    if args.datasets is None:
        args.datasets = VIGEO_DEFAULT_DATASETS[args.task]

    supported = VIGEO_SUPPORTED_DATASETS[args.task]
    unsupported = [dataset for dataset in args.datasets if dataset not in supported]
    if unsupported:
        raise ValueError(
            f"Task '{args.task}' does not support datasets: {unsupported}. Supported: {supported}"
        )

    if args.task == TASK_POINTMAP and args.align_method == 'affine':
        raise ValueError("Pointmap evaluation does not support affine alignment.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViGeo.")
    parser.add_argument('--task', type=str, required=True, choices=TASKS)
    parser.add_argument('--mode', type=str, default='chunk', choices=['offline', 'chunk', 'online'])
    parser.add_argument('--data_root', type=str, default='./benchmark_datasets')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=16)
    parser.add_argument('--total_budget', type=int, default=1200000)
    parser.add_argument('--align_method', type=str, default='scale', choices=['scale', 'affine', 'metric'])
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--pose_load_img_size', type=int, default=512)
    parser.add_argument('--pose_eval_stride', type=int, default=1)
    parser.add_argument('--sintel_pass', type=str, default='clean', choices=['clean', 'final'])
    parser.add_argument('--limit_sequences', type=int, default=None)
    parser.add_argument('--recon_width', type=int, default=518)
    parser.add_argument('--recon_height', type=int, default=392)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--icp_threshold', type=float, default=0.1)
    parser.add_argument('--seven_scenes_stride', type=int, default=200)
    parser.add_argument('--nrgbd_stride', type=int, default=500)
    parser.add_argument('--limit_scenes', type=int, default=None)
    args = parser.parse_args()
    validate_args(args)
    return args


if __name__ == '__main__':
    evaluate(parse_args())
