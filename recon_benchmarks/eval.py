from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon_benchmarks.datasets import DATASET_NAMES, DEFAULT_PI3_EVAL_DATASETS, DEFAULT_STRIDES, default_stride_for_dataset, discover_scenes, load_scene  # noqa: E402
from recon_benchmarks.metrics import evaluate_reconstruction  # noqa: E402
from recon_benchmarks.model_registry import infer_scene, load_model  # noqa: E402
from eval_utils import format_metrics, write_summary_table  # noqa: E402


TASK_RECONSTRUCTION = "reconstruction"
METRIC_COLUMNS = (
    "acc_mean",
    "acc_median",
    "comp_mean",
    "comp_median",
    "nc_mean",
    "nc_median",
)
SUMMARY_COLUMNS = ["task", "dataset", "benchmark", "num_scenes", *METRIC_COLUMNS]

# Values in the StreamVGGT paper table supplied with this benchmark request.
def _summary_row(model_name: str, dataset: str, scene_rows: list[dict]) -> dict:
    row = {
        "task": TASK_RECONSTRUCTION,
        "dataset": dataset,
        "benchmark": model_name,
        "num_scenes": len(scene_rows),
    }
    for metric in METRIC_COLUMNS:
        row[metric] = float(np.mean([scene[metric] for scene in scene_rows]))
    return row


def evaluate(args: argparse.Namespace) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("3D reconstruction model inference requires CUDA.")
    model = load_model(args.model, args.checkpoint)

    summary_rows = []
    for dataset in args.datasets:
        scenes = discover_scenes(args.data_root, dataset)
        if args.limit_scenes is not None:
            scenes = scenes[: args.limit_scenes]
        if not scenes:
            raise RuntimeError(f"No scenes discovered for {dataset}")

        dataset_rows = []
        stride = default_stride_for_dataset(dataset)
        if dataset == "7scenes":
            stride = args.seven_scenes_stride
        elif dataset == "nrgbd":
            stride = args.nrgbd_stride
        for scene_id in tqdm(scenes, desc=f"{args.model}:{dataset}"):
            scene = load_scene(
                args.data_root,
                dataset,
                scene_id,
                resolution=(args.width, args.height),
                stride=stride,
            )
            prediction = infer_scene(
                model,
                args.model,
                scene,
                chunk_size=args.chunk_size,
                total_budget=args.total_budget,
            )
            metrics = evaluate_reconstruction(
                prediction.points,
                scene.points_gt,
                scene.valid_masks,
                colors=scene.images.permute(0, 2, 3, 1),
                icp_threshold=args.icp_threshold,
                crop_size=args.crop_size,
            )
            row = {
                **metrics.as_dict(),
            }
            dataset_rows.append(row)
            tqdm.write(f"[{dataset} - {scene_id}] {format_metrics({k: row[k] for k in METRIC_COLUMNS})}")
            torch.cuda.empty_cache()

        summary = _summary_row(args.model, dataset, dataset_rows)
        summary_rows.append(summary)
        print(
            f"[SUMMARY {dataset}] Acc {summary['acc_mean']:.3f}/{summary['acc_median']:.3f}, "
            f"Comp {summary['comp_mean']:.3f}/{summary['comp_median']:.3f}, "
            f"NC {summary['nc_mean']:.3f}/{summary['nc_median']:.3f}"
        )

    write_summary_table(PROJECT_ROOT, "eval_results_reconstruction_summary", SUMMARY_COLUMNS, summary_rows)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scene-level 3D reconstruction on 7-Scenes and NRGBD.")
    parser.add_argument(
        "--model",
        required=True,
        choices=("pi3", "vggt", "da3", "streamvggt", "stream3r", "infinitevggt"),
    )
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", "--checkpoint-path", "--checkpoint_path", dest="checkpoint", default=None)
    parser.add_argument("--data-root", "--data_root", dest="data_root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DEFAULT_PI3_EVAL_DATASETS))
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--icp-threshold", type=float, default=0.1)
    parser.add_argument("--seven-scenes-stride", type=int, default=DEFAULT_STRIDES["7scenes"])
    parser.add_argument("--nrgbd-stride", type=int, default=DEFAULT_STRIDES["nrgbd"])
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--total-budget", type=int, default=1_200_000)
    parser.add_argument("--limit-scenes", type=int, default=None, help="Smoke-test only: evaluate the first N scenes.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    evaluate(parse_args())
