from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon_benchmarks.datasets import DATASET_NAMES, discover_scenes, load_scene  # noqa: E402
from recon_benchmarks.metrics import evaluate_reconstruction  # noqa: E402
from recon_benchmarks.model_registry import infer_scene, load_model  # noqa: E402
from eval_utils import format_metrics  # noqa: E402


TASK_RECONSTRUCTION = "reconstruction_long"
METRIC_COLUMNS = (
    "acc_mean",
    "acc_median",
    "comp_mean",
    "comp_median",
    "nc_mean",
    "nc_median",
)
SUMMARY_COLUMNS = ["task", "dataset", "benchmark", "input", "num_scenes", *METRIC_COLUMNS]
LONG_MODELS = ("infinitevggt",)


def _summary_row(model_name: str, dataset: str, input_length: int, scene_rows: list[dict]) -> dict:
    row = {
        "task": TASK_RECONSTRUCTION,
        "dataset": dataset,
        "benchmark": model_name,
        "input": input_length,
        "num_scenes": len(scene_rows),
    }
    for metric in METRIC_COLUMNS:
        row[metric] = float(np.mean([scene[metric] for scene in scene_rows]))
    return row


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUMMARY_COLUMNS:
            return []
        return [{column: row.get(column, "") for column in SUMMARY_COLUMNS} for row in reader]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_rows(path)
    new_keys = {(row["task"], row["dataset"], row["benchmark"], str(row["input"])) for row in rows}
    merged = [
        row for row in existing
        if (row.get("task"), row.get("dataset"), row.get("benchmark"), str(row.get("input"))) not in new_keys
    ]
    merged.extend(rows)
    merged = sorted(merged, key=lambda row: (row["task"], row["benchmark"], int(float(row["input"])), row["dataset"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(_format_long_rows(merged))


def _format_long_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        item = {column: row.get(column, "") for column in SUMMARY_COLUMNS}
        item["input"] = str(int(float(item["input"])))
        item["num_scenes"] = str(int(float(item["num_scenes"])))
        for metric in METRIC_COLUMNS:
            if item[metric] not in ("", None):
                item[metric] = f"{float(item[metric]):.3f}"
        formatted.append(item)
    return formatted


def evaluate(args: argparse.Namespace) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Long-sequence reconstruction evaluation requires CUDA.")

    model = load_model(args.model)

    summary_rows = []
    for input_length in args.input_lengths:
        for dataset in args.datasets:
            scenes = discover_scenes(args.data_root, dataset)
            if args.limit_scenes is not None:
                scenes = scenes[: args.limit_scenes]
            if not scenes:
                raise RuntimeError(f"No scenes discovered for {dataset}")

            dataset_rows = []
            for scene_id in tqdm(scenes, desc=f"{args.model}:{dataset}:{input_length}"):
                scene = load_scene(
                    args.data_root,
                    dataset,
                    scene_id,
                    resolution=(args.width, args.height),
                    stride=args.stride,
                    max_frames=input_length,
                    project_missing_depth=True,
                )
                if len(scene.image_paths) < input_length:
                    message = (
                        f"{dataset} - {scene_id}: only {len(scene.image_paths)} frames after stride={args.stride}, "
                        f"requested {input_length}."
                    )
                    if args.skip_short_scenes:
                        tqdm.write(f"[SKIP] {message}")
                        continue
                    tqdm.write(f"[WARN] {message}")

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
                    max_metric_points=args.max_metric_points,
                )
                row = metrics.as_dict()
                dataset_rows.append(row)
                tqdm.write(
                    f"[{dataset} - {scene_id} - {input_length}] "
                    f"{format_metrics({key: row[key] for key in METRIC_COLUMNS})}"
                )
                torch.cuda.empty_cache()

            if not dataset_rows:
                raise RuntimeError(
                    f"No valid scenes for {dataset} with input_length={input_length}. "
                    "Use --allow-short-scenes to evaluate shorter sequences."
                )

            summary = _summary_row(args.model, dataset, input_length, dataset_rows)
            summary_rows.append(summary)
            print(
                f"[SUMMARY {args.model}:{dataset}:{input_length}] "
                f"Acc {summary['acc_mean']:.3f}/{summary['acc_median']:.3f}, "
                f"Comp {summary['comp_mean']:.3f}/{summary['comp_median']:.3f}, "
                f"NC {summary['nc_mean']:.3f}/{summary['nc_median']:.3f}"
            )

    csv_path = Path(args.output_dir) / f"{args.output_prefix}.csv"
    _write_rows(csv_path, summary_rows)
    print(f"Saved summary table ({len(summary_rows)} new rows): {csv_path}")
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate long-sequence 3D reconstruction on 7-Scenes and NRGBD."
    )
    parser.add_argument("--model", required=True, choices=LONG_MODELS)
    parser.add_argument("--data-root", "--data_root", dest="data_root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=["7scenes", "nrgbd"])
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[300, 400, 500])
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--icp-threshold", type=float, default=0.1)
    parser.add_argument("--max-metric-points", type=int, default=500000)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--total-budget", type=int, default=1_200_000)
    parser.add_argument("--limit-scenes", type=int, default=None, help="Smoke-test only: evaluate the first N scenes.")
    parser.add_argument("--allow-short-scenes", action="store_false", dest="skip_short_scenes")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--output-prefix", type=str, default="eval_results_reconstruction_long_summary")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
