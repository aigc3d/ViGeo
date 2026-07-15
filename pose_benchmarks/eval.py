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

from pose_benchmarks.datasets import discover_sequences, load_sequence  # noqa: E402
from pose_benchmarks.metrics import evaluate_pose  # noqa: E402
from pose_benchmarks.model_registry import infer_pose, load_model  # noqa: E402
from eval_utils import format_metrics, write_summary_table  # noqa: E402


TASK_POSE = "pose_estimation"
SUMMARY_COLUMNS = ("task", "dataset", "benchmark", "num_sequences", "num_failed", "ate", "rpe_trans", "rpe_rot")


def evaluate(args: argparse.Namespace) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Camera pose benchmark requires CUDA.")
    model = load_model(args.model, args.checkpoint)

    summary_rows: list[dict] = []
    for dataset in args.datasets:
        sequences = args.sequences or discover_sequences(args.data_root, dataset, sintel_pass=args.sintel_pass)
        if args.limit_sequences is not None:
            sequences = sequences[: args.limit_sequences]
        if not sequences:
            raise RuntimeError(f"No pose sequences discovered for {dataset}.")

        dataset_rows = []
        for sequence_name in tqdm(sequences, desc=f"{args.model}:{dataset}"):
            sequence = load_sequence(
                args.data_root,
                dataset,
                sequence_name,
                stride=args.pose_eval_stride,
                sintel_pass=args.sintel_pass,
            )
            pred_poses, _ = infer_pose(
                model,
                args.model,
                sequence.image_paths,
                width=args.load_img_size,
                chunk_size=args.chunk_size,
                total_budget=args.total_budget,
            )
            try:
                metrics = evaluate_pose(
                    pred_poses,
                    sequence.gt_poses,
                    sequence.timestamps,
                )
                status = "ok"
                error = ""
            except Exception as exc:
                metrics = {"ate": float("nan"), "rpe_trans": float("nan"), "rpe_rot": float("nan")}
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "status": status,
                **metrics,
            }
            dataset_rows.append(row)
            if status == "ok":
                tqdm.write(f"[{dataset} - {sequence_name}] {format_metrics(metrics)}")
            else:
                tqdm.write(f"[{dataset} - {sequence_name}] failed: {error}")
            torch.cuda.empty_cache()

        ok_rows = [row for row in dataset_rows if row["status"] == "ok"]
        if not ok_rows:
            raise RuntimeError(f"All pose sequences failed for {args.model}:{dataset}")
        summary = {
            "task": TASK_POSE,
            "dataset": dataset,
            "benchmark": args.model,
            "num_sequences": len(ok_rows),
            "num_failed": len(dataset_rows) - len(ok_rows),
            "ate": float(np.mean([row["ate"] for row in ok_rows])),
            "rpe_trans": float(np.mean([row["rpe_trans"] for row in ok_rows])),
            "rpe_rot": float(np.mean([row["rpe_rot"] for row in ok_rows])),
        }
        summary_rows.append(summary)
        print(
            f"[SUMMARY {dataset}] ATE {summary['ate']:.4f}, "
            f"RPE trans {summary['rpe_trans']:.4f}, RPE rot {summary['rpe_rot']:.4f}"
        )

    write_summary_table(PROJECT_ROOT, "eval_results_pose_summary", list(SUMMARY_COLUMNS), summary_rows)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate camera pose estimation with the Pi3 relpose-distance protocol.")
    parser.add_argument("--model", required=True, choices=("vggt", "pi3", "da3", "streamvggt", "stream3r"))
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", "--checkpoint-path", "--checkpoint_path", dest="checkpoint", default=None)
    parser.add_argument("--data-root", "--data_root", dest="data_root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["sintel"], choices=("sintel",))
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--load-img-size", "--load_img_size", dest="load_img_size", type=int, default=512)
    parser.add_argument("--pose-eval-stride", "--pose_eval_stride", dest="pose_eval_stride", type=int, default=1)
    parser.add_argument("--sintel-pass", choices=("clean", "final"), default="clean")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--total-budget", type=int, default=1_200_000)
    parser.add_argument("--limit-sequences", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    evaluate(parse_args())
