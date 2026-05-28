import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(PROJECT_ROOT), str(CURRENT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_defs import NORMAL_DATASETS, NORMAL_METRICS, NORMAL_SUMMARY_COLUMNS, TASK_NORMAL
from dataset_io import get_normal_benchmark_dataset_config, read_normal_map
from eval_utils import format_metrics, has_valid_metrics, make_summary_row, print_summary, write_summary_table
from metric_func import as_normal_stack, compute_normal_metrics
try:
    from .inference import run_inference
    from .model_registry import NORMAL_MODELS, get_model
except ImportError:
    from inference import run_inference
    from model_registry import NORMAL_MODELS, get_model

SUMMARY_COLUMNS = NORMAL_SUMMARY_COLUMNS
OUTPUT_DIR = PROJECT_ROOT


# ==============================================================================
# 1. Main Evaluation Loop
# ==============================================================================

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_name)

    print("=" * 60)
    print(f"Baseline: {args.model_name.upper()} | Task: {TASK_NORMAL.upper()}")
    print("=" * 60)

    rows_written = 0
    config_dict = get_normal_benchmark_dataset_config(args.data_root)

    for ds_name in args.datasets:
        if ds_name not in config_dict:
            print(f"[SKIP] Dataset is not configured for normal: {ds_name}")
            continue
        cfg = config_dict[ds_name]
        if not os.path.exists(cfg['base']):
            print(f"[SKIP] Dataset path does not exist for {ds_name}: {cfg['base']}")
            continue

        scenes = cfg['scenes'](cfg['base']) if callable(cfg['scenes']) else cfg['scenes']
        val_metrics = defaultdict(list)

        for scene in tqdm(scenes, desc=ds_name):
            torch.cuda.empty_cache()
            all_imgs, _, all_norms, _ = cfg['get'](cfg['base'], scene)
            s = cfg['slice']
            imgs, norms = all_imgs[s], all_norms[s]

            valid_imgs, gts_tensor = [], []
            for i in range(len(imgs)):
                if norms[i] is None or not os.path.exists(norms[i]):
                    continue
                nml_np = read_normal_map(norms[i], ds_name, valid_threshold=0.5)
                if nml_np is None:
                    continue
                gts_tensor.append(as_normal_stack(nml_np))
                valid_imgs.append(imgs[i])

            if not valid_imgs:
                continue

            Y = torch.cat(gts_tensor, dim=0).to(device)
            pred = run_inference(model, args.model_name, valid_imgs, ds_name)

            pred = pred.to(device).float()
            Y = Y.float()

            metrics = compute_normal_metrics(pred, Y, metrics=NORMAL_METRICS)

            if has_valid_metrics(metrics, 'mean'):
                tqdm.write(f"[{ds_name.upper()} - {scene}] {format_metrics(metrics)}")
                for k, v in metrics.items():
                    val_metrics[k].append(float(v))
            else:
                tqdm.write(f"[{ds_name.upper()} - {scene}] No valid normal pixels found.")

        if val_metrics:
            summary_metrics = {k: float(np.mean(v)) for k, v in val_metrics.items()}
            print_summary(ds_name, summary_metrics)
            row = make_summary_row(TASK_NORMAL, ds_name, args.model_name, summary_metrics, SUMMARY_COLUMNS)
            write_summary_table(OUTPUT_DIR, "eval_results_normal_summary", SUMMARY_COLUMNS, [row])
            rows_written += 1

    if rows_written == 0:
        raise RuntimeError("No normal summary rows were produced; no CSV data was written.")

    print(f"\nWrote {rows_written} normal summary rows.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Baseline Normal Estimation Models")
    parser.add_argument('--model_name', type=str, required=True, choices=NORMAL_MODELS)
    parser.add_argument('--data_root', type=str, default='./benchmark_datasets')
    parser.add_argument('--datasets', nargs='+', default=NORMAL_DATASETS, choices=NORMAL_DATASETS)

    args = parser.parse_args()
    evaluate(args)
