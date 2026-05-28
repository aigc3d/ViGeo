#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --data-root" >&2; exit 2; }
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$DATA_ROOT" ]]; then
    echo "Missing required argument: --data-root" >&2
    exit 2
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Data root does not exist: $DATA_ROOT" >&2
    exit 2
fi

rm -f eval_results_normal_summary.csv

NORMAL_DATASETS=(hammer sintel nyuv2)
NORMAL_MODELS=(
    dsine
    normalcrafter
    stablenormal
    lotus
)
FAILURES=()

for model_name in "${NORMAL_MODELS[@]}"; do
    if ! python normal_benchmarks/eval.py \
        --model_name "$model_name" \
        --data_root "$DATA_ROOT" \
        --datasets "${NORMAL_DATASETS[@]}"; then
        FAILURES+=("$model_name")
    fi
done

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "Failed evaluations:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

if [[ ! -s eval_results_normal_summary.csv ]]; then
    echo "No summary CSV was produced. Check the eval logs for 'No summary rows were produced'." >&2
    exit 1
fi
