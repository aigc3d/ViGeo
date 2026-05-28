#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT=""
CHECKPOINT_PATH=""
CHUNK_SIZE="16"
USE_FP16="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --data-root" >&2; exit 2; }
            DATA_ROOT="$2"
            shift 2
            ;;
        --checkpoint-path)
            [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint-path" >&2; exit 2; }
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --chunk-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --chunk-size" >&2; exit 2; }
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --use-fp16)
            USE_FP16="1"
            shift
            ;;
        --no-fp16)
            USE_FP16="0"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Missing required argument: --checkpoint-path" >&2
    exit 2
fi

if [[ -z "$DATA_ROOT" ]]; then
    echo "Missing required argument: --data-root" >&2
    exit 2
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Data root does not exist: $DATA_ROOT" >&2
    exit 2
fi

rm -f eval_results_vigeo_summary.csv

COMMON_ARGS=(
    --data_root "$DATA_ROOT"
    --checkpoint_path "$CHECKPOINT_PATH"
    --chunk_size "$CHUNK_SIZE"
)

if [[ "$USE_FP16" == "1" || "$USE_FP16" == "true" ]]; then
    COMMON_ARGS+=(--use_fp16)
fi

STANDARD_DEPTH_DATASETS=(sintel bonn kitti)
LONG_DEPTH_DATASETS=(bonn_400 kitti_300 hammer)
NORMAL_DATASETS=(hammer sintel nyuv2)
FAILURES=()

run_vigeo_eval() {
    local task="$1"
    local mode="$2"
    local align_method="$3"
    shift 3

    if ! python eval.py \
        --task "$task" \
        --mode "$mode" \
        "${COMMON_ARGS[@]}" \
        --align_method "$align_method" \
        --datasets "$@"; then
        FAILURES+=("$task/$mode")
    fi
}

for mode in offline online; do
    run_vigeo_eval video_depth "$mode" scale "${STANDARD_DEPTH_DATASETS[@]}"
    run_vigeo_eval pointmap "$mode" scale "${STANDARD_DEPTH_DATASETS[@]}"
done

run_vigeo_eval mono_depth offline affine "${STANDARD_DEPTH_DATASETS[@]}"
run_vigeo_eval normal offline scale "${NORMAL_DATASETS[@]}"
run_vigeo_eval video_depth chunk scale "${LONG_DEPTH_DATASETS[@]}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "Failed evaluations:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

if [[ ! -s eval_results_vigeo_summary.csv ]]; then
    echo "No summary CSV was produced. Check the eval logs for 'No summary rows were produced'." >&2
    exit 1
fi
