#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT=""
CHECKPOINT_PATH="vigeo.pt"
CHUNK_SIZE="16"
TASKS=(depth normal pose reconstruction)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root|--data_root)
            [[ $# -ge 2 ]] || { echo "Missing value for --data-root" >&2; exit 2; }
            DATA_ROOT="$2"
            shift 2
            ;;
        --checkpoint|--checkpoint-path|--checkpoint_path)
            [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint-path" >&2; exit 2; }
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --chunk-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --chunk-size" >&2; exit 2; }
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --tasks)
            shift
            TASKS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TASKS+=("$1")
                shift
            done
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

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint does not exist: $CHECKPOINT_PATH" >&2
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

STANDARD_DEPTH_DATASETS=(sintel bonn kitti)
LONG_DEPTH_DATASETS=(bonn_400 kitti_300 hammer)
NORMAL_DATASETS=(hammer sintel nyuv2)
POSE_DATASETS=(sintel)
FAILURES=()

task_enabled() {
    local target="$1"
    local task
    for task in "${TASKS[@]}"; do
        [[ "$task" == "$target" ]] && return 0
    done
    return 1
}

run_geometry_eval() {
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

run_pose_eval() {
    local mode="$1"
    if ! python eval.py \
        --task pose_estimation \
        --mode "$mode" \
        "${COMMON_ARGS[@]}" \
        --datasets "${POSE_DATASETS[@]}"; then
        FAILURES+=("pose_estimation/$mode")
    fi
}

run_reconstruction_eval() {
    local mode="$1"
    if ! python eval.py \
        --task reconstruction \
        --mode "$mode" \
        "${COMMON_ARGS[@]}"; then
        FAILURES+=("reconstruction/$mode")
    fi
}

if task_enabled depth; then
    for mode in offline online; do
        run_geometry_eval video_depth "$mode" scale "${STANDARD_DEPTH_DATASETS[@]}"
        run_geometry_eval pointmap "$mode" scale "${STANDARD_DEPTH_DATASETS[@]}"
    done
    run_geometry_eval mono_depth offline affine "${STANDARD_DEPTH_DATASETS[@]}"
    run_geometry_eval video_depth chunk scale "${LONG_DEPTH_DATASETS[@]}"
fi

if task_enabled normal; then
    run_geometry_eval normal offline scale "${NORMAL_DATASETS[@]}"
fi

if task_enabled pose; then
    for mode in offline online; do
        run_pose_eval "$mode"
    done
fi

if task_enabled reconstruction; then
    for mode in offline online chunk; do
        run_reconstruction_eval "$mode"
    done
fi

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "Failed evaluations:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

if [[ ! -s eval_results_vigeo_summary.csv ]]; then
    echo "No summary CSV was produced. Check the eval logs for 'No summary rows were produced'." >&2
    exit 1
fi
