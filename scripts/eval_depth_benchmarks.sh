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

rm -f \
    eval_results_video_depth_summary.csv \
    eval_results_mono_depth_summary.csv \
    eval_results_pointmap_summary.csv

STANDARD_DATASETS=(sintel bonn kitti)
LONG_DATASETS=(bonn_400 kitti_300 hammer)
FAILURES=()

VIDEO_DEPTH_MODELS=(
    da3
    vggt
    stream3r
    vda
    geometrycrafter
    streamvggt
    depthcrafter
    pi3
)

MONO_DEPTH_MODELS=(
    vda
    depthcrafter
    vggt
    vggt_omega
    pi3
    da3
    flashdepth
)

LONG_VIDEO_DEPTH_MODELS=(
    vda
    depthcrafter
    geometrycrafter
    infinitevggt
)

POINTMAP_MODELS=(
    geometrycrafter
    vggt
    vggt_omega
    pi3
    da3
    streamvggt
    stream3r
)

run_depth_eval() {
    local task="$1"
    local align_method="$2"
    local model_name="$3"
    shift 3

    if ! python depth_benchmarks/eval.py \
        --task "$task" \
        --model_name "$model_name" \
        --data_root "$DATA_ROOT" \
        --align_method "$align_method" \
        --datasets "$@"; then
        FAILURES+=("$task/$model_name")
    fi
}

for model_name in "${VIDEO_DEPTH_MODELS[@]}"; do
    run_depth_eval video_depth scale "$model_name" "${STANDARD_DATASETS[@]}"
done

for model_name in "${MONO_DEPTH_MODELS[@]}"; do
    run_depth_eval mono_depth affine "$model_name" "${STANDARD_DATASETS[@]}"
done

for model_name in "${LONG_VIDEO_DEPTH_MODELS[@]}"; do
    run_depth_eval video_depth scale "$model_name" "${LONG_DATASETS[@]}"
done

for model_name in "${POINTMAP_MODELS[@]}"; do
    run_depth_eval pointmap scale "$model_name" "${STANDARD_DATASETS[@]}"
done

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "Failed evaluations:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

if [[ ! -s eval_results_video_depth_summary.csv \
      && ! -s eval_results_mono_depth_summary.csv \
      && ! -s eval_results_pointmap_summary.csv ]]; then
    echo "No summary CSV was produced. Check the eval logs for 'No summary rows were produced'." >&2
    exit 1
fi
