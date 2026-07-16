#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT=""
CHECKPOINT_PATH="vigeo.pt"
CHUNK_SIZE="16"
MODELS=(infinitevggt vigeo_chunk)
INPUT_LENGTHS=(300 400 500)

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
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --input-lengths)
            shift
            INPUT_LENGTHS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                INPUT_LENGTHS+=("$1")
                shift
            done
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

for model in "${MODELS[@]}"; do
    if [[ "$model" == vigeo* ]]; then
        bash scripts/eval_vigeo.sh \
            --data-root "$DATA_ROOT" \
            --checkpoint-path "$CHECKPOINT_PATH" \
            --chunk-size "$CHUNK_SIZE" \
            --tasks reconstruction_long \
            --recon-input-lengths "${INPUT_LENGTHS[@]}"
    else
        python recon_benchmarks/eval_long.py \
            --model "$model" \
            --data-root "$DATA_ROOT" \
            --input-lengths "${INPUT_LENGTHS[@]}"
    fi
done
