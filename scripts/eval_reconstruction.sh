#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="streamvggt"
DATA_ROOT=""
CHECKPOINT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --checkpoint|--checkpoint-path|--checkpoint_path)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$DATA_ROOT" ]]; then
    echo "Missing required --data-root" >&2
    exit 2
fi
if [[ "$MODEL" == vigeo* ]]; then
    echo "ViGeo evaluation uses scripts/eval_vigeo.sh." >&2
    exit 2
fi

ARGS=(
    --model "$MODEL"
    --data_root "$DATA_ROOT"
)
if [[ -n "$CHECKPOINT" ]]; then
    ARGS+=(--checkpoint_path "$CHECKPOINT")
fi

python recon_benchmarks/eval.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
