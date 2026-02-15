#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-cook_hot_dogs}"
CONFIG="${CONFIG:-pi05_b1k-base}"
PATH_TO_CKPT="${PATH_TO_CKPT:-/workspace/comet-pi05-b1k-pt50-ptm}"
PORT="${PORT:-8000}"

exec uv run scripts/serve_b1k.py \
  --task_name "$TASK_NAME" \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config "$CONFIG" \
  --policy.dir "$PATH_TO_CKPT"