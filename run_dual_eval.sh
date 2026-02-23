#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="${TASK_NAME:-turning_on_radio}"
PATH_TO_CKPT="${PATH_TO_CKPT:-/opt/eval/models/cosmos-pi05-sft-pt12-20260203145532}"
CONDA_ROOT="${CONDA_ROOT:-/opt/miniconda3}"
CONDA_ENV="${CONDA_ENV:-behavior}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/opt/openpi-cache}"
BASE_PORT="${BASE_PORT:-8000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
EVAL_START="${EVAL_START:-0}"
EVAL_COUNT="${EVAL_COUNT:-10}"
NUM_GPU="${NUM_GPU:-1}"


LOG_DIR="${LOG_DIR:-/opt/eval/eval_output}"
mkdir -p "$OPENPI_DATA_HOME" "$LOG_DIR"
export OPENPI_DATA_HOME

echo "[INFO] task=$TASK_NAME gpus=$NUM_GPU eval_start=$EVAL_START eval_count=$EVAL_COUNT"

# Launch one openpi policy server per GPU
OPENPI_PIDS=()
for ((gpu = 0; gpu < NUM_GPU; gpu++)); do
  (
    cd /opt/eval/openpi-comet
    CUDA_VISIBLE_DEVICES="$gpu" \
    TASK_NAME="$TASK_NAME" \
    PATH_TO_CKPT="$PATH_TO_CKPT" \
    PORT="$((BASE_PORT + gpu))" \
      bash /opt/eval/openpi-comet/eval_openpi.sh
  ) >"$LOG_DIR/openpi_gpu${gpu}.log" 2>&1 &
  OPENPI_PIDS+=("$!")
done

# Run eval with episode wrapping
(
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  cd /opt/eval/BEHAVIOR-1K

  _remaining=$EVAL_COUNT
  _cursor=$EVAL_START

  while (( _remaining > 0 )); do
    _s=$((_cursor % EPISODES_PER_TASK))
    _chunk=$((EPISODES_PER_TASK - _s))
    (( _chunk > _remaining )) && _chunk=$_remaining

    echo "[INFO] eval chunk: episodes ${_s}..$((_s + _chunk - 1)) (${_chunk} trials)"

    TASK_NAME="$TASK_NAME" \
    LOG_PATH="$LOG_DIR" \
    NUM_GPU="$NUM_GPU" \
    BASE_PORT="$BASE_PORT" \
    EVAL_START_IDX="$_s" \
    EVAL_COUNT="$_chunk" \
      bash /opt/eval/BEHAVIOR-1K/eval_b1k.sh

    _cursor=$((_cursor + _chunk))
    _remaining=$((_remaining - _chunk))
  done
) >"$LOG_DIR/behavior_eval.log" 2>&1 &
B1K_PID=$!

wait "$B1K_PID" || true
for pid in "${OPENPI_PIDS[@]}"; do
  kill "$pid" 2>/dev/null || true
  wait "$pid" || true
done

echo "[INFO] done."
