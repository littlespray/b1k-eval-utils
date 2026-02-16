#!/usr/bin/env bash
set -euo pipefail

# Configurable parameters (can be overridden via env).
TASK_NAME="${TASK_NAME:-turning_on_radio}"
PATH_TO_CKPT="${PATH_TO_CKPT:-/opt/eval/models/cosmos-pi05-sft-pt12-20260203145532}"
CONDA_ROOT="${CONDA_ROOT:-/opt/miniconda3}"
CONDA_ENV="${CONDA_ENV:-behavior}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/opt/openpi-cache}"
NUM_GPU="${NUM_GPU:-1}"
BASE_PORT="${BASE_PORT:-8000}"

CKPT_NAME="$(basename "$PATH_TO_CKPT")"
LOG_DIR="${LOG_DIR:-/opt/eval/eval_output/${TASK_NAME}_${CKPT_NAME}_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$OPENPI_DATA_HOME"
mkdir -p "$LOG_DIR"
export OPENPI_DATA_HOME

echo "[INFO] task_name=$TASK_NAME"
echo "[INFO] checkpoint=$PATH_TO_CKPT"
echo "[INFO] openpi_data_home=$OPENPI_DATA_HOME"
echo "[INFO] logs=$LOG_DIR"
echo "[INFO] num_gpu=$NUM_GPU"
echo "[INFO] base_port=$BASE_PORT"

OPENPI_PIDS=()

for ((gpu=0; gpu<NUM_GPU; gpu++)); do
  PORT=$((BASE_PORT + gpu))
  OPENPI_LOG="$LOG_DIR/openpi_gpu${gpu}.log"
  echo "[INFO] starting policy server gpu=$gpu port=$PORT (log=$OPENPI_LOG)"

  (
    set -euo pipefail
    cd /opt/eval/openpi-comet
    CUDA_VISIBLE_DEVICES="$gpu" \
    TASK_NAME="$TASK_NAME" \
    PATH_TO_CKPT="$PATH_TO_CKPT" \
    PORT="$PORT" \
      bash /opt/eval/openpi-comet/eval_openpi.sh
  ) >"$OPENPI_LOG" 2>&1 &
  OPENPI_PIDS+=("$!")
done

(
  set -euo pipefail
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  cd /opt/eval/BEHAVIOR-1K
  echo "[INFO] starting behavior eval..."
  TASK_NAME="$TASK_NAME" \
  LOG_PATH="$LOG_DIR" \
  NUM_GPU="$NUM_GPU" \
  BASE_PORT="$BASE_PORT" \
    bash /opt/eval/BEHAVIOR-1K/eval_b1k.sh
) >"$LOG_DIR/behavior_eval.log" 2>&1 &
B1K_PID=$!

echo "[INFO] openpi pids=${OPENPI_PIDS[*]:-(none)}"
echo "[INFO] behavior eval pid=$B1K_PID"
echo "[INFO] tail logs:"
echo "       ls -1 \"$LOG_DIR\"/openpi_gpu*.log"
echo "       tail -f \"$LOG_DIR/behavior_eval.log\""

wait "$B1K_PID" || true
for pid in "${OPENPI_PIDS[@]:-}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
done
for pid in "${OPENPI_PIDS[@]:-}"; do
  wait "$pid" || true
done

echo "[INFO] both processes finished."
