#!/usr/bin/env bash
set -euo pipefail

# Configurable parameters (can be overridden via env).
TASK_NAME="${TASK_NAME:-turning_on_radio}"
PATH_TO_CKPT="${PATH_TO_CKPT:-/opt/eval/models/cosmos-pi05-sft-pt12-20260203145532}"
CONDA_ROOT="${CONDA_ROOT:-/opt/miniconda3}"
CONDA_ENV="${CONDA_ENV:-behavior}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/opt/openpi-cache}"

CKPT_NAME="$(basename "$PATH_TO_CKPT")"
LOG_DIR="${LOG_DIR:-/opt/eval/eval_output/${TASK_NAME}_${CKPT_NAME}_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$OPENPI_DATA_HOME"
mkdir -p "$LOG_DIR"
export OPENPI_DATA_HOME

echo "[INFO] task_name=$TASK_NAME"
echo "[INFO] checkpoint=$PATH_TO_CKPT"
echo "[INFO] openpi_data_home=$OPENPI_DATA_HOME"
echo "[INFO] logs=$LOG_DIR"

(
  set -euo pipefail
  cd /opt/eval/openpi-comet
  echo "[INFO] starting policy server..."
  TASK_NAME="$TASK_NAME" PATH_TO_CKPT="$PATH_TO_CKPT" \
    bash /opt/eval/openpi-comet/eval_openpi.sh
) >"$LOG_DIR/openpi.log" 2>&1 &
OPENPI_PID=$!

(
  set -euo pipefail
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  cd /opt/eval/BEHAVIOR-1K
  echo "[INFO] starting behavior eval..."
  TASK_NAME="$TASK_NAME" LOG_PATH="$LOG_DIR" bash /opt/eval/BEHAVIOR-1K/eval_b1k.sh
) >"$LOG_DIR/behavior_eval.log" 2>&1 &
B1K_PID=$!

echo "[INFO] openpi pid=$OPENPI_PID"
echo "[INFO] behavior eval pid=$B1K_PID"
echo "[INFO] tail logs:"
echo "       tail -f \"$LOG_DIR/openpi.log\""
echo "       tail -f \"$LOG_DIR/behavior_eval.log\""

set +e
wait "$B1K_PID"
B1K_STATUS=$?
if kill -0 "$OPENPI_PID" 2>/dev/null; then
  kill "$OPENPI_PID" 2>/dev/null
  wait "$OPENPI_PID"
  OPENPI_STATUS=0
else
  wait "$OPENPI_PID"
  OPENPI_STATUS=$?
fi
set -e

if [[ "$OPENPI_STATUS" -ne 0 || "$B1K_STATUS" -ne 0 ]]; then
  echo "[ERROR] openpi exit=$OPENPI_STATUS, behavior exit=$B1K_STATUS"
  exit 1
fi

echo "[INFO] both processes finished successfully."
