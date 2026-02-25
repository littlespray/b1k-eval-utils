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


NODE_ID="${NODE_ID:-0}"
CKPT_NAME="${CKPT_NAME:-ckpt}"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
LOG_BASE="/opt/eval/eval_output"
LOG_PREFIX="${TASK_NAME}_${CKPT_NAME}_node${NODE_ID}"
mkdir -p "$OPENPI_DATA_HOME" "$LOG_BASE"
export OPENPI_DATA_HOME

echo "[INFO] task=$TASK_NAME gpus=$NUM_GPU eval_start=$EVAL_START eval_count=$EVAL_COUNT"

OPENPI_PIDS=()
B1K_PID=""

cleanup() {
  echo "[INFO] cleanup: killing background processes …"
  if [[ -n "$B1K_PID" ]]; then
    kill "$B1K_PID" 2>/dev/null || true
    wait "$B1K_PID" 2>/dev/null || true
  fi
  for pid in "${OPENPI_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  echo "[INFO] cleanup: done."
}
trap cleanup EXIT INT TERM HUP

# Launch one openpi policy server per GPU
for ((gpu = 0; gpu < NUM_GPU; gpu++)); do
  (
    cd /opt/eval/openpi-comet
    CUDA_VISIBLE_DEVICES="$gpu" \
    TASK_NAME="$TASK_NAME" \
    PATH_TO_CKPT="$PATH_TO_CKPT" \
    PORT="$((BASE_PORT + gpu))" \
      bash /opt/eval/openpi-comet/eval_openpi.sh
  ) >"$LOG_BASE/${LOG_PREFIX}_gpu${gpu}_openpi.log" 2>&1 &
  OPENPI_PIDS+=("$!")
done

# Run eval with episode wrapping
(
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  cd /opt/eval/BEHAVIOR-1K

  echo "[INFO] eval: start=${EVAL_START}, count=${EVAL_COUNT}, num_gpu=${NUM_GPU}"

  # Calculate total rounds needed (each round uses all GPUs in parallel)
  total_rounds=$(( (EVAL_COUNT + NUM_GPU - 1) / NUM_GPU ))

  for (( round = 0; round < total_rounds; round++ )); do
    # Calculate this round's trial range
    round_offset=$((round * NUM_GPU))
    round_start=$((EVAL_START + round_offset))
    remaining=$((EVAL_COUNT - round_offset))
    if (( remaining > NUM_GPU )); then
      round_count=$NUM_GPU
    else
      round_count=$remaining
    fi

    echo "[INFO] round=${round}: trials ${round_start}..$(( round_start + round_count - 1 )) (${round_count} trials)"

    TASK_NAME="$TASK_NAME" \
    CKPT_NAME="$CKPT_NAME" \
    LOG_BASE="$LOG_BASE" \
    TIMESTAMP="$TIMESTAMP" \
    NUM_GPU="$NUM_GPU" \
    BASE_PORT="$BASE_PORT" \
    EVAL_START="$round_start" \
    EVAL_COUNT="$round_count" \
    EPISODES_PER_TASK="$EPISODES_PER_TASK" \
    EVAL_ROUND="$round" \
    NODE_ID="$NODE_ID" \
      bash /opt/eval/BEHAVIOR-1K/eval_b1k.sh
  done
) >"$LOG_BASE/${LOG_PREFIX}_behavior_eval.log" 2>&1 &
B1K_PID=$!

wait "$B1K_PID" || true

echo "[INFO] done."
