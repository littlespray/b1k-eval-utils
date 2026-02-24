export ROOT_DIR=${ROOT_DIR:-/opt/eval}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export NUM_GPU=${NUM_GPU:-1}
export BASE_PORT=${BASE_PORT:-8000}
export TASK_NAME=${TASK_NAME:-turning_on_radio}
export CKPT_NAME=${CKPT_NAME:-ckpt}
export LOG_BASE=${LOG_BASE:-${ROOT_DIR}/eval_output}
export TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}
mkdir -p "$LOG_BASE"

TOTAL=${EVAL_COUNT:-4}
START=${EVAL_START_IDX:-0}

BASE=$((TOTAL / NUM_GPU))
REM=$((TOTAL % NUM_GPU))

for ((gpu = 0; gpu < NUM_GPU; gpu++)); do
  if ((gpu < REM)); then
    count=$((BASE + 1))
    s=$((START + gpu * (BASE + 1)))
  else
    count=$BASE
    s=$((START + REM * (BASE + 1) + (gpu - REM) * BASE))
  fi
  (( count == 0 )) && continue

  ROUND=${EVAL_ROUND:-0}
  NODE=${NODE_ID:-0}
  GPU_LOG_PATH="$LOG_BASE/${TASK_NAME}_${CKPT_NAME}_node${NODE}_gpu${gpu}_round${ROUND}_${TIMESTAMP}"
  mkdir -p "$GPU_LOG_PATH"

  CUDA_VISIBLE_DEVICES="$gpu" \
    python "${ROOT_DIR}/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_custom.py" \
      policy=websocket \
      task.name=$TASK_NAME \
      log_path=$GPU_LOG_PATH \
      env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
      save_rollout=false \
      perturb_pose=false \
      use_parallel_evaluator=true \
      parallel_evaluator_start_idx=$s \
      parallel_evaluator_end_idx=$((s + count)) \
      model.port=$((BASE_PORT + gpu)) \
      >> "$GPU_LOG_PATH/stdout.log" 2>&1 &
done
wait
