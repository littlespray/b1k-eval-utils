export ROOT_DIR=${ROOT_DIR:-/opt/eval}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export NUM_GPU=${NUM_GPU:-1}
export BASE_PORT=${BASE_PORT:-8000}
export TASK_NAME=${TASK_NAME:-turning_on_radio}
export CKPT_NAME=${CKPT_NAME:-ckpt}
export LOG_BASE=${LOG_BASE:-${ROOT_DIR}/eval_output}
export TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}
export EPISODES_PER_TASK=${EPISODES_PER_TASK:-10}
mkdir -p "$LOG_BASE"

TOTAL=${EVAL_COUNT:-4}
START=${EVAL_START:-0}
NODE=${NODE_ID:-0}
ROUND=${EVAL_ROUND:-0}

# Each GPU gets one trial in this round (TOTAL <= NUM_GPU per round)
for ((gpu = 0; gpu < TOTAL; gpu++)); do
  # Calculate episode index with wrap-around
  trial_idx=$((START + gpu))
  episode_idx=$((trial_idx % EPISODES_PER_TASK))

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
      parallel_evaluator_start_idx=$episode_idx \
      parallel_evaluator_end_idx=$((episode_idx + 1)) \
      model.port=$((BASE_PORT + gpu)) \
      >> "$GPU_LOG_PATH/stdout.log" 2>&1 &
done
wait
