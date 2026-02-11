export ROOT_DIR=${ROOT_DIR:-/opt/eval}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export NUM_GPU=${NUM_GPU:-4}
export BASE_PORT=${BASE_PORT:-8000}
export TASK_NAME=${TASK_NAME:-turning_on_radio}
export LOG_PATH=${LOG_PATH:-${ROOT_DIR}/eval_output}
mkdir -p $LOG_PATH

TOTAL=10
BASE=$((TOTAL / NUM_GPU))
REM=$((TOTAL % NUM_GPU))

for ((gpu=0; gpu<NUM_GPU; gpu++)); do
  s=$((gpu * BASE))
  e=$((s + BASE))
  [ "$gpu" -eq $((NUM_GPU - 1)) ] && e=$((e + REM))
  CUDA_VISIBLE_DEVICES=$gpu \
    python ${ROOT_DIR}/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_custom.py \
      policy=websocket \
      task.name=$TASK_NAME \
      log_path=$LOG_PATH \
      env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
      save_rollout=false \
      perturb_pose=false \
      use_parallel_evaluator=true \
      parallel_evaluator_start_idx=$s \
      parallel_evaluator_end_idx=$e \
      model.port=$((BASE_PORT + gpu)) \
      > $LOG_PATH/${gpu}_stdout.log 2>&1 &
done
wait