#!/usr/bin/env bash

set -euo pipefail

# Tasks to evaluate (each item maps to TASK_NAME in job command).
task_names=(
  "turning_on_radio"
)

# Number of jobs created per task.
gpu_per_task=5

# Number of trials assigned to each GPU.
trials_per_gpu=8

# Number of unique episodes per task (wraps around when exceeded).
episodes_per_task=10

ckpt_basename="pt12-3w-comet2-20260204022222"

if (( gpu_per_task <= 0 )); then
  echo "gpu_per_task must be > 0"
  exit 1
fi

if (( trials_per_gpu <= 0 )); then
  echo "trials_per_gpu must be > 0"
  exit 1
fi

submitted_jobs=0

for task_name in "${task_names[@]}"; do
  for (( gpu_slot = 0; gpu_slot < gpu_per_task; gpu_slot++ )); do
    slot_offset=$(( (gpu_slot * trials_per_gpu) % episodes_per_task ))

    job_command="$(cat <<EOF
export TASK_NAME="${task_name}"
ckpt_basename="${ckpt_basename}"
export PATH_TO_CKPT=/tmp/\${ckpt_basename}
export OPENPI_DATA_HOME=/opt/openpi-cache
export NUM_GPU=1
export TOTAL_TRIAL=${trials_per_gpu}
export EVAL_START_IDX=${slot_offset}
export EPISODES_PER_TASK=${episodes_per_task}
export EVAL_ROOT=/opt/eval

# prepare datasets
cd \${EVAL_ROOT}/BEHAVIOR-1K
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate behavior
python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
conda deactivate

# prepare ckpt
hf download shangkuns/\${ckpt_basename} --local-dir \${PATH_TO_CKPT}
mkdir -p \${PATH_TO_CKPT}/assets/behavior-1k/2025-challenge-demos
cp \${EVAL_ROOT}/b1k-eval-utils/norm_stats.json \${PATH_TO_CKPT}/assets/behavior-1k/2025-challenge-demos/

# run eval
bash /opt/eval/b1k-eval-utils/run_dual_eval.sh

# upload eval results
hf upload shangkuns/\${ckpt_basename} /opt/eval/eval_output \${TASK_NAME} --repo-type dataset

EOF
)"

    lep job create \
      --resource-shape my.1xl40s \
      --node-group oci-ord-lepton-001 \
      --num-workers 1 \
      --container-image littlespray/b1k-eval-light:v3 \
      --command "${job_command}" \
      --intra-job-communication=true \
      --env NVIDIA_DRIVER_CAPABILITIES=all \
      --env DEBIAN_FRONTEND=noninteractive \
      --secret HUGGING_FACE_HUB_TOKEN=SHANGKUN_HF_NV.shangkuns \
      --ttl-seconds-after-finished 259200 \
      --log-collection true \
      --queue-priority 9 \
      --can-preempt \
      --name "${task_name}-${gpu_slot}"

    submitted_jobs=$((submitted_jobs + 1))
  done
done

echo "Submitted ${submitted_jobs} lep jobs."