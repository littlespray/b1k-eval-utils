#!/usr/bin/env bash
set -euo pipefail

task_names=(
  "turning_on_radio"
  # "picking_up_trash"
  # "picking_up_toys"
  # "rearranging_kitchen_furniture"
  # "putting_up_Christmas_decorations_inside"
  # "sorting_vegetables"
  # "putting_shoes_on_rack"
  # "boxing_books_up_for_storage"
  # "storing_food"
  # "sorting_household_items"
  # "wash_a_baseball_cap"
  # "wash_dog_toys"
  # "hanging_pictures"
  # "attach_a_camera_to_a_tripod"
  # "clean_a_trumpet"
  # "spraying_for_bugs"
  # "make_microwave_popcorn"
  # "freeze_pies"
)

node_per_task=2
gpu_per_node=4
episodes_per_task=10
num_test_per_episode=2
ckpt_basename="pi05-b1kpt12-cs32"

total_work=$((episodes_per_task * num_test_per_episode))
base=$((total_work / node_per_task))
rem=$((total_work % node_per_task))

submitted=0

for task_name in "${task_names[@]}"; do
  for (( node = 0; node < node_per_task; node++ )); do
    if (( node < rem )); then
      node_trials=$((base + 1))
      node_start=$((node * (base + 1)))
    else
      node_trials=$base
      node_start=$((rem * (base + 1) + (node - rem) * base))
    fi

    job_command="$(cat <<EOF
export TASK_NAME="${task_name}"
export CKPT_BASENAME="${ckpt_basename}"
export PATH_TO_CKPT=/tmp/\${CKPT_BASENAME}
export OPENPI_DATA_HOME=/opt/openpi-cache
export EPISODES_PER_TASK=${episodes_per_task}
export EVAL_START=${node_start}
export EVAL_COUNT=${node_trials}
export NUM_GPU=${gpu_per_node}

EVAL_ROOT=/opt/eval

cd \${EVAL_ROOT}/BEHAVIOR-1K
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate behavior
python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
conda deactivate

hf download shangkuns/\${CKPT_BASENAME} --local-dir \${PATH_TO_CKPT}
mkdir -p \${PATH_TO_CKPT}/assets/behavior-1k/2025-challenge-demos
cp \${EVAL_ROOT}/b1k-eval-utils/norm_stats.json \${PATH_TO_CKPT}/assets/behavior-1k/2025-challenge-demos/

cd \${EVAL_ROOT}/b1k-eval-utils
bash patch_walltime.sh \${EVAL_ROOT}/BEHAVIOR-1K/OmniGibson/omnigibson/learning
bash run_dual_eval.sh

hf upload shangkuns/\${CKPT_BASENAME} /opt/eval/eval_output \${TASK_NAME} --repo-type dataset
EOF
)"

    lep job create \
      --resource-shape "my.${gpu_per_node}xl40s" \
      --node-group oci-ord-lepton-001 \
      --num-workers 1 \
      --container-image littlespray/b1k-eval-light:v4 \
      --command "${job_command}" \
      --intra-job-communication=true \
      --env NVIDIA_DRIVER_CAPABILITIES=all \
      --env DEBIAN_FRONTEND=noninteractive \
      --secret HUGGING_FACE_HUB_TOKEN=SHANGKUN_HF_NV.shangkuns \
      --ttl-seconds-after-finished 259200 \
      --log-collection true \
      --queue-priority 8 \
      --can-preempt \
      --name "${task_name//_/-}-n${node}"

    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} jobs."
