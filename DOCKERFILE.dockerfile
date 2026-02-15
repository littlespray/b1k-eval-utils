ARG COSMOS_RL_BUILD_MODE=efa
ARG COSMOS_RL_EXTRAS=""

ARG CUDA_VERSION=12.8.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

# System dependencies for eval
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10-dev python3-dev python3-distutils \
    libgl1 libglib2.0-0 libglu1-mesa \
    curl git ca-certificates \
    tmux zip unzip vim \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Miniconda
RUN curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda3 \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/miniconda3/bin:$PATH"

ENV EVAL_ROOT=/opt/eval

# Clone b1k-eval-utils
RUN mkdir -p ${EVAL_ROOT} \
    && git clone https://github.com/littlespray/b1k-eval-utils.git ${EVAL_ROOT}/b1k-eval-utils

# Clone and setup openpi-comet
SHELL ["/bin/bash", "-c"]
RUN cd ${EVAL_ROOT} \
    && git clone https://github.com/mli0603/openpi-comet.git --depth=1 \
    && cd openpi-comet \
    && cp ${EVAL_ROOT}/b1k-eval-utils/eval_openpi.sh ${EVAL_ROOT}/openpi-comet/eval_openpi.sh \
    && cp ${EVAL_ROOT}/b1k-eval-utils/patch_comet_safetensors.sh ${EVAL_ROOT}/openpi-comet/patch_comet_safetensors.sh \
    && GIT_LFS_SKIP_SMUDGE=1 uv sync \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . \
    && uv pip install transformers==4.53.2 \
    && cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ \
    && bash patch_comet_safetensors.sh


# Clone BEHAVIOR-1K and install bddl/OmniGibson into openpi-comet venv
RUN git clone -b v3.7.2 --single-branch --depth=1 \
    https://github.com/StanfordVL/BEHAVIOR-1K.git ${EVAL_ROOT}/BEHAVIOR-1K

RUN cd ${EVAL_ROOT}/BEHAVIOR-1K \
    && source ${EVAL_ROOT}/openpi-comet/.venv/bin/activate \
    && uv pip install -e bddl3 \
    && uv pip install -e "OmniGibson[eval]" \
    && deactivate

# Prepare b1k environment
# RUN cd ${EVAL_ROOT}/BEHAVIOR-1K \
#     && cp ${EVAL_ROOT}/b1k-eval-utils/patch_setup.sh patch_setup.sh \
#     && bash patch_setup.sh \

WORKDIR ${EVAL_ROOT}/BEHAVIOR-1K

# Run b1k setup.sh (separate layer — most likely to fail, easy to retry)
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"
RUN cd ${EVAL_ROOT}/BEHAVIOR-1K \
    && source /opt/miniconda3/etc/profile.d/conda.sh \
    && conda deactivate \
    && ./setup.sh --new-env --omnigibson --bddl --joylo --eval --primitives \
        --accept-conda-tos --accept-nvidia-eula
        
RUN source /opt/miniconda3/etc/profile.d/conda.sh \
    && conda activate behavior \
    && pip install scipy==1.11.4 \
    && pip uninstall -y numpy \
    && pip uninstall -y numpy \
    && pip install numpy==1.26.4 opencv-contrib-python==4.10.0.84



# RUN cd ${EVAL_ROOT}/BEHAVIOR-1K \
#     && source /opt/miniconda3/etc/profile.d/conda.sh \
#     && conda activate behavior \
#     && python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
#     && python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
#     && python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"

# Copy eval script
RUN cp ${EVAL_ROOT}/b1k-eval-utils/eval_b1k.sh ${EVAL_ROOT}/BEHAVIOR-1K/eval_b1k.sh

# Download HF models into the image for offline eval.
# Append new model repos here, separated by spaces.

# ENV HF_MODEL_REPOS="shangkuns/pt12-3w-comet2-20260204022222 shangkuns/pt12-2w-comet2-20260205151209 shangkuns/cosmos-pi05-sft-pt12-20260203145532 shangkuns/pt12-3w-20260204011934"
# RUN pip install --no-cache-dir "huggingface_hub[cli]" \
#     && mkdir -p /opt/eval/models \
#     && for repo in $HF_MODEL_REPOS; do \
#         name="${repo#*/}"; \
#         hf download "$repo" --repo-type model --local-dir "/opt/eval/models/$name"; \
#     done


RUN cd ${EVAL_ROOT} \
&& cp -r openpi-comet/src/behavior/learning/* BEHAVIOR-1K/OmniGibson/omnigibson/learning/

# 
ENV OPENPI_DATA_HOME=/opt/openpi-cache
RUN mkdir -p /opt/openpi-cache/big_vision \
    && cp ${EVAL_ROOT}/b1k-eval-utils/tokenizer.model /opt/openpi-cache/big_vision/paligemma_tokenizer.model

# Install huggingface-hub in the default Python environment
RUN python -m pip install huggingface-hub wandb
