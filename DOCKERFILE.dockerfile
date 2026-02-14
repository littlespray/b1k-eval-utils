# Usage:
# To build without AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=no-efa .
# To build with AWS-EFA:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_BUILD_MODE=efa .
# To build with specific dependency groups:
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_EXTRAS=all .
#   docker build -t cosmos_rl:latest -f Dockerfile --build-arg COSMOS_RL_EXTRAS=wfm,vla .

ARG COSMOS_RL_BUILD_MODE=efa
ARG COSMOS_RL_EXTRAS=""

ARG CUDA_VERSION=12.8.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS no-efa-base

ARG GDRCOPY_VERSION=v2.4.4
ARG EFA_INSTALLER_VERSION=1.42.0
ARG AWS_OFI_NCCL_VERSION=v1.16.0
# NCCL version, should be found at https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG FLASH_ATTN_VERSION=2.8.3
ARG PYTHON_VERSION=3.12

ENV TZ=Etc/UTC

RUN apt-get update -y && apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    curl git gpg lsb-release tzdata wget
RUN apt-get purge -y cuda-compat-*
RUN apt-get update && apt-get install -y dnsutils

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV PATH=/opt/gdrcopy/bin:$PATH

###################################################
## Install NCCL with specific version
RUN apt-get remove -y --purge --allow-change-held-packages \
    libnccl2 \
    libnccl-dev
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update -y \
    && apt-get install -y libnccl2=${NCCL_VERSION} libnccl-dev=${NCCL_VERSION}

###################################################
## Install cuDNN
RUN apt-get update -y && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

###################################################
## Install redis
# Download and add Redis GPG key, Redis APT repository
RUN curl -fsSL https://packages.redis.io/gpg  | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb  $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

# Update package list
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y redis-server

###################################################
RUN apt-get install -qq -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
## Install python
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y --allow-change-held-packages \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
## Create a virtual environment

# RUN python${PYTHON_VERSION} -m venv /opt/venv/cosmos_rl
# ENV PATH="/opt/venv/cosmos_rl/bin:$PATH"

# RUN pip install -U pip setuptools wheel packaging psutil

# # even though we don't depend on torchaudio, vllm does. in order to
# # make sure the cuda version matches, we install it here.
# RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# # Install flash_attn separately
# # RUN pip install flash_attn==2.8.2 --no-build-isolation

# RUN pip install \
#     torchao==0.13.0 \
#     flash_attn==${FLASH_ATTN_VERSION} \
#     vllm==0.11.0 \
#     flashinfer-python==0.6.1 \
#     transformer_engine[pytorch] --no-build-isolation

# # install apex
# RUN APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex@bf903a2

# ###################################################

# # Install nvshmem grouped_gemm and DeepEP for MoE
# RUN pip install nvidia-nvshmem-cu12==3.4.5
# RUN TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4 --no-build-isolation
# RUN apt-get update && apt-get install -y  libibverbs-dev
# RUN git clone https://github.com/deepseek-ai/DeepEP.git /tmp/deepep \
#     && cd /tmp/deepep \
#     && python setup.py build \
#     && python setup.py install

# # Phase for building any lib that we want to builf from source
FROM no-efa-base AS source-build

# # install git
# RUN apt-get update -y && apt-get install -y git

# WORKDIR /workspace

# RUN git clone --branch v${FLASH_ATTN_VERSION} --single-branch https://github.com/Dao-AILab/flash-attention.git

# WORKDIR /workspace/flash-attention/hopper

# RUN python setup.py bdist_wheel


FROM no-efa-base AS efa-base

# # Remove HPCX and MPI to avoid conflicts with AWS-EFA
# RUN rm -rf /opt/hpcx \
#     && rm -rf /usr/local/mpi \
#     && rm -f /etc/ld.so.conf.d/hpcx.conf \
#     && ldconfig

RUN apt-get remove -y --purge --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1

###################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install AWS-OFI-NCCL plugin
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
#Switch from sh to bash to allow parameter expansion
SHELL ["/bin/bash", "-c"]
RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH


###################################################
# ## Image target: cosmos_rl
FROM ${COSMOS_RL_BUILD_MODE}-base AS pre-package

# WORKDIR /workspace

# # install fa3
# COPY --from=source-build /workspace/flash-attention/hopper/dist/*.whl /workspace
# RUN pip install /workspace/*.whl
# RUN rm /workspace/*.whl

###################################################
## Image target: cosmos_rl
FROM pre-package AS package-base

ARG COSMOS_RL_EXTRAS

# COPY . /workspace/cosmos_rl
# RUN apt install -y cmake && \
#     pip install /workspace/cosmos_rl${COSMOS_RL_EXTRAS:+[$COSMOS_RL_EXTRAS]} && \
#     if [[ ",$COSMOS_RL_EXTRAS," == *,vla,* ]]; then \
#         bash /workspace/cosmos_rl/tools/scripts/setup_vla.sh; \
#     fi && \
#     rm -rf /workspace/cosmos_rl
# RUN pip uninstall -y xformers

###################################################
## Install eval environment
###################################################

# System dependencies for eval
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10-dev python3-dev python3-distutils \
    libgl1 libglib2.0-0 libglu1-mesa \
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
    && git clone https://github.com/mli0603/openpi-comet.git \
    && cd openpi-comet \
    && cp ${EVAL_ROOT}/b1k-eval-utils/eval_openpi.sh ${EVAL_ROOT}/openpi-comet/eval_openpi.sh \
    && cp ${EVAL_ROOT}/b1k-eval-utils/patch_comet_safetensors.sh ${EVAL_ROOT}/openpi-comet/patch_comet_safetensors.sh \
    && GIT_LFS_SKIP_SMUDGE=1 uv sync \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . \
    && uv pip install transformers==4.53.2 \
    && cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/ \
    && bash patch_comet_safetensors.sh



# Clone BEHAVIOR-1K and install bddl/OmniGibson into openpi-comet venv
RUN git clone -b v3.7.2 --single-branch \
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

# Debug target: stop before b1k setup.sh
FROM package-base AS debug
WORKDIR ${EVAL_ROOT}/BEHAVIOR-1K
CMD ["/bin/bash"]

## Final image target: cosmos_rl with eval setup
FROM package-base AS package

# Run b1k setup.sh (separate layer — most likely to fail, easy to retry)
ENV TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"
RUN cd ${EVAL_ROOT}/BEHAVIOR-1K \
    && source /opt/miniconda3/etc/profile.d/conda.sh \
    && conda deactivate \
    && ./setup.sh --new-env --omnigibson --bddl --joylo --eval --primitives \
        --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

RUN cd ${EVAL_ROOT}/BEHAVIOR-1K \
    && source /opt/miniconda3/etc/profile.d/conda.sh \
    && conda activate behavior \
    && pip install --force-reinstall numpy==1.26.4 scipy==1.11.4 \
    && ./setup.sh --dataset --accept-dataset-tos

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

# Convenience CLI tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tmux zip unzip \
    && rm -rf /var/lib/apt/lists/*

# 
ENV OPENPI_DATA_HOME=/opt/openpi-cache
RUN mkdir -p /opt/openpi-cache/big_vision \
    && cp ${EVAL_ROOT}/b1k-eval-utils/tokenizer.model /opt/openpi-cache/big_vision/paligemma_tokenizer.model

# Install huggingface-hub in the default Python environment
RUN python -m pip install huggingface-hub wandb
