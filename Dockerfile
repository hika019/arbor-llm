# syntax=docker/dockerfile:1
#
# arbor-llm v2 学習/推論用イメージ。
# v2 は自己完結実装 (torch のみ依存) なので xformers のソースビルドは不要
# (旧 BLT fork 時代の名残を撤去)。torch は再現性のため stable に固定する:
# nightly (--pre) はビルド日で中身が変わり、長期 run の途中で環境を作り直すと
# 別バージョンになるため使わない。
#
# build:  docker build -t arbor-llm .
# run:    docker run --gpus all -it -v $(pwd):/workspace/arbor-llm arbor-llm

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_BIN=python3.12
ARG VENV_PATH=/opt/arbor-venv
# cu128 index にある最新 stable (2026-06 時点)。上げるときはここだけ変える
ARG TORCH_VERSION=2.11.0
ARG TORCH_CUDA_ARCH_LIST=8.9

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=${VENV_PATH} \
    PATH=${VENV_PATH}/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ${PYTHON_BIN} \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ${PYTHON_BIN} -m venv ${VENV_PATH} \
    && ${VENV_PATH}/bin/python -m pip install -U pip setuptools wheel

# torch は CUDA wheel を固定版で (ランタイム同梱なのでホストはドライバのみ必要)
RUN ${VENV_PATH}/bin/python -m pip install \
    torch==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cu128

# 残りは requirements.txt と同内容 (ソースは /workspace/arbor-llm にマウント)
RUN ${VENV_PATH}/bin/python -m pip install \
    "pyyaml>=6.0" \
    "numpy>=1.26" \
    "safetensors>=0.4.3" \
    "datasets>=2.18" \
    "bitsandbytes>=0.43" \
    "transformers>=4.45" \
    "accelerate>=0.33" \
    "huggingface_hub>=0.23" \
    "pytest>=8.2" \
    "ruff>=0.6"

# ビルド時の import 検証 (GPU 無しでも torch import と版だけ確認できる)
RUN ${VENV_PATH}/bin/python - <<'PY'
import torch, datasets, bitsandbytes, safetensors, transformers
print("torch:", torch.__version__, torch.version.cuda)
print("datasets:", datasets.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
assert torch.version.cuda == "12.8"
PY

RUN mkdir -p /workspace/arbor-llm \
    && printf '%s\n' \
      'source /opt/arbor-venv/bin/activate' \
      'export TORCH_CUDA_ARCH_LIST="8.9"' \
      'cd /workspace/arbor-llm 2>/dev/null || cd /workspace' \
      > /root/.bashrc

WORKDIR /workspace/arbor-llm

CMD ["/bin/bash"]
