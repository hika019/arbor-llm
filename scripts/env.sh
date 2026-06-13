#!/usr/bin/env bash
# venv 用の環境変数. source して使う:
#   source scripts/env.sh
# 効果:
#   - .venv を有効化
#   - micromamba 環境の gcc/g++ を PATH に通し torch.compile / FlexAttention を動かせるように
#   - BLT 由来の attention 警告を抑止
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_ENV="${HOME}/.mamba/envs/arbor-build"

if [[ -d "${MAMBA_ENV}/bin" ]]; then
  export PATH="${MAMBA_ENV}/bin:${PATH}"
  export CC="${MAMBA_ENV}/bin/gcc"
  export CXX="${MAMBA_ENV}/bin/g++"
  # Python.h を triton/JIT が探せるよう include path を補う
  export C_INCLUDE_PATH="${MAMBA_ENV}/include/python3.12:${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="${MAMBA_ENV}/include/python3.12:${CPLUS_INCLUDE_PATH:-}"
fi

if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

# CUDA caching allocator を expandable segments 化し VRAM 断片化を抑止.
# 可変 batch/seq・大きな activation で OOM 余地を減らす (torch>=2.1, 単一GPUなら無害).
# CUDA 初期化前 = torch import 前に効かせる必要があるので shell 側で設定.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# compile worker を増やしすぎると WSL RAM を食い潰す。
# FX graph cache は2回目以降の起動待ちを短縮する。
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
echo "[env] venv + micromamba gcc 有効化済み. gcc=$(gcc --version | head -1)"
