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

export BLT_SUPPRESS_ATTN_ERROR=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/third_party/blt:${PYTHONPATH:-}"
echo "[env] venv + micromamba gcc 有効化済み. gcc=$(gcc --version | head -1)"
