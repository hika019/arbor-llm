#!/usr/bin/env bash
# venv (Python 3.12 / torch 2.11) 用の環境変数. source して使う:
#   source scripts/env.sh
# 本走は Python 3.13 側なので、本走を再現するときは scripts/env313.sh を使うこと。
# 効果:
#   - .venv を有効化
#   - micromamba 環境の gcc/g++ と inductor 設定 (env_common.sh)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARBOR_PY_INCLUDE="${HOME}/.mamba/envs/arbor-build/include/python3.12"
# shellcheck source=scripts/env_common.sh
source "${PROJECT_DIR}/scripts/env_common.sh"

if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

echo "[env] venv + micromamba gcc 有効化済み. gcc=$(gcc --version | head -1)"
echo "[env] python=$(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null) torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '(未導入)')"
