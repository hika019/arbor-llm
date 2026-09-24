#!/usr/bin/env bash
# Project-local Python 3.13 (torch 2.14) — 本走はこちら。source して使う:
#   source scripts/env313.sh
#
# 依存の導入:
#   python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
#   python -m pip install -r requirements.txt
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY313_PREFIX="${PROJECT_DIR}/.python/py313"

if [[ ! -x "${PY313_PREFIX}/bin/python3.13" ]]; then
  echo "[env313] Python 3.13 is missing: ${PY313_PREFIX}" >&2
  echo "[env313] run: micromamba create -y -p '${PY313_PREFIX}' -c conda-forge python=3.13 pip" >&2
  return 1 2>/dev/null || exit 1
fi

# gcc / inductor 設定は env.sh と共有する。これを入れ忘れると torch.compile が
# 黙って eager に落ちる (BitNet では step ~12 倍)。
ARBOR_PY_INCLUDE="${PY313_PREFIX}/include/python3.13"
# shellcheck source=scripts/env_common.sh
source "${PROJECT_DIR}/scripts/env_common.sh"

# env_common.sh が micromamba を PATH 先頭に置くので、その後に py313 を被せる。
export PATH="${PY313_PREFIX}/bin:${PATH}"

echo "[env313] $(python3.13 --version) prefix=${PY313_PREFIX}"
echo "[env313] torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '(未導入)') cc=${CC:-(なし)}"
