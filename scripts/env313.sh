#!/usr/bin/env bash
# Project-local Python 3.13 installed by micromamba.
# Usage:
#   source scripts/env313.sh
#
# This selects only the interpreter. Install project dependencies into the
# environment before training:
#   python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
#   python -m pip install -r requirements.txt
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY313_PREFIX="${PROJECT_DIR}/.python/py313"

if [[ ! -x "${PY313_PREFIX}/bin/python3.13" ]]; then
  echo "[env313] Python 3.13 is missing: ${PY313_PREFIX}" >&2
  echo "[env313] run: micromamba create -y -p '${PY313_PREFIX}' -c conda-forge python=3.13 pip" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${PY313_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export C_INCLUDE_PATH="${PY313_PREFIX}/include/python3.13:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${PY313_PREFIX}/include/python3.13:${CPLUS_INCLUDE_PATH:-}"
export PYTHONUNBUFFERED=1

echo "[env313] $(python3.13 --version) prefix=${PY313_PREFIX}"
