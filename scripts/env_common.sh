#!/usr/bin/env bash
# env.sh / env313.sh が共有する toolchain + inductor 設定。
# 単体では source しない。呼び出し側が ARBOR_PY_INCLUDE (Python.h のあるディレクトリ)
# を設定してから source する。
#
# ここを env.sh 側にだけ置いていた結果、env313.sh (本走の Python 3.13) では
# gcc も inductor の設定も効かず、torch.compile が黙って eager に落ちる /
# nsys の計測が本走と別環境になる、という事故が起きた (2026-09-24)。
_ENVC_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ENVC_MAMBA_ENV="${HOME}/.mamba/envs/arbor-build"

if [[ -d "${_ENVC_MAMBA_ENV}/bin" ]]; then
  # micromamba の gcc/g++ を通す。これが無いと inductor はエラーを出さずに
  # eager へフォールバックし、BitNet では step が ~12 倍になる。
  export PATH="${_ENVC_MAMBA_ENV}/bin:${PATH}"
  export CC="${_ENVC_MAMBA_ENV}/bin/gcc"
  export CXX="${_ENVC_MAMBA_ENV}/bin/g++"
  # Python.h を triton/JIT が探せるよう include path を補う。multiarch ヘッダを
  # 相対 include するケースがあるため /usr/include も必要 (無いと新しい compile
  # グラフ形状が出る度に triton launcher のビルドが失敗する)。
  export C_INCLUDE_PATH="${ARBOR_PY_INCLUDE:-}:/usr/include:${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="${ARBOR_PY_INCLUDE:-}:/usr/include:${CPLUS_INCLUDE_PATH:-}"
fi

# CUDA caching allocator を expandable segments 化し VRAM 断片化を抑止.
# CUDA 初期化前 = torch import 前に効かせる必要があるので shell 側で設定.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1B モデルの初回 compile は host RAM の peak が大きいため WSL 既定では控えめに。
# 潤沢な RAM の環境では source 前に上書きしてよい。
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-6}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export PYTHONPATH="${_ENVC_PROJECT_DIR}:${PYTHONPATH:-}"
# stdout をファイルにリダイレクトする本番 run 向け: block buffering だと print が
# 溜まるまでログに出ず監視しづらいので常に unbuffered にする。
export PYTHONUNBUFFERED=1

unset _ENVC_PROJECT_DIR _ENVC_MAMBA_ENV
