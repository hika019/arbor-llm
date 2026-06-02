#!/usr/bin/env bash
# BLT を自分の GitHub アカウントに fork した後、リモートを差し替える手順。
#   1. https://github.com/facebookresearch/blt を Web UI で fork
#   2. このスクリプトを fork 先 URL 付きで実行
#       例) ./scripts/setup_blt_fork.sh git@github.com:<your-account>/blt.git
set -euo pipefail

FORK_URL="${1:-}"
if [[ -z "$FORK_URL" ]]; then
  echo "usage: $0 <your-blt-fork-url>" >&2
  exit 1
fi

cd "$(dirname "$0")/../third_party/blt"

# 既存 upstream を保持し、origin に fork を入れる
git remote remove origin 2>/dev/null || true
git remote add origin "$FORK_URL"
git fetch origin
echo
git remote -v
echo
echo "OK: origin=fork, upstream=facebookresearch/blt"
