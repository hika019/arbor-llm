#!/usr/bin/env python3
"""ReazonSpeech v2 の書き起こしを学習用 parquet に手動で用意する (学習時は自動なので通常不要).

学習 config の source に ``prepare: reazonspeech_v2_text`` があれば、データ読み込みの初期化時に
同じ処理 (src/data/prepare.py) が走り、完了印があればスキップする。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare import ensure_prepared  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "datasets" / "reazonspeech_text")
    args = ap.parse_args()
    ensure_prepared("reazonspeech_v2_text", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
