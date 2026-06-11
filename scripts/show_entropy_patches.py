"""ByteLM checkpoint のエントロピー境界を可視化する (CPU で動く読み取り専用ツール).

使い方:
    python scripts/show_entropy_patches.py                          # 内蔵サンプル文
    python scripts/show_entropy_patches.py --text "任意のテキスト"
    python scripts/show_entropy_patches.py --threshold 1.5 --ckpt best

本体の entropy patching と同じ compute_patch_starts (min/max patch 長込み) で
境界を計算するので、学習で実際に使われる区切りと一致する。
`|` が patch 境界。日本語で UTF-8 文字の途中 (置換文字 � が出る位置) で
切れている間は ByteLM の学習不足。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.arbor import build_byte_lm, compute_patch_starts  # noqa: E402

BYTE_OFFSET = 4

DEFAULT_TEXTS = [
    "The most important thing in life is happiness.",
    "def fibonacci(n):\n    return n if n < 2 else fib(n-1)",
    "日本の四季は美しいことで知られています。",
    "今日は天気が良いので散歩に行きました。",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="latest", help="'latest' | 'best' | step dir 名")
    p.add_argument("--ckpt-dir", default="./checkpoints/entropy_lm", type=Path)
    p.add_argument("--threshold", default=2.0, type=float, help="境界判定 (nats)")
    p.add_argument("--min-patch-len", default=2, type=int)
    p.add_argument("--max-patch-len", default=16, type=int)
    p.add_argument("--text", action="append", default=None, help="複数指定可")
    args = p.parse_args()

    from safetensors.torch import load_file

    ckpt = (args.ckpt_dir / args.ckpt).resolve()
    cfg = yaml.safe_load((ckpt / "config.yaml").read_text())["model"]
    model = build_byte_lm(cfg)
    model.load_state_dict(load_file(str(ckpt / "model.safetensors"), device="cpu"), strict=True)
    model = model.float().eval()
    print(f"[show] ckpt={ckpt} threshold={args.threshold} "
          f"min/max_patch_len={args.min_patch_len}/{args.max_patch_len}")

    for text in args.text or DEFAULT_TEXTS:
        bs = text.encode("utf-8")
        ids = torch.tensor([[b + BYTE_OFFSET for b in bs]])
        with torch.no_grad():
            ent = model.next_byte_entropy(ids)
            starts = compute_patch_starts(
                ids, mode="entropy",
                min_len=args.min_patch_len, max_len=args.max_patch_len,
                entropy_model=model, threshold=args.threshold,
            )[0]
        idx = [i for i in range(len(bs)) if starts[i]]
        patches = [bs[a:b] for a, b in zip(idx, idx[1:] + [len(bs)])]
        print(f"mean_H={ent.mean():.2f} nats | patches={len(patches)} "
              f"avg_len={len(bs) / len(patches):.1f}B")
        print("  " + "|".join(p.decode("utf-8", "replace") for p in patches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
