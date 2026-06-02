"""1B 規模 BLT × BitLinear のパラメータ数と学習時間の試算.

使い方:
    python scripts/size_1b.py [--config configs/arbor_1b.yaml]

1B BLT の代表的なパラメータを変えながら, 実 BLT を構築してパラメータ数を
測定し, 4090 上の現状スループット (FineWeb-Edu mini 実測) からトークン量
あたりの所要時間を概算する.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "third_party" / "blt"))

from src.model.arbor_blt import build_arbor_blt  # noqa: E402


def fmt(n: int) -> str:
    for u in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{u}"
        n /= 1000
    return f"{n:.2f}T"


def measure_params(cfg: dict) -> int:
    model = build_arbor_blt(cfg)
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    presets = [
        # (name, hidden, intermediate, n_layers, n_heads)
        ("400M-target", 1024, 2816, 18, 16),
        ("700M-target", 1280, 3520, 22, 16),
        ("1B-target",   1536, 4224, 24, 16),
        ("1.5B-target", 1792, 4928, 26, 16),
    ]
    base_data = {
        "vocab_size": 260,
        "patch_size": 4,
        "patching_mode": "space",
        "max_position_embeddings": 2048,
        "num_local_layers": 1,
        "num_key_value_heads": 16,
        "bitlinear_in_global": True,
        "fp_in_local": True,
        "backend": "blt",
    }
    print(f"{'preset':14s} {'hidden':>6s} {'inter':>6s} {'layers':>6s} {'heads':>5s} {'params':>10s}")
    print("-" * 60)
    for name, h, i, n, nh in presets:
        cfg = {
            **base_data,
            "hidden_size": h,
            "intermediate_size": i,
            "num_hidden_layers": n,
            "num_attention_heads": nh,
            "num_key_value_heads": nh,
        }
        try:
            n_params = measure_params(cfg)
            print(f"{name:14s} {h:>6d} {i:>6d} {n:>6d} {nh:>5d} {fmt(n_params):>10s}")
        except Exception as e:
            print(f"{name:14s} ERR: {type(e).__name__}: {e}")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print()
    # 実測スループット (FineWeb-Edu mini, hidden=256, BitLinear ON, 8bit Adam, BF16):
    # 安定後 28-33k tok/s, 平均 ~25k tok/s. ただしモデルサイズに反比例.
    # 4090 でのざっくり ETA:
    #   1B 目安: ~3-5k tok/s (パラメータ計算量比, 経験的)
    #   100B tokens (BitNet 標準的な学習量) 学習に:
    #     5k tok/s -> 100B / 5k = 2e7 sec = 230 day  (単機 4090)
    #     ※ Compile / Flash Attn / 速度最適化で 2-3x 改善余地はあり
    print("== ETA 試算 (4090 単機, 100B tokens 想定) ==")
    for tps in (3_000, 5_000, 10_000, 20_000):
        days = 100_000_000_000 / tps / 86400
        print(f"  {tps:>7d} tok/s  -> {days:6.1f} day ({days/30:.1f} month)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
