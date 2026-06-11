"""学習済みチェックポイントから perplexity を測る評価エントリポイント.

実行:
    python -m src.eval.run_eval --config configs/smoke.yaml --ckpt latest --max-batches 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.byte_dataset import build_byte_dataloader  # noqa: E402
from src.eval.perplexity import byte_perplexity  # noqa: E402
from src.model.arbor import build_arbor  # noqa: E402
from src.train.checkpoint import CheckpointManager  # noqa: E402
from src.train.train import apply_speed_settings, load_config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--ckpt", default="latest", help="'latest' | 'best' | path")
    p.add_argument("--max-batches", type=int, default=20)
    args = p.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))
    apply_speed_settings(cfg.get("speed", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_arbor(cfg["model"]).to(device=device, dtype=torch.bfloat16)

    ckpt = CheckpointManager(cfg["checkpoint"]["dir"], async_save=False)
    meta, _ = ckpt.load(args.ckpt, model, map_location=device)
    print(f"[eval] loaded {args.ckpt}: step={meta.global_step} best_loss={meta.best_loss:.4f}")

    data_cfg = dict(cfg["data"])
    data_cfg.setdefault("micro_batch_size", cfg.get("speed", {}).get("micro_batch_size", 4))
    loader = build_byte_dataloader(data_cfg, split="train")
    ppl = byte_perplexity(model, loader, device, max_batches=args.max_batches)
    print(f"[eval] byte_perplexity ({args.max_batches} batches) = {ppl:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
