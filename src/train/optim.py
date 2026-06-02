"""Optimizer / LR scheduler ファクトリ。

8bit Adam (bitsandbytes) を既定。state を 1/4 に圧縮し VRAM を稼ぐ。
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


def build_optimizer(params: Iterable[torch.nn.Parameter], cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "bnb_adamw_8bit")
    lr = cfg["lr"]
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = cfg.get("eps", 1e-8)
    wd = cfg.get("weight_decay", 0.0)

    if name == "bnb_adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        except ImportError:
            # bnb 不在環境では fused AdamW にフォールバック (smoke / CPU 用).
            print("[optim] bitsandbytes 未導入: AdamW(fused) にフォールバック")
            name = "adamw_fused"
    if name == "adamw_fused":
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, fused=fused)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    name = cfg.get("scheduler", "cosine_warmup")
    warmup = cfg.get("warmup_steps", 0)
    total = cfg["total_steps"]
    if name != "cosine_warmup":
        raise ValueError(f"unknown scheduler: {name}")

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
