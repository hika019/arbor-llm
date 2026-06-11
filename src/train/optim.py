"""Optimizer / LR scheduler ファクトリ。

8bit Adam (bitsandbytes) を既定。state を 1/4 に圧縮し VRAM を稼ぐ。
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


class Lion(torch.optim.Optimizer):
    """Lion optimizer with decoupled weight decay.

    This is a lightweight experiment option: one momentum tensor per parameter
    and sign-based updates. It is not a drop-in quality equivalent to AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        state_dtype: torch.dtype | None = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive: {lr}")
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"betas must be in [0, 1): {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, state_dtype=state_dtype)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            state_dtype = group["state_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                grad = p.grad
                if wd:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    dtype = state_dtype or p.dtype
                    state["exp_avg"] = torch.zeros_like(p, dtype=dtype)
                exp_avg = state["exp_avg"]
                grad_for_state = grad.to(exp_avg.dtype)

                update = exp_avg.mul(beta1).add(grad_for_state, alpha=1.0 - beta1)
                p.add_(update.sign().to(p.dtype), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad_for_state, alpha=1.0 - beta2)

        return loss


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
    if name == "lion":
        state_dtype_name = cfg.get("state_dtype")
        state_dtype = None
        if state_dtype_name:
            state_dtype = getattr(torch, state_dtype_name)
        return Lion(params, lr=lr, betas=betas, weight_decay=wd, state_dtype=state_dtype)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    name = cfg.get("scheduler", "cosine_warmup")
    warmup = cfg.get("warmup_steps", 0)
    total = cfg["total_steps"]
    # 最終 step での lr 下限 (ピーク lr に対する比率)。0 で従来どおり 0 まで減衰。
    # 下限を残しておくと total_steps を増やした resume での追加学習が素直に効く。
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))
    if not 0.0 <= min_ratio < 1.0:
        raise ValueError(f"min_lr_ratio は [0, 1) で指定: {min_ratio}")
    if name != "cosine_warmup":
        raise ValueError(f"unknown scheduler: {name}")

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_ratio + (1.0 - min_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
