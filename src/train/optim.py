"""Optimizer / LR scheduler ファクトリ。

optimizer は `optim.optimizer` (adamw | lion) と `optim.state_precision`
(fp32 | int8) で選ぶ。state_precision は adamw の optimizer state dtype を表し、
全 parameter に一様に適用される (小さい層も除外しない)。指定した実装/精度が
使えない場合に別 optimizer や別精度へ暗黙フォールバックしてはいけない。
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


_INT8_STATE_BLOCK_SIZE = 2048


def _quantize_int8_state(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """blockwise signed int8 + FP32 scale に量子化する。

    optimizer state 専用。scale=0 を避けるため scale に FP32 の最小正規値を使う。
    量子化 state は実際に torch.int8 で保持し、更新時だけ FP32 に戻す。
    """
    flat = x.float().reshape(-1)
    n = flat.numel()
    pad = (-n) % _INT8_STATE_BLOCK_SIZE
    if pad:
        flat = torch.cat((flat, flat.new_zeros(pad)))
    blocks = flat.view(-1, _INT8_STATE_BLOCK_SIZE)
    scales = (blocks.abs().amax(dim=1) / 127.0).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    q = (blocks / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(-1)[:n].reshape_as(x), scales


def _dequantize_int8_state(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    flat_scale = scale.repeat_interleave(_INT8_STATE_BLOCK_SIZE)[: q.numel()]
    return (q.reshape(-1).float() * flat_scale).reshape_as(q)


class AdamW8bit(torch.optim.Optimizer):
    """CUDA/MPS/CPU 共通の、INT8 state を持つ AdamW。

    ``exp_avg`` と ``exp_avg_sq`` は signed INT8、2048 要素 block ごとに FP32
    scale を持つ。演算時だけ FP32 に dequantize し、step 後に明示的に再量子化する。
    パラメータ・勾配・BitNet の W1.58/A8 forward を別形式へ置換しない。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive: {lr}")
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"betas must be in [0, 1): {betas}")
        if eps <= 0:
            raise ValueError(f"eps must be positive: {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        """PyTorch の parameter-dtype cast 後に、宣言した state dtype を復元する。"""
        super().load_state_dict(state_dict)
        for p, state in self.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key].to(device=p.device, dtype=torch.int8)
            for key in ("exp_avg_scale", "exp_avg_sq_scale"):
                if key in state:
                    state[key] = state[key].to(device=p.device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdamW8bit does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    n_blocks = math.ceil(p.numel() / _INT8_STATE_BLOCK_SIZE)
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.int8)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.int8)
                    state["exp_avg_scale"] = torch.ones(
                        n_blocks, device=p.device, dtype=torch.float32
                    )
                    state["exp_avg_sq_scale"] = torch.ones(
                        n_blocks, device=p.device, dtype=torch.float32
                    )

                state["step"] += 1
                step = state["step"]
                exp_avg = _dequantize_int8_state(
                    state["exp_avg"], state["exp_avg_scale"]
                )
                exp_avg_sq = _dequantize_int8_state(
                    state["exp_avg_sq"], state["exp_avg_sq_scale"]
                )
                grad32 = grad.float()

                exp_avg.mul_(beta1).add_(grad32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad32, grad32, value=1.0 - beta2
                )

                if wd:
                    p.mul_(1.0 - lr * wd)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.div(denom).mul_(lr / bias_correction1)
                p.add_(update.to(dtype=p.dtype), alpha=-1.0)

                state["exp_avg"], state["exp_avg_scale"] = _quantize_int8_state(exp_avg)
                state["exp_avg_sq"], state["exp_avg_sq_scale"] = _quantize_int8_state(
                    exp_avg_sq
                )

        return loss


class AdamWFP32(torch.optim.Optimizer):
    """全 parameter の AdamW moment state を厳密に FP32 で保持する実装。"""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive: {lr}")
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"betas must be in [0, 1): {betas}")
        if eps <= 0:
            raise ValueError(f"eps must be positive: {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        """parameter が BF16/FP16 でも checkpoint state を FP32 に戻す。"""
        super().load_state_dict(state_dict)
        for p, state in self.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key].to(device=p.device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdamWFP32 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad32 = grad.float()

                exp_avg.mul_(beta1).add_(grad32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad32, grad32, value=1.0 - beta2
                )

                if wd:
                    p.mul_(1.0 - lr * wd)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.div(denom).mul_(lr / bias_correction1)
                p.add_(update.to(dtype=p.dtype), alpha=-1.0)

        return loss


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


_STATE_PRECISIONS = ("fp32", "int8")


def resolve_state_precision(value: str | None) -> str:
    """optim.state_precision を正規化する (fp32 | int8)。別値は暗黙変換せずエラー。"""
    if value is None:
        return "fp32"
    normalized = str(value).lower()
    if normalized in ("fp32", "float32"):
        return "fp32"
    if normalized == "int8":
        return "int8"
    raise ValueError(
        f"unknown optim.state_precision: {value!r} (choices: {_STATE_PRECISIONS})"
    )


def _build_adamw(
    params: list[torch.nn.Parameter],
    *,
    state_precision: str,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    wd: float,
) -> torch.optim.Optimizer:
    """state_precision に従い、全 parameter で一様な AdamW を作る。

    - fp32: 本モジュールの AdamWFP32 (全 parameter の moment を厳密に FP32)。
    - int8: 本モジュールの AdamW8bit (全 parameter の moment を blockwise INT8)。
    どちらも parameter を個別 dtype 例外なく一様に扱う (小さい層も除外しない)。
    別 device への暗黙フォールバックや別 optimizer への置換はしない。
    """
    if state_precision == "int8":
        return AdamW8bit(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    return AdamWFP32(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)


def build_optimizer(params: Iterable[torch.nn.Parameter], cfg: dict) -> torch.optim.Optimizer:
    params = list(params)
    if not params:
        raise ValueError("optimizer parameter list is empty")
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = cfg["lr"]
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = cfg.get("eps", 1e-8)
    wd = cfg.get("weight_decay", 0.0)
    cfg_precision = cfg.get("state_precision")

    if name == "lion":
        if cfg_precision is not None:
            raise ValueError(
                "optim.state_precision は adamw 系専用です。lion では指定できません"
            )
        state_dtype_name = cfg.get("state_dtype")
        state_dtype = getattr(torch, state_dtype_name) if state_dtype_name else None
        return Lion(params, lr=lr, betas=betas, weight_decay=wd, state_dtype=state_dtype)

    if name != "adamw":
        raise ValueError(f"unknown optimizer: {name}")

    return _build_adamw(
        params,
        state_precision=resolve_state_precision(cfg_precision),
        lr=lr,
        betas=betas,
        eps=eps,
        wd=wd,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    name = cfg.get("scheduler", "cosine_warmup")
    warmup = cfg.get("warmup_steps", 0)
    total = cfg["total_steps"]
    # 最終 step での lr 下限 (ピーク lr に対する比率)。0 で従来どおり 0 まで減衰。
    # 下限を残しておくと total_steps を増やした resume での追加学習が素直に効く。
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))
    if not 0.0 <= min_ratio < 1.0:
        raise ValueError(f"min_lr_ratio は [0, 1) で指定: {min_ratio}")
    # cosine 減衰を total_steps のどの時点で終えるか (比率)。0.8 なら総 step の
    # 80% で min_lr に到達し、残り 20% は min_lr で一定 (終盤の lr を低く保つ)。
    decay_end_ratio = float(cfg.get("decay_end_ratio", 1.0))
    if not 0.0 < decay_end_ratio <= 1.0:
        raise ValueError(f"decay_end_ratio は (0, 1] で指定: {decay_end_ratio}")
    decay_end = max(warmup + 1, round(total * decay_end_ratio))
    if name != "cosine_warmup":
        raise ValueError(f"unknown scheduler: {name}")

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, decay_end - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_ratio + (1.0 - min_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
