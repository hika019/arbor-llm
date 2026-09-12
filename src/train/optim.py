"""Optimizer / LR scheduler ファクトリ。

optimizer は `optim.optimizer` (adamw | lion) と `optim.state_precision`
(fp32 | int8 | bf8) で選ぶ。state_precision は adamw の optimizer state形式を表し、
全 parameter に一様に適用される (小さい層も除外しない)。指定した実装/精度が
使えない場合に別 optimizer や別精度へ暗黙フォールバックしてはいけない。
int8 は blockwise scale + 非線形dynamic符号帳、bf8 はfloat8_e5m2でmomentを保持し、
更新演算はどちらもFP32で行う。
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import torch


_INT8_STATE_BLOCK_SIZE = 2048
_BF8_DTYPE = torch.float8_e5m2  # E5M2: bf16 と同じ指数幅で range 重視の 8bit float

# ---- dynamic (非線形) 8bit 符号帳 --------------------------------------------
# 旧実装は blockwise absmax の「線形」int8 だった。二次モーメント (常に正で裾が重い)
# を線形量子化すると、block 内の大きな要素が scale を支配し、小さな v が 0 へ
# underflow する。すると denom=sqrt(v)+eps≈eps となり update=m/eps が爆発して発散した
# (実データ A/B で確認)。bitsandbytes 8bit Adam と同様に、対数(指数)間隔の符号帳を使い
# 相対精度を保ちつつ広いレンジを表現して underflow を避ける。
#
#   signed   … 一次モーメント用。[-1,1] を対数間隔 (0 を厳密表現)。zero index=127。
#   unsigned … 二次モーメント用。[0,1] を対数間隔 (0 を厳密表現)。zero index=0。
# 量子化は block absmax で [-1,1]/[0,1] に正規化してから最近傍符号へ丸める。
_SIGNED_MIN_LOG10 = -16.0   # 一次モーメント: absmax の 1e-16 まで表現
_UNSIGNED_MIN_LOG10 = -12.0  # 二次モーメント: absmax の 1e-12 まで表現 (二乗でレンジ広い)
_SIGNED_ZERO_INDEX = 127
_UNSIGNED_ZERO_INDEX = 0


@lru_cache(maxsize=8)
def _dynamic_codebook(signed: bool, device_str: str) -> torch.Tensor:
    """昇順ソート済みの 256 要素符号帳を返す (端点は厳密に ±1 / 0 / 1)。"""
    device = torch.device(device_str)
    if signed:
        mag_pos = torch.logspace(_SIGNED_MIN_LOG10, 0.0, steps=128, device=device)
        mag_neg = torch.logspace(_SIGNED_MIN_LOG10, 0.0, steps=127, device=device)
        codes = torch.cat([-mag_neg.flip(0), torch.zeros(1, device=device), mag_pos])
    else:
        mag = torch.logspace(_UNSIGNED_MIN_LOG10, 0.0, steps=255, device=device)
        codes = torch.cat([torch.zeros(1, device=device), mag])
    return codes.contiguous().float()


def _quantize_dynamic_state(
    x: torch.Tensor, *, signed: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """block absmax + 非線形符号帳で uint8 index へ量子化する。返り値 (index, scale)。"""
    flat = x.float().reshape(-1)
    n = flat.numel()
    pad = (-n) % _INT8_STATE_BLOCK_SIZE
    if pad:
        flat = torch.cat((flat, flat.new_zeros(pad)))
    blocks = flat.view(-1, _INT8_STATE_BLOCK_SIZE)
    scales = blocks.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    norm = blocks / scales[:, None]  # signed:[-1,1] / unsigned:[0,1]
    codes = _dynamic_codebook(signed, str(x.device))
    # 最近傍符号: searchsorted で挟む 2 符号のうち近い方を選ぶ
    idx = torch.searchsorted(codes, norm.reshape(-1).contiguous())
    idx = idx.clamp_(1, codes.numel() - 1)
    left = codes[idx - 1]
    right = codes[idx]
    take_left = (norm.reshape(-1) - left).abs() <= (right - norm.reshape(-1)).abs()
    q = torch.where(take_left, idx - 1, idx).to(torch.uint8)
    return q.reshape(-1)[:n].reshape_as(x), scales


def _dequantize_dynamic_state(
    q: torch.Tensor, scale: torch.Tensor, *, signed: bool
) -> torch.Tensor:
    codes = _dynamic_codebook(signed, str(q.device))
    vals = codes[q.reshape(-1).long()]
    flat_scale = scale.repeat_interleave(_INT8_STATE_BLOCK_SIZE)[: q.numel()]
    return (vals * flat_scale).reshape_as(q)


def _quantize_int8_state(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """後方互換用エイリアス: 一次モーメント (signed) の dynamic 量子化。"""
    return _quantize_dynamic_state(x, signed=True)


def _dequantize_int8_state(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """後方互換用エイリアス: 一次モーメント (signed) の dynamic 逆量子化。"""
    return _dequantize_dynamic_state(q, scale, signed=True)


class AdamW8bit(torch.optim.Optimizer):
    """CUDA/MPS/CPU 共通の、8bit state を持つ AdamW。

    ``exp_avg`` (一次) と ``exp_avg_sq`` (二次) は 2048 要素 block ごとに FP32 scale を
    持つ uint8 index として保持する。量子化は「線形」ではなく対数間隔の dynamic 符号帳
    (`_dynamic_codebook`) を使う: 一次は signed、二次は unsigned。二次モーメントの小さな
    値が underflow して denom≈eps → update 爆発する旧線形実装の発散を避ける。
    演算時だけ FP32 に dequantize し、step 後に明示的に再量子化する。パラメータ・勾配・
    BitNet の W1.58/A8 forward を別形式へ置換しない。
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
        """dynamic uint8 stateを復元する。旧linear int8 stateは明示エラーにする."""
        for raw_state in state_dict.get("state", {}).values():
            for key in ("exp_avg", "exp_avg_sq"):
                value = raw_state.get(key)
                if torch.is_tensor(value) and value.dtype == torch.int8:
                    raise ValueError(
                        "旧linear-int8 optimizer stateはdynamic-int8と非互換です。"
                        "optimizer stateを初期化して再開してください。"
                        "暗黙変換は行いません"
                    )
        super().load_state_dict(state_dict)
        for p, state in self.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key].to(device=p.device, dtype=torch.uint8)
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
                    # 一次モーメントの零は signed 符号帳の zero index、二次は unsigned の 0。
                    state["exp_avg"] = torch.full_like(
                        p, _SIGNED_ZERO_INDEX, dtype=torch.uint8
                    )
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint8)
                    state["exp_avg_scale"] = torch.ones(
                        n_blocks, device=p.device, dtype=torch.float32
                    )
                    state["exp_avg_sq_scale"] = torch.ones(
                        n_blocks, device=p.device, dtype=torch.float32
                    )

                state["step"] += 1
                step = state["step"]
                exp_avg = _dequantize_dynamic_state(
                    state["exp_avg"], state["exp_avg_scale"], signed=True
                )
                exp_avg_sq = _dequantize_dynamic_state(
                    state["exp_avg_sq"], state["exp_avg_sq_scale"], signed=False
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

                state["exp_avg"], state["exp_avg_scale"] = _quantize_dynamic_state(
                    exp_avg, signed=True
                )
                state["exp_avg_sq"], state["exp_avg_sq_scale"] = _quantize_dynamic_state(
                    exp_avg_sq, signed=False
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
        backend: str = "auto",
    ) -> None:
        if backend not in ("auto", "eager", "triton"):
            raise ValueError(f"unknown AdamWFP32 backend: {backend!r}")
        self.backend = backend
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
        """Restore FP32 moments without a lossy intermediate parameter-dtype cast."""
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for saved_id, p in zip(saved_group["params"], group["params"]):
                saved = state_dict["state"].get(saved_id, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in saved:
                        self.state[p][key] = saved[key].to(device=p.device, dtype=torch.float32)

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
                if self.backend != "eager":
                    from src.train.adamw_triton import adamw_update, supported

                    can_fuse = (
                        supported(p, grad)
                        and exp_avg.is_contiguous()
                        and exp_avg_sq.is_contiguous()
                    )
                    if can_fuse:
                        adamw_update(
                            p, grad, exp_avg, exp_avg_sq,
                            lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd, step=step,
                        )
                        continue
                    if self.backend == "triton":
                        raise RuntimeError(
                            "AdamWFP32 triton requires contiguous CUDA floating tensors "
                            "(NVIDIA; FP32/FP16/BF16)"
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

        return loss


class AdamWBF8(torch.optim.Optimizer):
    """全 parameter の AdamW moment state を bf8 (float8_e5m2) で保持する実装。

    float8 は指数を持つので int8 のような per-block scale は要らない。moment を
    bf8 で保存し (1 byte/要素)、更新のたびに FP32 へ dequantize して計算する。
    float8 には elementwise 演算カーネルが無いため、演算は必ず FP32 で行う。
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
        """checkpoint の parameter-dtype cast 後に bf8 state を復元する。"""
        super().load_state_dict(state_dict)
        for p, state in self.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key].to(device=p.device, dtype=_BF8_DTYPE)

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
                    raise RuntimeError("AdamWBF8 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=_BF8_DTYPE)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=_BF8_DTYPE)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()
                grad32 = grad.float()

                exp_avg.mul_(beta1).add_(grad32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad32, grad32, value=1.0 - beta2)

                if wd:
                    p.mul_(1.0 - lr * wd)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.div(denom).mul_(lr / bias_correction1)
                p.add_(update.to(dtype=p.dtype), alpha=-1.0)

                state["exp_avg"] = exp_avg.to(_BF8_DTYPE)
                state["exp_avg_sq"] = exp_avg_sq.to(_BF8_DTYPE)

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


_STATE_PRECISIONS = ("fp32", "int8", "bf8")


def resolve_state_precision(value: str | None) -> str:
    """optim.state_precision を正規化する (fp32 | int8 | bf8)。別値は暗黙変換せずエラー。"""
    if value is None:
        return "fp32"
    normalized = str(value).lower()
    if normalized in ("fp32", "float32"):
        return "fp32"
    if normalized == "int8":
        return "int8"
    if normalized == "bf8":
        return "bf8"
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
    fp32_backend: str = "auto",
) -> torch.optim.Optimizer:
    """state_precision に従い、全 parameter で一様な AdamW を作る。

    - fp32: 本モジュールの AdamWFP32 (全 parameter の moment を厳密に FP32)。
    - int8: 本モジュールの AdamW8bit (全 parameter の moment を blockwise INT8)。
    - bf8:  本モジュールの AdamWBF8 (全 parameter の moment を float8_e5m2)。
    どちらも parameter を個別 dtype 例外なく一様に扱う (小さい層も除外しない)。
    別 device への暗黙フォールバックや別 optimizer への置換はしない。
    """
    if state_precision == "int8":
        return AdamW8bit(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    if state_precision == "bf8":
        return AdamWBF8(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    return AdamWFP32(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, backend=fp32_backend)


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
        fp32_backend=str(cfg.get("fp32_backend", "auto")),
    )


class TwoStageCooldownLR(torch.optim.lr_scheduler.LambdaLR):
    """LR を cosine 2 段に、weight decay を 2 段に切り替える scheduler.

    BitNet b1.58 公式レシピ (2B4T) の 2 段構成に寄せたもの:
      - Stage 1 [warmup, stage2_start]: cosine で 1.0 -> stage2_peak_lr_ratio。
        前半は比較的高い LR を保つ (WD は stage1 値)。
      - Stage 2 [stage2_start, decay_end]: cosine で stage2_peak_lr_ratio ->
        min_lr_ratio まで cooldown。WD は stage2 値 (公式は 0)。
      - [decay_end, total]: min_lr_ratio で一定。
    LambdaLR を継承するので base_lrs / lr_lambdas / state_dict / get_last_lr /
    rebase_scheduler_lr との互換をそのまま保つ。WD は last_epoch (= step) から
    毎 step 再計算して optimizer.param_groups へ書き戻す (state を増やさない)。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda,
        *,
        wd_stage1: float,
        wd_stage2: float,
        stage2_start_step: int,
    ) -> None:
        self._wd_stage1 = float(wd_stage1)
        self._wd_stage2 = float(wd_stage2)
        self._stage2_start_step = int(stage2_start_step)
        super().__init__(optimizer, lr_lambda)
        self._apply_weight_decay()

    def _current_weight_decay(self) -> float:
        return (
            self._wd_stage2
            if self.last_epoch >= self._stage2_start_step
            else self._wd_stage1
        )

    def _apply_weight_decay(self) -> None:
        wd = self._current_weight_decay()
        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd

    def step(self, epoch=None):  # type: ignore[override]
        super().step(epoch)
        self._apply_weight_decay()


def _scheduler_common(cfg: dict) -> tuple[int, int, float, float]:
    """warmup / total / min_lr_ratio / decay_end_ratio を検証して返す."""
    warmup = cfg.get("warmup_steps", 0)
    total = cfg["total_steps"]
    # 最終 step での lr 下限 (ピーク lr に対する比率)。0 で従来どおり 0 まで減衰。
    # 下限を残すと total_steps を増やした resume での加学習が素直に効く。
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))
    if not 0.0 <= min_ratio < 1.0:
        raise ValueError(f"min_lr_ratio は [0, 1) で指定: {min_ratio}")
    # cosine 減衰を total_steps のどの時点で終えるか (比率)。0.8 なら総 step の
    # 80% で min_lr に到達し、残り 20% は min_lr で一定 (終盤の lr を低く保つ)。
    decay_end_ratio = float(cfg.get("decay_end_ratio", 1.0))
    if not 0.0 < decay_end_ratio <= 1.0:
        raise ValueError(f"decay_end_ratio は (0, 1] で指定: {decay_end_ratio}")
    return warmup, total, min_ratio, decay_end_ratio


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    name = cfg.get("scheduler", "cosine_warmup")
    warmup, total, min_ratio, decay_end_ratio = _scheduler_common(cfg)

    if name == "cosine_warmup":
        decay_end = max(warmup + 1, round(total * decay_end_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, decay_end - warmup)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cos

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if name == "two_stage":
        # BitNet 公式 2 段レシピ。stage2_start_ratio で stage を切り替え、stage1 は
        # cosine で stage2_peak_lr_ratio まで、stage2 はそこから min_lr_ratio まで
        # cooldown する。weight decay も stage1/stage2 で切り替える (公式は stage2=0)。
        stage2_start_ratio = float(cfg.get("stage2_start_ratio", 0.5))
        if not 0.0 < stage2_start_ratio < 1.0:
            raise ValueError(f"stage2_start_ratio は (0, 1) で指定: {stage2_start_ratio}")
        if decay_end_ratio <= stage2_start_ratio:
            raise ValueError(
                "decay_end_ratio は stage2_start_ratio より大きくすること "
                f"(decay_end_ratio={decay_end_ratio}, stage2_start_ratio={stage2_start_ratio})"
            )
        stage2_peak = float(cfg.get("stage2_peak_lr_ratio", 0.5))
        if not min_ratio < stage2_peak <= 1.0:
            raise ValueError(
                f"stage2_peak_lr_ratio は (min_lr_ratio, 1] で指定: {stage2_peak}"
            )
        wd_stage1 = float(cfg.get("weight_decay", 0.0))
        wd_stage2 = float(cfg.get("weight_decay_stage2", 0.0))
        if wd_stage2 < 0:
            raise ValueError(f"weight_decay_stage2 は非負で指定: {wd_stage2}")
        s2 = max(warmup + 1, round(total * stage2_start_ratio))
        decay_end = max(s2 + 1, round(total * decay_end_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            if step < s2:
                progress = (step - warmup) / max(1, s2 - warmup)
                cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return stage2_peak + (1.0 - stage2_peak) * cos
            progress = (step - s2) / max(1, decay_end - s2)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (stage2_peak - min_ratio) * cos

        return TwoStageCooldownLR(
            optimizer, lr_lambda,
            wd_stage1=wd_stage1, wd_stage2=wd_stage2, stage2_start_step=s2,
        )

    raise ValueError(f"unknown scheduler: {name}")
