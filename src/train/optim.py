"""Optimizer / LR scheduler ファクトリ。

optimizer は `optim.optimizer` (adamw | muon | lion) と `optim.state_precision`
(fp32 | int8 | bf8) で選ぶ。muon は transformer Block 内の 2D 重みだけを Muon で更新し、
残り (embedding / head / norm / patch_proj 等) は fp32 state の AdamW で更新する。state_precision は adamw の optimizer state形式を表し、
全 parameter に一様に適用される (小さい層も除外しない)。指定した実装/精度が
使えない場合に別 optimizer や別精度へ暗黙フォールバックしてはいけない。
int8 は blockwise scale + 非線形dynamic符号帳、bf8 はfloat8_e5m2でmomentを保持し、
更新演算はどちらもFP32で行う。
parameter への書き戻しは `optim.param_rounding` (nearest | stochastic) で丸める。
fp32 master を持たない BF16 parameter では stochastic が既定 (src/train/rounding.py)。
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Iterable

import torch

from src.train.rounding import (
    check_param_rounding_dtype,
    new_base_seed,
    resolve_param_rounding,
    round_to_param_dtype,
    step_seed,
)


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


def _apply_update(p: torch.Tensor, update32: torch.Tensor, decay: float, rounding: str) -> None:
    """decoupled weight decay と update を parameter へ書き戻す。

    nearest は従来どおり decay 後と update をそれぞれ parameter dtype へ丸める
    (Triton 版と bit 一致させる契約)。stochastic は fp32 で new = p*(1-decay) - update
    を作り、1 回だけ確率的に丸める。
    """
    if rounding == "nearest":
        if decay:
            p.mul_(1.0 - decay)
        p.add_(update32.to(dtype=p.dtype), alpha=-1.0)
        return
    check_param_rounding_dtype(rounding, p.dtype)
    new32 = p.float().mul_(1.0 - decay).sub_(update32)
    p.copy_(round_to_param_dtype(new32, p.dtype, rounding))


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
        param_rounding: str = "nearest",
    ) -> None:
        self.param_rounding = resolve_param_rounding(param_rounding)
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

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.div(denom).mul_(lr / bias_correction1)
                _apply_update(p, update, lr * wd, self.param_rounding)

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
        param_rounding: str = "nearest",
    ) -> None:
        if backend not in ("auto", "eager", "triton"):
            raise ValueError(f"unknown AdamWFP32 backend: {backend!r}")
        self.backend = backend
        self.param_rounding = resolve_param_rounding(param_rounding)
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
                stochastic = self.param_rounding == "stochastic" and p.dtype != torch.float32
                if stochastic:
                    check_param_rounding_dtype(self.param_rounding, p.dtype)
                    # parameter ごとの base seed は state に置き checkpoint に載る。
                    # step と合成するので resume 後も同じ乱数列になる。
                    if "sr_seed" not in state:
                        state["sr_seed"] = new_base_seed()
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
                            rounding=self.param_rounding,
                            seed=step_seed(state["sr_seed"], step) if stochastic else 0,
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

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.div(denom).mul_(lr / bias_correction1)
                _apply_update(p, update, lr * wd, self.param_rounding)

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
        param_rounding: str = "nearest",
    ) -> None:
        self.param_rounding = resolve_param_rounding(param_rounding)
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

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.div(denom).mul_(lr / bias_correction1)
                _apply_update(p, update, lr * wd, self.param_rounding)

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


# ---- Muon (Moonshot 版: weight decay + AdamW と RMS を揃えるスケール) -------------
# 出典: "Muon is Scalable for LLM Training" (arXiv 2502.16982)。更新は
#   M = μM + G,  U = G + μM (nesterov),  O = NewtonSchulz5(U),
#   W = W (1 - lr·wd) - lr · 0.2·√max(rows, cols) · O
# 0.2·√max(A,B) は AdamW の更新 RMS (~0.2〜0.4) に合わせる係数で、これにより
# lr / weight_decay を AdamW と共有できる。Newton-Schulz は bf16 で回す (公式実装と同じ)。
# 2D でない parameter や Block 外 (embedding / head / patch_proj / global_to_local /
# RMSNorm) は Moonshot / スピードランと同じく AdamW に任せる。
_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_MUON_RMS_MATCH = 0.2


def _newton_schulz_orthogonalize(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """G の極分解の直交因子 U·Vᵀ を Newton-Schulz 5 次反復で近似する (bf16)。

    係数は 5 反復で特異値を [0.7, 1.2] 程度へ寄せる KellerJordan 版 (厳密な 1 には
    収束しない代わりに反復が少なくて済む)。細長い行列は転置して小さい側で反復する。
    """
    if g.ndim != 2:
        raise ValueError(f"Muon expects 2D parameters, got shape {tuple(g.shape)}")
    a, b, c = _NS_COEFFS
    x = g.to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm() + eps)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.mT
    return x


def _adamw_update_eager(
    p: torch.Tensor, grad: torch.Tensor, state: dict, *,
    lr: float, beta1: float, beta2: float, eps: float, wd: float, rounding: str,
) -> None:
    """fp32 moment state の AdamW 1 step (eager)。AdamWFP32 の非融合経路と同じ式。"""
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
    exp_avg_sq.mul_(beta2).addcmul_(grad32, grad32, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
    update = exp_avg.div(denom).mul_(lr / bias_correction1)
    _apply_update(p, update, lr * wd, rounding)


def muon_param_split(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(Muon 対象, AdamW 対象) に分ける。Muon 対象 = transformer Block 内の 2D 重み。

    ByteLM / Arbor とも attention (wq/wk/wv/wo) と FFN (gate/up/down) は Block の中に
    あり、embedding / head / patch_proj / global_to_local / RMSNorm は外にあるので、
    名前に依存せず Block の所属だけで Moonshot と同じ分担になる。
    """
    from src.model.arbor import Block

    muon_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, Block):
            for p in module.parameters():
                if p.ndim == 2:
                    muon_ids.add(id(p))
    muon: list[torch.nn.Parameter] = []
    adamw: list[torch.nn.Parameter] = []
    for p in model.parameters():
        (muon if id(p) in muon_ids else adamw).append(p)
    return muon, adamw


class Muon(torch.optim.Optimizer):
    """Muon (Block 内 2D 重み) + AdamW (それ以外) の複合 optimizer。

    param_groups は 2 つ: group[0] が Muon (`use_muon=True`)、group[1] が AdamW。
    LambdaLR は各 group の initial_lr に同じ倍率を掛けるので、scheduler /
    rebase_scheduler_lr / WeightDecaySwitchLR の weight decay 切替はそのまま効く。
    Muon の momentum は fp32 (AdamW の半分の state)。AdamW 側は fp32 state の eager 経路
    (対象が embedding / head / norm 程度で小さいため融合は不要)。
    parameter への書き戻しは両 group とも `_apply_update` (param_rounding 対応)。
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_lr: float | None = None,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        param_rounding: str = "stochastic",
    ) -> None:
        self.param_rounding = resolve_param_rounding(param_rounding)
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        if not muon_params:
            raise ValueError("Muon: 対象 parameter (Block 内の 2D 重み) がありません")
        for p in muon_params:
            if p.ndim != 2:
                raise ValueError(f"Muon: 2D 以外の parameter が対象に含まれています: {tuple(p.shape)}")
        if lr <= 0:
            raise ValueError(f"lr must be positive: {lr}")
        if adamw_lr is not None and adamw_lr <= 0:
            raise ValueError(f"adamw_lr must be positive: {adamw_lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1): {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1: {ns_steps}")
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"betas must be in [0, 1): {betas}")
        if eps <= 0:
            raise ValueError(f"eps must be positive: {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative: {weight_decay}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=bool(nesterov),
            ns_steps=int(ns_steps), betas=tuple(betas), eps=eps, use_muon=False,
        )
        groups = [dict(params=muon_params, use_muon=True)]
        if adamw_params:
            groups.append(dict(params=adamw_params, lr=adamw_lr if adamw_lr is not None else lr))
        super().__init__(groups, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        """fp32 state を parameter dtype へ落とさずに復元する (AdamWFP32 と同じ契約)。"""
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for saved_id, p in zip(saved_group["params"], group["params"]):
                saved = state_dict["state"].get(saved_id, {})
                for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
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
            wd = group["weight_decay"]
            if group["use_muon"]:
                mu = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    grad32 = grad.float()
                    buf.mul_(mu).add_(grad32)
                    u = grad32.add(buf, alpha=mu) if nesterov else buf
                    ortho = _newton_schulz_orthogonalize(u, ns_steps).float()
                    scale = lr * _MUON_RMS_MATCH * math.sqrt(max(p.shape))
                    _apply_update(p, ortho.mul_(scale), lr * wd, self.param_rounding)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradients")
                    _adamw_update_eager(
                        p, grad, self.state[p],
                        lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd,
                        rounding=self.param_rounding,
                    )
        return loss


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
    param_rounding: str = "stochastic",
) -> torch.optim.Optimizer:
    """state_precision に従い、全 parameter で一様な AdamW を作る。

    - fp32: 本モジュールの AdamWFP32 (全 parameter の moment を厳密に FP32)。
    - int8: 本モジュールの AdamW8bit (全 parameter の moment を blockwise INT8)。
    - bf8:  本モジュールの AdamWBF8 (全 parameter の moment を float8_e5m2)。
    どちらも parameter を個別 dtype 例外なく一様に扱う (小さい層も除外しない)。
    別 device への暗黙フォールバックや別 optimizer への置換はしない。
    """
    kwargs = dict(lr=lr, betas=betas, eps=eps, weight_decay=wd, param_rounding=param_rounding)
    if state_precision == "int8":
        return AdamW8bit(params, **kwargs)
    if state_precision == "bf8":
        return AdamWBF8(params, **kwargs)
    return AdamWFP32(params, backend=fp32_backend, **kwargs)


def build_optimizer(
    params: Iterable[torch.nn.Parameter], cfg: dict, model: torch.nn.Module | None = None,
) -> torch.optim.Optimizer:
    """`model` は muon の parameter 分担 (Block 内 2D 重み / それ以外) に使う。muon 以外は不要。"""
    params = list(params)
    if not params:
        raise ValueError("optimizer parameter list is empty")
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = cfg["lr"]
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = cfg.get("eps", 1e-8)
    wd = cfg.get("weight_decay", 0.0)
    cfg_precision = cfg.get("state_precision")

    if name == "muon":
        if model is None:
            raise ValueError("optim.optimizer: muon は build_optimizer(model=...) が必要です")
        if resolve_state_precision(cfg_precision) != "fp32":
            raise ValueError(
                "optim.state_precision は muon では fp32 のみ対応 (momentum / AdamW state を fp32 で保持)"
            )
        muon_params, adamw_params = muon_param_split(model)
        given = {id(p) for p in params}
        muon_params = [p for p in muon_params if id(p) in given]
        adamw_params = [p for p in adamw_params if id(p) in given]
        return Muon(
            muon_params, adamw_params,
            lr=lr,
            momentum=float(cfg.get("muon_momentum", 0.95)),
            nesterov=bool(cfg.get("muon_nesterov", True)),
            ns_steps=int(cfg.get("muon_ns_steps", 5)),
            weight_decay=wd,
            adamw_lr=cfg.get("muon_adamw_lr"),
            betas=betas,
            eps=eps,
            param_rounding=resolve_param_rounding(cfg.get("param_rounding")),
        )

    if name == "lion":
        if cfg_precision is not None:
            raise ValueError(
                "optim.state_precision は adamw 系専用です。lion では指定できません"
            )
        if cfg.get("param_rounding") is not None:
            raise ValueError(
                "optim.param_rounding は adamw 系専用です。lion では指定できません"
            )
        state_dtype_name = cfg.get("state_dtype")
        state_dtype = getattr(torch, state_dtype_name) if state_dtype_name else None
        return Lion(params, lr=lr, betas=betas, weight_decay=wd, state_dtype=state_dtype)

    if name != "adamw":
        raise ValueError(f"unknown optimizer: {name} (choices: adamw | muon | lion)")

    return _build_adamw(
        params,
        state_precision=resolve_state_precision(cfg_precision),
        lr=lr,
        betas=betas,
        eps=eps,
        wd=wd,
        fp32_backend=str(cfg.get("fp32_backend", "auto")),
        param_rounding=resolve_param_rounding(cfg.get("param_rounding")),
    )


class WeightDecaySwitchLR(torch.optim.lr_scheduler.LambdaLR):
    """LambdaLR + 指定 step 以降で weight decay を切り替える scheduler.

    WSD の decay 区間で WD を 0 にする (BitNet b1.58 公式レシピの後半 WD=0 に対応) ために
    使う。LambdaLR を継承するので base_lrs / lr_lambdas / state_dict / get_last_lr /
    rebase_scheduler_lr との互換をそのまま保つ。WD は last_epoch (= step) から毎 step
    再計算して optimizer.param_groups へ書き戻す (state を増やさない)。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda,
        *,
        wd_before: float,
        wd_after: float,
        switch_step: int,
    ) -> None:
        self._wd_before = float(wd_before)
        self._wd_after = float(wd_after)
        self._switch_step = int(switch_step)
        super().__init__(optimizer, lr_lambda)
        self._apply_weight_decay()

    def _current_weight_decay(self) -> float:
        return self._wd_after if self.last_epoch >= self._switch_step else self._wd_before

    def _apply_weight_decay(self) -> None:
        wd = self._current_weight_decay()
        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd

    def step(self, epoch=None):  # type: ignore[override]
        super().step(epoch)
        self._apply_weight_decay()


_WSD_DECAY_SHAPES = ("inv_sqrt", "linear", "cosine")


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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    lr_scale: Callable[[int], float] = lambda step: 1.0,
    progress: Callable[[int], float] | None = None,
):
    """lr_scale(step) は lr_lambda に乗算する係数 (batch size warmup の √(accum/final))。
    progress(step) は学習の進行 [0,1] (消費 bytes 割合。省略時は step/total)。cosine /
    decay_start_ratio / decay_end_ratio はこの進行で測るので、accum が変わる run でも
    固定 accum と「同じ bytes で同じ lr」になる。warmup は step。LambdaLR の base_lrs /
    state_dict / rebase_scheduler_lr との互換はそのまま。"""
    name = cfg.get("scheduler", "cosine_warmup")
    warmup, total, min_ratio, decay_end_ratio = _scheduler_common(cfg)
    if progress is None:
        progress = lambda step: min(1.0, step / total)  # noqa: E731

    def step_at(ratio: float, after: int) -> int:
        """進行が ratio に最初に達する step (after より後)。"""
        return next((k for k in range(after + 1, total + 1) if progress(k) >= ratio), total)

    def segment(step: int, start: int, end: int) -> float:
        """[start, end] 区間内の進行 [0,1] (進行の差で測る)。"""
        f0, f1 = progress(start), progress(end)
        return min(1.0, max(0.0, (progress(step) - f0) / max(1e-9, f1 - f0)))

    def cosine(p: float, hi: float, lo: float) -> float:
        return lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * p))

    if name == "cosine_warmup":
        decay_end = step_at(decay_end_ratio, warmup)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            return cosine(segment(step, warmup, decay_end), 1.0, min_ratio)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step) * lr_scale(step))

    if name == "wsd":
        # Warmup-Stable-Decay (MiniCPM / Hägele+ 2024 "Scaling Laws and Compute-Optimal
        # Training Beyond Fixed Training Durations")。warmup 後は decay_start_ratio まで
        # ピーク lr で一定 (stable)、そこから decay_end_ratio まで min_lr_ratio へ decay、
        # 以後は下限で一定。stable 区間は lr が定数なので total_steps を増やしても過去の
        # lr 履歴と矛盾せず、run を後から延長できる (cosine では不可)。
        # stable 区間の任意 checkpoint から短い decay を枝分かれさせて「今止めたらどれ
        # くらいか」を測ることもできる (効率化メモ §3 の anneal ×N)。
        # weight decay は stable 区間 weight_decay、decay 区間 weight_decay_decay_phase
        # (BitNet 公式の後半 WD=0 に対応)。
        decay_start_ratio = float(cfg.get("decay_start_ratio", 0.8))
        if not 0.0 < decay_start_ratio < decay_end_ratio:
            raise ValueError(
                "decay_start_ratio は (0, decay_end_ratio) で指定: "
                f"decay_start_ratio={decay_start_ratio}, decay_end_ratio={decay_end_ratio}"
            )
        decay_shape = str(cfg.get("decay_shape", "inv_sqrt")).lower()
        if decay_shape not in _WSD_DECAY_SHAPES:
            raise ValueError(
                f"decay_shape は {' | '.join(_WSD_DECAY_SHAPES)} から選ぶ: {decay_shape!r}"
            )
        wd_stable = float(cfg.get("weight_decay", 0.0))
        wd_decay = float(cfg.get("weight_decay_decay_phase", 0.0))
        if wd_decay < 0:
            raise ValueError(f"weight_decay_decay_phase は非負で指定: {wd_decay}")
        ds = step_at(decay_start_ratio, warmup)
        decay_end = step_at(decay_end_ratio, ds)

        def decay(p: float) -> float:
            if decay_shape == "inv_sqrt":
                # 1-sqrt: 序盤に速く落として終盤を長く低 lr で回す。Hägele+ 2024 が
                # cosine / linear より僅かに良いと報告した形。
                return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(p))
            if decay_shape == "linear":
                return min_ratio + (1.0 - min_ratio) * (1.0 - p)
            return cosine(p, 1.0, min_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            if step < ds:
                return 1.0
            return decay(segment(step, ds, decay_end))

        return WeightDecaySwitchLR(
            optimizer, lambda step: lr_lambda(step) * lr_scale(step),
            wd_before=wd_stable, wd_after=wd_decay, switch_step=ds,
        )

    raise ValueError(f"unknown scheduler: {name}")
