"""parameter dtype への書き戻し丸め (optim.param_rounding)。

BF16 parameter を fp32 master 無しで学習すると、更新量が parameter の半 ulp
(相対 ~0.2%) を下回った時点で最近接丸め (nearest) が更新を 0 に潰す。lr が
下がる学習後半は大半の更新が消え、weight decay (相対 lr*wd ~ 1e-5) は最初から
一度も効かない。BitNet の latent weight は微小更新の蓄積で三値の閾値を跨ぐので
影響が大きい。

stochastic は fp32 で計算した新しい値を bf16 へ書き戻す時に、切り捨て/切り上げを
端数に比例した確率で選ぶ (不偏)。期待値では fp32 更新と一致し、追加メモリ無し。
BF16 は FP32 の上位 16bit なので、下位 16bit に一様乱数を足して切り捨てるだけで
実装できる (torchao の bf16_stochastic_round と同じ)。
"""
from __future__ import annotations

import torch

PARAM_ROUNDINGS = ("nearest", "stochastic")

# BF16 = FP32 の上位 16bit。下位 16bit を落とすマスク (int32 で 0xFFFF0000)。
_BF16_TRUNC_MASK = -65536
_BF16_NOISE_BOUND = 1 << 16


def resolve_param_rounding(value: str | None) -> str:
    """optim.param_rounding を正規化する。既定は stochastic。"""
    if value is None:
        return "stochastic"
    normalized = str(value).lower()
    if normalized not in PARAM_ROUNDINGS:
        raise ValueError(
            f"unknown optim.param_rounding: {value!r} (choices: {PARAM_ROUNDINGS})"
        )
    return normalized


def check_param_rounding_dtype(rounding: str, dtype: torch.dtype) -> None:
    """stochastic は bf16 (と丸めの要らない fp32) だけ対応。fp16 は暗黙に nearest へ落とさない。"""
    if rounding == "stochastic" and dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"param_rounding=stochastic は bf16/fp32 parameter 専用です (got {dtype})"
        )


def round_to_param_dtype(x32: torch.Tensor, dtype: torch.dtype, rounding: str) -> torch.Tensor:
    """fp32 の新しい値を parameter dtype へ丸める (eager 版)。"""
    if x32.dtype != torch.float32:
        raise TypeError(f"round_to_param_dtype expects float32 input, got {x32.dtype}")
    if dtype == torch.float32 or rounding == "nearest":
        return x32.to(dtype)
    check_param_rounding_dtype(rounding, dtype)
    bits = x32.contiguous().view(torch.int32)
    noise = torch.randint(
        0, _BF16_NOISE_BOUND, bits.shape, device=bits.device, dtype=torch.int32
    )
    return ((bits + noise) & _BF16_TRUNC_MASK).view(torch.float32).to(dtype)


def step_seed(base_seed: int, step: int) -> int:
    """parameter ごとの base seed と step から、Triton Philox 用の 31bit seed を作る。"""
    return (base_seed ^ (step * 0x9E3779B1)) & 0x7FFFFFFF


def new_base_seed() -> int:
    """torch の CPU RNG から parameter 用 base seed を引く (checkpoint の rng 状態で再現可能)。"""
    return int(torch.randint(0, 0x7FFFFFFF, (1,)).item())
