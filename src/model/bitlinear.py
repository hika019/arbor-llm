"""BitLinear (BitNet b1.58, W1.58 / A8) with STE.

BF16 shadow weights are kept as the master parameter. On the forward pass the
weights are quantized to ternary {-1, 0, +1} via absmean rounding, and the
activations are quantized per-token to int8 via absmax. The backward pass
uses a straight-through estimator (STE).

NOTE: This is a numerically faithful but unoptimized reference implementation.
Replace the W * x matmul with a packed-ternary kernel once integration with the
training loop is verified.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(torch.round(x), lo, hi)


def quantize_weight_ternary(w: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """absmean ternary quantization. Returns (w_q in {-1,0,1}, scale)."""
    scale = w.abs().mean().clamp_min(eps)
    w_q = _round_clip(w / scale, -1.0, 1.0)
    return w_q, scale


def quantize_activation_int8(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """absmax per-token int8 quantization. Returns (x_q in [-127,127], scale).
    Scale shape: (..., 1) so it broadcasts on the last (feature) dim.
    """
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    x_q = _round_clip(x / scale, -128.0, 127.0)
    return x_q, scale


class _BitLinearSTE(torch.autograd.Function):
    """STE wrapper: forward uses quantized w & x, backward passes grads through."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor):
        x_q, sx = quantize_activation_int8(x)
        w_q, sw = quantize_weight_ternary(w)
        # ternary matmul in fp space (reference): scale-merged GEMM
        y = F.linear(x_q, w_q) * sx * sw
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        # STE: 量子化を恒等視して勾配を素通し. autocast 由来の dtype 混在を吸収する.
        g = grad_out.to(w.dtype)
        grad_x = (g @ w).to(x.dtype)
        flat_g = g.reshape(-1, g.size(-1))
        flat_x = x.reshape(-1, x.size(-1)).to(w.dtype)
        grad_w = flat_g.t() @ flat_x
        return grad_x, grad_w


class BitLinear(nn.Module):
    """Drop-in replacement for nn.Linear with W1.58 / A8 + STE.

    bias is intentionally not supported (matches MS BitNet b1.58 2B4T design:
    Linear layers carry no bias).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if bias:
            raise ValueError("BitLinear is bias-free (matches BitNet b1.58 spec).")
        self.in_features = in_features
        self.out_features = out_features
        # BF16 シャドウ重みを master とする (学習対象).
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        # 一部の nn 標準モジュール (MultiheadAttention 等) が .bias を直接参照するため
        # nn.Linear と同じ属性を None で公開する.
        self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _BitLinearSTE.apply(x, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False, w_dtype={self.weight.dtype}"
