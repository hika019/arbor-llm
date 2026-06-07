"""BitLinear (BitNet b1.58, W1.58 / A8) with STE.

BF16 shadow weights are kept as the master parameter. On the forward pass the
weights are quantized to ternary {-1, 0, +1} in int8 storage via absmean
rounding, and the activations are quantized per-token to int8 via absmax. The
backward pass uses a straight-through estimator (STE).

On CUDA, the forward path uses an optional Triton kernel that stores ternary
weights packed 4 values per byte and multiplies them with int8 activations. On
CPU or when Triton is unavailable, it falls back to the PyTorch reference path.
Backward is still an STE reference implementation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional CUDA stack
    triton = None
    tl = None


def _round_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(torch.round(x), lo, hi)


def quantize_weight_ternary(w: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """absmean ternary quantization. Returns (int8 w_q in {-1,0,1}, scale)."""
    scale = w.abs().mean().clamp_min(eps)
    w_q = _round_clip(w / scale, -1.0, 1.0).to(torch.int8)
    return w_q, scale


def quantize_activation_int8(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """absmax per-token int8 quantization. Returns (int8 x_q in [-127,127], scale).
    Scale shape: (..., 1) so it broadcasts on the last (feature) dim.
    """
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    x_q = _round_clip(x / scale, -127.0, 127.0).to(torch.int8)
    return x_q, scale


def pack_ternary_weight(w_q: torch.Tensor) -> torch.Tensor:
    """Pack int8 ternary weights {-1,0,1} into uint8, 4 weights per byte."""
    if w_q.dtype != torch.int8:
        raise TypeError(f"w_q must be torch.int8, got {w_q.dtype}")
    if w_q.dim() != 2:
        raise ValueError(f"w_q must be 2D, got shape={tuple(w_q.shape)}")
    n, k = w_q.shape
    k_packed = math.ceil(k / 4)
    padded = w_q.new_zeros((n, k_packed * 4), dtype=torch.int8)
    padded[:, :k] = w_q
    codes = (padded + 1).to(torch.uint8).view(n, k_packed, 4)
    return (
        codes[:, :, 0]
        | (codes[:, :, 1] << 2)
        | (codes[:, :, 2] << 4)
        | (codes[:, :, 3] << 6)
    ).contiguous()


def unpack_ternary_weight(w_packed: torch.Tensor, k: int) -> torch.Tensor:
    """Unpack uint8 ternary weights back to int8 {-1,0,1}."""
    if w_packed.dtype != torch.uint8:
        raise TypeError(f"w_packed must be torch.uint8, got {w_packed.dtype}")
    codes = torch.stack(
        (
            w_packed & 0x03,
            (w_packed >> 2) & 0x03,
            (w_packed >> 4) & 0x03,
            (w_packed >> 6) & 0x03,
        ),
        dim=-1,
    )
    return (codes.reshape(w_packed.size(0), -1)[:, :k].to(torch.int8) - 1).contiguous()


if triton is not None:

    @triton.jit
    def _packed_bitlinear_kernel(
        x_ptr, w_ptr, sx_ptr, sw_ptr, y_ptr,
        m: tl.constexpr, n: tl.constexpr, k: tl.constexpr, k_packed: tl.constexpr,
        stride_xm: tl.constexpr, stride_ym: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, k, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :],
                mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
                other=0,
            )

            pack_idxs = k_idxs // 4
            shifts = (k_idxs % 4) * 2
            packed = tl.load(
                w_ptr + offs_n[None, :] * k_packed + pack_idxs[:, None],
                mask=(offs_n[None, :] < n) & (k_idxs[:, None] < k),
                other=1,
            )
            codes = ((packed >> shifts[:, None]) & 3).to(tl.int32)
            w = (codes - 1).to(tl.int8)
            acc += tl.dot(x, w, out_dtype=tl.float32)

        sx = tl.load(sx_ptr + offs_m, mask=offs_m < m, other=0.0).to(tl.float32)
        sw = tl.load(sw_ptr).to(tl.float32)
        y = acc * sx[:, None] * sw
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
            y,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )


def _packed_bitlinear_forward(
    x_q: torch.Tensor,
    sx: torch.Tensor,
    w_q: torch.Tensor,
    sw: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    if triton is None or not x_q.is_cuda:
        return None
    if x_q.dim() != 2 or w_q.dim() != 2:
        return None
    m, k = x_q.shape
    n = w_q.size(0)
    w_packed = pack_ternary_weight(w_q).contiguous()
    y = torch.empty((m, n), device=x_q.device, dtype=out_dtype)
    block_m, block_n, block_k = 16, 32, 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _packed_bitlinear_kernel[grid](
        x_q.contiguous(), w_packed, sx.reshape(m).contiguous(), sw.contiguous(), y,
        m, n, k, w_packed.size(1),
        x_q.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
    )
    return y


class _BitLinearSTE(torch.autograd.Function):
    """STE wrapper: forward uses quantized w & x, backward passes grads through."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor):
        orig_shape = x.shape[:-1]
        x_q, sx = quantize_activation_int8(x)
        w_q, sw = quantize_weight_ternary(w)
        x_q_2d = x_q.reshape(-1, x_q.size(-1))
        sx_2d = sx.reshape(-1, 1)
        y = _packed_bitlinear_forward(x_q_2d, sx_2d, w_q, sw, x.dtype)
        if y is None:
            y = F.linear(x_q.to(x.dtype), w_q.to(x.dtype)) * sx * sw
        else:
            y = y.reshape(*orig_shape, w.size(0))
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
