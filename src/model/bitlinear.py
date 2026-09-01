"""BitLinear — BitNet b1.58 (W1.58 / A8) 公式レシピ準拠.

"The Era of 1-bit LLMs: Training Tips, Code and FAQ" (Microsoft) の実装に従う:

- 重み: per-tensor absmean scale で ternary {-1, 0, +1} に丸める (1.58 bit)。
- 活性: per-token absmax scale で int8 [-128, 127] に丸める。
- STE は detach トリックで実装する:
      x_q = x + (Q(x) - x).detach()
  これにより forward は量子化値を使い、backward は量子化を恒等写像とみなした
  勾配が「量子化後の重み・活性」で計算される (生の重み/活性で勾配を取るのは
  公式レシピと異なる)。
- bias は持たない (BitNet b1.58 2B4T 仕様)。
- 量子化前の正規化 (SubLN) は層構造側 (arbor.py) が担う。BitNet 2B4T と同じく、
  すべての BitLinear の入力は直前に RMSNorm を通る:
      q/k/v <- input_layernorm,  o <- attn_sub_norm,
      gate/up <- post_attention_layernorm,  down <- ffn_sub_norm

学習パスは既定では純 PyTorch (BF16 fake-quant) で、torch.compile が全体を融合
でき、CPU でもそのまま動く。学習は BF16 シャドウ重みが master。

低ビット GEMM (任意, sm89+):
`set_bitlinear_fp8_mode(model, "bwd"|"full"|"int8")` で学習 GEMM を置き換える。
ternary 重みは optimizer step 後に INT8 {-1,0,+1} と FP8 の両レイアウトへ
一度だけ変換し、gradient accumulation 中は再量子化・transpose しない。
  - "bwd":  forward は BF16 のまま (数値は既定パスと同一)。dgrad/wgrad のみ FP8。
            勾配 g と保存活性 x_q の e4m3 丸めが新規ノイズ。
  - "full": forward も FP8。x_q の e4m3 再丸め (tensorwise) が forward に乗る。
            丸めは STE 扱い。validation/eval は常に既定 (BF16) パス。
  - "int8": forward は A8 INT8 × ternary INT8、INT32 accumulation の native
            CUDA GEMM。per-token activation scale × per-weight scale で BF16 に戻す。
            backward は cached FP8 weight と FP8 GEMM を使う。RTX 5090向け既定候補。

推論パス (任意): `freeze_for_inference()` を呼ぶと dequantize 済み ternary 重みを
キャッシュし、以後の eval forward で毎回の重み再量子化を省く。実験的な
packed ternary Triton カーネルは `ARBOR_PACKED_BITLINEAR_INFERENCE=1` のときだけ
有効にする。packed 経路は速いが、行数によって通常 eval forward と丸めが変わり、
逐次生成の argmax が分岐し得るため既定では使わない。
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CUDA スタック依存
    triton = None
    tl = None


def activation_quant(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """per-token absmax int8 量子化 (値は dequantize して返す)."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return (x * scale).round().clamp(-128, 127) / scale


def activation_quant_ste(x: torch.Tensor) -> torch.Tensor:
    """activation_quant の detach-STE 版."""
    return x + (activation_quant(x) - x).detach()


# BitLinear の活性量子化。重みは常に W1.58 ternary (BitNet コア) で固定。
#   int8 … BitNet b1.58 公式 A8 (per-token absmax int8)。既定。
#   bf8  … 8bit float (float8_e5m2) fake-quant。指数を持つので per-token scale 不要。
#   bf16 … 量子化しない (活性は計算 dtype のまま = A8 を無効化)。
ACTIVATION_PRECISIONS = ("int8", "bf8", "bf16")
_BF8_ACT_DTYPE = torch.float8_e5m2


def check_activation_precision(precision: str) -> str:
    if precision not in ACTIVATION_PRECISIONS:
        raise ValueError(
            f"unknown activation_precision: {precision!r} (choices: {ACTIVATION_PRECISIONS})"
        )
    return precision


def quantize_activation(x: torch.Tensor, precision: str, eps: float = 1e-5) -> torch.Tensor:
    """activation を指定精度へ fake-quant して dequantize 値 (同 dtype) を返す."""
    if precision == "int8":
        return activation_quant(x, eps)
    if precision == "bf8":
        # float8 の丸めは cast で行い、演算は元 dtype に戻してから。
        return x.to(_BF8_ACT_DTYPE).to(x.dtype)
    if precision == "bf16":
        return x
    raise ValueError(
        f"unknown activation_precision: {precision!r} (choices: {ACTIVATION_PRECISIONS})"
    )


def quantize_activation_ste(x: torch.Tensor, precision: str = "int8") -> torch.Tensor:
    """quantize_activation の detach-STE 版 (bf16 は恒等)."""
    if precision == "bf16":
        return x
    return x + (quantize_activation(x, precision) - x).detach()


def weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """per-tensor absmean ternary 量子化 (値は dequantize して返す)."""
    scale = w.abs().mean().clamp_min(eps)
    return (w / scale).round().clamp(-1, 1) * scale


def ternary_quantize_int8(
    w: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16/FP32 shadow weight を INT8 ternary と FP32 scalar scale に分解する."""
    # 既定fake-quant経路と量子化境界・BF16丸めを一致させるため、absmeanと
    # divisionはshadow weight自身のdtypeで行う。scaleだけcache用にFP32保持する。
    scale = w.detach().abs().mean().clamp_min(eps)
    w_int8 = (w.detach() / scale).round().clamp(-1, 1).to(torch.int8)
    return w_int8, scale.float()


def _dequantize_ternary(
    w_int8: torch.Tensor, row_scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """row ごとの scale を持つ INT8 ternary cache を計算 dtype へ戻す."""
    return w_int8.to(dtype) * row_scale.to(dtype).unsqueeze(1)


def _dequantize_ternary_matching_ste(
    w_int8: torch.Tensor,
    row_scale: torch.Tensor,
    shadow_weight: torch.Tensor,
) -> torch.Tensor:
    """`w + (Q(w) - w)` と同じ演算順にして既存BF16 forwardをbit一致させる."""
    w = shadow_weight.detach()
    q = _dequantize_ternary(w_int8, row_scale, w.dtype)
    return w + (q - w)


def pack_ternary_weight(w_q: torch.Tensor) -> torch.Tensor:
    """int8 ternary {-1,0,1} を uint8 に 4 値/byte で pack する."""
    if w_q.dtype != torch.int8:
        raise TypeError(f"w_q must be torch.int8, got {w_q.dtype}")
    if w_q.dim() != 2:
        raise ValueError(f"w_q must be 2D, got shape={tuple(w_q.shape)}")
    n, k = w_q.shape
    k_packed = math.ceil(k / 4)
    padded = w_q.new_zeros((n, k_packed * 4))
    padded[:, :k] = w_q
    codes = (padded + 1).to(torch.uint8).view(n, k_packed, 4)
    return (
        codes[:, :, 0]
        | (codes[:, :, 1] << 2)
        | (codes[:, :, 2] << 4)
        | (codes[:, :, 3] << 6)
    ).contiguous()


def unpack_ternary_weight(w_packed: torch.Tensor, k: int) -> torch.Tensor:
    """pack_ternary_weight の逆変換 (検証用)."""
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
                other=1,  # code 1 = ternary 0
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

    @triton.jit
    def _a8_quantize_rows_kernel(
        x_ptr, q_ptr, inv_scale_ptr,
        m: tl.constexpr, k: tl.constexpr,
        stride_xm: tl.constexpr, stride_qm: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """1 program/row で absmax reduction と INT8 write をまとめる."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < k
        x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        inv_scale = tl.maximum(amax / 127.0, 1.0e-5 / 127.0)
        scaled = x / inv_scale
        # 外部libdevice aliasはtorch.compileのTriton source抽出で失われるため、
        # kernel内演算だけでround-to-nearestを行う。
        q = tl.where(
            scaled >= 0,
            tl.floor(scaled + 0.5),
            tl.ceil(scaled - 0.5),
        )
        q = tl.maximum(-128.0, tl.minimum(127.0, q)).to(tl.int8)
        tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)
        tl.store(inv_scale_ptr + row, inv_scale)

    @triton.jit
    def _int8_bitlinear_kernel(
        x_ptr, w_ptr, inv_sx_ptr, sw_ptr, y_ptr,
        m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
        stride_xm: tl.constexpr, stride_wn: tl.constexpr,
        stride_ym: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """A8 INT8 × ternary INT8 -> INT32 accumulate -> scaled output."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

        for k0 in range(0, k, BLOCK_K):
            k_idx = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :],
                mask=(offs_m[:, None] < m) & (k_idx[None, :] < k),
                other=0,
            )
            w = tl.load(
                w_ptr + offs_n[None, :] * stride_wn + k_idx[:, None],
                mask=(offs_n[None, :] < n) & (k_idx[:, None] < k),
                other=0,
            )
            acc += tl.dot(x, w, out_dtype=tl.int32)

        inv_sx = tl.load(inv_sx_ptr + offs_m, mask=offs_m < m, other=0.0)
        sw = tl.load(sw_ptr + offs_n, mask=offs_n < n, other=0.0)
        y = acc.to(tl.float32) * inv_sx[:, None] * sw[None, :]
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
            y,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )

    @triton.jit
    def _fp8_cast_transpose_kernel(
        x_ptr, out_ptr, scale_ptr,
        rows: tl.constexpr, cols: tl.constexpr,
        stride_xr: tl.constexpr, stride_or: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        """tensorwise FP8 cast を transpose layout へ直接 write する."""
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (r[:, None] < rows) & (c[None, :] < cols)
        x = tl.load(
            x_ptr + r[:, None] * stride_xr + c[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr)
        q = tl.maximum(-448.0, tl.minimum(448.0, x / scale))
        # out は (cols, rows) row-major。store時にFP8へcastされる。
        tl.store(
            out_ptr + c[None, :] * stride_or + r[:, None],
            q,
            mask=mask,
        )


def _packed_linear(
    x_q: torch.Tensor,      # (M, K) int8
    inv_sx: torch.Tensor,   # (M,) float32: 行ごとの 1/activation_scale
    w_packed: torch.Tensor, # (N, ceil(K/4)) uint8
    sw: torch.Tensor,       # () float32: weight scale
    k: int,
    n: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m = x_q.size(0)
    y = torch.empty((m, n), device=x_q.device, dtype=out_dtype)
    block_m, block_n, block_k = 16, 32, 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _packed_bitlinear_kernel[grid](
        x_q.contiguous(), w_packed, inv_sx.contiguous(), sw.contiguous(), y,
        m, n, k, w_packed.size(1),
        x_q.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
    )
    return y


def _quantize_a8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA/Triton fused per-token A8 quantization.

    戻り値は INT8 activation と dequantization scale (1/quant scale)。
    """
    if triton is None or not x.is_cuda:
        raise RuntimeError("native INT8 BitLinear には CUDA + Triton が必要です")
    x2 = x.contiguous()
    m, k = x2.shape
    q = torch.empty_like(x2, dtype=torch.int8)
    inv_scale = torch.empty(m, device=x.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(k)
    _a8_quantize_rows_kernel[(m,)](
        x2, q, inv_scale,
        m, k, x2.stride(0), q.stride(0),
        BLOCK_K=block_k,
        num_warps=8 if block_k >= 2048 else 4,
    )
    return q, inv_scale


def _int8_linear(
    x_int8: torch.Tensor,
    inv_sx: torch.Tensor,
    w_int8: torch.Tensor,
    row_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """INT8 tensor-core GEMM with INT32 accumulation."""
    if not x_int8.is_cuda:
        raise RuntimeError("native INT8 BitLinear には CUDA が必要です")
    # PyTorchのCUDA INT8 GEMM dispatcherはcuBLASLt/CUTLASSのdevice最適kernelを
    # 選び、手書きTritonよりAda実測で高速。公開APIがまだ無いため局所wrapperに
    # 隔離し、未提供buildでは下のTriton kernelへ戻す。
    # CUDA _int_mm currently rejects M<=16; small local/global batches use Triton.
    if hasattr(torch, "_int_mm") and x_int8.size(0) > 16:
        acc = torch._int_mm(x_int8, w_int8.t())
        return (
            acc.float()
            * inv_sx.float().unsqueeze(1)
            * row_scale.float().unsqueeze(0)
        ).to(out_dtype)
    if triton is None:
        raise RuntimeError("native INT8 BitLinear には torch._int_mm または Triton が必要です")
    m, k = x_int8.shape
    n = w_int8.size(0)
    y = torch.empty((m, n), device=x_int8.device, dtype=out_dtype)
    block_m, block_n, block_k = 32, 64, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _int8_bitlinear_kernel[grid](
        x_int8, w_int8, inv_sx, row_scale, y,
        m, n, k,
        x_int8.stride(0), w_int8.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
    )
    return y


class _CachedBitLinearSTE(torch.autograd.Function):
    """キャッシュ済み量子化重みを使い、元の重みにSTE勾配を返す."""

    @staticmethod
    def forward(
        ctx,
        x_q: torch.Tensor,
        shadow_weight: torch.Tensor,
        cached_weight_q: torch.Tensor,
    ) -> torch.Tensor:
        del shadow_weight
        ctx.input_shape = tuple(x_q.shape)
        ctx.save_for_backward(x_q, cached_weight_q)
        return F.linear(x_q, cached_weight_q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_q, weight_q = ctx.saved_tensors
        grad2 = grad_output.reshape(-1, weight_q.size(0))
        x2 = x_q.reshape(-1, weight_q.size(1))
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = grad2.matmul(weight_q).reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            grad_w = grad2.transpose(0, 1).matmul(x2)
        return grad_x, grad_w, None


class _CachedBitLinearGroupSTE(torch.autograd.Function):
    """複数の BitLinear 射影を1回のGEMMにまとめ、各重みに勾配を返す."""

    @staticmethod
    def forward(
        ctx,
        x_q: torch.Tensor,
        cached_weight_q: torch.Tensor,
        *shadow_weights: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_shape = tuple(x_q.shape)
        ctx.out_sizes = tuple(int(weight.size(0)) for weight in shadow_weights)
        ctx.save_for_backward(x_q, cached_weight_q)
        return F.linear(x_q, cached_weight_q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_q, weight_q = ctx.saved_tensors
        grad2 = grad_output.reshape(-1, weight_q.size(0))
        x2 = x_q.reshape(-1, weight_q.size(1))
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad2.matmul(weight_q).reshape(ctx.input_shape)

        needs = ctx.needs_input_grad[2:]
        if any(needs):
            grad_all = grad2.transpose(0, 1).matmul(x2)
            raw = grad_all.split(ctx.out_sizes, dim=0)
            grad_weights = tuple(value if need else None for value, need in zip(raw, needs))
        else:
            grad_weights = tuple(None for _ in ctx.out_sizes)
        return (grad_x, None, *grad_weights)


# ---------------------------------------------------------------- FP8 GEMM
_FP8_E4M3 = torch.float8_e4m3fn
_FP8_MAX = 448.0  # e4m3fn の最大有限値
_FP8_MODES = ("off", "bwd", "full", "int8")


def fp8_gemm_supported() -> bool:
    """FP8 scaled GEMM 経路が使えるか (Ada sm89 以上)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


def _fp8_dims_ok(m: int, k: int, n: int) -> bool:
    # cuBLASLt FP8 GEMM は 16 要素アラインメントを要求する。満たさない形状は
    # 呼び出し側が BF16 パスへフォールバックする。
    return m % 16 == 0 and k % 16 == 0 and n % 16 == 0


def _cast_fp8_tensorwise(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """per-tensor dynamic scale で e4m3 化する。t ≈ 返値[0] × 返値[1]."""
    scale = (t.abs().amax().float() / _FP8_MAX).clamp_min(1e-12)
    t_f8 = (t / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_E4M3)
    return t_f8, scale


def _cast_fp8_tensorwise_transposed(
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """tensorwise FP8 cast を転置済み row-major layout へ直接書く."""
    scale = (t.abs().amax().float() / _FP8_MAX).clamp_min(1e-12)
    if triton is None or not t.is_cuda:
        return (
            (t / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_E4M3).t().contiguous(),
            scale,
        )
    rows, cols = t.shape
    out = torch.empty((cols, rows), device=t.device, dtype=_FP8_E4M3)
    block_r, block_c = 32, 32
    grid = (triton.cdiv(rows, block_r), triton.cdiv(cols, block_c))
    _fp8_cast_transpose_kernel[grid](
        t, out, scale,
        rows, cols, t.stride(0), out.stride(0),
        BLOCK_R=block_r, BLOCK_C=block_c,
        num_warps=4,
    )
    return out, scale


def _scaled_mm_tensorwise(
    a: torch.Tensor,
    b_col_major: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """公開 F.scaled_mm を優先し、古い torch だけ private API へ戻す."""
    if hasattr(F, "scaled_mm") and hasattr(F, "ScalingType"):
        return F.scaled_mm(
            a,
            b_col_major,
            scale_a,
            F.ScalingType.TensorWise,
            scale_b,
            F.ScalingType.TensorWise,
            output_dtype=out_dtype,
        )
    return torch._scaled_mm(
        a,
        b_col_major,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
    )


class _Fp8BitLinearSTE(torch.autograd.Function):
    """cached FP8 weight で GEMM を実行する BitLinear/Group 共用 STE."""

    @staticmethod
    def forward(
        ctx,
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        w_fp8: torch.Tensor,
        w_fp8_t: torch.Tensor,
        row_scale: torch.Tensor,
        fwd_fp8: bool,
        out_sizes: tuple[int, ...],
        *shadow_weights: torch.Tensor,
    ) -> torch.Tensor:
        del shadow_weights
        ctx.input_shape = tuple(x_q.shape)
        ctx.out_sizes = out_sizes
        x2 = x_q.reshape(-1, w_q.size(1))
        ctx.save_for_backward(x2, w_fp8_t, row_scale)
        if fwd_fp8:
            x_f8, sx = _cast_fp8_tensorwise(x2)
            y = _scaled_mm_tensorwise(
                x_f8,
                w_fp8.t(),
                sx,
                sx.new_ones(()),
                x_q.dtype,
            )
            y = y * row_scale.to(y.dtype)
            return y.reshape(*ctx.input_shape[:-1], w_q.size(0))
        return F.linear(x_q, w_q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x2, w_fp8_t, row_scale = ctx.saved_tensors
        n = row_scale.numel()
        one = row_scale.new_ones(())
        g2 = grad_output.reshape(-1, n)
        grad_x = None
        if ctx.needs_input_grad[0]:
            # dgrad: (g × w_scale) [m,n] @ ternary [n,k]。
            gs_f8, sgs = _cast_fp8_tensorwise(g2 * row_scale.to(g2.dtype))
            grad_x = _scaled_mm_tensorwise(
                gs_f8,
                w_fp8_t.t(),
                sgs,
                one,
                grad_output.dtype,
            ).reshape(ctx.input_shape)

        needs_w = ctx.needs_input_grad[7:]
        if any(needs_w):
            # wgrad: gᵀ [n,m] @ x_q [m,k]。shadow weight へ STE 勾配を返す。
            gt_f8, sg = _cast_fp8_tensorwise_transposed(g2)
            x_km, sx = _cast_fp8_tensorwise_transposed(x2)
            grad_w = _scaled_mm_tensorwise(
                gt_f8,
                x_km.t(),
                sg,
                sx,
                grad_output.dtype,
            )
            raw = grad_w.split(ctx.out_sizes, dim=0)
            grad_weights = tuple(v if need else None for v, need in zip(raw, needs_w))
        else:
            grad_weights = tuple(None for _ in ctx.out_sizes)
        return (grad_x, None, None, None, None, None, None, *grad_weights)


class _Int8BitLinearSTE(torch.autograd.Function):
    """A8×W1.58 native INT8 forward + cached FP8 backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_int8: torch.Tensor,
        row_scale: torch.Tensor,
        w_fp8: torch.Tensor,
        w_fp8_t: torch.Tensor,
        out_sizes: tuple[int, ...],
        *shadow_weights: torch.Tensor,
    ) -> torch.Tensor:
        del w_fp8, shadow_weights
        ctx.input_shape = tuple(x.shape)
        ctx.out_sizes = out_sizes
        x2 = x.reshape(-1, w_int8.size(1))
        x_int8, inv_sx = _quantize_a8_rows(x2)
        ctx.save_for_backward(x_int8, inv_sx, w_fp8_t, row_scale)
        y = _int8_linear(x_int8, inv_sx, w_int8, row_scale, x.dtype)
        return y.reshape(*ctx.input_shape[:-1], w_int8.size(0))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_int8, inv_sx, w_fp8_t, row_scale = ctx.saved_tensors
        n = row_scale.numel()
        one = row_scale.new_ones(())
        g2 = grad_output.reshape(-1, n)
        grad_x = None
        if ctx.needs_input_grad[0]:
            gs_f8, sgs = _cast_fp8_tensorwise(g2 * row_scale.to(g2.dtype))
            grad_x = _scaled_mm_tensorwise(
                gs_f8,
                w_fp8_t.t(),
                sgs,
                one,
                grad_output.dtype,
            ).reshape(ctx.input_shape)

        needs_w = ctx.needs_input_grad[6:]
        if any(needs_w):
            # A8 dequantized activation を再構築し、転置済みFP8へ直接castする。
            x_q = x_int8.to(grad_output.dtype) * inv_sx.to(
                grad_output.dtype
            ).unsqueeze(1)
            gt_f8, sg = _cast_fp8_tensorwise_transposed(g2)
            x_km, sx = _cast_fp8_tensorwise_transposed(x_q)
            grad_w = _scaled_mm_tensorwise(
                gt_f8,
                x_km.t(),
                sg,
                sx,
                grad_output.dtype,
            )
            raw = grad_w.split(ctx.out_sizes, dim=0)
            grad_weights = tuple(v if need else None for v, need in zip(raw, needs_w))
        else:
            grad_weights = tuple(None for _ in ctx.out_sizes)
        return (grad_x, None, None, None, None, None, *grad_weights)


def set_bitlinear_fp8_mode(module: nn.Module, mode: str) -> dict[str, int | str]:
    """module 以下の BitLinear / BitLinearGroup に FP8 モードを設定する.

    QKV/gate-up 融合後 (install_arbor_projection_fusions 後) に呼ぶこと。
    """
    normalized = str(mode).lower()
    if normalized == "native":
        normalized = "int8"
    if normalized not in _FP8_MODES:
        raise ValueError(f"unknown bitlinear fp8 mode: {mode!r} (choices: {_FP8_MODES})")
    targets = [
        child for child in module.modules()
        if isinstance(child, (BitLinear, BitLinearGroup))
    ]
    has_cuda_weights = any(
        (child.weight if isinstance(child, BitLinear) else child.members()[0].weight).is_cuda
        for child in targets
    )
    if normalized != "off" and not (has_cuda_weights and fp8_gemm_supported()):
        raise RuntimeError(
            "bitlinear_fp8 には compute capability 8.9 以上の CUDA GPU が必要 "
            "(FP8 scaled GEMM)。暗黙フォールバックは行いません"
        )
    if normalized == "int8" and triton is None:
        raise RuntimeError("bitlinear_fp8=int8 には Triton が必要です")
    for child in targets:
        child._fp8_mode = normalized
    return {"mode": normalized, "layers": len(targets)}


class BitLinear(nn.Module):
    """nn.Linear (bias 無し) の drop-in 置換. W1.58 ternary weight + STE.

    活性量子化は activation_precision (int8=A8 公式 | bf8 | bf16) で選ぶ。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_precision: str = "int8",
    ):
        super().__init__()
        if bias:
            raise ValueError("BitLinear is bias-free (BitNet b1.58 spec).")
        self.in_features = in_features
        self.out_features = out_features
        self.activation_precision = check_activation_precision(activation_precision)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # 出力側 (wo / down) は builder 側で 1/sqrt(2*n_layers) に再スケールする
        nn.init.trunc_normal_(self.weight, std=0.02, a=-0.06, b=0.06)
        self.register_parameter("bias", None)
        # optimizer step 間だけ使う low-bit cache。BF16 shadow weight は Parameter。
        self.register_buffer("_train_w_int8", None, persistent=False)
        self.register_buffer("_train_w_scale", None, persistent=False)
        self.register_buffer("_train_w_fp8", None, persistent=False)
        self.register_buffer("_train_w_fp8_t", None, persistent=False)
        self._train_cache_enabled = False
        self._fp8_mode = "off"  # set_bitlinear_fp8_mode で設定
        # 推論凍結用 (freeze_for_inference 後のみ非 None)
        self._w_packed: torch.Tensor | None = None
        self._w_scale: torch.Tensor | None = None
        self._w_dq: torch.Tensor | None = None  # CPU/Triton 無し環境のフォールバック
        self.register_load_state_dict_post_hook(self._after_load_state_dict)

    # --------------------------------------------------------- training cache
    def _after_load_state_dict(self, module: nn.Module, incompatible_keys: Any) -> None:
        del module, incompatible_keys
        if self._train_cache_enabled:
            self.refresh_training_weight_cache()
        self.unfreeze()

    @property
    def training_weight_cache_enabled(self) -> bool:
        return (
            self._train_cache_enabled
            and self._train_w_int8 is not None
            and self._train_w_scale is not None
        )

    @property
    def training_cache_bytes(self) -> int:
        caches = (
            self._train_w_int8,
            self._train_w_scale,
            self._train_w_fp8,
            self._train_w_fp8_t,
        )
        return sum(
            cache.numel() * cache.element_size()
            for cache in caches
            if cache is not None
        )

    @property
    def cache_cost_bytes(self) -> int:
        # INT8 ternary + row scale。FP8 backward時は N×K / K×N の両layout。
        cost = self.weight.numel() + self.out_features * 4
        if self._fp8_mode != "off":
            cost += 2 * self.weight.numel()
        return cost

    @torch.no_grad()
    def enable_training_weight_cache(self, enabled: bool = True) -> None:
        if not enabled:
            self.disable_training_weight_cache()
            return
        if not self.weight.requires_grad:
            return
        self._train_cache_enabled = True
        self.refresh_training_weight_cache()

    @torch.no_grad()
    def disable_training_weight_cache(self) -> None:
        self._train_cache_enabled = False
        self._train_w_int8 = None
        self._train_w_scale = None
        self._train_w_fp8 = None
        self._train_w_fp8_t = None

    @torch.no_grad()
    def refresh_training_weight_cache(self) -> None:
        if not self._train_cache_enabled:
            return
        w_int8, scale = ternary_quantize_int8(self.weight)
        row_scale = scale.expand(self.out_features).contiguous()
        self._train_w_int8 = w_int8.contiguous()
        self._train_w_scale = row_scale
        if self._fp8_mode != "off":
            self._train_w_fp8 = w_int8.to(_FP8_E4M3).contiguous()
            self._train_w_fp8_t = self._train_w_fp8.t().contiguous()
        else:
            self._train_w_fp8 = None
            self._train_w_fp8_t = None

    def _cached_fp8_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training_weight_cache_enabled:
            raise RuntimeError("BitLinear low-bit cache is not enabled")
        if self._train_w_fp8 is None or self._train_w_fp8_t is None:
            raise RuntimeError(
                f"bitlinear_fp8={self._fp8_mode} was configured after cache allocation; "
                "refresh_bitlinear_training_cache() を実行してください"
            )
        return (
            self._train_w_int8,
            self._train_w_scale,
            self._train_w_fp8,
            self._train_w_fp8_t,
        )

    def _fp8_applicable(self, x_q: torch.Tensor) -> bool:
        if self._fp8_mode == "off":
            return False
        if not x_q.is_cuda:
            raise RuntimeError(
                f"bitlinear_fp8={self._fp8_mode} received non-CUDA input; "
                "暗黙フォールバックは禁止"
            )
        dims = (
            x_q.numel() // x_q.size(-1), self.in_features, self.out_features
        )
        if not _fp8_dims_ok(*dims):
            raise RuntimeError(
                f"bitlinear_fp8={self._fp8_mode} requires M/K/N multiples of 16, "
                f"got {dims}; BF16への暗黙フォールバックは禁止"
            )
        return True

    def forward_prequantized(self, x_q: torch.Tensor) -> torch.Tensor:
        if self.training and self._fp8_applicable(x_q):
            if self.training_weight_cache_enabled:
                w_int8, row_scale, w_fp8, w_fp8_t = self._cached_fp8_weights()
                w_q = _dequantize_ternary_matching_ste(
                    w_int8, row_scale, self.weight
                )
            else:
                w_int8, scale = ternary_quantize_int8(self.weight)
                row_scale = scale.expand(self.out_features).contiguous()
                w_fp8 = w_int8.to(_FP8_E4M3).contiguous()
                w_fp8_t = w_fp8.t().contiguous()
                w_q = _dequantize_ternary_matching_ste(
                    w_int8, row_scale, self.weight
                )
            return _Fp8BitLinearSTE.apply(
                x_q,
                w_q,
                w_fp8,
                w_fp8_t,
                row_scale,
                self._fp8_mode == "full",
                (self.out_features,),
                self.weight,
            )
        if self.training and self.training_weight_cache_enabled:
            w_q = _dequantize_ternary(
                self._train_w_int8, self._train_w_scale, self.weight.dtype
            )
            return _CachedBitLinearSTE.apply(x_q, self.weight, w_q)
        w = self.weight
        w_q = w + (weight_quant(w) - w).detach()
        return F.linear(x_q, w_q)

    # ------------------------------------------------------------ inference
    # 行数がこれ以下なら packed カーネルより cuBLAS GEMV の方が速い
    # (batch=1 生成はカーネル起動オーバーヘッド律速のため)
    _SMALL_M = 16

    def freeze_for_inference(self) -> None:
        """ternary 重みを事前計算して以後の eval forward を高速化する (重みは凍結前提).

        - 既定: dequantize 済み重みキャッシュ + cuBLAS。
        - `ARBOR_PACKED_BITLINEAR_INFERENCE=1`: 大バッチで packed ternary Triton
          カーネルを使う。速度診断用で、品質比較では既定経路を使う。
        """
        with torch.no_grad():
            scale = self.weight.abs().mean().clamp_min(1e-5).float()
            w_int = (self.weight.float() / scale).round().clamp(-1, 1).to(torch.int8)
            self._w_dq = (w_int.float() * scale).to(self.weight.dtype)
            use_packed = os.environ.get("ARBOR_PACKED_BITLINEAR_INFERENCE", "0") == "1"
            if use_packed and triton is not None and self.weight.is_cuda:
                self._w_packed = pack_ternary_weight(w_int)
            else:
                self._w_packed = None
            self._w_scale = scale

    def unfreeze(self) -> None:
        self._w_packed = None
        self._w_scale = None
        self._w_dq = None

    @property
    def frozen(self) -> bool:
        return self._w_scale is not None

    def _quantize_act(self, x2: torch.Tensor) -> torch.Tensor:
        """activation を activation_precision に従って fake-quant する (同 dtype).

        注意: torch.fake_quantize_per_channel_affine は 1 カーネルに見えて
        実測 ~335us (素の F.linear の 8 倍) かかるため使わない。
        """
        return quantize_activation(x2, self.activation_precision)

    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, self.in_features)
        m = x2.size(0)
        # packed ternary カーネルは int8 活性専用。bf8/bf16 では通常経路を使う。
        if (
            self._w_packed is not None
            and m > self._SMALL_M
            and self.activation_precision == "int8"
        ):
            scale = 127.0 / x2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5).float()
            x_q = (x2.float() * scale).round().clamp(-128, 127).to(torch.int8)
            y = _packed_linear(
                x_q, (1.0 / scale).reshape(-1), self._w_packed, self._w_scale,
                self.in_features, self.out_features, torch.float32,
            ).to(x.dtype)
        else:
            y = F.linear(self._quantize_act(x2), self._w_dq)
        return y.reshape(*x.shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen and not self.training:
            return self._forward_inference(x)
        if self.training and self._fp8_mode == "int8":
            if self.activation_precision != "int8":
                raise RuntimeError(
                    "bitlinear_fp8=int8 は activation_precision=int8 専用です"
                )
            self._fp8_applicable(x)
            w_int8, row_scale, w_fp8, w_fp8_t = self._cached_fp8_weights()
            return _Int8BitLinearSTE.apply(
                x,
                w_int8,
                row_scale,
                w_fp8,
                w_fp8_t,
                (self.out_features,),
                self.weight,
            )
        return self.forward_prequantized(
            quantize_activation_ste(x, self.activation_precision)
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, act={self.activation_precision}, "
            f"train_cache={self.training_weight_cache_enabled}, frozen={self.frozen}"
        )


class BitLinearGroup(nn.Module):
    """既存 BitLinear Parameter を保持したまま射影をまとめて計算する補助層."""

    def __init__(self, members: Sequence[BitLinear], kind: str = "group"):
        super().__init__()
        if len(members) < 2:
            raise ValueError("BitLinearGroup requires at least two members")
        if not all(isinstance(member, BitLinear) for member in members):
            raise TypeError("all BitLinearGroup members must be BitLinear")
        in_features = members[0].in_features
        if any(member.in_features != in_features for member in members):
            raise ValueError("all BitLinearGroup members must share in_features")
        activation_precision = members[0].activation_precision
        if any(member.activation_precision != activation_precision for member in members):
            raise ValueError("all BitLinearGroup members must share activation_precision")
        self.activation_precision = activation_precision
        self.in_features = in_features
        self.out_splits = tuple(member.out_features for member in members)
        self.out_features = sum(self.out_splits)
        self.kind = str(kind)
        object.__setattr__(self, "_members", tuple(members))
        self.register_buffer("_train_w_int8", None, persistent=False)
        self.register_buffer("_train_w_scale", None, persistent=False)
        self.register_buffer("_train_w_fp8", None, persistent=False)
        self.register_buffer("_train_w_fp8_t", None, persistent=False)
        self._train_cache_enabled = False
        self._fp8_mode = "off"  # set_bitlinear_fp8_mode で設定

    def members(self) -> tuple[BitLinear, ...]:
        return self._members

    @property
    def matrix_count(self) -> int:
        return len(self._members)

    @property
    def cache_cost_bytes(self) -> int:
        numel = sum(member.weight.numel() for member in self.members())
        cost = numel + self.out_features * 4
        if self._fp8_mode != "off":
            cost += 2 * numel
        return cost

    @property
    def training_weight_cache_enabled(self) -> bool:
        return (
            self._train_cache_enabled
            and self._train_w_int8 is not None
            and self._train_w_scale is not None
        )

    @property
    def training_cache_bytes(self) -> int:
        caches = (
            self._train_w_int8,
            self._train_w_scale,
            self._train_w_fp8,
            self._train_w_fp8_t,
        )
        return sum(
            cache.numel() * cache.element_size()
            for cache in caches
            if cache is not None
        )

    @torch.no_grad()
    def enable_training_weight_cache(self, enabled: bool = True) -> None:
        if not enabled:
            self.disable_training_weight_cache()
            return
        self._train_cache_enabled = True
        self.refresh_training_weight_cache()

    @torch.no_grad()
    def disable_training_weight_cache(self) -> None:
        self._train_cache_enabled = False
        self._train_w_int8 = None
        self._train_w_scale = None
        self._train_w_fp8 = None
        self._train_w_fp8_t = None

    @torch.no_grad()
    def refresh_training_weight_cache(self) -> None:
        if not self._train_cache_enabled:
            return
        quantized: list[torch.Tensor] = []
        row_scales: list[torch.Tensor] = []
        for member in self.members():
            w_int8, scale = ternary_quantize_int8(member.weight)
            quantized.append(w_int8)
            row_scales.append(scale.expand(member.out_features))
        self._train_w_int8 = torch.cat(quantized, dim=0).contiguous()
        self._train_w_scale = torch.cat(row_scales, dim=0).float().contiguous()
        if self._fp8_mode != "off":
            self._train_w_fp8 = self._train_w_int8.to(_FP8_E4M3).contiguous()
            self._train_w_fp8_t = self._train_w_fp8.t().contiguous()
        else:
            self._train_w_fp8 = None
            self._train_w_fp8_t = None

    def _cached_fp8_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training_weight_cache_enabled:
            raise RuntimeError("BitLinearGroup low-bit cache is not enabled")
        if self._train_w_fp8 is None or self._train_w_fp8_t is None:
            raise RuntimeError(
                f"bitlinear_fp8={self._fp8_mode} was configured after cache allocation; "
                "refresh_bitlinear_training_cache() を実行してください"
            )
        return (
            self._train_w_int8,
            self._train_w_scale,
            self._train_w_fp8,
            self._train_w_fp8_t,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        members = self.members()
        if self.training and self._fp8_mode == "int8":
            if self.activation_precision != "int8":
                raise RuntimeError(
                    "bitlinear_fp8=int8 は activation_precision=int8 専用です"
                )
            if not x.is_cuda:
                raise RuntimeError(
                    "bitlinear_fp8=int8 received non-CUDA input; 暗黙フォールバックは禁止"
                )
            dims = (x.numel() // x.size(-1), self.in_features, self.out_features)
            if not _fp8_dims_ok(*dims):
                raise RuntimeError(
                    "bitlinear_fp8=int8 requires M/K/N multiples of 16, "
                    f"got {dims}; BF16への暗黙フォールバックは禁止"
                )
            w_int8, row_scale, w_fp8, w_fp8_t = self._cached_fp8_weights()
            weights = tuple(member.weight for member in members)
            return _Int8BitLinearSTE.apply(
                x,
                w_int8,
                row_scale,
                w_fp8,
                w_fp8_t,
                self.out_splits,
                *weights,
            )
        x_q = quantize_activation_ste(x, self.activation_precision)
        use_fp8 = False
        if self.training and self._fp8_mode != "off":
            if not x_q.is_cuda:
                raise RuntimeError(
                    f"bitlinear_fp8={self._fp8_mode} received non-CUDA input; "
                    "暗黙フォールバックは禁止"
                )
            dims = (
                x_q.numel() // x_q.size(-1), self.in_features, self.out_features
            )
            if not _fp8_dims_ok(*dims):
                raise RuntimeError(
                    f"bitlinear_fp8={self._fp8_mode} requires M/K/N multiples of 16, "
                    f"got {dims}; BF16への暗黙フォールバックは禁止"
                )
            use_fp8 = True
        if use_fp8:
            if self.training_weight_cache_enabled:
                w_int8, row_scale, w_fp8, w_fp8_t = self._cached_fp8_weights()
                shadow = torch.cat(
                    [member.weight.detach() for member in members], dim=0
                )
                w_q = _dequantize_ternary_matching_ste(
                    w_int8, row_scale, shadow
                )
            else:
                quantized_int8 = []
                row_scales = []
                for member in members:
                    w_int8, scale = ternary_quantize_int8(member.weight)
                    quantized_int8.append(w_int8)
                    row_scales.append(scale.expand(member.out_features))
                w_int8 = torch.cat(quantized_int8, dim=0).contiguous()
                row_scale = torch.cat(row_scales, dim=0).float().contiguous()
                w_fp8 = w_int8.to(_FP8_E4M3).contiguous()
                w_fp8_t = w_fp8.t().contiguous()
                shadow = torch.cat(
                    [member.weight.detach() for member in members], dim=0
                )
                w_q = _dequantize_ternary_matching_ste(
                    w_int8, row_scale, shadow
                )
            weights = tuple(member.weight for member in members)
            return _Fp8BitLinearSTE.apply(
                x_q,
                w_q,
                w_fp8,
                w_fp8_t,
                row_scale,
                self._fp8_mode == "full",
                self.out_splits,
                *weights,
            )
        if self.training and self.training_weight_cache_enabled:
            weights = tuple(member.weight for member in members)
            w_q = _dequantize_ternary(
                self._train_w_int8, self._train_w_scale, members[0].weight.dtype
            )
            return _CachedBitLinearGroupSTE.apply(x_q, w_q, *weights)
        quantized = [
            member.weight + (weight_quant(member.weight) - member.weight).detach()
            for member in members
        ]
        return F.linear(x_q, torch.cat(quantized, dim=0))


def install_arbor_projection_fusions(module: nn.Module) -> dict[str, int]:
    """Arbor の QKV / gate-up 射影を checkpoint 互換のまま融合可能にする."""
    qkv_groups = 0
    gate_up_groups = 0
    for child in list(module.modules()):
        if all(hasattr(child, name) for name in ("wq", "wk", "wv", "wo", "n_heads")):
            if all(isinstance(getattr(child, name), BitLinear) for name in ("wq", "wk", "wv")):
                if not hasattr(child, "_fast_qkv_group"):
                    child._fast_qkv_group = BitLinearGroup(
                        (child.wq, child.wk, child.wv), kind="qkv"
                    )
                qkv_groups += 1
        if all(hasattr(child, name) for name in ("gate", "up", "down", "ffn_sub_norm")):
            if isinstance(child.gate, BitLinear) and isinstance(child.up, BitLinear):
                if not hasattr(child, "_fast_gate_up_group"):
                    child._fast_gate_up_group = BitLinearGroup(
                        (child.gate, child.up), kind="gate_up"
                    )
                gate_up_groups += 1
    return {"qkv_groups": qkv_groups, "gate_up_groups": gate_up_groups}


def _parse_cache_mode(
    value: bool | str | None,
    *,
    grad_accum_steps: int,
    has_cuda: bool,
) -> str:
    if value is None:
        value = "auto"
    if isinstance(value, bool):
        return "full" if value else "off"
    normalized = str(value).lower().replace("-", "_")
    if normalized == "auto":
        return "full" if has_cuda and grad_accum_steps > 1 else "off"
    if normalized in {"1", "true", "yes", "on", "full", "all"}:
        return "full"
    if normalized in {"fused", "groups", "projection_groups"}:
        return "fused"
    if normalized in {"0", "false", "no", "off", "none", "disabled"}:
        return "off"
    raise ValueError(f"unknown bitnet_weight_cache setting: {value!r}")


@torch.no_grad()
def configure_bitlinear_training_cache(
    module: nn.Module,
    *,
    enabled: bool | str | None = "auto",
    grad_accum_steps: int = 1,
    max_cache_gib: float | None = 1.25,
    min_numel: int = 0,
) -> dict[str, int | float | bool | str]:
    """torch.compile 前に射影融合を取り付け、訓練用量子化重みcacheを選ぶ."""
    fusion_info = install_arbor_projection_fusions(module)
    all_modules = list(module.modules())
    groups = [child for child in all_modules if isinstance(child, BitLinearGroup)]
    native_int8 = any(
        isinstance(child, (BitLinear, BitLinearGroup))
        and child._fp8_mode == "int8"
        for child in all_modules
    )
    effective_min_numel = 0 if native_int8 else int(min_numel)
    linears = [
        child
        for child in all_modules
        if isinstance(child, BitLinear)
        and child.weight.requires_grad
        and child.weight.numel() >= effective_min_numel
    ]
    mode = _parse_cache_mode(
        enabled,
        grad_accum_steps=max(1, int(grad_accum_steps)),
        has_cuda=any(layer.weight.is_cuda for layer in linears),
    )
    for group in groups:
        group.disable_training_weight_cache()
    for layer in linears:
        layer.disable_training_weight_cache()

    budget = None if max_cache_gib is None else max(0, int(float(max_cache_gib) * 2**30))
    used = 0
    selected_groups: list[BitLinearGroup] = []
    selected_linears: list[BitLinear] = []
    covered: set[int] = set()

    def fits(cost: int) -> bool:
        return budget is None or used + cost <= budget

    if mode in {"fused", "full"}:
        groups.sort(
            key=lambda group: (
                (group.matrix_count - 1) / max(group.cache_cost_bytes, 1),
                group.cache_cost_bytes,
            ),
            reverse=True,
        )
        for group in groups:
            if any(
                member.weight.numel() < effective_min_numel
                for member in group.members()
            ):
                continue
            cost = group.cache_cost_bytes
            if fits(cost):
                group.enable_training_weight_cache(True)
                selected_groups.append(group)
                used += cost
                covered.update(id(member) for member in group.members())

    if mode == "full":
        for layer in sorted(linears, key=lambda item: item.weight.numel(), reverse=True):
            if id(layer) in covered:
                continue
            cost = layer.cache_cost_bytes
            if fits(cost):
                layer.enable_training_weight_cache(True)
                selected_linears.append(layer)
                used += cost

    cached_matrices = sum(group.matrix_count for group in selected_groups) + len(
        selected_linears
    )
    if native_int8 and cached_matrices != len(linears):
        required = sum(group.cache_cost_bytes for group in groups)
        covered_by_any_group = {
            id(member) for group in groups for member in group.members()
        }
        required += sum(
            layer.cache_cost_bytes
            for layer in linears
            if id(layer) not in covered_by_any_group
        )
        raise RuntimeError(
            "bitlinear_fp8=int8 は全BitLinearのlow-bit cacheを必要とします: "
            f"cached={cached_matrices}/{len(linears)}, "
            f"budget={0 if budget is None else budget / 2**30:.2f}GiB, "
            f"required≈{required / 2**30:.2f}GiB。"
            "bitnet_weight_cache=full, min_numel=0, 十分なcache_gibを指定してください"
        )
    return {
        "enabled": bool(selected_groups or selected_linears),
        "mode": mode,
        "eligible_layers": len(linears),
        "cached_layers": cached_matrices,
        "fused_groups": len(selected_groups),
        "standalone_layers": len(selected_linears),
        "cache_bytes": used,
        "cache_gib": used / 2**30,
        "grad_accum_steps": int(grad_accum_steps),
        "cache_format": "int8+fp8_dual_layout" if any(
            isinstance(child, (BitLinear, BitLinearGroup))
            and child._fp8_mode != "off"
            for child in all_modules
        ) else "int8",
        **fusion_info,
    }


@torch.no_grad()
def refresh_bitlinear_training_cache(module: nn.Module) -> int:
    """optimizer.step 後に有効な量子化重みcacheを更新する."""
    all_modules = list(module.modules())
    covered: set[int] = set()
    count = 0
    for child in all_modules:
        if isinstance(child, BitLinearGroup) and child.training_weight_cache_enabled:
            child.refresh_training_weight_cache()
            covered.update(id(member) for member in child.members())
            count += child.matrix_count
    for child in all_modules:
        if (
            isinstance(child, BitLinear)
            and id(child) not in covered
            and child.training_weight_cache_enabled
        ):
            child.refresh_training_weight_cache()
            count += 1
    return count


def freeze_bitlinear_for_inference(module: nn.Module) -> int:
    """module 以下の全 BitLinear を推論凍結する。凍結した層数を返す."""
    count = 0
    for child in module.modules():
        if isinstance(child, BitLinear):
            child.freeze_for_inference()
            count += 1
    return count
