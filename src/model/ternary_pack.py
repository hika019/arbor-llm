"""BitLinear 訓練用 packed ternary cache を 1 kernel で再生成する Triton 実装.

`refresh_training_weight_cache` の既定 (純 PyTorch) 経路は、1 層あたり
absmean → 除算 → round → clamp → int8 化 → zero pad → +1 → uint8 化 →
shift/or ×3 → transpose copy を forward 用と dX 用の 2 layout について
個別 kernel として発行する (約 20〜30 launch/層、1B model で毎 update 約
3,600 launch)。各 kernel は数 µs で終わるため GPU は launch 待ちになり、
optimizer 区間の GPU idle の主因になる。

ここでは shadow weight の tile を 1 回 load し、同じ tile から
「k 方向 4 値/byte」(forward 用) と「n 方向 4 値/byte」(dX 用) の両 layout と
row scale を書き出す。GPU 機種に依存しない単純な elementwise/reduction のみで、
Tensor Core も特定 SM 数も前提にしない。

数値は純 PyTorch 経路 (`ternary_quantize_int8` + `pack_ternary_weight*`) と
bit 一致させる:

- absmean scale は呼び出し側が PyTorch で weight dtype のまま計算する
  (reduction 順序差による丸め差を持ち込まない)。
- 除算は IEEE RN (`libdevice.div_rn`) で行い、BF16/FP16 weight では PyTorch の
  BF16/FP16 二項演算と同じく結果を weight dtype へ丸めてから `rint` する。
- pad 位置は PyTorch 経路と同じく zero weight (code 1) として pack する。
"""
# Triton の constexpr 注釈は JIT DSL の値であり Python 型式ではない。
# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - CUDA スタック依存
    triton = None


_BLOCK_N = 64
_BLOCK_K = 64


if triton is not None:

    @triton.jit
    def _ternary_pack_dual_kernel(
        w_ptr, scale_ptr, pk_ptr, pn_ptr, row_scale_ptr,
        N, K, stride_wn, stride_wk,
        stride_pk_n, stride_pk_j, stride_pn_i, stride_pn_k,
        ROUND_BF16: tl.constexpr, ROUND_FP16: tl.constexpr,
        PK_TRANS: tl.constexpr, PN_TRANS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """W[N,K] の tile から pk[N, K/4] と pn[N/4, K] を同時に書く.

        pk[n, j] = Σ_r code[n, 4j+r] << 2r  (k 方向 pack, forward 用)
        pn[i, k] = Σ_r code[4i+r, k] << 2r  (n 方向 pack, dX 用)
        code = round(w / scale).clamp(-1, 1) + 1 ∈ {0, 1, 2}
        *_TRANS は store 時に tile を転置し、出力の連続軸へ coalesced に書く。
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (n[:, None] < N) & (k[None, :] < K)
        # 範囲外は zero weight → code 1 (PyTorch 経路の zero pad + 1 と同じ)。
        w = tl.load(
            w_ptr + n[:, None] * stride_wn + k[None, :] * stride_wk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
        q = libdevice.div_rn(w, scale)
        if ROUND_BF16:
            q = q.to(tl.bfloat16).to(tl.float32)
        if ROUND_FP16:
            q = q.to(tl.float16).to(tl.float32)
        q = libdevice.rint(q)
        q = tl.minimum(tl.maximum(q, -1.0), 1.0)
        code = (q + 1.0).to(tl.int32)

        shifts = 1 << (2 * tl.arange(0, 4))
        # k 方向 pack: [BLOCK_N, BLOCK_K/4, 4] の最終軸を重み付き和で 1 byte にする。
        pk = tl.sum(
            tl.reshape(code, [BLOCK_N, BLOCK_K // 4, 4]) * shifts[None, None, :],
            axis=2,
        ).to(tl.uint8)
        # n 方向 pack: [BLOCK_N/4, 4, BLOCK_K] の中央軸を同様に潰す。
        pn = tl.sum(
            tl.reshape(code, [BLOCK_N // 4, 4, BLOCK_K]) * shifts[None, :, None],
            axis=1,
        ).to(tl.uint8)

        j = pid_k * (BLOCK_K // 4) + tl.arange(0, BLOCK_K // 4)
        i = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)
        k4 = (K + 3) // 4
        n4 = (N + 3) // 4
        if PK_TRANS:
            tl.store(
                pk_ptr + j[:, None] * stride_pk_j + n[None, :] * stride_pk_n,
                tl.trans(pk),
                mask=(j[:, None] < k4) & (n[None, :] < N),
            )
        else:
            tl.store(
                pk_ptr + n[:, None] * stride_pk_n + j[None, :] * stride_pk_j,
                pk,
                mask=(n[:, None] < N) & (j[None, :] < k4),
            )
        if PN_TRANS:
            tl.store(
                pn_ptr + k[:, None] * stride_pn_k + i[None, :] * stride_pn_i,
                tl.trans(pn),
                mask=(k[:, None] < K) & (i[None, :] < n4),
            )
        else:
            tl.store(
                pn_ptr + i[:, None] * stride_pn_i + k[None, :] * stride_pn_k,
                pn,
                mask=(i[:, None] < n4) & (k[None, :] < K),
            )
        if pid_k == 0:
            tl.store(
                row_scale_ptr + n,
                scale + tl.zeros([BLOCK_N], dtype=tl.float32),
                mask=n < N,
            )


def fused_pack_supported(weight: torch.Tensor) -> bool:
    """この weight に fused kernel を使えるか (CUDA + Triton + libdevice RN 除算)."""
    return (
        triton is not None
        and weight.is_cuda
        # HIP は device.type='cuda' だが CUDA 用 libdevice の RN 除算を持たない。
        and torch.version.hip is None
        and weight.dim() == 2
        and weight.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def _layout_strides(
    packed: torch.Tensor,
    packed_t: torch.Tensor,
    backend: str,
    n_offset: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
    """backend の cache layout を論理行列 pk[N,K/4] / pn[N/4,K] の base と stride に直す."""
    k4 = math.ceil(k / 4)
    n4 = math.ceil(n / 4)
    if backend in ("kmajor_current", "kmajor_single_dot"):
        # packed = pack(W).t() → [K/4, N_total], packed_t = pack(W^T).t() → [N_total/4, K]
        if packed.shape[0] != k4 or packed_t.shape[1] != k:
            raise ValueError(
                f"kmajor cache shape mismatch: packed={tuple(packed.shape)} "
                f"packed_t={tuple(packed_t.shape)} for N={n} K={k}"
            )
        n_total, n4_total = packed.shape[1], packed_t.shape[0]
        pk_base, s_pk = packed[:, n_offset:], (packed.stride(1), packed.stride(0))
        pn_base, s_pn = packed_t[n_offset // 4:], (packed_t.stride(0), packed_t.stride(1))
    else:
        # packed = pack(W) → [N_total, K/4], packed_t = pack(W^T) → [K, N_total/4]
        if packed.shape[1] != k4 or packed_t.shape[0] != k:
            raise ValueError(
                f"n-major cache shape mismatch: packed={tuple(packed.shape)} "
                f"packed_t={tuple(packed_t.shape)} for N={n} K={k}"
            )
        n_total, n4_total = packed.shape[0], packed_t.shape[1]
        pk_base, s_pk = packed[n_offset:], (packed.stride(0), packed.stride(1))
        pn_base, s_pn = packed_t[:, n_offset // 4:], (packed_t.stride(1), packed_t.stride(0))
    if n_offset + n > n_total or n_offset // 4 + n4 > n4_total:
        raise ValueError(
            f"packed cache has no room for N={n} at offset={n_offset} "
            f"(N_total={n_total}, N_total/4={n4_total})"
        )
    return pk_base, s_pk[0], s_pk[1], pn_base, s_pn[0], s_pn[1]


@torch.no_grad()
def pack_ternary_cache_(
    weight: torch.Tensor,
    scale: torch.Tensor,
    packed: torch.Tensor,
    packed_t: torch.Tensor,
    row_scale: torch.Tensor,
    *,
    backend: str,
    n_offset: int = 0,
) -> None:
    """既存の cache buffer へ ternary pack 結果を in-place で書き込む.

    ``scale`` は `ternary_quantize_int8` と同じ weight dtype の 0-d tensor
    (``w.abs().mean().clamp_min(eps)``)。``n_offset`` は BitLinearGroup の
    member 先頭行 (4 の倍数) で、cache は member 連結後の全体 buffer を渡す。
    row_scale は FP32 で ``scale.float()`` と一致する値を ``[n_offset, n_offset+N)``
    へ書く。
    """
    if not fused_pack_supported(weight):
        raise RuntimeError("fused ternary pack requires a CUDA weight with Triton")
    n, k = weight.shape
    if n_offset % 4 != 0:
        raise ValueError(f"n_offset must be a multiple of 4, got {n_offset}")
    if scale.dtype != weight.dtype or scale.numel() != 1:
        raise ValueError("scale must be a 0-d tensor in the weight dtype")
    for name, buf in (("packed", packed), ("packed_t", packed_t)):
        if buf.dtype != torch.uint8 or buf.device != weight.device:
            raise ValueError(f"{name} must be a uint8 tensor on the weight device")
    if row_scale.dtype != torch.float32 or row_scale.numel() < n_offset + n:
        raise ValueError("row_scale must be FP32 with room for this member")
    pk_base, s_pk_n, s_pk_j, pn_base, s_pn_i, s_pn_k = _layout_strides(
        packed, packed_t, backend, n_offset, n, k
    )
    grid = (triton.cdiv(n, _BLOCK_N), triton.cdiv(k, _BLOCK_K))
    with torch.cuda.device(weight.device):
        _ternary_pack_dual_kernel[grid](
            weight, scale, pk_base, pn_base, row_scale[n_offset:],
            n, k, weight.stride(0), weight.stride(1),
            s_pk_n, s_pk_j, s_pn_i, s_pn_k,
            ROUND_BF16=weight.dtype == torch.bfloat16,
            ROUND_FP16=weight.dtype == torch.float16,
            PK_TRANS=(s_pk_n == 1 and s_pk_j != 1),
            PN_TRANS=(s_pn_i == 1 and s_pn_k != 1),
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=4,
        )
    # raw kernel は dispatcher 外で storage を書き換えるため、version counter を
    # in-place 演算と同様に進めておく。
    torch.autograd.graph.increment_version((packed, packed_t, row_scale))
