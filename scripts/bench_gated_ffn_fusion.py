"""Microbenchmark for Phase 3 candidate A: fused gate/up + ReLU^2 FFN.

Reference:
    gate = packed_gemm(x, W_gate)   # BF16 [M, I]
    up   = packed_gemm(x, W_up)     # BF16 [M, I]
    out  = relu(gate)^2 * up

Fused:
    one Triton kernel loads the x tile once, decodes both K-major packed
    weights, keeps the two INT32 accumulators in registers, and computes
    relu(gate)^2 * up in the epilogue before a single [M, I] store.

This measures the kernel-level ceiling.  Wiring it into the compiled training
FFN (which currently fuses gate/up through the FP8 path) is a follow-up.
"""
from __future__ import annotations

import argparse
import statistics

import torch
import triton
import triton.language as tl

from src.model.bitlinear import (
    _packed_linear,
    _quantize_a8_rows,
    pack_ternary_weight_kmajor,
    ternary_quantize_int8,
)


@triton.jit
def _decode_kmajor_wtile(packed, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    w0 = ((packed & 3).to(tl.int32) - 1).to(tl.int8)
    w1 = (((packed >> 2) & 3).to(tl.int32) - 1).to(tl.int8)
    w2 = (((packed >> 4) & 3).to(tl.int32) - 1).to(tl.int8)
    w3 = (((packed >> 6) & 3).to(tl.int32) - 1).to(tl.int8)
    lo = tl.join(w0, w1)
    hi = tl.join(w2, w3)
    full = tl.join(lo, hi)
    return tl.reshape(tl.permute(full, (0, 3, 2, 1)), (BLOCK_K, BLOCK_N))


@triton.jit
def _packed_gated_ffn_kernel(
    x_ptr, wg_ptr, wu_ptr, sx_ptr, swg_ptr, swu_ptr, y_ptr,
    m: tl.constexpr, i: tl.constexpr, k: tl.constexpr,
    k_packed: tl.constexpr, stride_xm: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    offs_pk = tl.arange(0, BLOCK_K // 4)
    offs_k = tl.arange(0, BLOCK_K)

    for k0 in range(0, k, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :],
            mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
            other=0,
        )
        pk = k0 // 4 + offs_pk
        wmask = (pk[:, None] < k_packed) & (offs_n[None, :] < i)
        packed_g = tl.load(wg_ptr + pk[:, None] * i + offs_n[None, :], mask=wmask, other=0x55)
        packed_u = tl.load(wu_ptr + pk[:, None] * i + offs_n[None, :], mask=wmask, other=0x55)
        wg = _decode_kmajor_wtile(packed_g, BLOCK_K, BLOCK_N)
        wu = _decode_kmajor_wtile(packed_u, BLOCK_K, BLOCK_N)
        acc_g += tl.dot(x, wg, out_dtype=tl.int32)
        acc_u += tl.dot(x, wu, out_dtype=tl.int32)

    sx = tl.load(sx_ptr + offs_m, mask=offs_m < m, other=0.0).to(tl.float32)
    swg = tl.load(swg_ptr + offs_n, mask=offs_n < i, other=0.0).to(tl.float32)
    swu = tl.load(swu_ptr + offs_n, mask=offs_n < i, other=0.0).to(tl.float32)
    gate = acc_g.to(tl.float32) * sx[:, None] * swg[None, :]
    up = acc_u.to(tl.float32) * sx[:, None] * swu[None, :]
    a = tl.maximum(gate, 0.0)
    tl.store(
        y_ptr + offs_m[:, None] * i + offs_n[None, :],
        (a * a * up).to(tl.bfloat16),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < i),
    )


def fused_gated_ffn(
    x_q: torch.Tensor,
    inv_sx: torch.Tensor,
    wg_packed: torch.Tensor,
    wu_packed: torch.Tensor,
    swg: torch.Tensor,
    swu: torch.Tensor,
    *,
    k: int,
    i: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m = x_q.size(0)
    k_packed = triton.cdiv(k, 4)
    y = torch.empty((m, i), device=x_q.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, block_m), triton.cdiv(i, block_n))
    _packed_gated_ffn_kernel[grid](
        x_q.contiguous(), wg_packed, wu_packed, inv_sx.contiguous(),
        swg.contiguous(), swu.contiguous(), y,
        m, i, k, k_packed, x_q.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


def reference_gated_ffn(x_q, inv_sx, wg_packed, wu_packed, swg, swu, k, i, dtype):
    gate = _packed_linear(
        x_q, inv_sx, wg_packed, swg, k, i, dtype,
        grouped_decode=False, kmajor_layout=True, decode_v2=True,
        backend="kmajor_single_dot",
        launch_config=None,
    )
    up = _packed_linear(
        x_q, inv_sx, wu_packed, swu, k, i, dtype,
        grouped_decode=False, kmajor_layout=True, decode_v2=True,
        backend="kmajor_single_dot",
        launch_config=None,
    )
    a = torch.relu(gate)
    return a * a * up


def _measure(fn, *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    ordered = sorted(times)
    return {
        "median": statistics.median(ordered),
        "p10": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="2048,2048,5632", help="M,K,I")
    parser.add_argument("--tile", type=str, default="64,64,64")
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    m, k, i = (int(part) for part in args.shape.split(","))
    bm, bn, bk = (int(part) for part in args.tile.split(","))
    dtype = torch.bfloat16
    torch.manual_seed(0)

    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    x_q, inv_sx = _quantize_a8_rows(x)
    wg_int8, sg = ternary_quantize_int8(torch.randn((i, k), device="cuda", dtype=torch.float32))
    wu_int8, su = ternary_quantize_int8(torch.randn((i, k), device="cuda", dtype=torch.float32))
    wg_packed = pack_ternary_weight_kmajor(wg_int8)
    wu_packed = pack_ternary_weight_kmajor(wu_int8)
    swg = sg.expand(i).contiguous()
    swu = su.expand(i).contiguous()

    fused = fused_gated_ffn(x_q, inv_sx, wg_packed, wu_packed, swg, swu, k=k, i=i,
                            block_m=bm, block_n=bn, block_k=bk,
                            num_warps=args.warps, num_stages=args.stages)
    ref = reference_gated_ffn(x_q, inv_sx, wg_packed, wu_packed, swg, swu, k, i, dtype)
    if not args.no_check:
        diff = (fused.float() - ref.float()).abs()
        print(f"[check] gated_ffn_relu2 max_abs_diff={diff.max().item():.6g} "
              f"mean_abs_diff={diff.mean().item():.6g}")

    ref_t = _measure(lambda: reference_gated_ffn(x_q, inv_sx, wg_packed, wu_packed,
                                                  swg, swu, k, i, dtype),
                    warmup=args.warmup, iters=args.iters)
    fused_t = _measure(lambda: fused_gated_ffn(x_q, inv_sx, wg_packed, wu_packed,
                                                swg, swu, k=k, i=i, block_m=bm,
                                                block_n=bn, block_k=bk,
                                                num_warps=args.warps, num_stages=args.stages),
                      warmup=args.warmup, iters=args.iters)
    print(f"[A/B] shape={m}x{k}x{i} tile={bm}x{bn}x{bk} warps={args.warps} stages={args.stages}")
    print(f"  reference 2xGEMM+act  median={ref_t['median']:.4f}ms "
          f"p10={ref_t['p10']:.4f} p90={ref_t['p90']:.4f}")
    print(f"  fused gate/up+relu2  median={fused_t['median']:.4f}ms "
          f"p10={fused_t['p10']:.4f} p90={fused_t['p90']:.4f}")
    print(f"  speedup={ref_t['median'] / fused_t['median']:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
