"""BitLinear low-bit kernel microbenchmarks.

Measures the decode question directly:

  A. unpacked INT8 W + torch._int_mm
  B. unpacked INT8 W + Triton tl.dot
  C. packed2 W + current logical-K decode + tl.dot
  D. packed2 W + grouped 4-way decode + tl.dot (A/B only)

Example:
  python -m scripts.bench_bitlinear_kernels --shape 1024,2048,11264
"""
from __future__ import annotations

import argparse
import contextlib
import statistics

import torch

import src.model.bitlinear as bitlinear_mod
from src.model.bitlinear import (
    _FP8_E4M3,
    _int8_wgrad_kernel,
    _int8_linear,
    _cast_fp8_tensorwise,
    _cast_fp8_tensorwise_transposed,
    _lowbit_wgrad,
    _packed_linear,
    _quantize_a8_rows,
    _quantize_a8_rows_scaled,
    _quantize_int8_tensorwise,
    _scaled_mm_tensorwise,
    _wgrad_tile,
    fp8_gemm_supported,
    pack_ternary_weight,
    set_bitlinear_int8_backend,
    ternary_quantize_int8,
)

try:
    import triton
except Exception:  # pragma: no cover - CUDA stack dependent
    triton = None


DEFAULT_SHAPES = (
    "1024,2048,11264",
    "1024,5632,2048",
    "1024,2048,3072",
    "1024,2048,2048",
)


def _parse_shape(spec: str) -> tuple[int, int, int]:
    try:
        m, k, n = (int(part) for part in spec.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape must be M,K,N, got {spec!r}"
        ) from exc
    if min(m, k, n) <= 0:
        raise argparse.ArgumentTypeError(f"shape values must be positive: {spec!r}")
    return m, k, n


def _parse_tile(spec: str) -> tuple[int, int, int, int]:
    try:
        bm, bn, bk, warps = (int(part) for part in spec.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"tile must be BM,BN,BK,WARPS, got {spec!r}"
        ) from exc
    if min(bm, bn, bk, warps) <= 0:
        raise argparse.ArgumentTypeError(f"tile values must be positive: {spec!r}")
    return bm, bn, bk, warps


@contextlib.contextmanager
def _override_dot_current_tile(tile: tuple[int, int, int, int] | None):
    if tile is None:
        yield
        return
    original = bitlinear_mod._packed_linear_tile

    def forced_tile(m: int, n: int, k: int, *, grouped_decode: bool):
        if grouped_decode:
            return original(m, n, k, grouped_decode=grouped_decode)
        return tile

    bitlinear_mod._packed_linear_tile = forced_tile
    try:
        yield
    finally:
        bitlinear_mod._packed_linear_tile = original


def _sync() -> None:
    torch.cuda.synchronize()


def _measure(fn, *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync()
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


def _fmt(result: dict[str, float], baseline_ms: float) -> str:
    ratio = result["median"] / baseline_ms if baseline_ms > 0 else float("nan")
    return (
        f"{result['median']:8.3f} ms  "
        f"p10={result['p10']:8.3f}  p90={result['p90']:8.3f}  "
        f"x{ratio:5.2f}"
    )


def _wgrad_gemm_from_quantized(
    g_int8: torch.Tensor,
    sg: torch.Tensor,
    x_int8: torch.Tensor,
    sx: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required")
    m, n = g_int8.shape
    k = x_int8.size(1)
    out = torch.empty((n, k), device=g_int8.device, dtype=out_dtype)
    block_n, block_k, block_m, num_warps = _wgrad_tile(m, n, k)
    grid = (triton.cdiv(n, block_n), triton.cdiv(k, block_k))
    _int8_wgrad_kernel[grid](
        g_int8,
        x_int8,
        sg,
        sx,
        out,
        m,
        n,
        k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return out


def _int8_fp8_dx(
    grad: torch.Tensor,
    row_scale: torch.Tensor,
    w_fp8_t: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Replicate _Int8BitLinearSTE.backward dX: cast(g*w_scale)->FP8 scaled_mm."""
    n = row_scale.numel()
    g2 = grad.reshape(-1, n)
    one = row_scale.new_ones(())
    gs_f8, sgs = _cast_fp8_tensorwise(g2 * row_scale.to(g2.dtype))
    return _scaled_mm_tensorwise(gs_f8, w_fp8_t.t(), sgs, one, out_dtype)


def _int8_fp8_wgrad(
    grad: torch.Tensor,
    x_int8: torch.Tensor,
    inv_sx: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Replicate the full _Int8BitLinearSTE.backward dW path."""
    n = grad.size(-1)
    g2 = grad.reshape(-1, n)
    x_q = x_int8.to(out_dtype) * inv_sx.to(out_dtype).unsqueeze(1)
    gt_f8, sg = _cast_fp8_tensorwise_transposed(g2)
    x_km, sx = _cast_fp8_tensorwise_transposed(x_q)
    return _scaled_mm_tensorwise(gt_f8, x_km.t(), sg, sx, out_dtype)


def _ternary_dx(
    grad: torch.Tensor,
    row_scale: torch.Tensor,
    w_packed_t: torch.Tensor,
    k: int,
    n: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Replicate the full TernaryBitLinearSTE.backward dX path."""
    g_int8, inv_sg = _quantize_a8_rows_scaled(grad, row_scale)
    return _packed_linear(
        g_int8,
        inv_sg,
        w_packed_t,
        row_scale.new_ones(()),
        n,
        k,
        out_dtype,
        grouped_decode=False,
    )


def _ternary_wgrad(
    grad: torch.Tensor,
    x_int8: torch.Tensor,
    inv_sx: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Replicate the full TernaryBitLinearSTE.backward dW path."""
    x_q = x_int8.to(out_dtype) * inv_sx.to(out_dtype).unsqueeze(1)
    return _lowbit_wgrad(grad, x_q, out_dtype)


def _print_results(
    results: dict[str, dict[str, float]],
    *,
    baseline_name: str,
) -> None:
    baseline = results[baseline_name]["median"]
    for name, result in results.items():
        print(f"  {name:22s} {_fmt(result, baseline)}")


def _bench_shape(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    check: bool,
    include_wgrad: bool,
    breakdown: bool,
) -> None:
    m, k, n = shape
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    x_int8, inv_sx = _quantize_a8_rows(x)
    w = torch.randn((n, k), device="cuda", dtype=dtype)
    w_int8, scale = ternary_quantize_int8(w)
    row_scale = scale.expand(n).contiguous()
    w_packed = pack_ternary_weight(w_int8)
    w_packed_t = pack_ternary_weight(w_int8.t().contiguous())

    def int8_linear_backend(backend: str) -> torch.Tensor:
        set_bitlinear_int8_backend(backend)
        return _int8_linear(x_int8, inv_sx, w_int8, row_scale, dtype)

    kernels = {
        "int8_int_mm": lambda: int8_linear_backend("int_mm"),
        "int8_triton": lambda: int8_linear_backend("triton"),
        "packed_dot_current": lambda: _packed_linear(
            x_int8,
            inv_sx,
            w_packed,
            row_scale,
            k,
            n,
            dtype,
            grouped_decode=False,
        ),
        "packed_dot": lambda: _packed_linear(
            x_int8,
            inv_sx,
            w_packed,
            row_scale,
            k,
            n,
            dtype,
            grouped_decode=True,
        ),
    }
    set_bitlinear_int8_backend("triton")
    if check:
        reference = kernels["int8_int_mm"]().float()
        for name, fn in kernels.items():
            diff = (fn().float() - reference).abs().max().item()
            print(f"[check] shape={m},{k},{n} {name} max_abs_diff={diff:.6g}")

    print(f"\n[shape] M={m} K={k} N={n} dtype={dtype}")
    results = {}
    for name, fn in kernels.items():
        results[name] = _measure(fn, warmup=warmup, iters=iters)
    _print_results(results, baseline_name="int8_int_mm")

    if include_wgrad:
        grad = torch.randn((m, n), device="cuda", dtype=dtype)
        x_q = x_int8.to(dtype) * inv_sx.to(dtype).unsqueeze(1)
        result = _measure(
            lambda: _lowbit_wgrad(grad, x_q, dtype),
            warmup=warmup,
            iters=iters,
        )
        print(f"  {'lowbit_wgrad':22s} {result['median']:8.3f} ms  "
              f"p10={result['p10']:8.3f}  p90={result['p90']:8.3f}")

    if breakdown:
        grad = torch.randn((m, n), device="cuda", dtype=dtype)
        g_int8, inv_sg = _quantize_a8_rows_scaled(grad, row_scale)
        one = row_scale.new_ones(())
        x_q = x_int8.to(dtype) * inv_sx.to(dtype).unsqueeze(1)
        wg_int8, wg_sg = _quantize_int8_tensorwise(grad)
        wx_int8, wx_sx = _quantize_int8_tensorwise(x_q)
        breakdown_kernels = {
            "a8_quant_fwd": lambda: _quantize_a8_rows(x),
            "fwd_int8_int_mm": lambda: int8_linear_backend("int_mm"),
            "fwd_packed_current": lambda: _packed_linear(
                x_int8,
                inv_sx,
                w_packed,
                row_scale,
                k,
                n,
                dtype,
                grouped_decode=False,
            ),
            "dx_quant_scaled": lambda: _quantize_a8_rows_scaled(grad, row_scale),
            "dx_packed_current": lambda: _packed_linear(
                g_int8,
                inv_sg,
                w_packed_t,
                one,
                n,
                k,
                dtype,
                grouped_decode=False,
            ),
            "ternary_dx_total": lambda: _ternary_dx(
                grad,
                row_scale,
                w_packed_t,
                k,
                n,
                dtype,
            ),
            "x_reconstruct": lambda: (
                x_int8.to(dtype) * inv_sx.to(dtype).unsqueeze(1)
            ),
            "wgrad_quant": lambda: (
                _quantize_int8_tensorwise(grad),
                _quantize_int8_tensorwise(x_q),
            ),
            "wgrad_gemm": lambda: _wgrad_gemm_from_quantized(
                wg_int8,
                wg_sg,
                wx_int8,
                wx_sx,
                dtype,
            ),
            "wgrad_core": lambda: _lowbit_wgrad(grad, x_q, dtype),
            "ternary_wgrad_total": lambda: _ternary_wgrad(
                grad,
                x_int8,
                inv_sx,
                dtype,
            ),
        }
        # INT8 training mode の実運用 backward (FP8 dX / FP8 dW) を同一 shape で
        # 比較対象へ追加する。full-training 差の主犯 (dX か dW か) を切り分ける。
        fp8_ok = (
            fp8_gemm_supported()
            and m % 16 == 0
            and k % 16 == 0
            and n % 16 == 0
        )
        if fp8_ok:
            w_fp8 = w_int8.to(_FP8_E4M3).contiguous()
            w_fp8_t = w_fp8.t().contiguous()
            breakdown_kernels["int8_fp8_dx"] = (
                lambda: _int8_fp8_dx(grad, row_scale, w_fp8_t, dtype)
            )
            breakdown_kernels["int8_fp8_wgrad"] = (
                lambda: _int8_fp8_wgrad(grad, x_int8, inv_sx, dtype)
            )
        else:
            print(
                "  [info] FP8 backward skipped "
                f"(fp8_supported={fp8_gemm_supported()}, dims_ok="
                f"{m % 16 == 0 and k % 16 == 0 and n % 16 == 0})"
            )
        print("\n[breakdown]")
        breakdown_results = {
            name: _measure(fn, warmup=warmup, iters=iters)
            for name, fn in breakdown_kernels.items()
        }
        _print_results(breakdown_results, baseline_name="fwd_int8_int_mm")
        if fp8_ok:
            ternary_dx_ms = breakdown_results["ternary_dx_total"]["median"]
            fp8_dx_ms = breakdown_results["int8_fp8_dx"]["median"]
            ternary_dw_ms = breakdown_results["ternary_wgrad_total"]["median"]
            fp8_dw_ms = breakdown_results["int8_fp8_wgrad"]["median"]
            print("\n[backward comparison]")
            print(
                "  dX  ternary/FP8: "
                f"{ternary_dx_ms:.3f} / {fp8_dx_ms:.3f} ms  "
                f"x{ternary_dx_ms / fp8_dx_ms:.2f}"
            )
            print(
                "  dW  ternary/FP8: "
                f"{ternary_dw_ms:.3f} / {fp8_dw_ms:.3f} ms  "
                f"x{ternary_dw_ms / fp8_dw_ms:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="M,K,N. May be passed multiple times.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--wgrad", action="store_true")
    parser.add_argument(
        "--packed-tile",
        type=_parse_tile,
        default=None,
        help="Override dot_current packed tile as BM,BN,BK,WARPS for experiments.",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Also measure fwd/dX/dW quantization and GEMM components.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >=0 and --iters must be >0")

    shapes = args.shape or [_parse_shape(spec) for spec in DEFAULT_SHAPES]
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    if args.packed_tile is not None:
        bm, bn, bk, warps = args.packed_tile
        print(f"[override] dot_current tile BM={bm} BN={bn} BK={bk} warps={warps}")
    with _override_dot_current_tile(args.packed_tile):
        for shape in shapes:
            _bench_shape(
                shape,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                check=not args.no_check,
                include_wgrad=args.wgrad,
                breakdown=args.breakdown,
            )


if __name__ == "__main__":
    main()
