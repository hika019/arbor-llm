"""BitLinear low-bit kernel microbenchmarks.

Measures the decode question directly:

  A. unpacked INT8 W + torch._int_mm
  B. unpacked INT8 W + Triton tl.dot
  C. packed2 W + current logical-K decode + tl.dot
  D. K-major packed2 W + current logical-K decode + tl.dot
  E. K-major packed2 W + single-load decode + dense fragment + tl.dot x1
  F. packed2 W + grouped 4-way decode + tl.dot (A/B only)

Example:
  python -m scripts.bench_bitlinear_kernels --breakdown --no-check
"""
from __future__ import annotations

import argparse
import contextlib
import statistics

import torch

import src.model.bitlinear as bitlinear_mod
from src.model.bitlinear import (
    _FP8_E4M3,
    _FP8_MAX,
    _int8_wgrad_kernel,
    _int8_linear,
    _cast_a8_dequant_fp8_transposed,
    _cast_fp8_tensorwise,
    _cast_fp8_tensorwise_transposed,
    _fp8_wgrad_from_a8,
    _lowbit_wgrad,
    _packed_linear,
    _quantize_a8_rows,
    _quantize_a8_rows_scaled,
    _quantize_int8_tensorwise,
    _scaled_mm_tensorwise,
    _wgrad_tile,
    fp8_gemm_supported,
    pack_ternary_weight,
    pack_ternary_weight_kmajor,
    set_bitlinear_int8_backend,
    ternary_quantize_int8,
)

try:
    import triton
except Exception:  # pragma: no cover - CUDA stack dependent
    triton = None


DEFAULT_SHAPES = (
    "2048,2048,11264",
    "2048,5632,2048",
    "2048,2048,3072",
    "2048,2048,2048",
)

DEFAULT_CALLS_PER_STEP = {
    (2048, 2048, 11264): 320,
    (2048, 5632, 2048): 320,
    (2048, 2048, 2048): 320,
    (2048, 2048, 3072): 320,
    (32768, 768, 4096): 48,
    (32768, 768, 2304): 48,
    (32768, 2048, 768): 48,
    (32768, 768, 768): 48,
}

PACKED_TILE_SWEEP = (
    (32, 64, 32, 4),
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (64, 128, 64, 4),
    (128, 64, 64, 4),
    (128, 128, 64, 4),
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


def _dense_equiv_tflops(m: int, k: int, n: int, ms: float) -> float:
    return (2.0 * m * k * n) / (ms * 1.0e9)


def _grid_size(m: int, n: int, tile: tuple[int, int, int, int]) -> tuple[int, int]:
    if triton is None:
        return 0, 0
    bm, bn, _bk, _warps = tile
    return triton.cdiv(m, bm), triton.cdiv(n, bn)


def _print_dense_efficiency(
    title: str,
    results: dict[str, dict[str, float]],
    rows: tuple[tuple[str, str], ...],
    *,
    m: int,
    k: int,
    n: int,
    calls_per_step: int,
) -> None:
    print(f"\n[{title}]")
    for label, key in rows:
        if key not in results:
            continue
        ms = results[key]["median"]
        print(
            f"  {label:22s} {ms:8.3f} ms  "
            f"{_dense_equiv_tflops(m, k, n, ms):8.3f} dense-eq TF/s  "
            f"step={ms * calls_per_step:8.1f} ms @ calls={calls_per_step}"
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


def _fp8_tensorwise_scale(t: torch.Tensor) -> torch.Tensor:
    return (t.detach().abs().amax().float() / _FP8_MAX).clamp_min(1e-12)


def _fp8_cast_transposed_with_scale(
    t: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Measure the production FP8 cast+transpose kernel without scale reduction."""
    t = t.contiguous()
    if triton is None or not t.is_cuda:
        return (
            (t / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_E4M3).t().contiguous()
        )
    rows, cols = t.shape
    out = torch.empty((cols, rows), device=t.device, dtype=_FP8_E4M3)
    block_r, block_c = 32, 32
    grid = (triton.cdiv(rows, block_r), triton.cdiv(cols, block_c))
    bitlinear_mod._fp8_cast_transpose_kernel[grid](
        t,
        out,
        scale,
        rows,
        cols,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


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


def _ternary_forward(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    row_scale: torch.Tensor,
    k: int,
    n: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Replicate the full TernaryBitLinearSTE.forward quantize + packed GEMM path."""
    x_int8, inv_sx = _quantize_a8_rows(x)
    return _packed_linear(
        x_int8,
        inv_sx,
        w_packed,
        row_scale,
        k,
        n,
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
    tile_sweep: bool,
    calls_per_step: int,
) -> None:
    m, k, n = shape
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    x_int8, inv_sx = _quantize_a8_rows(x)
    w = torch.randn((n, k), device="cuda", dtype=dtype)
    w_int8, scale = ternary_quantize_int8(w)
    row_scale = scale.expand(n).contiguous()
    w_packed = pack_ternary_weight(w_int8)
    w_packed_t = pack_ternary_weight(w_int8.t().contiguous())
    w_packed_kmajor = pack_ternary_weight_kmajor(w_int8)
    w_packed_t_kmajor = pack_ternary_weight_kmajor(w_int8.t().contiguous())

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
        "packed_kmajor_current": lambda: _packed_linear(
            x_int8,
            inv_sx,
            w_packed_kmajor,
            row_scale,
            k,
            n,
            dtype,
            grouped_decode=False,
            kmajor_layout=True,
        ),
        "packed_kmajor_single_dot": lambda: _packed_linear(
            x_int8,
            inv_sx,
            w_packed_kmajor,
            row_scale,
            k,
            n,
            dtype,
            grouped_decode=False,
            kmajor_layout=True,
            decode_v2=True,
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
    _print_dense_efficiency(
        "forward dense-equivalent throughput",
        results,
        (
            ("torch._int_mm", "int8_int_mm"),
            ("triton int8", "int8_triton"),
            ("packed current", "packed_dot_current"),
            ("packed K-major", "packed_kmajor_current"),
            ("packed K-major v2", "packed_kmajor_single_dot"),
            ("packed grouped", "packed_dot"),
        ),
        m=m,
        k=k,
        n=n,
        calls_per_step=calls_per_step,
    )

    if tile_sweep:
        print("\n[packed tile sweep]")
        for tile in PACKED_TILE_SWEEP:
            with _override_dot_current_tile(tile):
                fwd = _measure(
                    lambda: _packed_linear(
                        x_int8,
                        inv_sx,
                        w_packed,
                        row_scale,
                        k,
                        n,
                        dtype,
                        grouped_decode=False,
                    ),
                    warmup=warmup,
                    iters=iters,
                )
                grid_m, grid_n = _grid_size(m, n, tile)
                bm, bn, bk, warps = tile
                print(
                    f"  fwd BM={bm:3d} BN={bn:3d} BK={bk:3d} W={warps} "
                    f"grid={grid_m:4d}x{grid_n:<4d} "
                    f"{fwd['median']:8.3f} ms  "
                    f"{_dense_equiv_tflops(m, k, n, fwd['median']):8.3f} TF/s"
                )

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
        g2 = grad.reshape(-1, n)
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
            "fwd_packed_kmajor_current": lambda: _packed_linear(
                x_int8,
                inv_sx,
                w_packed_kmajor,
                row_scale,
                k,
                n,
                dtype,
                grouped_decode=False,
                kmajor_layout=True,
            ),
            "fwd_packed_kmajor_single_dot": lambda: _packed_linear(
                x_int8,
                inv_sx,
                w_packed_kmajor,
                row_scale,
                k,
                n,
                dtype,
                grouped_decode=False,
                kmajor_layout=True,
                decode_v2=True,
            ),
            "ternary_forward_total": lambda: _ternary_forward(
                x,
                w_packed,
                row_scale,
                k,
                n,
                dtype,
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
            "dx_packed_kmajor_current": lambda: _packed_linear(
                g_int8,
                inv_sg,
                w_packed_t_kmajor,
                one,
                n,
                k,
                dtype,
                grouped_decode=False,
                kmajor_layout=True,
            ),
            "dx_packed_kmajor_single_dot": lambda: _packed_linear(
                g_int8,
                inv_sg,
                w_packed_t_kmajor,
                one,
                n,
                k,
                dtype,
                grouped_decode=False,
                kmajor_layout=True,
                decode_v2=True,
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
            fp8_dw_sg = _fp8_tensorwise_scale(g2)
            fp8_dw_sx = _fp8_tensorwise_scale(x_q)
            fp8_dw_gt = _fp8_cast_transposed_with_scale(g2, fp8_dw_sg)
            fp8_dw_x = _fp8_cast_transposed_with_scale(x_q, fp8_dw_sx)
            torch.cuda.synchronize()
            breakdown_kernels["int8_fp8_dx"] = (
                lambda: _int8_fp8_dx(grad, row_scale, w_fp8_t, dtype)
            )
            breakdown_kernels["fp8_dw_reconstruct_x"] = (
                lambda: x_int8.to(dtype) * inv_sx.to(dtype).unsqueeze(1)
            )
            breakdown_kernels["fp8_dw_scale_grad"] = (
                lambda: _fp8_tensorwise_scale(g2)
            )
            breakdown_kernels["fp8_dw_scale_x"] = (
                lambda: _fp8_tensorwise_scale(x_q)
            )
            breakdown_kernels["fp8_dw_scale_both"] = (
                lambda: (_fp8_tensorwise_scale(g2), _fp8_tensorwise_scale(x_q))
            )
            breakdown_kernels["fp8_dw_cast_grad_t"] = (
                lambda: _fp8_cast_transposed_with_scale(g2, fp8_dw_sg)
            )
            breakdown_kernels["fp8_dw_cast_x_t"] = (
                lambda: _fp8_cast_transposed_with_scale(x_q, fp8_dw_sx)
            )
            breakdown_kernels["fp8_dw_direct_cast_x_t"] = (
                lambda: _cast_a8_dequant_fp8_transposed(x_int8, inv_sx)
            )
            breakdown_kernels["fp8_dw_cast_both"] = (
                lambda: (
                    _fp8_cast_transposed_with_scale(g2, fp8_dw_sg),
                    _fp8_cast_transposed_with_scale(x_q, fp8_dw_sx),
                )
            )
            breakdown_kernels["fp8_dw_gemm_only"] = (
                lambda: _scaled_mm_tensorwise(
                    fp8_dw_gt,
                    fp8_dw_x.t(),
                    fp8_dw_sg,
                    fp8_dw_sx,
                    dtype,
                )
            )
            breakdown_kernels["fp8_dw_old_total"] = (
                lambda: _int8_fp8_wgrad(grad, x_int8, inv_sx, dtype)
            )
            breakdown_kernels["fp8_dw_direct_total"] = (
                lambda: _fp8_wgrad_from_a8(g2, x_int8, inv_sx, dtype)
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
            fp8_dw_old_ms = breakdown_results["fp8_dw_old_total"]["median"]
            fp8_dw_ms = breakdown_results["fp8_dw_direct_total"]["median"]
            fp8_grad_scale_ms = breakdown_results["fp8_dw_scale_grad"]["median"]
            fp8_grad_cast_ms = breakdown_results["fp8_dw_cast_grad_t"]["median"]
            fp8_gemm_ms = breakdown_results["fp8_dw_gemm_only"]["median"]
            fp8_direct_x_ms = breakdown_results["fp8_dw_direct_cast_x_t"]["median"]
            fp8_other_ms = fp8_dw_ms - (
                fp8_grad_scale_ms
                + fp8_grad_cast_ms
                + fp8_direct_x_ms
                + fp8_gemm_ms
            )
            print("\n[fp8 dW split]")
            print(
                "  old_total / direct_total / grad_scale / grad_cast / "
                "direct_x_scale_cast / gemm_only / other: "
                f"{fp8_dw_old_ms:.3f} / {fp8_dw_ms:.3f} / "
                f"{fp8_grad_scale_ms:.3f} / {fp8_grad_cast_ms:.3f} / "
                f"{fp8_direct_x_ms:.3f} / {fp8_gemm_ms:.3f} / "
                f"{fp8_other_ms:.3f} ms"
            )
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
        _print_dense_efficiency(
            "backward dense-equivalent throughput",
            breakdown_results,
            (
                ("dX packed gemm", "dx_packed_current"),
                ("dX packed K-major", "dx_packed_kmajor_current"),
                ("dX packed K-major v2", "dx_packed_kmajor_single_dot"),
                ("dX ternary total", "ternary_dx_total"),
                ("dX FP8 total", "int8_fp8_dx"),
                ("dW INT8 gemm", "wgrad_gemm"),
                ("dW INT8 total", "ternary_wgrad_total"),
                ("dW FP8 gemm", "fp8_dw_gemm_only"),
                ("dW FP8 old total", "fp8_dw_old_total"),
                ("dW FP8 direct total", "fp8_dw_direct_total"),
            ),
            m=m,
            k=k,
            n=n,
            calls_per_step=calls_per_step,
        )

        if tile_sweep:
            print("\n[packed dX tile sweep]")
            for tile in PACKED_TILE_SWEEP:
                with _override_dot_current_tile(tile):
                    dx = _measure(
                        lambda: _packed_linear(
                            g_int8,
                            inv_sg,
                            w_packed_t,
                            one,
                            n,
                            k,
                            dtype,
                            grouped_decode=False,
                        ),
                        warmup=warmup,
                        iters=iters,
                    )
                    grid_m, grid_n = _grid_size(m, k, tile)
                    bm, bn, bk, warps = tile
                    print(
                        f"  dX  BM={bm:3d} BN={bn:3d} BK={bk:3d} W={warps} "
                        f"grid={grid_m:4d}x{grid_n:<4d} "
                        f"{dx['median']:8.3f} ms  "
                        f"{_dense_equiv_tflops(m, k, n, dx['median']):8.3f} TF/s"
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
    parser.add_argument(
        "--tile-sweep",
        action="store_true",
        help="Sweep candidate dot_current packed tiles for forward and, with --breakdown, dX.",
    )
    parser.add_argument(
        "--calls-per-step",
        type=int,
        default=None,
        help="Override MB4 profiler calls/step used for step contribution reporting.",
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
            calls_per_step = (
                args.calls_per_step
                if args.calls_per_step is not None
                else DEFAULT_CALLS_PER_STEP.get(shape, 1)
            )
            _bench_shape(
                shape,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                check=not args.no_check,
                include_wgrad=args.wgrad,
                breakdown=args.breakdown,
                tile_sweep=args.tile_sweep,
                calls_per_step=calls_per_step,
            )


if __name__ == "__main__":
    main()
