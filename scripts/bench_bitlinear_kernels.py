"""BitLinear low-bit kernel microbenchmarks.

Measures the decode question directly:

  A. unpacked INT8 W + Triton tl.dot
  B. packed2 W + current logical-K decode + tl.dot
  C. packed2 W + grouped 4-way decode + tl.dot (A/B only)

Example:
  python -m scripts.bench_bitlinear_kernels --shape 1024,2048,2048
"""
from __future__ import annotations

import argparse
import statistics

import torch

from src.model.bitlinear import (
    _int8_linear,
    _lowbit_wgrad,
    _packed_linear,
    _quantize_a8_rows,
    pack_ternary_weight,
    set_bitlinear_int8_backend,
    ternary_quantize_int8,
)


DEFAULT_SHAPES = (
    "1024,2048,2048",
    "1024,2048,5632",
    "1024,5632,2048",
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


def _bench_shape(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    check: bool,
    include_wgrad: bool,
) -> None:
    m, k, n = shape
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    x_int8, inv_sx = _quantize_a8_rows(x)
    w = torch.randn((n, k), device="cuda", dtype=dtype)
    w_int8, scale = ternary_quantize_int8(w)
    row_scale = scale.expand(n).contiguous()
    w_packed = pack_ternary_weight(w_int8)

    set_bitlinear_int8_backend("triton")
    kernels = {
        "int8_triton": lambda: _int8_linear(
            x_int8, inv_sx, w_int8, row_scale, dtype
        ),
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
    if check:
        reference = kernels["int8_triton"]().float()
        for name, fn in kernels.items():
            diff = (fn().float() - reference).abs().max().item()
            print(f"[check] shape={m},{k},{n} {name} max_abs_diff={diff:.6g}")

    print(f"\n[shape] M={m} K={k} N={n} dtype={dtype}")
    results = {}
    for name, fn in kernels.items():
        results[name] = _measure(fn, warmup=warmup, iters=iters)
    baseline = results["int8_triton"]["median"]
    for name, result in results.items():
        print(f"  {name:19s} {_fmt(result, baseline)}")

    if include_wgrad:
        grad = torch.randn((m, n), device="cuda", dtype=dtype)
        x_q = x_int8.to(dtype) * inv_sx.to(dtype).unsqueeze(1)
        result = _measure(
            lambda: _lowbit_wgrad(grad, x_q, dtype),
            warmup=warmup,
            iters=iters,
        )
        print(f"  {'lowbit_wgrad':19s} {result['median']:8.3f} ms  "
              f"p10={result['p10']:8.3f}  p90={result['p90']:8.3f}")


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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >=0 and --iters must be >0")

    shapes = args.shape or [_parse_shape(spec) for spec in DEFAULT_SHAPES]
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    for shape in shapes:
        _bench_shape(
            shape,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            check=not args.no_check,
            include_wgrad=args.wgrad,
        )


if __name__ == "__main__":
    main()
