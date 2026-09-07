"""Investigate packed ternary dot_current tile performance cliffs.

This is a diagnostic-only benchmark. It directly launches the production
packed ternary Triton kernel with explicit tile/warp/stage choices, and can
compare it with an unpacked INT8 tl.dot kernel using the same tile.

Examples:
  python -m scripts.investigate_packed_tile_cliff --suite bn --warmup 50 --iters 500
  python -m scripts.investigate_packed_tile_cliff --suite k --warmup 50 --iters 500
  python -m scripts.investigate_packed_tile_cliff --single --tile 128,64,64,4 \
    --m 2048 --k 11264 --n 2048 --iters 1
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from src.model.bitlinear import (
    _int8_bitlinear_kernel,
    _packed_bitlinear_kernel,
    _quantize_a8_rows,
    _quantize_a8_rows_scaled,
    pack_ternary_weight,
)

try:
    import triton
except Exception:  # pragma: no cover - CUDA stack dependent
    triton = None


@dataclass(frozen=True)
class Tile:
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int = 3

    def text(self) -> str:
        return (
            f"BM={self.bm:3d} BN={self.bn:3d} BK={self.bk:3d} "
            f"W={self.warps} S={self.stages}"
        )


@dataclass(frozen=True)
class Timing:
    median: float
    p10: float
    p90: float


MAIN4_TILES = (
    Tile(128, 64, 64, 4),
    Tile(128, 64, 128, 4),
    Tile(128, 128, 32, 4),
    Tile(128, 128, 64, 8),
)

GLOBAL_PHYSICAL_SHAPES = (
    (2048, 2048, 11264, "global fwd K2048 N11264"),
    (2048, 11264, 2048, "global dX K11264 N2048"),
    (2048, 5632, 2048, "global fwd K5632 N2048"),
    (2048, 2048, 5632, "global dX K2048 N5632"),
    (2048, 2048, 3072, "global fwd K2048 N3072"),
    (2048, 3072, 2048, "global dX K3072 N2048"),
    (2048, 2048, 2048, "global square K2048 N2048"),
)


def _parse_tile(spec: str) -> Tile:
    parts = [int(part) for part in spec.split(",")]
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            f"tile must be BM,BN,BK,WARPS[,STAGES], got {spec!r}"
        )
    if min(parts) <= 0:
        raise argparse.ArgumentTypeError(f"tile values must be positive: {spec!r}")
    if len(parts) == 4:
        parts.append(3)
    return Tile(*parts)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _tile_supported(tile: Tile) -> bool:
    return (
        _is_power_of_two(tile.bm)
        and _is_power_of_two(tile.bn)
        and _is_power_of_two(tile.bk)
    )


def _measure(fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]], *, warmup: int, iters: int) -> Timing:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return Timing(
        median=statistics.median(samples),
        p10=samples[max(0, int(0.10 * (len(samples) - 1)))],
        p90=samples[min(len(samples) - 1, int(0.90 * (len(samples) - 1)))],
    )


def _dense_equiv_tflops(m: int, k: int, n: int, ms: float) -> float:
    return (2.0 * m * k * n) / (ms * 1.0e9)


def _print_timing(label: str, m: int, k: int, n: int, timing: Timing, base: Timing | None = None) -> None:
    ratio = "" if base is None else f" x{timing.median / base.median:5.2f}"
    print(
        f"  {label:34s} {timing.median:8.3f} ms  "
        f"p10={timing.p10:8.3f} p90={timing.p90:8.3f}  "
        f"{_dense_equiv_tflops(m, k, n, timing.median):8.3f} TF/s{ratio}"
    )


def _make_inputs(m: int, k: int, n: int, dtype: torch.dtype, *, scale_per_output: bool):
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    grad = torch.randn((m, n), device="cuda", dtype=dtype)
    x_int8, inv_sx = _quantize_a8_rows(x)
    w_int8 = torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8)
    w_packed = pack_ternary_weight(w_int8)
    if scale_per_output:
        sw = torch.rand((n,), device="cuda", dtype=torch.float32) * 0.01 + 0.01
    else:
        sw = torch.ones((), device="cuda", dtype=torch.float32)
    return x, grad, x_int8, inv_sx, w_int8, w_packed, sw


def _packed_direct(
    x_int8: torch.Tensor,
    inv_sx: torch.Tensor,
    w_packed: torch.Tensor,
    sw: torch.Tensor,
    k: int,
    n: int,
    out_dtype: torch.dtype,
    tile: Tile,
) -> torch.Tensor:
    m = x_int8.size(0)
    y = torch.empty((m, n), device=x_int8.device, dtype=out_dtype)
    grid = (triton.cdiv(m, tile.bm), triton.cdiv(n, tile.bn))
    _packed_bitlinear_kernel[grid](
        x_int8.contiguous(),
        w_packed,
        inv_sx.contiguous(),
        sw.contiguous(),
        y,
        m,
        n,
        k,
        w_packed.size(1),
        x_int8.stride(0),
        y.stride(0),
        SCALE_PER_OUTPUT=sw.numel() != 1,
        GROUPED_DECODE=False,
        BLOCK_M=tile.bm,
        BLOCK_N=tile.bn,
        BLOCK_K=tile.bk,
        num_warps=tile.warps,
        num_stages=tile.stages,
    )
    return y


def _compile_packed(
    x_int8: torch.Tensor,
    inv_sx: torch.Tensor,
    w_packed: torch.Tensor,
    sw: torch.Tensor,
    k: int,
    n: int,
    out_dtype: torch.dtype,
    tile: Tile,
):
    m = x_int8.size(0)
    y = torch.empty((m, n), device=x_int8.device, dtype=out_dtype)
    grid = (triton.cdiv(m, tile.bm), triton.cdiv(n, tile.bn))
    return _packed_bitlinear_kernel.warmup(
        x_int8.contiguous(),
        w_packed,
        inv_sx.contiguous(),
        sw.contiguous(),
        y,
        m,
        n,
        k,
        w_packed.size(1),
        x_int8.stride(0),
        y.stride(0),
        SCALE_PER_OUTPUT=sw.numel() != 1,
        GROUPED_DECODE=False,
        BLOCK_M=tile.bm,
        BLOCK_N=tile.bn,
        BLOCK_K=tile.bk,
        num_warps=tile.warps,
        num_stages=tile.stages,
        grid=grid,
    )


def _ptx_register_count(ptx: str) -> int:
    total = 0
    for match in re.finditer(r"\.reg\s+\.\w+\s+%[a-z]+<(\d+)>;", ptx):
        total += int(match.group(1))
    return total


def _print_compiled_resource(
    label: str,
    m: int,
    k: int,
    n: int,
    tile: Tile,
    *,
    dtype: torch.dtype,
    scale_per_output: bool,
) -> None:
    _x, _grad, x_int8, inv_sx, _w_int8, w_packed, sw = _make_inputs(
        m, k, n, dtype, scale_per_output=scale_per_output
    )
    compiled = _compile_packed(x_int8, inv_sx, w_packed, sw, k, n, dtype, tile)
    metadata = compiled.metadata
    ptx = compiled.asm.get("ptx", "")
    reg_count = _ptx_register_count(ptx)
    ld_local = ptx.count("ld.local")
    st_local = ptx.count("st.local")
    print(
        f"  {label:12s} {tile.text()} "
        f"shared={int(getattr(metadata, 'shared', 0)):6d}B "
        f"ptx_regs_decl={reg_count:4d} "
        f"ld.local={ld_local:3d} st.local={st_local:3d}"
    )


def _unpacked_int8_direct(
    x_int8: torch.Tensor,
    inv_sx: torch.Tensor,
    w_int8: torch.Tensor,
    sw: torch.Tensor,
    out_dtype: torch.dtype,
    tile: Tile,
) -> torch.Tensor:
    if sw.numel() != w_int8.size(0):
        raise ValueError("unpacked INT8 diagnostic currently requires per-output scale")
    m, k = x_int8.shape
    n = w_int8.size(0)
    y = torch.empty((m, n), device=x_int8.device, dtype=out_dtype)
    grid = (triton.cdiv(m, tile.bm), triton.cdiv(n, tile.bn))
    _int8_bitlinear_kernel[grid](
        x_int8.contiguous(),
        w_int8.contiguous(),
        inv_sx.contiguous(),
        sw.contiguous(),
        y,
        m,
        n,
        k,
        BLOCK_M=tile.bm,
        BLOCK_N=tile.bn,
        BLOCK_K=tile.bk,
        num_warps=tile.warps,
        num_stages=tile.stages,
    )
    return y


def _bench_packed(
    m: int,
    k: int,
    n: int,
    tile: Tile,
    *,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    scale_per_output: bool,
) -> Timing:
    _x, _grad, x_int8, inv_sx, _w_int8, w_packed, sw = _make_inputs(
        m, k, n, dtype, scale_per_output=scale_per_output
    )
    return _measure(
        lambda: _packed_direct(x_int8, inv_sx, w_packed, sw, k, n, dtype, tile),
        warmup=warmup,
        iters=iters,
    )


def _bench_unpacked(
    m: int,
    k: int,
    n: int,
    tile: Tile,
    *,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> Timing:
    _x, _grad, x_int8, inv_sx, w_int8, _w_packed, sw = _make_inputs(
        m, k, n, dtype, scale_per_output=True
    )
    return _measure(
        lambda: _unpacked_int8_direct(x_int8, inv_sx, w_int8, sw, dtype, tile),
        warmup=warmup,
        iters=iters,
    )


def _bench_a8_total(
    m: int,
    k: int,
    n: int,
    tile: Tile,
    *,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    scale_per_output: bool,
) -> None:
    x, grad, x_int8, inv_sx, _w_int8, w_packed, sw = _make_inputs(
        m, k, n, dtype, scale_per_output=scale_per_output
    )
    timing_quant = _measure(lambda: _quantize_a8_rows(x), warmup=warmup, iters=iters)
    timing_gemm = _measure(
        lambda: _packed_direct(x_int8, inv_sx, w_packed, sw, k, n, dtype, tile),
        warmup=warmup,
        iters=iters,
    )
    timing_total = _measure(
        lambda: _packed_direct(*_quantize_a8_rows(x), w_packed, sw, k, n, dtype, tile),
        warmup=warmup,
        iters=iters,
    )
    row_scale = sw if sw.numel() == n else torch.ones((n,), device="cuda", dtype=torch.float32)
    w_packed_t = pack_ternary_weight(torch.randint(-1, 2, (k, n), device="cuda", dtype=torch.int8))
    g_int8, inv_sg = _quantize_a8_rows_scaled(grad, row_scale)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    dx_quant = _measure(
        lambda: _quantize_a8_rows_scaled(grad, row_scale),
        warmup=warmup,
        iters=iters,
    )
    dx_gemm = _measure(
        lambda: _packed_direct(g_int8, inv_sg, w_packed_t, one, n, k, dtype, tile),
        warmup=warmup,
        iters=iters,
    )
    dx_total = _measure(
        lambda: _packed_direct(
            *_quantize_a8_rows_scaled(grad, row_scale), w_packed_t, one, n, k, dtype, tile
        ),
        warmup=warmup,
        iters=iters,
    )
    print(f"\n[A8 total] M={m} K={k} N={n} {tile.text()}")
    _print_timing("fwd quant only", m, k, n, timing_quant)
    _print_timing("fwd packed gemm only", m, k, n, timing_gemm)
    _print_timing("fwd quant + packed gemm", m, k, n, timing_total)
    _print_timing("dx scaled quant only", m, k, n, dx_quant)
    _print_timing("dx packed gemm only", m, k, n, dx_gemm)
    _print_timing("dx quant + packed gemm", m, k, n, dx_total)


def _run_table(
    title: str,
    cases: Iterable[tuple[str, int, int, int, Tile, bool]],
    *,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> None:
    print(f"\n[{title}]")
    base: Timing | None = None
    for label, m, k, n, tile, scale_per_output in cases:
        if not _tile_supported(tile):
            print(f"  {label:34s} {tile.text()} skipped: tile sizes must be powers of two")
            continue
        timing = _bench_packed(
            m,
            k,
            n,
            tile,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            scale_per_output=scale_per_output,
        )
        if base is None:
            base = timing
        _print_timing(f"{label} {tile.text()}", m, k, n, timing, base)


def _suite_bn(args, dtype: torch.dtype) -> None:
    cases = [
        (f"BN={bn}", args.m, args.k, args.n, Tile(128, bn, 64, 4, args.stages), True)
        for bn in (32, 64, 96, 128, 256)
    ]
    _run_table("BN sweep", cases, dtype=dtype, warmup=args.warmup, iters=args.iters)


def _suite_k(args, dtype: torch.dtype) -> None:
    cases = []
    for k in (1024, 2048, 3072, 4096, 5632, 8192, 11264):
        cases.append((f"K={k} BN64 ", args.m, k, args.n, Tile(128, 64, 64, 4, args.stages), True))
        cases.append((f"K={k} BN128", args.m, k, args.n, Tile(128, 128, 64, 4, args.stages), True))
    _run_table("K sweep", cases, dtype=dtype, warmup=args.warmup, iters=args.iters)


def _suite_bk(args, dtype: torch.dtype) -> None:
    cases = []
    for bn in (64, 128):
        for bk in (32, 64, 128):
            cases.append((f"BN={bn} BK={bk}", args.m, args.k, args.n, Tile(128, bn, bk, 4, args.stages), True))
    _run_table("BLOCK_K sweep", cases, dtype=dtype, warmup=args.warmup, iters=args.iters)


def _suite_warps(args, dtype: torch.dtype) -> None:
    cases = []
    for bn in (64, 128):
        for warps in (2, 4, 8):
            cases.append((f"BN={bn} W={warps}", args.m, args.k, args.n, Tile(128, bn, 64, warps, args.stages), True))
    _run_table("num_warps sweep", cases, dtype=dtype, warmup=args.warmup, iters=args.iters)


def _suite_stages(args, dtype: torch.dtype) -> None:
    cases = []
    for bn in (64, 128):
        for stages in (1, 2, 3, 4, 5):
            cases.append((f"BN={bn} S={stages}", args.m, args.k, args.n, Tile(128, bn, 64, 4, stages), True))
    _run_table("num_stages sweep", cases, dtype=dtype, warmup=args.warmup, iters=args.iters)


def _suite_scale(args, dtype: torch.dtype) -> None:
    cases = []
    for scale_per_output in (False, True):
        label = "scale=per-output" if scale_per_output else "scale=scalar"
        cases.append((label, args.m, 2048, 2048, Tile(128, 128, 64, 4, args.stages), scale_per_output))
        cases.append((label, args.m, 2048, 2048, Tile(128, 64, 64, 4, args.stages), scale_per_output))
    _run_table("SCALE_PER_OUTPUT sweep", cases, dtype=dtype, warmup=args.warmup, iters=args.iters)


def _suite_unpacked(args, dtype: torch.dtype) -> None:
    print(f"\n[packed vs unpacked INT8] M={args.m} K={args.k} N={args.n}")
    base: Timing | None = None
    for bn in (64, 128):
        tile = Tile(128, bn, 64, 4, args.stages)
        packed = _bench_packed(
            args.m,
            args.k,
            args.n,
            tile,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            scale_per_output=True,
        )
        unpacked = _bench_unpacked(
            args.m,
            args.k,
            args.n,
            tile,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        if base is None:
            base = packed
        _print_timing(f"packed   {tile.text()}", args.m, args.k, args.n, packed, base)
        _print_timing(f"unpacked {tile.text()}", args.m, args.k, args.n, unpacked, base)


def _suite_main4(args, dtype: torch.dtype) -> None:
    print("\n[main 4 tile sweep]")
    for m, k, n, name in GLOBAL_PHYSICAL_SHAPES:
        rows = []
        print(f"\n[physical shape] {name}: M={m} K={k} N={n}")
        for tile in MAIN4_TILES:
            timing = _bench_packed(
                m,
                k,
                n,
                tile,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                scale_per_output=not args.scalar_scale,
            )
            rows.append((timing.median, tile, timing))
            _print_timing(tile.text(), m, k, n, timing)
        rows.sort(key=lambda row: row[0])
        best_ms, best_tile, best_timing = rows[0]
        second_ms = rows[1][0] if len(rows) > 1 else float("nan")
        print(
            f"  WINNER {best_tile.text()} median={best_ms:.3f} ms "
            f"margin={second_ms / best_ms:.2f}x vs second "
            f"p10={best_timing.p10:.3f} p90={best_timing.p90:.3f}"
        )


def _suite_resource(args, dtype: torch.dtype) -> None:
    print(f"\n[compile resources] M={args.m} K={args.k} N={args.n}")
    for tile in (
        Tile(128, 64, 64, 4, args.stages),
        Tile(128, 128, 64, 4, args.stages),
        Tile(128, 128, 64, 8, args.stages),
        Tile(128, 64, 128, 4, args.stages),
        Tile(128, 128, 32, 4, args.stages),
    ):
        _print_compiled_resource(
            "packed",
            args.m,
            args.k,
            args.n,
            tile,
            dtype=dtype,
            scale_per_output=not args.scalar_scale,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "bn",
            "k",
            "bk",
            "warps",
            "stages",
            "scale",
            "unpacked",
            "a8",
            "resource",
            "main4",
        ],
        default="bn",
    )
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=11264)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--tile", type=_parse_tile, default=Tile(128, 64, 64, 4, 3))
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--scalar-scale", action="store_true")
    args = parser.parse_args()

    if triton is None or not torch.cuda.is_available():
        raise SystemExit("CUDA + Triton are required")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >=0 and --iters must be >0")

    torch.manual_seed(20260907)
    dtype = getattr(torch, args.dtype)
    if args.single:
        timing = _bench_packed(
            args.m,
            args.k,
            args.n,
            args.tile,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            scale_per_output=not args.scalar_scale,
        )
        print(f"\n[single] M={args.m} K={args.k} N={args.n}")
        _print_timing(args.tile.text(), args.m, args.k, args.n, timing)
        return

    suites = (
        [
            "bn",
            "k",
            "bk",
            "warps",
            "stages",
            "scale",
            "unpacked",
            "a8",
            "resource",
            "main4",
        ]
        if args.suite == "all"
        else [args.suite]
    )
    for suite in suites:
        if suite == "bn":
            _suite_bn(args, dtype)
        elif suite == "k":
            _suite_k(args, dtype)
        elif suite == "bk":
            _suite_bk(args, dtype)
        elif suite == "warps":
            _suite_warps(args, dtype)
        elif suite == "stages":
            _suite_stages(args, dtype)
        elif suite == "scale":
            _suite_scale(args, dtype)
        elif suite == "unpacked":
            _suite_unpacked(args, dtype)
        elif suite == "a8":
            _bench_a8_total(
                args.m,
                args.k,
                args.n,
                args.tile,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                scale_per_output=not args.scalar_scale,
            )
        elif suite == "resource":
            _suite_resource(args, dtype)
        elif suite == "main4":
            _suite_main4(args, dtype)


if __name__ == "__main__":
    main()
