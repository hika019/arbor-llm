"""Profile Arbor packed-ternary BitLinear shapes without modifying existing source files.

This script:
1. Builds the real Arbor model in ternary mode.
2. Monkey-patches BitLinear internals only in this process to count logical
   (M, K, N) calls for forward, dX, and dW during one real fwd+bwd.
3. Benchmarks each unique shape in isolation with the same production kernels.
4. Multiplies median kernel time by calls/optimizer-step and prints estimated
   contribution per shape.
5. Optionally compares the same shapes against the INT8 training path and
   ranks estimated extra cost per optimizer step.

Example:
    python -m scripts.profile_bitlinear_shapes \
      --micro-batch 2 --grad-accum 32 --warmup 50 --iters 500 \
      --ternary-backend dot_current --wgrad-backend fp8
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml

from src.model.arbor import build_arbor
import src.model.bitlinear as bl


@dataclass(frozen=True, order=True)
class Shape:
    m: int
    k: int
    n: int

    def text(self) -> str:
        return f"{self.m}x{self.k}->{self.n}"


@dataclass
class Timing:
    median: float
    p10: float
    p90: float


@dataclass
class PhaseTimings:
    fwd: Timing
    dx: Timing
    dw: Timing


_COUNTS = {
    "fwd": collections.Counter(),
    "dx": collections.Counter(),
    "dw": collections.Counter(),
}
_PHASE: str | None = None


@contextlib.contextmanager
def _phase(name: str):
    global _PHASE
    old = _PHASE
    _PHASE = name
    try:
        yield
    finally:
        _PHASE = old


def _install_shape_counter():
    """Install process-local wrappers and return restore callback."""
    orig_forward = bl.TernaryBitLinearSTE.forward
    orig_backward = bl.TernaryBitLinearSTE.backward
    orig_packed = bl._packed_linear
    orig_wgrad_from_a8 = bl._ternary_wgrad_from_a8

    def counted_forward(ctx, *args, **kwargs):
        with _phase("fwd"):
            return orig_forward(ctx, *args, **kwargs)

    def counted_backward(ctx, *args, **kwargs):
        with _phase("bwd"):
            return orig_backward(ctx, *args, **kwargs)

    def counted_packed(
        x_int8,
        inv_sx,
        w_packed,
        row_scale,
        k,
        n,
        out_dtype,
        *,
        grouped_decode,
        kmajor_layout=False,
        decode_v2=False,
        backend=None,
        launch_config=None,
    ):
        if _PHASE == "fwd":
            # Forward: [M,K] @ W[N,K]^T -> [M,N]
            _COUNTS["fwd"][Shape(int(x_int8.size(0)), int(k), int(n))] += 1
        elif _PHASE == "bwd":
            # dX call is [M,N] @ W[N,K] -> [M,K]. Report original K->N.
            _COUNTS["dx"][Shape(int(x_int8.size(0)), int(n), int(k))] += 1
        return orig_packed(
            x_int8,
            inv_sx,
            w_packed,
            row_scale,
            k,
            n,
            out_dtype,
            grouped_decode=grouped_decode,
            kmajor_layout=kmajor_layout,
            decode_v2=decode_v2,
            backend=backend,
            launch_config=launch_config,
        )

    def counted_wgrad_from_a8(grad_output, x_int8, inv_sx, out_dtype):
        # dW = dY^T [N,M] @ X [M,K] -> [N,K]
        _COUNTS["dw"][
            Shape(
                int(grad_output.size(0)),
                int(x_int8.size(1)),
                int(grad_output.size(1)),
            )
        ] += 1
        return orig_wgrad_from_a8(grad_output, x_int8, inv_sx, out_dtype)

    bl.TernaryBitLinearSTE.forward = staticmethod(counted_forward)
    bl.TernaryBitLinearSTE.backward = staticmethod(counted_backward)
    bl._packed_linear = counted_packed
    bl._ternary_wgrad_from_a8 = counted_wgrad_from_a8

    def restore():
        bl.TernaryBitLinearSTE.forward = orig_forward
        bl.TernaryBitLinearSTE.backward = orig_backward
        bl._packed_linear = orig_packed
        bl._ternary_wgrad_from_a8 = orig_wgrad_from_a8

    return restore


def _measure(fn, *, warmup: int, iters: int) -> Timing:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    xs = sorted(samples)
    return Timing(
        median=statistics.median(xs),
        p10=xs[max(0, int(0.10 * (len(xs) - 1)))],
        p90=xs[min(len(xs) - 1, int(0.90 * (len(xs) - 1)))],
    )


def _bench_ternary_shape(
    shape, *, dtype, grouped_decode, kmajor_layout, decode_v2, warmup, iters
):
    m, k, n = shape.m, shape.k, shape.n
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    grad = torch.randn((m, n), device="cuda", dtype=dtype)
    w_int8 = torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8)
    row_scale = torch.ones((n,), device="cuda", dtype=torch.float32)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    pack = (
        bl.pack_ternary_weight_kmajor
        if kmajor_layout
        else bl.pack_ternary_weight
    )
    w_packed = pack(w_int8)
    w_packed_t = pack(w_int8.t().contiguous())
    x_int8_saved, inv_sx_saved = bl._quantize_a8_rows(x)

    def fwd_total():
        x_int8, inv_sx = bl._quantize_a8_rows(x)
        return bl._packed_linear(
            x_int8, inv_sx, w_packed, row_scale, k, n, dtype,
            grouped_decode=grouped_decode,
            kmajor_layout=kmajor_layout,
            decode_v2=decode_v2,
        )

    def dx_total():
        g_int8, inv_sg = bl._quantize_a8_rows_scaled(grad, row_scale)
        return bl._packed_linear(
            g_int8, inv_sg, w_packed_t, one, n, k, dtype,
            grouped_decode=grouped_decode,
            kmajor_layout=kmajor_layout,
            decode_v2=decode_v2,
        )

    def dw_total():
        return bl._ternary_wgrad_from_a8(
            grad, x_int8_saved, inv_sx_saved, dtype
        )

    return PhaseTimings(
        fwd=_measure(fwd_total, warmup=warmup, iters=iters),
        dx=_measure(dx_total, warmup=warmup, iters=iters),
        dw=_measure(dw_total, warmup=warmup, iters=iters),
    )


def _bench_int8_shape(shape: Shape, *, dtype, int8_backend, warmup, iters):
    if not bl.fp8_gemm_supported():
        raise RuntimeError("INT8 comparison requires FP8 backward support")

    m, k, n = shape.m, shape.k, shape.n
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    grad = torch.randn((m, n), device="cuda", dtype=dtype)
    w_int8 = torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8)
    row_scale = torch.ones((n,), device="cuda", dtype=torch.float32)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    w_fp8 = w_int8.to(bl._FP8_E4M3).contiguous()
    w_fp8_t = w_fp8.t().contiguous()
    x_int8_saved, inv_sx_saved = bl._quantize_a8_rows(x)

    def fwd_total():
        bl.set_bitlinear_int8_backend(int8_backend)
        x_int8, inv_sx = bl._quantize_a8_rows(x)
        return bl._int8_linear(x_int8, inv_sx, w_int8, row_scale, dtype)

    def dx_total():
        g2 = grad.reshape(-1, n)
        gs_f8, sgs = bl._cast_fp8_tensorwise(g2 * row_scale.to(g2.dtype))
        return bl._scaled_mm_tensorwise(gs_f8, w_fp8_t.t(), sgs, one, dtype)

    def dw_total():
        x_q = x_int8_saved.to(dtype) * inv_sx_saved.to(dtype).unsqueeze(1)
        return bl._fp8_wgrad(grad, x_q, dtype)

    return PhaseTimings(
        fwd=_measure(fwd_total, warmup=warmup, iters=iters),
        dx=_measure(dx_total, warmup=warmup, iters=iters),
        dw=_measure(dw_total, warmup=warmup, iters=iters),
    )


def _build_and_count(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mcfg = dict(cfg["model"])
    if args.seq is not None:
        mcfg["max_bytes"] = args.seq
    seq = int(mcfg.get("max_bytes", cfg.get("data", {}).get("context_length", 8192)))

    model = build_arbor(mcfg).to(device="cuda", dtype=torch.bfloat16)
    model.train()

    bl.set_bitlinear_int8_backend("auto")
    bl.set_bitlinear_ternary_backend(args.ternary_backend)
    bl.set_bitlinear_ternary_execution_path(args.execution_path)
    bl.configure_bitlinear_ternary_tuning(
        mode="auto" if args.execution_path == "raw_plan" else "off",
        cache_enabled=True,
        cache_path=args.tune_cache_path,
    )
    bl.set_bitlinear_ternary_wgrad_backend(args.wgrad_backend)
    bl.install_arbor_projection_fusions(model)
    bl.set_bitlinear_fp8_mode(model, "ternary")

    speed_cfg = cfg.get("speed", {})
    cache_gib = float(
        args.weight_cache_gib
        if args.weight_cache_gib is not None
        else speed_cfg.get("bitnet_weight_cache_gib", 3.25)
    )
    cache_info = bl.configure_bitlinear_training_cache(
        model,
        enabled="full",
        grad_accum_steps=args.grad_accum,
        max_cache_gib=cache_gib,
        min_numel=int(speed_cfg.get("bitnet_weight_cache_min_numel", 0)),
    )
    print(
        "[count] "
        f"seq={seq} micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
        f"cache={cache_info['cache_gib']:.2f}GiB "
        f"ternary_backend={args.ternary_backend} wgrad_backend={args.wgrad_backend}"
    )

    restore = _install_shape_counter()
    try:
        x = torch.randint(4, 260, (args.micro_batch, seq), device="cuda", dtype=torch.long)
        model.zero_grad(set_to_none=True)
        logits = model(x).logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            x[:, 1:].reshape(-1),
        )
        loss.backward()
        torch.cuda.synchronize()
    finally:
        restore()

    counts = {phase: collections.Counter(counter) for phase, counter in _COUNTS.items()}
    del model
    torch.cuda.empty_cache()
    return counts


def _print_count_sanity(counts):
    print("\n[counts per micro-batch]")
    for phase in ("fwd", "dx", "dw"):
        total = sum(counts[phase].values())
        print(f"  {phase:3s}: {total:5d} calls, {len(counts[phase]):3d} unique shapes")

    all_shapes = sorted(set().union(*(counts[p].keys() for p in ("fwd", "dx", "dw"))))
    mismatches = []
    for s in all_shapes:
        triple = tuple(counts[p][s] for p in ("fwd", "dx", "dw"))
        if len(set(triple)) != 1:
            mismatches.append((s, triple))
    if mismatches:
        print("  [warn] phase call-count mismatch:")
        for s, (f, dx, dw) in mismatches:
            print(f"    {s.text():20s} fwd={f} dx={dx} dw={dw}")


def _phase_counts_per_step(counts, shape: Shape, grad_accum: int) -> tuple[int, int, int]:
    return (
        counts["fwd"][shape] * grad_accum,
        counts["dx"][shape] * grad_accum,
        counts["dw"][shape] * grad_accum,
    )


def _phase_step_ms(timing: PhaseTimings, counts_per_step: tuple[int, int, int]):
    c_f, c_dx, c_dw = counts_per_step
    return (
        timing.fwd.median * c_f,
        timing.dx.median * c_dx,
        timing.dw.median * c_dw,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arbor.yaml")
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument(
        "--ternary-backend",
        default="dot_current",
        choices=["dot_current", "kmajor_current", "kmajor_single_dot", "dot"],
    )
    ap.add_argument("--wgrad-backend", default="fp8", choices=["int8", "fp8", "auto"])
    ap.add_argument(
        "--execution-path",
        default="legacy_raw",
        choices=["legacy_raw", "legacy_custom_op", "custom_op", "raw_plan"],
    )
    ap.add_argument("--tune-cache-path", default="auto")
    ap.add_argument("--compare-int8", action="store_true")
    ap.add_argument("--int8-backend", default="auto", choices=["auto", "int_mm", "triton"])
    ap.add_argument("--weight-cache-gib", type=float, default=None)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >=0 and --iters must be >0")
    if (
        args.execution_path in {"legacy_raw", "legacy_custom_op"}
        and args.ternary_backend != "dot_current"
    ):
        raise SystemExit(
            f"--execution-path {args.execution_path} requires "
            "--ternary-backend dot_current"
        )

    torch.manual_seed(1234)
    counts = _build_and_count(args)
    _print_count_sanity(counts)

    shapes = sorted(set().union(*(counts[p].keys() for p in ("fwd", "dx", "dw"))))
    if not shapes:
        raise SystemExit("No TernaryBitLinearSTE calls were observed")

    bl.set_bitlinear_ternary_backend(args.ternary_backend)
    bl.set_bitlinear_ternary_execution_path(args.execution_path)
    bl.configure_bitlinear_ternary_tuning(
        mode="auto" if args.execution_path == "raw_plan" else "off",
        cache_enabled=True,
        cache_path=args.tune_cache_path,
    )
    bl.set_bitlinear_ternary_wgrad_backend(args.wgrad_backend)
    bl.set_bitlinear_int8_backend(args.int8_backend)
    grouped = args.ternary_backend == "dot"
    kmajor = args.ternary_backend in ("kmajor_current", "kmajor_single_dot")
    decode_v2 = args.ternary_backend == "kmajor_single_dot"

    rows = []
    extra_rows = []
    print(f"\n[benchmark] {len(shapes)} unique logical shapes")
    for i, shape in enumerate(shapes, 1):
        timing = _bench_ternary_shape(
            shape,
            dtype=torch.bfloat16,
            grouped_decode=grouped,
            kmajor_layout=kmajor,
            decode_v2=decode_v2,
            warmup=args.warmup,
            iters=args.iters,
        )
        counts_per_step = _phase_counts_per_step(counts, shape, args.grad_accum)
        c_f, c_dx, c_dw = counts_per_step
        f_ms, dx_ms, dw_ms = _phase_step_ms(timing, counts_per_step)
        total_ms = f_ms + dx_ms + dw_ms
        rows.append((total_ms, shape, c_f, timing.fwd, f_ms, timing.dx, dx_ms, timing.dw, dw_ms))
        print(
            f"  [{i:2d}/{len(shapes):2d}] {shape.text():20s} "
            f"ternary fwd={timing.fwd.median:.3f} "
            f"dx={timing.dx.median:.3f} dw={timing.dw.median:.3f} ms"
        )
        if args.compare_int8:
            int8_timing = _bench_int8_shape(
                shape,
                dtype=torch.bfloat16,
                int8_backend=args.int8_backend,
                warmup=args.warmup,
                iters=args.iters,
            )
            int8_f_ms, int8_dx_ms, int8_dw_ms = _phase_step_ms(
                int8_timing, counts_per_step
            )
            extra_f = f_ms - int8_f_ms
            extra_dx = dx_ms - int8_dx_ms
            extra_dw = dw_ms - int8_dw_ms
            extra_total = extra_f + extra_dx + extra_dw
            extra_rows.append(
                (
                    extra_total,
                    shape,
                    counts_per_step,
                    timing,
                    int8_timing,
                    (extra_f, extra_dx, extra_dw),
                )
            )
            print(
                f"                       int8    fwd={int8_timing.fwd.median:.3f} "
                f"dx={int8_timing.dx.median:.3f} dw={int8_timing.dw.median:.3f} ms "
                f"extra/step={extra_total:.1f} ms"
            )

    rows.sort(key=lambda r: r[0], reverse=True)
    print("\n[estimated packed-BitLinear contribution per optimizer step]")
    print(
        "shape                 calls   fwd/kernel  fwd/step   "
        "dx/kernel   dx/step    dw/kernel   dw/step    total"
    )
    print("-" * 116)
    for row in rows[: args.top]:
        total_ms, shape, calls, tf, f_ms, tdx, dx_ms, tdw, dw_ms = row
        print(
            f"{shape.text():20s} {calls:6d} "
            f"{tf.median:10.3f} {f_ms:9.1f} "
            f"{tdx.median:10.3f} {dx_ms:9.1f} "
            f"{tdw.median:10.3f} {dw_ms:9.1f} {total_ms:9.1f}"
        )

    sum_f = sum(r[4] for r in rows)
    sum_dx = sum(r[6] for r in rows)
    sum_dw = sum(r[8] for r in rows)
    grand = sum_f + sum_dx + sum_dw
    print("-" * 116)
    print(
        f"ESTIMATED TOTAL: fwd={sum_f:.1f} ms  dx={sum_dx:.1f} ms  "
        f"dw={sum_dw:.1f} ms  all={grand:.1f} ms/opt-step"
    )
    if grand > 0:
        print(
            f"share: fwd={100*sum_f/grand:.1f}%  dx={100*sum_dx/grand:.1f}%  "
            f"dw={100*sum_dw/grand:.1f}%"
        )
    if args.compare_int8:
        extra_rows.sort(key=lambda r: r[0], reverse=True)
        print("\n[estimated extra cost vs INT8 per optimizer step]")
        print(
            "shape                 calls  "
            "fwd_i8 fwd_ter  dx_i8  dx_ter  dw_i8  dw_ter  "
            "extra_f extra_dx extra_dw    total"
        )
        print("-" * 116)
        for row in extra_rows[: args.top]:
            extra_total, shape, counts_per_step, ter, i8, extras = row
            c_f, c_dx, c_dw = counts_per_step
            extra_f, extra_dx, extra_dw = extras
            calls = c_f if c_f == c_dx == c_dw else f"{c_f}/{c_dx}/{c_dw}"
            print(
                f"{shape.text():20s} {str(calls):>6s} "
                f"{i8.fwd.median:7.3f} {ter.fwd.median:7.3f} "
                f"{i8.dx.median:7.3f} {ter.dx.median:7.3f} "
                f"{i8.dw.median:7.3f} {ter.dw.median:7.3f} "
                f"{extra_f:8.1f} {extra_dx:8.1f} {extra_dw:8.1f} "
                f"{extra_total:8.1f}"
            )
        sum_extra_f = sum(r[5][0] for r in extra_rows)
        sum_extra_dx = sum(r[5][1] for r in extra_rows)
        sum_extra_dw = sum(r[5][2] for r in extra_rows)
        sum_extra = sum_extra_f + sum_extra_dx + sum_extra_dw
        print("-" * 116)
        print(
            f"ESTIMATED EXTRA: fwd={sum_extra_f:.1f} ms  "
            f"dx={sum_extra_dx:.1f} ms  dw={sum_extra_dw:.1f} ms  "
            f"all={sum_extra:.1f} ms/opt-step"
        )
    print(
        "\n[note] Totals are estimates: isolated CUDA-event medians are multiplied "
        "by observed call counts. Counting is eager so Python instrumentation remains "
        "visible; attention/optimizer/compile overlap/non-BitLinear work are excluded."
    )


if __name__ == "__main__":
    main()
