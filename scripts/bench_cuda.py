"""Arbor 学習スループットの合成データ・マイクロベンチ (CUDA 実機向け).

本走 config のモデルを合成入力で fwd+bwd+opt し、per-step 時間と bytes/s を測る。
HF streaming を挟まないので純粋な計算速度を比較できる。global_attn_impl や
compile_mode, micro_batch, seq を切り替えて A/B するための最小ツール。

例:
  # flex vs sdpa (CUDA)
  python -m scripts.bench_cuda --global-attn-impl sdpa
  python -m scripts.bench_cuda --global-attn-impl flex
  # compile mode
  python -m scripts.bench_cuda --compile-mode max-autotune
  # 実効バッチを grad_accum で作る
  python -m scripts.bench_cuda --micro-batch 4 --grad-accum 16
"""
from __future__ import annotations

import argparse
import time

import torch
import yaml

from src.model.arbor import build_arbor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arbor.yaml")
    ap.add_argument("--seq", type=int, default=None, help="既定は config の max_bytes")
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--global-attn-impl", default=None, choices=["sdpa", "flex", "auto"])
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument(
        "--bitlinear-fp8",
        default=None,
        choices=["off", "bwd", "full", "auto"],
        help="既定は config speed.bitlinear_fp8",
    )
    ap.add_argument(
        "--weight-cache",
        default=None,
        choices=["off", "fused", "full", "auto"],
        help="既定は config speed.bitnet_weight_cache",
    )
    ap.add_argument("--weight-cache-gib", type=float, default=None)
    ap.add_argument("--fwd-only", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    cfg = yaml.safe_load(open(args.config))
    mcfg = dict(cfg["model"])
    if args.global_attn_impl is not None:
        mcfg["global_attn_impl"] = args.global_attn_impl
    if (
        mcfg.get("global_attn_impl", "sdpa") == "auto"
        and not (device.type == "cuda" and args.compile)
    ):
        mcfg["global_attn_impl"] = "sdpa"
    seq = args.seq or int(mcfg.get("max_bytes", 2048))
    mcfg["max_bytes"] = seq
    if device.type == "mps" and mcfg.get("activation_precision") == "bf8":
        mcfg["activation_precision"] = "bf16"

    model = build_arbor(mcfg).to(device=device, dtype=dtype)
    model.train()
    speed_cfg = cfg.get("speed", {})
    from src.model.bitlinear import (
        configure_bitlinear_training_cache,
        refresh_bitlinear_training_cache,
        set_bitlinear_fp8_mode,
    )

    cache_mode = args.weight_cache
    if cache_mode is None:
        cache_mode = speed_cfg.get("bitnet_weight_cache", "auto")
    cache_gib = (
        args.weight_cache_gib
        if args.weight_cache_gib is not None
        else speed_cfg.get("bitnet_weight_cache_gib", 1.25)
    )
    cache_info = configure_bitlinear_training_cache(
        model,
        enabled=cache_mode,
        grad_accum_steps=args.grad_accum,
        max_cache_gib=cache_gib,
        min_numel=int(speed_cfg.get("bitnet_weight_cache_min_numel", 65536)),
    )
    fp8_mode = args.bitlinear_fp8
    if fp8_mode is None:
        fp8_mode = speed_cfg.get("bitlinear_fp8", "off")
    if fp8_mode in (None, False):
        fp8_mode = "off"
    fp8_info = set_bitlinear_fp8_mode(model, str(fp8_mode))
    print(f"[bench] device={device} dtype={args.dtype} seq={seq} "
          f"micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
          f"global_attn_impl={mcfg.get('global_attn_impl','sdpa')} "
          f"compile={args.compile}({args.compile_mode})")
    print(
        "[bench] "
        f"weight_cache={cache_info['mode']} cached_layers={cache_info['cached_layers']} "
        f"cache={cache_info['cache_gib']:.2f}GiB "
        f"fp8={fp8_info['mode']} (requested={fp8_info['requested']})"
    )

    resolved_impl = mcfg.get("global_attn_impl", "sdpa")
    if resolved_impl == "auto":
        resolved_impl = "flex" if device.type == "cuda" else "sdpa"
    if resolved_impl == "flex" and device.type != "cuda" and not args.fwd_only:
        raise SystemExit(
            "[bench] global_attn_impl=flex の backward は CUDA 専用です "
            "(CPU/MPS では --fwd-only か sdpa を使ってください)"
        )

    run = model
    if args.compile and device.type == "cuda":
        run = torch.compile(model, mode=args.compile_mode)

    # fused AdamW (bench 用。学習の実 optimizer とは別。opt はボトルネックでない)
    opt = torch.optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1,
        fused=(device.type == "cuda"),
    )

    def rand_batch() -> torch.Tensor:
        return torch.randint(4, 260, (args.micro_batch, seq), device=device)

    def one_opt_step() -> float:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.grad_accum):
            x = rand_batch()
            logits = run(x).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1)
            ) / args.grad_accum
            if not args.fwd_only:
                loss.backward()
            total += float(loss.detach())
        if not args.fwd_only:
            opt.step()
            refresh_bitlinear_training_cache(model)
        return total

    for _ in range(args.warmup):
        one_opt_step()
    _sync(device)

    bytes_per_step = args.micro_batch * seq * args.grad_accum
    times = []
    for i in range(args.iters):
        _sync(device)
        t0 = time.perf_counter()
        loss = one_opt_step()
        _sync(device)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  iter {i}: {dt*1000:.0f} ms  loss={loss:.4f}")

    times.sort()
    med = times[len(times) // 2]
    print(f"[bench] per opt-step median={med*1000:.0f} ms  "
          f"{bytes_per_step/med:.0f} bytes/s  "
          f"=> 1000 steps ~= {med*1000/3600:.2f} h")
    if device.type == "cuda":
        print(f"[bench] peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
