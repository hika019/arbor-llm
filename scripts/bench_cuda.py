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
import math
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
    ap.add_argument("--global-attn-impl", default=None, choices=["sdpa", "flex"])
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument(
        "--bitlinear-fp8",
        default=None,
        choices=["off", "bwd", "full", "int8", "ternary"],
        help="既定は config speed.bitlinear_fp8",
    )
    ap.add_argument(
        "--int8-backend",
        default=None,
        choices=["auto", "int_mm", "triton"],
        help="native INT8 forward backend。既定は config speed.bitlinear_int8_backend",
    )
    ap.add_argument(
        "--weight-cache",
        default=None,
        choices=["off", "fused", "full", "auto"],
        help="既定は config speed.bitnet_weight_cache",
    )
    ap.add_argument("--weight-cache-gib", type=float, default=None)
    ap.add_argument(
        "--production-optimizer",
        action="store_true",
        help="configのoptimizer/scheduler/grad clipを使う",
    )
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument(
        "--check-params-every",
        type=int,
        default=0,
        help="N stepごとに全parameterのfiniteを検査。0で無効",
    )
    ap.add_argument("--fwd-only", action="store_true")
    args = ap.parse_args()
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")
    if args.check_params_every < 0:
        raise SystemExit("--check-params-every must be non-negative")

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
    seq = args.seq or int(mcfg.get("max_bytes", 2048))
    mcfg["max_bytes"] = seq
    resolved_impl = mcfg.get("global_attn_impl", "sdpa")
    if resolved_impl == "flex" and device.type != "cuda":
        raise SystemExit(
            "[bench] global_attn_impl=flex はCUDA専用です。"
            "暗黙フォールバックは行いません"
        )
    if resolved_impl == "flex" and not args.compile:
        raise SystemExit(
            "[bench] global_attn_impl=flex には --compile が必要です。"
            "暗黙フォールバックは行いません"
        )

    model = build_arbor(mcfg).to(device=device, dtype=dtype)
    model.train()
    speed_cfg = cfg.get("speed", {})
    from src.model.bitlinear import (
        configure_bitlinear_training_cache,
        install_arbor_projection_fusions,
        refresh_bitlinear_training_cache,
        set_bitlinear_fp8_mode,
        set_bitlinear_int8_backend,
    )

    int8_backend = set_bitlinear_int8_backend(
        args.int8_backend
        or str(speed_cfg.get("bitlinear_int8_backend", "auto"))
    )
    fp8_mode = args.bitlinear_fp8
    if fp8_mode is None:
        fp8_mode = speed_cfg.get("bitlinear_fp8", "off")
    if fp8_mode in (None, False):
        fp8_mode = "off"
    install_arbor_projection_fusions(model)
    fp8_info = set_bitlinear_fp8_mode(model, str(fp8_mode))
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
    print(f"[bench] device={device} dtype={args.dtype} seq={seq} "
          f"micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
          f"global_attn_impl={mcfg.get('global_attn_impl','sdpa')} "
          f"compile={args.compile}({args.compile_mode})")
    print(
        "[bench] "
        f"weight_cache={cache_info['mode']} cached_layers={cache_info['cached_layers']} "
        f"cache={cache_info['cache_gib']:.2f}GiB "
        f"format={cache_info['cache_format']} "
        f"fp8={fp8_info['mode']} int8_backend={int8_backend}"
    )

    run = model
    if args.compile and device.type == "cuda":
        run = torch.compile(model, mode=args.compile_mode)

    scheduler = None
    if args.production_optimizer:
        from src.train.optim import build_optimizer, build_scheduler

        opt = build_optimizer(model.parameters(), cfg["optim"])
        scheduler = build_scheduler(opt, cfg["optim"])
        print(
            f"[bench] optimizer=config:{type(opt).__name__} "
            f"scheduler={type(scheduler).__name__}"
        )
    else:
        # 純粋なmodel計算比較用。実optimizerを検証する時は --production-optimizer。
        opt = torch.optim.AdamW(
            model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1,
            fused=(device.type == "cuda"),
        )
        print("[bench] optimizer=fused AdamW (benchmark)")

    def rand_batch() -> torch.Tensor:
        return torch.randint(4, 260, (args.micro_batch, seq), device=device)

    def one_opt_step() -> tuple[torch.Tensor, torch.Tensor | None]:
        opt.zero_grad(set_to_none=True)
        total: torch.Tensor | None = None
        grad_norm: torch.Tensor | None = None
        for _ in range(args.grad_accum):
            x = rand_batch()
            logits = run(x).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1)
            ) / args.grad_accum
            if not args.fwd_only:
                loss.backward()
            detached = loss.detach()
            total = detached if total is None else total + detached
        if not args.fwd_only:
            if args.production_optimizer and cfg["optim"].get("grad_clip"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["optim"]["grad_clip"], foreach=True
                )
            opt.step()
            refresh_bitlinear_training_cache(model)
            if scheduler is not None:
                scheduler.step()
        assert total is not None
        return total, grad_norm

    for _ in range(args.warmup):
        one_opt_step()
    _sync(device)

    bytes_per_step = args.micro_batch * seq * args.grad_accum
    times = []
    losses = []
    for i in range(args.iters):
        _sync(device)
        t0 = time.perf_counter()
        loss_tensor, grad_norm_tensor = one_opt_step()
        _sync(device)
        dt = time.perf_counter() - t0
        loss = float(loss_tensor)
        grad_norm = (
            None if grad_norm_tensor is None else float(grad_norm_tensor)
        )
        if not math.isfinite(loss):
            bad_param = next(
                (
                    name for name, p in model.named_parameters()
                    if not bool(torch.isfinite(p).all().item())
                ),
                None,
            )
            raise RuntimeError(
                f"non-finite loss at step={i + 1}: {loss}; bad_param={bad_param}"
            )
        if grad_norm is not None and not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step={i + 1}: {grad_norm}")
        if args.check_params_every and (i + 1) % args.check_params_every == 0:
            finite = torch.stack(
                [torch.isfinite(p).all() for p in model.parameters()]
            ).all()
            if not bool(finite.item()):
                raise RuntimeError(f"non-finite parameter at iter={i}")
        times.append(dt)
        losses.append(loss)
        if i == 0 or (i + 1) % args.log_every == 0 or i + 1 == args.iters:
            lr = float(opt.param_groups[0]["lr"])
            grad_text = "" if grad_norm is None else f" grad_norm={grad_norm:.4f}"
            print(
                f"  iter {i + 1}/{args.iters}: {dt*1000:.0f} ms "
                f"loss={loss:.4f}{grad_text} lr={lr:.3e}"
            )

    times.sort()
    med = times[len(times) // 2]
    print(f"[bench] per opt-step median={med*1000:.0f} ms  "
          f"{bytes_per_step/med:.0f} bytes/s  "
          f"=> 1000 steps ~= {med*1000/3600:.2f} h")
    print(
        f"[bench] loss first={losses[0]:.4f} last={losses[-1]:.4f} "
        f"min={min(losses):.4f} max={max(losses):.4f} finite=OK"
    )
    if device.type == "cuda":
        print(f"[bench] peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
