"""Benchmark torch.compile scopes for Arbor BLT training.

This compares eager, BLT global_transformer-only compile, and full-model compile.
Each measured step generates a fresh random byte batch so BLT patch lengths vary
instead of accidentally benchmarking a single stable patch shape.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "third_party" / "blt") not in sys.path:
    sys.path.insert(0, str(_ROOT / "third_party" / "blt"))

from src.model.arbor_blt import build_arbor_blt  # noqa: E402
from src.train.train import apply_compile_settings  # noqa: E402


def _gb(n: int) -> str:
    return f"{n / 2**30:.2f}GiB"


def _model_cfg(size: str) -> dict[str, Any]:
    if size == "smoke":
        hidden, inter, layers, heads, seq = 128, 256, 2, 4, 128
    elif size == "mini":
        hidden, inter, layers, heads, seq = 256, 512, 4, 8, 256
    else:
        raise ValueError(f"unknown size: {size}")

    return {
        "backend": "blt",
        "vocab_size": 260,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": layers,
        "num_local_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "max_position_embeddings": seq,
        "patch_size": 4,
        "patching_mode": "space",
        "gradient_checkpointing": False,
        "bitlinear_in_global": True,
        "fp_in_local": True,
    }


def _speed_cfg(scope: str, mode: str, dynamic: bool) -> dict[str, Any]:
    return {
        "torch_compile": scope != "eager",
        "compile_scope": "off" if scope == "eager" else scope,
        "compile_mode": mode,
        "compile_dynamic": dynamic,
        "dynamo_cache_size_limit": 64,
    }


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_size: int, seq_len: int):
    device = next(model.parameters()).device
    tokens = torch.randint(4, 260, (batch_size, seq_len), device=device)
    labels = tokens.clone()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(tokens)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten()
        )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def run_one(
    *,
    scope: str,
    size: str,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
    mode: str,
    dynamic: bool,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _model_cfg(size)
    cfg["max_position_embeddings"] = seq_len
    model = build_arbor_blt(cfg).to(device=device, dtype=torch.bfloat16)
    model = apply_compile_settings(model, _speed_cfg(scope, mode, dynamic))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        fused=(device.type == "cuda"),
    )
    optimizer.zero_grad(set_to_none=True)

    compile_t0 = time.perf_counter()
    for _ in range(warmup):
        _step(model, optimizer, batch_size, seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - compile_t0

    t0 = time.perf_counter()
    last_loss = None
    for _ in range(iters):
        last_loss = _step(model, optimizer, batch_size, seq_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    tokens = batch_size * seq_len * iters
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "scope": scope,
        "warmup_seconds": warmup_seconds,
        "seconds": seconds,
        "tok_s": tokens / seconds,
        "peak": peak,
        "loss": float(last_loss.cpu()) if last_loss is not None else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("smoke", "mini"), default="mini")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", default="default")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scopes", nargs="+", default=["eager", "global", "model"])
    args = parser.parse_args()

    print(
        "scope,size,batch_size,seq_len,warmup,iters,mode,dynamic,"
        "warmup_seconds,seconds,tok_s,peak_mem,loss"
    )
    for scope in args.scopes:
        result = run_one(
            scope=scope,
            size=args.size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.warmup,
            iters=args.iters,
            mode=args.mode,
            dynamic=args.dynamic,
        )
        print(
            f"{result['scope']},{args.size},{args.batch_size},{args.seq_len},"
            f"{args.warmup},{args.iters},{args.mode},{args.dynamic},"
            f"{result['warmup_seconds']:.3f},{result['seconds']:.3f},"
            f"{result['tok_s']:.1f},{_gb(result['peak'])},{result['loss']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
