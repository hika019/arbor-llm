"""Synthetic training bench for Arbor BLT config variants.

This keeps data I/O out of the measurement and runs real forward/backward
micro-steps before each optimizer step, so BitLinear weight cache can be tested
under gradient accumulation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "third_party" / "blt") not in sys.path:
    sys.path.insert(0, str(_ROOT / "third_party" / "blt"))

from src.model.arbor_blt import build_arbor_blt  # noqa: E402
from src.model.bitlinear import clear_bitlinear_weight_cache, set_bitlinear_weight_cache  # noqa: E402
from src.train.optim import build_optimizer  # noqa: E402
from src.train.train import apply_compile_settings, apply_speed_settings, resolve_precision  # noqa: E402


def _gb(n: int) -> str:
    return f"{n / 2**30:.2f}GiB"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _variant_cfg(base: dict[str, Any], variant: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("speed", {})
    cfg.setdefault("model", {})

    if variant in ("base", "cache"):
        pass
    elif variant == "ffn4096":
        cfg["model"]["intermediate_size"] = 4096
    elif variant == "local1024":
        cfg["model"].update(
            local_hidden_size=1024,
            local_num_attention_heads=8,
            local_num_key_value_heads=4,
        )
    elif variant == "ffn4096_local1024":
        cfg["model"].update(
            intermediate_size=4096,
            local_hidden_size=1024,
            local_num_attention_heads=8,
            local_num_key_value_heads=4,
        )
    elif variant == "ffn6528_local1024":
        cfg["model"].update(
            intermediate_size=6528,
            local_hidden_size=1024,
            local_num_attention_heads=8,
            local_num_key_value_heads=4,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    cfg["speed"]["bitlinear_weight_cache"] = variant != "base"
    return cfg


def _build_optimizer(model: torch.nn.Module, optim_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    try:
        return build_optimizer(model.parameters(), optim_cfg)
    except Exception as e:
        print(f"[bench] optimizer fallback AdamW fused ({type(e).__name__}: {e})")
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(optim_cfg.get("lr", 1e-4)),
            fused=torch.cuda.is_available(),
        )


def _make_batch(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(4, 260, (batch_size, seq_len), device=device)
    labels = tokens.clone()
    return tokens, labels


def _run_micro(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    grad_accum: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.autocast(device_type=tokens.device.type, dtype=compute_dtype):
        out = model(tokens)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
        ) / grad_accum
    loss.backward()
    return loss.detach()


def run_one(
    *,
    base_cfg: dict[str, Any],
    variant: str,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    cfg = _variant_cfg(base_cfg, variant)
    cfg["model"]["max_position_embeddings"] = seq_len
    speed = cfg.get("speed", {})
    speed["torch_compile"] = bool(speed.get("torch_compile", False))
    apply_speed_settings(speed)
    compute_dtype, _ = resolve_precision(speed.get("precision", "bf16"))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    device = torch.device("cuda")
    print(
        f"[bench] variant={variant} batch={batch_size} seq={seq_len} "
        f"grad_accum={grad_accum} cache={speed.get('bitlinear_weight_cache', False)} "
        f"hidden={cfg['model']['hidden_size']} inter={cfg['model']['intermediate_size']} "
        f"local={cfg['model'].get('local_hidden_size', cfg['model']['hidden_size'])}"
    )
    model = build_arbor_blt(cfg["model"]).to(device=device, dtype=compute_dtype)
    cache_enabled = bool(speed.get("bitlinear_weight_cache", False))
    if cache_enabled:
        print(f"[bench] cache_layers={set_bitlinear_weight_cache(model, True)}")
    model.train()
    model = torch.compile(model, mode=speed.get("compile_mode", "default"), dynamic=True) if speed.get("torch_compile", False) else model
    optimizer = _build_optimizer(model, cfg["optim"])
    optimizer.zero_grad(set_to_none=True)

    tokens, labels = _make_batch(batch_size, seq_len, device)

    def step() -> torch.Tensor:
        loss = None
        for _ in range(grad_accum):
            loss = _run_micro(model, tokens, labels, grad_accum, compute_dtype)
        optimizer.step()
        if cache_enabled:
            clear_bitlinear_weight_cache(model)
        optimizer.zero_grad(set_to_none=True)
        assert loss is not None
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_loss = None
    for _ in range(iters):
        last_loss = step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    tokens_total = batch_size * seq_len * grad_accum * iters
    peak = torch.cuda.max_memory_allocated()

    del model, optimizer, tokens, labels
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "variant": variant,
        "seconds": seconds,
        "tok_s": tokens_total / seconds,
        "peak": peak,
        "loss": float(last_loss.cpu()) if last_loss is not None else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/arbor_1b.yaml"))
    p.add_argument("--variants", nargs="+", default=["base", "cache", "local1024", "ffn4096_local1024"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    base_cfg = _load_config(args.config)
    batch_size = args.batch_size or int(base_cfg["speed"].get("micro_batch_size", 2))
    seq_len = args.seq_len or int(base_cfg["data"].get("context_length", 2048))
    grad_accum = args.grad_accum or int(base_cfg["speed"].get("grad_accum_steps", 1))

    print("variant,batch_size,seq_len,grad_accum,warmup,iters,seconds,tok_s,peak_mem,loss")
    for variant in args.variants:
        try:
            result = run_one(
                base_cfg=base_cfg,
                variant=variant,
                batch_size=batch_size,
                seq_len=seq_len,
                grad_accum=grad_accum,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(
                f"{variant},{batch_size},{seq_len},{grad_accum},{args.warmup},{args.iters},"
                f"{result['seconds']:.3f},{result['tok_s']:.1f},{_gb(result['peak'])},"
                f"{result['loss']:.6f}",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"{variant},{batch_size},{seq_len},{grad_accum},{args.warmup},{args.iters},OOM,0,-,{e}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{variant},{batch_size},{seq_len},{grad_accum},{args.warmup},{args.iters},ERR,0,-,{type(e).__name__}: {e}")
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
