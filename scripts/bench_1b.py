"""1B BLT × BitLinear ベンチ. VRAM 占有と tok/s を測る.

設定:
  - hidden=1536, 24 層 Global, 1 層 Local Enc/Dec, 16 heads
  - vocab=260, patch_size=4, context=2048 (= 512 patch)
  - BF16 mixed precision, 8bit Adam, BitLinear ON
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "third_party" / "blt"))
os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")


def _gb(x: int) -> str:
    return f"{x / 2**30:5.2f} GB"


def build_1b_blt(attn_impl: str = "sdpa", grad_ckpt: bool = False) -> torch.nn.Module:
    from bytelatent.model.blt import ByteLatentTransformer, ByteLatentTransformerArgs
    from src.model.global_latent import swap_linear_to_bitlinear

    h = 1536
    args = ByteLatentTransformerArgs(
        vocab_size=260,
        dim=h, dim_global=h, dim_token=h,
        dim_local_encoder=h, dim_local_decoder=h,
        n_layers=24, n_layers_global=24,
        n_layers_local_encoder=1, n_layers_local_decoder=1,
        n_heads=16, n_heads_global=16,
        n_heads_local_encoder=16, n_heads_local_decoder=16,
        patch_size=4, patching_mode="space",
        max_encoder_seq_length=2048, max_seqlen=2048, max_length=512,
        use_local_encoder_transformer=True,
        cross_attn_encoder=False, cross_attn_decoder=False,
        cross_attn_all_layers_decoder=False, cross_attn_all_layers_encoder=False,
        cross_attn_init_by_pooling=True, cross_attn_use_flex_attention=False,
        attn_impl=attn_impl, attn_bias_type="causal",
        non_linearity="swiglu", use_rope=True,
        pad_to_max_length=False, downsampling_by_pooling="max",
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=50002,
        encoder_hash_byte_group_nb_functions=3,
        patch_in_forward=True,
        recompute_attn=grad_ckpt,
        recompute_fc1_out=grad_ckpt,
        recompute_fc3_out=grad_ckpt,
    )
    model = ByteLatentTransformer(args)
    n = swap_linear_to_bitlinear(model.global_transformer,
                                 skip_names=("output", "embed", "tok_embeddings"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BitLinear 置換: {n} 層, params={n_params/1e6:.1f}M")
    return model


def bench_step(model, optimizer, batch_size: int, seq_len: int,
               warmup: int = 3, iters: int = 10) -> tuple[float, float]:
    device = next(model.parameters()).device
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tokens = torch.randint(4, 256, (batch_size, seq_len), device=device)
    labels = tokens.clone()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(tokens)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), labels.flatten()
            )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    tps = batch_size * seq_len * iters / dt
    return tps, peak


def run_one(name: str, *, attn_impl: str, grad_ckpt: bool,
            compile_mode: str | None, batch_size: int, seq_len: int,
            grad_accum: int = 1, verbose_err: bool = False) -> dict:
    eff_tok = batch_size * grad_accum * seq_len
    print(f"\n=== {name} ===")
    print(f"  attn={attn_impl} grad_ckpt={grad_ckpt} compile={compile_mode} "
          f"bs={batch_size} (accum={grad_accum}, eff_tok/step={eff_tok}) seq={seq_len}")
    result = {"name": name, "tps": None, "peak": None, "err": None}
    try:
        gc.collect(); torch.cuda.empty_cache()
        model = build_1b_blt(attn_impl=attn_impl, grad_ckpt=grad_ckpt).cuda().to(torch.bfloat16)
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        if compile_mode:
            model = torch.compile(model, mode=compile_mode)
        tps, peak = bench_step(model, optimizer, batch_size, seq_len)
        print(f"  -> tok/s = {tps:>9.0f}   VRAM peak = {_gb(peak)}")
        result["tps"] = tps
        result["peak"] = peak
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM: {e}")
        result["err"] = "OOM"
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {e}")
        if verbose_err:
            traceback.print_exc()
        result["err"] = f"{type(e).__name__}: {e}"
    finally:
        if "model" in locals(): del model
        if "optimizer" in locals(): del optimizer
        gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--verbose-err", action="store_true")
    args = p.parse_args()
    print(f"== 1B BLT × BitLinear bench, RTX 4090, seq={args.seq} ==")

    results = []
    # bs スケーリング (sdpa + grad_ckpt)
    for bs in (4, 8, 12):
        r = run_one(f"sdpa + grad_ckpt + bs={bs}",
                    attn_impl="sdpa", grad_ckpt=True, compile_mode=None,
                    batch_size=bs, seq_len=args.seq, verbose_err=args.verbose_err)
        results.append(r)
    # bs=8 + compile を試す (頂上アタック)
    results.append(run_one("sdpa + grad_ckpt + bs=8 + compile",
                           attn_impl="sdpa", grad_ckpt=True, compile_mode="default",
                           batch_size=8, seq_len=args.seq, verbose_err=args.verbose_err))
    results.append(run_one("sdpa + grad_ckpt + bs=8 + compile(max-autotune)",
                           attn_impl="sdpa", grad_ckpt=True, compile_mode="max-autotune",
                           batch_size=8, seq_len=args.seq, verbose_err=args.verbose_err))

    print("\n== サマリ ==")
    print(f"{'name':50s} {'tok/s':>9s} {'VRAM':>9s}")
    for r in results:
        if r["tps"] is None:
            print(f"{r['name'][:50]:50s} {'-':>9s} {'-':>9s}  ({r['err']})")
        else:
            print(f"{r['name'][:50]:50s} {r['tps']:>9.0f} {_gb(r['peak'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
