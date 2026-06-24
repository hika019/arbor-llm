#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer.generate import BYTE_OFFSET, VOCAB_SIZE, load_checkpoint_config, load_inference_model, resolve_checkpoint


def ids(s: str) -> list[int]:
    return [b + BYTE_OFFSET for b in s.encode("utf-8")]


@torch.inference_mode()
def logits_for(model, text: str, device, dtype):
    x = torch.tensor([ids(text)], dtype=torch.long, device=device)
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.no_grad()
    with ctx:
        logits = model(x).logits[0, -1].float()
    logits = logits[:VOCAB_SIZE].clone()
    logits[:BYTE_OFFSET] = float("-inf")
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("./checkpoints"))
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--debug-context", action="store_true")
    ap.add_argument("--no-freeze-bitlinear", action="store_true")
    args = ap.parse_args()

    if args.debug_context:
        os.environ["ARBOR_DEBUG_CONTEXT"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    ckpt = resolve_checkpoint(args.ckpt, args.ckpt_dir)
    cfg = load_checkpoint_config(ckpt)
    model = load_inference_model(
        ckpt,
        cfg,
        device=device,
        dtype=dtype,
        freeze_bitlinear=not args.no_freeze_bitlinear,
    )
    model.eval()

    la = logits_for(model, args.a, device, dtype)
    lb = logits_for(model, args.b, device, dtype)

    pa = torch.softmax(la, dim=-1)
    pb = torch.softmax(lb, dim=-1)

    l1 = torch.sum(torch.abs(pa - pb)).item()
    kl_ab = torch.sum(pa * (torch.log(pa.clamp_min(1e-30)) - torch.log(pb.clamp_min(1e-30)))).item()
    kl_ba = torch.sum(pb * (torch.log(pb.clamp_min(1e-30)) - torch.log(pa.clamp_min(1e-30)))).item()

    print(f"L1_prob={l1:.6f}")
    print(f"KL(a||b)={kl_ab:.6f}")
    print(f"KL(b||a)={kl_ba:.6f}")

    for name, p in [("A", pa), ("B", pb)]:
        vals, idx = torch.topk(p, 20)
        print(f"\n{name} top bytes:")
        for prob, tok in zip(vals.cpu().tolist(), idx.cpu().tolist()):
            b = tok - BYTE_OFFSET
            print(f"  {b:02x} {bytes([b]).decode('utf-8', errors='backslashreplace')!r} {prob:.6f}")


if __name__ == "__main__":
    main()
