#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer.generate import (  # noqa: E402
    BYTE_OFFSET,
    VOCAB_SIZE,
    generate_text,
    load_checkpoint_config,
    load_inference_model,
    resolve_checkpoint,
)
from src.eval.probes import byte_kind, ids_from_text, score_completion  # noqa: E402


@torch.inference_mode()
def next_byte_probe(
    model: torch.nn.Module,
    prompt: str,
    expected_text: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    top_k: int = 20,
) -> dict:
    prompt_ids = ids_from_text(prompt)
    expected_bytes = expected_text.encode("utf-8")
    if not expected_bytes:
        return {}

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()

    with ctx:
        logits = model(x).logits[0, -1].float()

    logits = logits[:VOCAB_SIZE].clone()
    logits[:BYTE_OFFSET] = float("-inf")
    probs = torch.softmax(logits, dim=-1)

    expected_byte = expected_bytes[0]
    expected_id = expected_byte + BYTE_OFFSET
    expected_logit = logits[expected_id]
    rank = int((logits > expected_logit).sum().item()) + 1

    vals, idx = torch.topk(probs, k=min(top_k, VOCAB_SIZE - BYTE_OFFSET))
    top = []
    for p, tok in zip(vals.detach().cpu().tolist(), idx.detach().cpu().tolist()):
        b = tok - BYTE_OFFSET
        top.append({
            "token_id": tok,
            "byte_hex": f"0x{b:02x}",
            "byte_kind": byte_kind(b),
            "as_text": bytes([b]).decode("utf-8", errors="backslashreplace"),
            "prob": p,
        })

    return {
        "expected_first_byte_hex": f"0x{expected_byte:02x}",
        "expected_first_byte_kind": byte_kind(expected_byte),
        "expected_first_byte_rank": rank,
        "expected_first_byte_prob": float(probs[expected_id].detach().cpu()),
        "top": top,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("./checkpoints"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--bad-target", action="append", default=[])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy-bytes", type=int, default=120)
    ap.add_argument("--no-freeze-bitlinear", action="store_true")
    ap.add_argument("--no-utf8-mask", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    ckpt_dir = resolve_checkpoint(args.ckpt, args.ckpt_dir)
    cfg = load_checkpoint_config(ckpt_dir, args.config)

    print(f"[probe] checkpoint={ckpt_dir}", file=sys.stderr)
    model = load_inference_model(
        ckpt_dir,
        cfg,
        device=device,
        dtype=dtype,
        freeze_bitlinear=not args.no_freeze_bitlinear,
    )
    model.eval()

    good = score_completion(model, args.prompt, args.target, device=device, dtype=dtype)
    bad = [
        score_completion(model, args.prompt, t, device=device, dtype=dtype)
        for t in args.bad_target
    ]
    next_probe = next_byte_probe(
        model,
        args.prompt,
        args.target,
        device=device,
        dtype=dtype,
        top_k=args.top_k,
    )

    greedy = generate_text(
        model,
        args.prompt,
        max_new_bytes=args.greedy_bytes,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=42,
        max_context=int(
            cfg.get("model", {}).get(
                "max_bytes",
                cfg.get("data", {}).get("context_length", 2048),
            )
        ),
        utf8_mask=not args.no_utf8_mask,
    )

    result = {
        "prompt": args.prompt,
        "good_target": good,
        "bad_targets": bad,
        "next_byte": next_probe,
        "greedy_completion": greedy,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n[summary]")
    print(f"good: bpb={good['bpb']:.4f} nll={good['nll']:.4f} text={good['text']!r}")
    for i, b in enumerate(bad):
        delta = b["bpb"] - good["bpb"]
        sign = "+" if delta >= 0 else ""
        print(
            f"bad[{i}]: bpb={b['bpb']:.4f} nll={b['nll']:.4f} "
            f"delta_bpb={sign}{delta:.4f} text={b['text']!r}"
        )
    if next_probe:
        print(
            "next first byte: "
            f"rank={next_probe['expected_first_byte_rank']} "
            f"prob={next_probe['expected_first_byte_prob']:.6g} "
            f"byte={next_probe['expected_first_byte_hex']}"
        )
    print(f"greedy: {greedy!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
