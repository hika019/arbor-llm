"""Small fixed text probes for generation-quality diagnostics."""
from __future__ import annotations

import math
from typing import Any

import torch

from src.infer.generate import BYTE_OFFSET


def byte_kind(b: int) -> str:
    if b < 0x80:
        return "ascii"
    if 0x80 <= b <= 0xBF:
        return "utf8_cont"
    if 0xC0 <= b <= 0xF7:
        return "utf8_lead"
    return "other"


def ids_from_text(text: str) -> list[int]:
    return [b + BYTE_OFFSET for b in text.encode("utf-8")]


@torch.inference_mode()
def score_completion(
    model: torch.nn.Module,
    prompt: str,
    target: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Return byte-level NLL/BPB for ``target`` conditioned on ``prompt``."""
    prompt_ids = ids_from_text(prompt)
    target_ids = ids_from_text(target)
    if not prompt_ids:
        raise ValueError("prompt must not be empty")
    if not target_ids:
        raise ValueError("target must not be empty")

    full = prompt_ids + target_ids
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor(full[1:], dtype=torch.long, device=device)

    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()
    with ctx:
        logits = model(x).logits[0].float()

    valid_start = len(prompt_ids) - 1
    valid_logits = logits[valid_start : valid_start + len(target_ids)]
    valid_labels = labels[valid_start : valid_start + len(target_ids)]
    losses = torch.nn.functional.cross_entropy(
        valid_logits,
        valid_labels,
        reduction="none",
    )

    mean_nll = float(losses.mean().cpu())
    total_nll = float(losses.sum().cpu())
    bpb = mean_nll / math.log(2.0)

    by_kind: dict[str, list[float]] = {}
    for tok, loss in zip(valid_labels.detach().cpu().tolist(), losses.detach().cpu().tolist()):
        by_kind.setdefault(byte_kind(tok - BYTE_OFFSET), []).append(float(loss))

    by_kind_out = {}
    for kind, vals in by_kind.items():
        mean = sum(vals) / len(vals)
        by_kind_out[kind] = {
            "count": len(vals),
            "nll": mean,
            "bpb": mean / math.log(2.0),
        }

    return {
        "text": target,
        "bytes": len(target_ids),
        "nll": mean_nll,
        "total_nll": total_nll,
        "bpb": bpb,
        "ppl_per_byte": math.exp(mean_nll),
        "by_byte_kind": by_kind_out,
    }


@torch.inference_mode()
def run_text_probes(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, float]]:
    """Score config-defined good/bad target probes and return compact metrics."""
    was_training = model.training
    model.eval()
    out: dict[str, dict[str, float]] = {}
    try:
        for item in items:
            name = str(item["name"])
            prompt = str(item["prompt"])
            target = str(item["target"])
            bad_targets = [str(value) for value in item.get("bad_targets", [])]
            good = score_completion(model, prompt, target, device=device, dtype=dtype)
            bad = [
                score_completion(model, prompt, bad_target, device=device, dtype=dtype)
                for bad_target in bad_targets
            ]
            bad_min_bpb = min((value["bpb"] for value in bad), default=float("nan"))
            margin = bad_min_bpb - good["bpb"] if bad else float("nan")
            out[name] = {
                "good_bpb": round(float(good["bpb"]), 6),
                "bad_min_bpb": round(float(bad_min_bpb), 6),
                "margin": round(float(margin), 6),
            }
        return out
    finally:
        if was_training:
            model.train()
