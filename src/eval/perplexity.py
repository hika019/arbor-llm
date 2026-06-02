"""バイトレベル perplexity 評価."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def byte_perplexity(model: torch.nn.Module, loader: Iterable[dict[str, torch.Tensor]],
                    device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inputs)
        loss = F.cross_entropy(
            out.logits.flatten(0, 1), labels.flatten(),
            ignore_index=-100, reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()
    return math.exp(total_loss / max(1, total_tokens))
