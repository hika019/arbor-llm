#!/usr/bin/env python3
"""Entropy patching の threshold と max_patches を実データで校正する.

凍結済み ByteLM を学習データ混合の小サンプルに通し、Arbor 本体と同じ
min/max patch length ルールを再現して、目標 bytes/patch に近い
entropy_threshold を二分探索する。FastBitNet CUDA 拡張には依存しない。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file as safe_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target-bytes-per-patch", type=float, default=5.0)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--binary-search-steps", type=int, default=28)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--write", action="store_true", help="config を直接更新する")
    return parser.parse_args()


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    z = logits.float()
    logp = F.log_softmax(z, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def resolve_weights(checkpoint: Path) -> Path:
    path = checkpoint.expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"model.safetensors not found: {path}")
    return path


def load_entropy_model(model_cfg: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from src.model.arbor import ByteLM

    entropy_cfg = dict(model_cfg["entropy_model"])
    entropy_cfg.setdefault("max_bytes", model_cfg["max_bytes"])
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = ByteLM(entropy_cfg).to(device=device, dtype=dtype).eval()
    state = safe_load(str(resolve_weights(checkpoint)), device="cpu")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.removeprefix("_orig_mod.")
        if key.startswith("entropy_model."):
            key = key.removeprefix("entropy_model.")
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)
    return model


def patch_count(entropy: np.ndarray, threshold: float, min_len: int, max_len: int) -> int:
    t = int(entropy.shape[0])
    raw = np.zeros(t, dtype=np.bool_)
    if t > 1:
        raw[1:] = entropy[:-1] > threshold
    count = 0
    index = 0
    while index < t:
        count += 1
        lo = index + min_len
        if lo >= t:
            break
        hi = min(index + max_len, t)
        candidates = np.flatnonzero(raw[lo:hi])
        index = lo + int(candidates[0]) if candidates.size else hi
    return count


def evaluate(
    samples: list[np.ndarray], threshold: float, min_len: int, max_len: int
) -> tuple[float, np.ndarray]:
    counts = np.asarray(
        [patch_count(sample, threshold, min_len, max_len) for sample in samples],
        dtype=np.int64,
    )
    total_bytes = sum(int(sample.shape[0]) for sample in samples)
    return total_bytes / int(counts.sum()), counts


def recommend_capacity(counts: np.ndarray, context: int, min_len: int) -> int:
    worst_case = math.ceil(context / min_len)
    observed = float(counts.max())
    p999 = float(np.quantile(counts, 0.999))
    desired = max(observed * 1.125, p999 * 1.15)
    rounded = int(math.ceil(desired / 128.0) * 128)
    return max(1, min(worst_case, rounded))


def update_config(path: Path, threshold: float, max_patches: int) -> None:
    text = path.read_text(encoding="utf-8")
    threshold_pattern = r"(?m)^(\s*entropy_threshold:\s*)[^\s#]+(.*)$"
    text, count = re.subn(
        threshold_pattern,
        lambda match: f"{match.group(1)}{threshold:.6f}{match.group(2)}",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("entropy_threshold line not found")
    max_pattern = r"(?m)^(\s*max_patches:\s*)[^\s#]+(.*)$"
    if re.search(max_pattern, text):
        text = re.sub(
            max_pattern,
            lambda match: f"{match.group(1)}{max_patches}{match.group(2)}",
            text,
            count=1,
        )
    else:
        anchor = re.search(r"(?m)^(\s*max_patch_len:.*\n)", text)
        if anchor is None:
            raise RuntimeError("max_patch_len line not found")
        indent = re.match(r"\s*", anchor.group(1)).group(0)
        insertion = anchor.group(1) + f"{indent}max_patches: {max_patches}        # calibrated capacity\n"
        text = text[: anchor.start()] + insertion + text[anchor.end() :]
    backup = path.with_suffix(path.suffix + ".before-calibration")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.target_bytes_per_patch <= 0:
        raise SystemExit("--target-bytes-per-patch must be positive")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    if model_cfg.get("patching_mode") != "entropy":
        raise SystemExit("the config must use model.patching_mode: entropy")
    checkpoint = args.checkpoint or Path(model_cfg["entropy_model_ckpt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_entropy_model(model_cfg, checkpoint, device)
    score_model = torch.compile(model, mode=args.compile_mode) if args.compile else model

    from src.data.byte_dataset import build_byte_dataloader

    data_cfg = dict(cfg["data"])
    data_cfg["micro_batch_size"] = args.batch_size
    data_cfg["num_workers"] = 0
    loader = build_byte_dataloader(data_cfg, split="train")
    iterator = iter(loader)
    samples: list[np.ndarray] = []
    entropy_min = math.inf
    entropy_max = -math.inf
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.inference_mode():
        for batch_index in range(args.batches):
            batch = next(iterator)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                logits = score_model(input_ids).logits
            entropy = entropy_from_logits(logits).float().cpu().numpy()
            entropy_min = min(entropy_min, float(entropy.min()))
            entropy_max = max(entropy_max, float(entropy.max()))
            samples.extend(np.ascontiguousarray(row) for row in entropy)
            print(
                f"[calibrate] batch={batch_index + 1}/{args.batches} "
                f"samples={len(samples)} entropy=[{entropy_min:.3f}, {entropy_max:.3f}]",
                flush=True,
            )

    min_len = int(model_cfg["min_patch_len"])
    max_len = int(model_cfg["max_patch_len"])
    low = entropy_min - 1e-5
    high = entropy_max + 1e-5
    min_bpp, _ = evaluate(samples, low, min_len, max_len)
    max_bpp, _ = evaluate(samples, high, min_len, max_len)
    target = min(max(args.target_bytes_per_patch, min_bpp), max_bpp)
    for _ in range(args.binary_search_steps):
        middle = (low + high) / 2.0
        bytes_per_patch, _ = evaluate(samples, middle, min_len, max_len)
        if bytes_per_patch < target:
            low = middle
        else:
            high = middle
    threshold = (low + high) / 2.0
    bytes_per_patch, counts = evaluate(samples, threshold, min_len, max_len)
    context = int(cfg["data"]["context_length"])
    max_patches = recommend_capacity(counts, context, min_len)

    result = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "samples": len(samples),
        "context_length": context,
        "target_bytes_per_patch": args.target_bytes_per_patch,
        "reachable_bytes_per_patch": [min_bpp, max_bpp],
        "entropy_threshold": threshold,
        "measured_bytes_per_patch": bytes_per_patch,
        "patches_per_seq_mean": float(counts.mean()),
        "patches_per_seq_p99": float(np.quantile(counts, 0.99)),
        "patches_per_seq_p999": float(np.quantile(counts, 0.999)),
        "patches_per_seq_max": int(counts.max()),
        "recommended_max_patches": max_patches,
        "min_patch_len": min_len,
        "max_patch_len": max_len,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nmodel:")
    print(f"  entropy_threshold: {threshold:.6f}")
    print(f"  max_patches: {max_patches}")
    if args.write:
        update_config(args.config, threshold, max_patches)
        print(f"[calibrate] updated {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
