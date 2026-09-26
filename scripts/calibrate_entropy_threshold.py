#!/usr/bin/env python3
"""Entropy patching の entropy_threshold を実データで校正する.

patch 数の上限 model.max_patches は「global の計算予算」として先に決める値
(境界規則の予算ガードにより patch 数はこれを超えない)。このスクリプトは
凍結済み ByteLM を学習データ混合の小サンプルに通し、本体と同じ境界規則
(min/max patch 長・文書先頭の強制境界・予算ガード。src/model/arbor.py の
compute_patch_starts) で、平均 patch 数が予算の --target-fill (既定 0.9) になる
entropy_threshold を二分探索する。

予算を使い切る系列 (ガードが効いて後半が max_patch_len 区切りに寄る系列) の
割合も表示する。多すぎるなら target-fill を下げるか max_patches を増やす。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file as safe_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target-fill", type=float, default=0.9,
                        help="平均 patch 数 / max_patches の目標 (pad 無駄 = 1 - fill)")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--binary-search-steps", type=int, default=24)
    parser.add_argument("--write", action="store_true", help="config の entropy_threshold を更新する")
    return parser.parse_args()


def resolve_weights(checkpoint: Path) -> Path:
    path = checkpoint.expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"model.safetensors not found: {path}")
    return path


def load_entropy_model(entropy_cfg: dict, max_bytes: int, checkpoint: Path, device: torch.device):
    from src.model.arbor import ByteLM

    entropy_cfg = dict(entropy_cfg)
    entropy_cfg.setdefault("max_bytes", max_bytes)
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


def update_config(path: Path, threshold: float) -> None:
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(
        r"(?m)^(\s*entropy_threshold:\s*)[^\s#]+(.*)$",
        lambda match: f"{match.group(1)}{threshold:.6f}{match.group(2)}",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("entropy_threshold line not found")
    backup = path.with_suffix(path.suffix + ".before-calibration")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def search_threshold(counts_at, target: float, low: float, high: float, steps: int) -> float:
    """平均 patch 数が target になる閾値を二分探索する (閾値を上げるほど patch は減る)."""
    for _ in range(steps):
        middle = (low + high) / 2.0
        if float(counts_at(middle).mean()) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def main() -> int:
    args = parse_args()
    if not 0 < args.target_fill <= 1:
        raise SystemExit("--target-fill must be in (0, 1]")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    if model_cfg.get("patching_mode") != "entropy":
        raise SystemExit("the config must use model.patching_mode: entropy")

    from src.model.arbor import compute_patch_starts
    from src.train.train import resolve_entropy_lm_reference

    # entropy_lm_config / entropy_model_ckpt の解決は学習と同じ経路を使う
    resolved = resolve_entropy_lm_reference(cfg, args.config)
    model_cfg = resolved["model"]
    checkpoint = args.checkpoint or Path(model_cfg["entropy_model_ckpt"])
    max_bytes = int(model_cfg["max_bytes"])
    min_len, max_len = int(model_cfg["min_patch_len"]), int(model_cfg["max_patch_len"])
    budget = model_cfg.get("max_patches")
    if not budget:
        raise SystemExit("model.max_patches (global の計算予算) を先に決めて config に書くこと")
    budget = int(budget)
    eos = int(model_cfg.get("eos_token_id", 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = load_entropy_model(model_cfg["entropy_model"], max_bytes, checkpoint, device)

    from src.data.byte_dataset import build_byte_dataloader

    data_cfg = dict(cfg["data"])
    data_cfg["micro_batch_size"] = args.batch_size
    data_cfg["num_workers"] = 0
    iterator = iter(build_byte_dataloader(data_cfg, split="train"))
    ids_list, ent_list = [], []
    with torch.inference_mode():
        for batch_index in range(args.batches):
            ids = next(iterator)["input_ids"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                ent = lm.next_byte_entropy(ids).float()
            ids_list.append(ids)
            ent_list.append(ent)
            print(f"[calibrate] batch={batch_index + 1}/{args.batches} "
                  f"entropy=[{float(ent.min()):.3f}, {float(ent.max()):.3f}]", flush=True)
    ids_all, ent_all = torch.cat(ids_list), torch.cat(ent_list)

    def counts_at(threshold: float) -> torch.Tensor:
        starts = compute_patch_starts(
            ids_all, "entropy", min_len, max_len, entropy_values=ent_all, threshold=threshold,
            eos_token_id=eos, budget=budget, horizon=max_bytes,
        )
        return starts.sum(1).float()

    threshold = search_threshold(
        counts_at, args.target_fill * budget,
        float(ent_all.min()) - 1e-5, float(ent_all.max()) + 1e-5, args.binary_search_steps,
    )
    counts = counts_at(threshold)
    tokens = ids_all.size(1)
    result = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "sequences": int(ids_all.size(0)),
        "max_patches(budget)": budget,
        "min_budget": math.ceil(max_bytes / max_len),
        "entropy_threshold": threshold,
        "patches_per_seq_mean": float(counts.mean()),
        "fill_mean": float(counts.mean()) / budget,
        "bytes_per_patch_mean": tokens / float(counts.mean()),
        "budget_exhausted_ratio": float((counts >= budget).float().mean()),
        "min_patch_len": min_len,
        "max_patch_len": max_len,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write:
        update_config(args.config, threshold)
        print(f"[calibrate] updated entropy_threshold in {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
