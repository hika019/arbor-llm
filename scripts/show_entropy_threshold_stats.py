#!/usr/bin/env python3
"""閾値ごとの entropy patching の区切り方を実データで表示する (config は書き換えない).

凍結済み ByteLM を学習データ混合の小サンプルに通し、本体と同じ境界規則
(src/model/arbor.py の compute_patch_starts。予算ガードは掛けない) で閾値ごとに区切って、
patch 長の分布・系列あたりの patch 数・長さ上限で切れた patch の割合・config の
max_patches を超える系列の割合を出す。閾値は区切りの細かさ (= global の計算量) を決める
つまみで、どれを使うかは同じ計算時間あたりの本体 bpb の A/B で決める。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file as safe_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/arbor.yaml"), help="既定は本走")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="表示する閾値 (既定: entropy_char は -1〜3 を 0.5 刻み + config の値)")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def resolve_weights(checkpoint: Path) -> Path:
    path = checkpoint.expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"model.safetensors not found: {path}")
    return path


def load_entropy_model(
    entropy_cfg: dict, max_bytes: int, checkpoint: Path, device: torch.device, char_rest: bool = False,
):
    from src.model.arbor import CHAR_REST_HEAD_FILE, ByteLM

    entropy_cfg = dict(entropy_cfg)
    entropy_cfg.setdefault("max_bytes", max_bytes)
    entropy_cfg["char_rest_head"] = char_rest
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = ByteLM(entropy_cfg).to(device=device, dtype=dtype).eval()
    state = safe_load(str(resolve_weights(checkpoint)), device="cpu")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.removeprefix("_orig_mod.")
        if key.startswith("entropy_model."):
            key = key.removeprefix("entropy_model.")
        normalized[key] = value
    if char_rest and not any(k.startswith("char_rest.") for k in normalized):
        head_file = resolve_weights(checkpoint).parent / CHAR_REST_HEAD_FILE
        if not head_file.is_file():
            raise FileNotFoundError(f"entropy_char_rest=true だが {head_file} が無い (scripts/train_char_rest_head.py)")
        normalized.update({f"char_rest.{k}": v for k, v in safe_load(str(head_file), device="cpu").items()})
    model.load_state_dict(normalized, strict=True)
    return model


def patch_stats(starts: torch.Tensor, soft_len: int, budget: int | None) -> dict:
    tokens = starts.size(1)
    lengths = []
    for row in starts:
        pos = torch.nonzero(row).flatten()
        lengths.append(torch.diff(pos, append=pos.new_tensor([tokens])))
    lengths = torch.cat(lengths).float()
    counts = starts.sum(1).float()
    q = torch.quantile(lengths, lengths.new_tensor([0.1, 0.5, 0.9]))
    return {
        "bytes_per_patch": float(lengths.mean()),
        "len_p10": float(q[0]), "len_p50": float(q[1]), "len_p90": float(q[2]),
        "patches_mean": float(counts.mean()),
        "patches_max": int(counts.max()),
        "capped_ratio": float((lengths >= soft_len).float().mean()),
        "over_budget_ratio": float((counts > budget).float().mean()) if budget else None,
    }


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    mode = model_cfg.get("patching_mode")
    if mode not in ("entropy", "entropy_char"):
        raise SystemExit("the config must use model.patching_mode: entropy | entropy_char")

    from src.model.arbor import compute_patch_starts
    from src.train.train import resolve_entropy_lm_reference

    # entropy_lm_config / entropy_model_ckpt の解決は学習と同じ経路を使う
    resolved = resolve_entropy_lm_reference(cfg, args.config)
    model_cfg = resolved["model"]
    checkpoint = args.checkpoint or Path(model_cfg["entropy_model_ckpt"])
    max_bytes = int(model_cfg["max_bytes"])
    min_len, max_len = int(model_cfg["min_patch_len"]), int(model_cfg["max_patch_len"])
    budget = int(model_cfg["max_patches"]) if model_cfg.get("max_patches") else None
    eos = int(model_cfg.get("eos_token_id", 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_rest = bool(model_cfg.get("entropy_char_rest", False))
    lm = load_entropy_model(model_cfg["entropy_model"], max_bytes, checkpoint, device, char_rest)

    from src.data.byte_dataset import build_byte_dataloader

    data_cfg = dict(cfg["data"])
    data_cfg["micro_batch_size"] = args.batch_size
    data_cfg["num_workers"] = 0
    iterator = iter(build_byte_dataloader(data_cfg, split="train"))
    ids_list, ent_list, rest_list = [], [], []
    with torch.inference_mode():
        for batch_index in range(args.batches):
            ids = next(iterator)["input_ids"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                ent, rest = lm.boundary_entropy(ids)
            ids_list.append(ids)
            ent_list.append(ent.float())
            if rest is not None:
                rest_list.append(rest.float())
            print(f"[stats] batch={batch_index + 1}/{args.batches} "
                  f"entropy=[{float(ent.min()):.3f}, {float(ent.max()):.3f}]", flush=True)
    ids_all, ent_all = torch.cat(ids_list), torch.cat(ent_list)
    rest_all = torch.cat(rest_list) if rest_list else None

    if args.thresholds:
        thresholds = sorted(args.thresholds)
    elif mode == "entropy_char":
        thresholds = sorted({x * 0.5 for x in range(-2, 7)} | {float(model_cfg["entropy_threshold"])})
    else:
        lo, hi = float(ent_all.min()), float(ent_all.max())
        thresholds = sorted({lo + (hi - lo) * i / 8 for i in range(1, 8)} | {float(model_cfg["entropy_threshold"])})
    soft_len = max_len - 3 if mode == "entropy_char" else max_len

    print(f"[stats] {args.config} / ByteLM {checkpoint}")
    print(f"[stats] {ids_all.size(0)} 系列 x {ids_all.size(1)} byte, min/max_patch_len={min_len}/{max_len}, "
          f"char_rest={char_rest}, config の閾値={model_cfg['entropy_threshold']}, max_patches={budget}")
    print(f"{'閾値':>8} {'B/patch':>8} {'長さ p10/p50/p90':>17} {'patch/系列 平均':>15} {'最大':>6} "
          f"{'上限切れ':>8} {'予算超え系列':>12}")
    for threshold in thresholds:
        starts = compute_patch_starts(
            ids_all, mode, min_len, max_len, entropy_values=ent_all, rest_values=rest_all,
            threshold=threshold, eos_token_id=eos,
        )
        st = patch_stats(starts, soft_len, budget)
        over = "-" if st["over_budget_ratio"] is None else f"{st['over_budget_ratio']:.1%}"
        lens = f"{st['len_p10']:.0f}/{st['len_p50']:.0f}/{st['len_p90']:.0f}"
        print(f"{threshold:8.3f} {st['bytes_per_patch']:8.2f} {lens:>17} {st['patches_mean']:15.1f} "
              f"{st['patches_max']:6d} {st['capped_ratio']:8.1%} {over:>12}")
    print(f"[stats] 上限切れ = 長さ {soft_len} byte 以上の patch (予測しにくさでなく長さで切れたもの)。"
          "予算超え系列 = max_patches を超える系列 (学習時は後半が予算ガードで機械的に区切られる)")
    return 0


if __name__ == "__main__":
    rc = main()
    # 読みかけの HF streaming を interpreter 終了時に破棄すると pyarrow の I/O スレッドが
    # EBADF の再試行で詰まりクラッシュ・終了待ちになる (src/train/train.py と同じ対処)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
