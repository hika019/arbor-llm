#!/usr/bin/env python3
"""閾値ごとの entropy patching の区切り方と、seq_patches 個の patch の系列の byte 数を実データで表示する.

凍結済み ByteLM を学習データ混合の小サンプルに通し、学習と同じ PatchPacker (source ごとに流れを
続けて区切る) で閾値ごとに区切って、patch 長の分布・長さ上限で切れた patch / byte の割合と、
seq_patches 個の patch がそのまま並んだときの系列の byte 数の分布・config の max_bytes の枠に
収まらない系列の割合・枠の余り (PAD) の割合を出す。config は書き換えない。

閾値は区切りの細かさを決めるつまみで、どれを使うかは本体 bpb の A/B で決める。max_bytes は
「系列の byte 数の p99 程度」が目安 (小さいと枠で切れて patch が seq_patches 未満の系列が増え、
大きいと local 層が PAD の分も計算する)。
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
                        help="表示する閾値 (既定: entropy_char は 1〜3 を 0.25 刻み + config の値)")
    parser.add_argument("--batches", type=int, default=64)
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


class CachedEntropy:
    """閾値を変えても packer が ByteLM に渡す入力は同じ順に同じなので、1 回目の結果を使い回す."""

    def __init__(self, fn):
        self.fn, self.cache, self.calls = fn, [], 0

    def rewind(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor, lengths: list, states: list):
        self.calls += 1
        if self.calls > len(self.cache):
            ent, rest, new_states = self.fn(x, lengths, states)
            self.cache.append((ent, rest))
            return ent, rest, new_states
        return (*self.cache[self.calls - 1], [None] * len(states))


def pack_rows(packer, batches: list[dict], rows: int) -> list[dict]:
    packer.set_source(iter(batches))
    out = []
    try:
        while True:
            out.append(packer.next_batch(rows))
    except StopIteration:
        pass
    packer.flush()
    return out + list(packer.drain(rows))


def stats(batches: list[dict], seq_patches: int, soft_len: int | None, frame: int) -> dict:
    lengths, row_bytes = [], []
    for b in batches:
        for r in range(b["input_ids"].size(0)):
            n = int(b["n_bytes"][r])
            pos = torch.nonzero(b["patch_starts"][r, :n]).flatten()
            lengths.append(torch.diff(pos, append=pos.new_tensor([n])))
            if int(b["n_patches"][r]) == seq_patches:
                row_bytes.append(n)
    lengths = torch.cat(lengths).float()
    rb = torch.tensor(row_bytes, dtype=torch.float32) if row_bytes else torch.zeros(1)
    q = torch.quantile(lengths, lengths.new_tensor([0.1, 0.5, 0.9, 0.99]))
    qr = torch.quantile(rb, rb.new_tensor([0.5, 0.9, 0.99]))
    capped = lengths >= soft_len if soft_len is not None else torch.zeros_like(lengths, dtype=torch.bool)
    return {
        "bytes_per_patch": float(lengths.mean()),
        "len_p10": float(q[0]), "len_p50": float(q[1]), "len_p90": float(q[2]), "len_p99": float(q[3]),
        "len_max": float(lengths.max()),
        "capped_patches": float(capped.float().mean()),
        "capped_bytes": float(lengths[capped].sum() / lengths.sum()),
        "rows": len(row_bytes),
        "row_p50": float(qr[0]), "row_p90": float(qr[1]), "row_p99": float(qr[2]), "row_max": float(rb.max()),
        "over_frame": float((rb > frame).float().mean()),
        "frame_pad": float(1 - rb.clamp(max=frame).mean() / frame),
    }


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mode = cfg["model"].get("patching_mode")
    if mode not in ("entropy", "entropy_char"):
        raise SystemExit("the config must use model.patching_mode: entropy | entropy_char")

    from src.data.byte_dataset import build_byte_dataloader
    from src.data.patch_packer import PatchPacker
    from src.model.arbor import ArborConfig, patch_len_bounds
    from src.train.train import resolve_entropy_lm_reference

    # entropy_lm_config / entropy_model_ckpt の解決は学習と同じ経路を使う
    resolved = resolve_entropy_lm_reference(cfg, args.config)
    model_cfg = resolved["model"]
    arbor_cfg = ArborConfig.from_dict(model_cfg)
    checkpoint = args.checkpoint or Path(model_cfg["entropy_model_ckpt"])
    _, max_len = patch_len_bounds(arbor_cfg)
    soft_len = None if max_len is None else max_len - 3 if mode == "entropy_char" else max_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = load_entropy_model(model_cfg["entropy_model"], arbor_cfg.max_bytes, checkpoint, device,
                            arbor_cfg.entropy_char_rest)

    data_cfg = dict(cfg["data"])
    data_cfg.update(micro_batch_size=args.batch_size, num_workers=0, contiguous=True)
    iterator = iter(build_byte_dataloader(data_cfg, split="train"))
    chunks = []
    for i in range(args.batches):
        chunks.append(next(iterator))
        if (i + 1) % 16 == 0:
            print(f"[stats] chunk batch {i + 1}/{args.batches}", flush=True)
    chunk_len = int(data_cfg["context_length"])

    if args.thresholds:
        thresholds = sorted(args.thresholds)
    elif mode == "entropy_char":
        thresholds = sorted({1 + x * 0.25 for x in range(9)} | {float(arbor_cfg.entropy_threshold)})
    else:
        thresholds = sorted({0.5 + x * 0.25 for x in range(9)} | {float(arbor_cfg.entropy_threshold)})

    entropy = None
    frame, seq_patches = arbor_cfg.max_bytes, arbor_cfg.seq_patches
    print(f"[stats] {args.config} / ByteLM {checkpoint}")
    print(f"[stats] {len(chunks) * args.batch_size} chunk x {chunk_len} byte, max_patch_len={max_len}, "
          f"config の閾値={arbor_cfg.entropy_threshold}, seq_patches={seq_patches}, max_bytes={frame}")
    print(f"{'閾値':>7} {'B/patch':>8} {'長さ p10/50/90/99/最大':>22} {'上限切れ patch/byte':>19} {'系列数':>6} "
          f"{'系列 byte p50/p90/p99/最大':>27} {'枠超え':>7} {'枠の余り':>8}")
    for threshold in thresholds:
        # 枠で切らずに系列の byte 数を測る
        packer = PatchPacker(arbor_cfg, lm, chunk_len=chunk_len, device=device, threshold=threshold,
                             max_bytes=seq_patches * max_len if max_len else 16 * frame)
        if entropy is None:
            entropy = CachedEntropy(packer.entropy_fn)
        entropy.rewind()
        packer.entropy_fn = entropy
        st = stats(pack_rows(packer, chunks, args.batch_size), seq_patches, soft_len, frame)
        lens = "/".join(f"{st[k]:.0f}" for k in ("len_p10", "len_p50", "len_p90", "len_p99", "len_max"))
        capped = f"{st['capped_patches']:.1%}/{st['capped_bytes']:.1%}" if soft_len is not None else "上限なし"
        rows = f"{st['row_p50']:.0f}/{st['row_p90']:.0f}/{st['row_p99']:.0f}/{st['row_max']:.0f}"
        print(f"{threshold:7.3f} {st['bytes_per_patch']:8.2f} {lens:>22} {capped:>19} {st['rows']:6d} "
              f"{rows:>27} {st['over_frame']:7.1%} {st['frame_pad']:8.1%}")
    if soft_len is not None:
        print(f"[stats] 上限切れ = 長さ {soft_len} byte 以上の patch (予測しにくさでなく長さで切れたもの) の割合と、"
              "その patch に入る byte の割合")
    print(f"[stats] 系列 byte = {seq_patches} patch の系列の byte 数。枠超え = max_bytes ({frame}) を超えて"
          "学習時は patch が枠で切られる系列の割合。枠の余り = 枠のうち PAD の割合 (local 層の無駄な計算)")
    return 0


if __name__ == "__main__":
    rc = main()
    # 読みかけの HF streaming を interpreter 終了時に破棄すると pyarrow の I/O スレッドが
    # EBADF の再試行で詰まりクラッシュ・終了待ちになる (src/train/train.py と同じ対処)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
