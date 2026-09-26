"""逐語コピーの成績が static patch の位相 (コピー元とコピー先の patch 境界のずれ) で変わるかを測る.

段落 P を 1 回見せ、区切り (改行 2 つ + 空白 m 個) の後にもう一度 P の前半を書かせて
続き 32 byte を当てさせる (cloze.py の repeat と同じ)。m を 0..15 で動かすと
2 回目の開始位置が patch_size で割った余り r だけずれる。1 回目は位置 0 (境界) から
始まるので、r=0 ならコピー元とコピー先で patch の切れ目が完全に一致する。

r=0 だけ成績が良ければ「大域層が patch 単位でしか照合できず、位相がずれると
コピーできない」= static patching の構造的弱点。r によらず同じなら構造は無関係。
基準として 1 回目を見せない (区切り + 前半 + 続き だけの) 場合も測る。

実行:
    source scripts/env313.sh
    python scripts/probe_patch_alignment.py \
      --ckpt checkpoints/arbor2_1b_8k_lowbit/latest --ckpt checkpoints/arbor2_1b_8k_hybrid/latest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.cloze import ClozeDoc, _passages, evaluate_cloze  # noqa: E402
from src.eval.run_downstream import _load_model  # noqa: E402


def build(paras: list[str], patch: int) -> dict[str, list[ClozeDoc]]:
    """条件名 → 問題リスト。条件: none (1 回目なし), r00..r15 (2 回目開始位置の余り)."""
    out: dict[str, list[ClozeDoc]] = {"none": []}
    for p in paras:
        raw = p.encode("utf-8")
        cut = len(raw) // 2
        while cut < len(raw) and (raw[cut] & 0xC0) == 0x80:
            cut += 1
        head = raw[:cut].decode("utf-8")
        tail = raw[cut:cut + 32].decode("utf-8", errors="ignore")
        out["none"].append(ClozeDoc(context=head, target=tail))
        for m in range(patch):
            sep = "\n\n" + " " * m
            start2 = len(raw) + len(sep.encode())
            r = start2 % patch
            out.setdefault(f"r{r:02d}", []).append(ClozeDoc(context=p + sep + head, target=tail))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, type=Path)
    ap.add_argument("--lang", default="en,ja")
    ap.add_argument("--n", type=int, default=200, help="段落数 (条件ごとの問題数)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    langs = [s.strip() for s in args.lang.split(",") if s.strip()]
    paras = {lang: _passages(lang)[: args.n] for lang in langs}

    results: dict[str, dict] = {}
    for ckpt in args.ckpt:
        model, meta, patch = _load_model(ckpt, device, torch.bfloat16)
        tag = f"{ckpt.resolve().parent.name}@{meta['global_step']}"
        results[tag] = {}
        print(f"\n[ckpt] {tag} patch_size={patch}", flush=True)
        for lang in langs:
            conds = build(paras[lang], patch)
            res = {}
            for name in ["none", *sorted(k for k in conds if k != "none")]:
                res[name] = evaluate_cloze(model, conds[name], device=device, dtype=torch.bfloat16,
                                           batch_size=args.batch_size, patch_size=patch)
            results[tag][lang] = res
            rs = [res[k] for k in sorted(res) if k != "none"]
            print(f"  [{lang}] none: acc={res['none']['acc']:.3f} bpb={res['none']['target_bpb']:.3f}")
            for k in sorted(k for k in res if k != "none"):
                m = res[k]
                print(f"  [{lang}] {k}: acc={m['acc']:.3f} bpb={m['target_bpb']:.3f}", flush=True)
            others = [res[k] for k in res if k not in ("none", "r00")]
            print(f"  [{lang}] r00 vs 他 15 位相の平均: acc {res['r00']['acc']:.3f} vs "
                  f"{sum(m['acc'] for m in others) / len(others):.3f} / bpb {res['r00']['target_bpb']:.3f} vs "
                  f"{sum(m['target_bpb'] for m in others) / len(others):.3f}  (位相 {len(rs)} 通り)")
        del model
        torch.cuda.empty_cache()

    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
