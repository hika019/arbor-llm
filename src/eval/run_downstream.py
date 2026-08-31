"""下流タスク評価: checkpoint を並べて多肢選択タスクの正解率/margin を比較する.

val bpb では測れない「知識の正確さ」を測るためのエントリポイント。
モデル形状は checkpoint 同梱の config.yaml から読むので、base/CPT を同じ
コマンドで比較できる。

実行:
    source scripts/env.sh
    python -m src.eval.run_downstream \
      --ckpt checkpoints/arbor2_1b_8k_filter/latest \
      --ckpt checkpoints/arbor2_1b_8k_cpt/latest \
      --tasks jcommonsenseqa --num-fewshot 3 --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval.multiple_choice import evaluate_mc, selftest  # noqa: E402
from src.eval.tasks import TASKS, format_fewshot_prefix, sample_fewshot  # noqa: E402
from src.model.arbor import build_arbor  # noqa: E402


def _load_model(ckpt_dir: Path, device: torch.device, dtype: torch.dtype):
    """checkpoint ディレクトリ (step_XXXX または latest symlink) からモデルを復元."""
    from safetensors.torch import load_file

    ckpt_dir = ckpt_dir.resolve()
    cfg = yaml.safe_load((ckpt_dir / "config.yaml").read_text())
    meta = json.loads((ckpt_dir / "meta.json").read_text())

    model = build_arbor(cfg["model"]).to(device=device, dtype=dtype)
    weights = load_file(str(ckpt_dir / "model.safetensors"), device=str(device))
    if any(k.startswith("_orig_mod.") for k in weights):
        weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, meta, int(cfg["model"].get("patch_size", 8))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, type=Path,
                   help="checkpoint ディレクトリ。複数指定で横並び比較")
    p.add_argument("--tasks", default="jcommonsenseqa", help="カンマ区切り")
    p.add_argument("--num-fewshot", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="評価問題数の上限 (試走用)")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true",
                   help="バッチ右 pad が採点値を変えないか実測して終了")
    p.add_argument("--out", type=Path, default=None, help="結果 JSON の出力先")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for name in task_names:
        if name not in TASKS:
            raise SystemExit(f"unknown task: {name} (available: {', '.join(TASKS)})")

    # データは checkpoint 間で共有 (同じ問題・同じ few-shot 例で比較する)
    prepared = {}
    for name in task_names:
        task = TASKS[name]
        eval_docs = task.load(task.eval_split)
        shots = sample_fewshot(task.load(task.fewshot_split), args.num_fewshot, args.seed)
        if args.limit:
            eval_docs = eval_docs[: args.limit]
        prepared[name] = (eval_docs, format_fewshot_prefix(shots))
        print(f"[data] {name}: eval={len(eval_docs)}問 fewshot={len(shots)}例")

    results: dict[str, dict] = {}
    for ckpt in args.ckpt:
        model, meta, patch_size = _load_model(ckpt, device, dtype)
        tag = f"{ckpt.parent.name}@{meta['global_step']}"
        print(f"\n[ckpt] {tag}  ({ckpt})")

        if args.selftest:
            docs, prefix = prepared[task_names[0]]
            st = selftest(model, docs[:40], prefix, device=device, dtype=dtype,
                          batch_size=args.batch_size, patch_size=patch_size)
            print(f"  順序不変性 (バッチ構成を変える)   : {st['order_gap_nats']:.2e} nats"
                  "  ← 0 なら比較が成立")
            print(f"  形状ノイズ (batch_size を +1)     : {st['shape_gap_nats']:.3f} nats"
                  f"  / margin が {st['margin_shape_noise']:+.4f} 動く")
            print(f"    margin: b={args.batch_size} → {st['margin_at_b']:+.4f} / "
                  f"b={args.batch_size + 1} → {st['margin_at_b_plus_1']:+.4f}")
            print(f"    acc   : b={args.batch_size} → {st['acc_at_b']:.3f} / "
                  f"b={args.batch_size + 1} → {st['acc_at_b_plus_1']:.3f}")
            print("  → checkpoint 間の差がこの形状ノイズより小さければ判定不能")
            del model
            torch.cuda.empty_cache()
            continue

        results[tag] = {}
        for name in task_names:
            docs, prefix = prepared[name]
            m = evaluate_mc(model, docs, prefix, device=device, dtype=dtype,
                            batch_size=args.batch_size, patch_size=patch_size)
            results[tag][name] = m
            print(f"  {name}: acc={m['acc']:.3f} acc_norm={m['acc_norm']:.3f} "
                  f"margin_mean={m['margin_mean']:+.4f} margin={m['margin']:+.4f} "
                  f"gold_bpb={m['gold_bpb']:.4f} (n={m['n']})")
        del model
        torch.cuda.empty_cache()

    if args.selftest:
        return 0

    if len(results) > 1:
        print("\n=== 比較 (margin_mean: 偶然=0, 正が良い / gold_bpb は低いほど良い) ===")
        for name in task_names:
            print(f"--- {name}")
            for tag, r in results.items():
                m = r[name]
                print(f"  {tag:>28}  acc={m['acc']:.3f}  acc_norm={m['acc_norm']:.3f}  "
                      f"margin_mean={m['margin_mean']:+.4f}  margin={m['margin']:+.4f}  "
                      f"gold_bpb={m['gold_bpb']:.4f}")

    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
