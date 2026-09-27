"""ByteLM の char_rest head (entropy_char の 3 byte 目以降の予測エントロピー推定、2 層 MLP) を学習する.

本体は凍結し、head だけを学習する。正解は同じ ByteLM が実際に出す
3〜4 byte 目の予測エントロピーの和で、その 2 byte 目に関する期待値が
「1 byte 目まで見た時点での残りの不確かさ」になる。重みは ByteLM の step dir に
char_rest_head.safetensors として保存する (ByteLM の step が変わったら学習し直す)。

    python scripts/train_char_rest_head.py --ckpt latest --steps 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.byte_dataset import build_byte_dataloader  # noqa: E402
from src.model.arbor import BYTE_OFFSET, CHAR_REST_HEAD_FILE, build_byte_lm  # noqa: E402


def rest_targets(input_ids: torch.Tensor, ent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(正解, 対象位置 mask)。位置 t が 3〜4 byte 文字の 1 byte 目なら ent[t+1] (+ 4 byte は ent[t+2])."""
    cur = input_ids - BYTE_OFFSET
    lead3 = (cur >= 0xE0) & (cur <= 0xEF)
    lead4 = (cur >= 0xF0) & (cur <= 0xF4)
    e1 = F.pad(ent[:, 1:], (0, 1))
    e2 = F.pad(ent[:, 2:], (0, 2))
    target = torch.where(lead4, e1 + e2, e1)
    t = input_ids.shape[1]
    pos = torch.arange(t, device=input_ids.device)
    mask = (lead3 & (pos < t - 1)) | (lead4 & (pos < t - 2))
    return target, mask


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="./checkpoints/entropy_lm", type=Path)
    p.add_argument("--ckpt", default="latest")
    p.add_argument("--config", default="configs/entropy_lm.yaml", type=Path, help="data 節を使う")
    p.add_argument("--steps", default=2000, type=int)
    p.add_argument("--micro-batch", default=8, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--eval-batches", default=16, type=int)
    p.add_argument("--seed", default=1234, type=int)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    ckpt = (args.ckpt_dir / args.ckpt).resolve()
    from safetensors.torch import load_file, save_file

    model_cfg = yaml.safe_load((ckpt / "config.yaml").read_text())["model"]
    model_cfg = {k: v for k, v in model_cfg.items() if k != "arch"}
    model_cfg["char_rest_head"] = True
    lm = build_byte_lm(model_cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in load_file(str(ckpt / "model.safetensors")).items()}
    missing, unexpected = lm.load_state_dict(state, strict=False)
    if unexpected or not missing or any(not k.startswith("char_rest.") for k in missing):
        raise RuntimeError(f"ByteLM の重みが合わない: missing={missing} unexpected={unexpected}")
    lm = lm.to(device)
    lm.requires_grad_(False)
    lm.char_rest.requires_grad_(True)
    torch.nn.init.zeros_(lm.char_rest[-1].weight)
    torch.nn.init.constant_(lm.char_rest[-1].bias, 1.0)
    lm.eval()
    trunk = torch.compile(lambda ids: lm.norm(lm._hidden(ids)))

    data_cfg = dict(yaml.safe_load(args.config.read_text())["data"])
    data_cfg.update(micro_batch_size=args.micro_batch, num_workers=0, seed=args.seed)
    train_it = iter(build_byte_dataloader(data_cfg, split="train"))
    # 評価は先頭の batch を取り分けて学習には使わない (head は 20 万 param で、学習は数千万箇所なので過学習しない)
    eval_batches = [next(train_it)["input_ids"] for _ in range(args.eval_batches)]

    def features(ids: torch.Tensor):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = trunk(ids)
        with torch.no_grad():
            logp = F.log_softmax(lm.head(x.float()), dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
        target, mask = rest_targets(ids, ent)
        return x.float()[mask], target[mask], ids[mask] - BYTE_OFFSET

    def evaluate() -> dict[str, float]:
        preds, targets, leads = [], [], []
        with torch.no_grad():
            for ids in eval_batches:
                x, tgt, lead = features(ids.to(device))
                preds.append(lm.char_rest_from_hidden(x))
                targets.append(tgt)
                leads.append(lead)
        pred, tgt, lead = torch.cat(preds), torch.cat(targets), torch.cat(leads)
        # 比較基準: 文字の長さ (3 / 4 byte) ごとの平均値だけで当てる定数予測
        const = torch.where(lead >= 0xF0, tgt[lead >= 0xF0].mean(), tgt[lead < 0xF0].mean())
        mse = F.mse_loss(pred, tgt).item()
        mse_const = F.mse_loss(const, tgt).item()
        return {
            "n": int(tgt.numel()), "mse": mse, "mae": (pred - tgt).abs().mean().item(),
            "mse_const": mse_const, "r2_vs_const": 1 - mse / mse_const,
            "corr": torch.corrcoef(torch.stack([pred, tgt]))[0, 1].item(),
            "target_mean": tgt.mean().item(), "pred_mean": pred.mean().item(),
        }

    opt = torch.optim.AdamW(lm.char_rest.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 1 - s / args.steps)
    )
    print(f"[rest] ckpt={ckpt} eval_n={evaluate()['n']} (初期値) {evaluate()}")
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, tgt, _ = features(next(train_it)["input_ids"].to(device))
        loss = F.mse_loss(lm.char_rest_from_hidden(x), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            print(f"[rest] step={step} train_mse={loss.item():.4f} {time.perf_counter() - t0:.0f}s eval={evaluate()}")
    metrics = evaluate()
    head = {k: v.detach().cpu().contiguous() for k, v in lm.char_rest.state_dict().items()}
    save_file(head, str(ckpt / CHAR_REST_HEAD_FILE))
    (ckpt / "char_rest_head.json").write_text(json.dumps(
        {"steps": args.steps, "lr": args.lr, "micro_batch": args.micro_batch, "eval": metrics},
        ensure_ascii=False, indent=2,
    ))
    print(f"[rest] saved {ckpt / CHAR_REST_HEAD_FILE} eval={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
