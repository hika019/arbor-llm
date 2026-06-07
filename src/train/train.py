"""学習エントリポイント.

実行:
    python -m src.train.train --config configs/arbor_1b.yaml
    python -m src.train.train --config configs/arbor_1b.yaml --resume latest

設計方針:
- データはストリーミング (HF datasets `streaming=True` 等)。全件メモリ展開しない。
- BF16 mixed precision + Flash Attn + torch.compile + 8bit Adam で速度を稼ぐ。
- SIGINT/SIGTERM で次 step 境界に安全保存して終了 (二重押しで強制終了)。
- チェックポイントは外部 dir に safetensors + 状態一式をアトミック保存。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# torch import / CUDA 初期化より前に効かせる必要がある env (env.sh と二重で保険).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._dynamo
import yaml

# プロジェクト root を import path に追加
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train.checkpoint import CheckpointManager, CheckpointMeta  # noqa: E402
from src.train.optim import build_optimizer, build_scheduler  # noqa: E402
from src.train.signals import StopFlag  # noqa: E402
from src.train.throughput import ThroughputMeter  # noqa: E402


# --------------------------------------------------------------- 引数 / 設定
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", default=None, help="'latest' | 'best' | step | path")
    p.add_argument("--dry-run", action="store_true", help="1 step だけ走らせて即終了")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------- グローバル最適化
def apply_speed_settings(speed: dict) -> None:
    """学習開始前に効かせるスループット系の設定をまとめて適用."""
    if speed.get("tf32_matmul", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if speed.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get("seed", 42))
    apply_speed_settings(cfg.get("speed", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} torch={torch.__version__}")

    # ---- モデル組み立て (BLT + BitLinear) は src.model.arbor_blt に集約 ----
    from src.model.arbor_blt import build_arbor_blt  # 遅延 import (BLT 取り込み後に有効化)
    model = build_arbor_blt(cfg["model"]).to(device=device, dtype=torch.bfloat16)

    if cfg["speed"].get("torch_compile", False):
        mode = cfg["speed"].get("compile_mode", "default")
        # BLT の動的パッチングで seq 長が毎 step 変動し dynamo が recompile を繰り返す
        # (cache_size_limit 既定 8 に到達→eager fallback) のを抑える:
        #   dynamo_cache_size_limit: cache 上限 (recompile を許容する形状数)
        #   compile_dynamic: None=auto / True=seq 次元を symbolic に1グラフ / False=静的
        csl = cfg["speed"].get("dynamo_cache_size_limit")
        if csl:
            torch._dynamo.config.cache_size_limit = int(csl)
        dynamic = cfg["speed"].get("compile_dynamic", None)
        model = torch.compile(model, mode=mode, dynamic=dynamic)

    # ---- データ (streaming, メモリに全部載せない) ----
    from src.data.byte_dataset import build_byte_dataloader
    data_cfg = dict(cfg["data"])
    data_cfg.setdefault("micro_batch_size", cfg.get("speed", {}).get("micro_batch_size", 4))
    train_loader = build_byte_dataloader(data_cfg, split="train")

    # ---- optimizer / scheduler ----
    optimizer = build_optimizer(model.parameters(), cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["optim"])

    # ---- チェックポイント ----
    ckpt_cfg = cfg["checkpoint"]
    ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", ckpt_cfg["dir"]))
    ckpt = CheckpointManager(
        ckpt_dir,
        keep_last_k=ckpt_cfg.get("keep_last_k", 3),
        keep_every_n_steps=ckpt_cfg.get("keep_every_n_steps"),
        async_save=ckpt_cfg.get("async_save", True),
    )

    # ---- 再開処理 ----
    global_step = 0
    best_loss = float("inf")
    if args.resume:
        meta, dl_state = ckpt.load(args.resume, model, optimizer, scheduler, map_location=device)
        global_step = meta.global_step
        best_loss = meta.best_loss
        if dl_state is not None:
            train_loader.load_state_dict(dl_state)
        print(f"[train] resumed from step={global_step}, best_loss={best_loss:.4f}")

    # ---- 学習ループ ----
    stop = StopFlag()
    meter = ThroughputMeter(window=cfg["logging"].get("throughput_window", 50))
    cfg_hash = config_hash(cfg)
    save_every = ckpt_cfg["save_every_steps"]
    grad_accum = cfg["speed"].get("grad_accum_steps", 1)
    log_every = cfg["logging"].get("log_every_steps", 20)
    total_steps = cfg["optim"]["total_steps"]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    data_iter = iter(train_loader)
    while global_step < total_steps:
        try:
            for micro in range(grad_accum):
                batch = next(data_iter)
                inputs = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        out.logits.flatten(0, 1), labels.flatten(), ignore_index=-100
                    ) / grad_accum
                loss.backward()
                accum_loss += loss.item()

            if cfg["optim"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            meter.step(inputs.numel() * grad_accum)

            if global_step % log_every == 0:
                print(f"step={global_step} loss={accum_loss:.4f} "
                      f"tok/s={meter.tokens_per_sec():.0f} lr={scheduler.get_last_lr()[0]:.2e}")
            cur_loss = accum_loss
            accum_loss = 0.0

            # best はトラッキングのみ。実保存は定期 / 中断 / 最終 step に限定する.
            # 毎 step ベスト更新で save するとディスクを食いつぶすので分離.
            is_best = cur_loss < best_loss
            if is_best:
                best_loss = cur_loss
            should_save = (
                global_step % save_every == 0
                or stop.requested
                or global_step >= total_steps
            )
            if should_save:
                meta = CheckpointMeta(
                    global_step=global_step,
                    best_loss=best_loss,
                    config_hash=cfg_hash,
                    wandb_run_id=os.environ.get("WANDB_RUN_ID"),
                )
                dl_state = train_loader.state_dict() if hasattr(train_loader, "state_dict") else None
                ckpt.save(
                    model, optimizer, scheduler, dl_state, meta,
                    is_best=is_best,
                    is_final=global_step >= total_steps,
                )
                print(f"[train] saved checkpoint @ step={global_step}{' (best)' if is_best else ''}")

            if stop.requested:
                print("[train] stop requested, exiting cleanly.")
                break
            if args.dry_run:
                break

        except StopIteration:
            data_iter = iter(train_loader)
            continue

    # 最終 prune の daemon スレッドが途中終了して checkpoint を部分削除しないよう待つ
    ckpt._await_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
