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
import copy
from contextlib import nullcontext
import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
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
    p.add_argument(
        "--allow-config-mismatch", action="store_true",
        help="resume 時に checkpoint の model 設定と現在の config が違っても続行する",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _git_output(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_metadata(repo_root: Path) -> dict[str, object]:
    status = _git_output(["status", "--porcelain=v1"], repo_root)
    return {
        "sha": _git_output(["rev-parse", "HEAD"], repo_root),
        "branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def run_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    return {
        "argv": list(sys.argv),
        "config_path": str(args.config),
        "resume": args.resume,
        "dry_run": bool(args.dry_run),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }


# ---------------------------------------------------------- グローバル最適化
def apply_speed_settings(speed: dict) -> None:
    """学習開始前に効かせるスループット系の設定をまとめて適用."""
    if speed.get("tf32_matmul", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if speed.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(name: str) -> tuple[torch.dtype, bool]:
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16, True
    if normalized in ("fp16", "float16"):
        return torch.float16, True
    if normalized in ("fp32", "float32"):
        return torch.float32, False
    raise ValueError(f"unknown speed.precision: {name}")


def apply_compile_settings(model: torch.nn.Module, speed: dict) -> torch.nn.Module:
    """Apply torch.compile according to speed config and return the trainable model."""
    # Arbor v2 は静的 patching で形状固定なので compile が素直に効く (既定 ON)
    if not speed.get("torch_compile", True):
        print("[train] torch_compile=OFF")
        return model
    mode = speed.get("compile_mode", "default")
    print(f"[train] torch_compile=ON mode={mode}")
    return torch.compile(model, mode=mode)

# ------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    git_info = git_metadata(_ROOT)

    torch.manual_seed(cfg.get("seed", 42))
    apply_speed_settings(cfg.get("speed", {}))

    device = pick_device()
    print(f"[train] device={device} torch={torch.__version__}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[train] cuda_mem_free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB")

    # ---- モデル組み立て (Arbor v2: 自己完結 BitNet 階層 Transformer) ----
    # model.arch: arbor (既定) | byte_lm (entropy patching 用の小型バイト LM)
    arch = cfg["model"].get("arch", "arbor")
    if arch == "byte_lm":
        from src.model.arbor import build_byte_lm as build_model
    elif arch == "arbor":
        from src.model.arbor import build_arbor as build_model
    else:
        raise ValueError(f"unknown model.arch: {arch}")
    compute_dtype, use_autocast = resolve_precision(cfg.get("speed", {}).get("precision", "bf16"))
    print(f"[train] arch={arch} precision={compute_dtype} autocast={use_autocast}")
    model = build_model(cfg["model"]).to(device=device, dtype=compute_dtype)
    # checkpoint 保存とサンプル生成は compile 前のモデルで行う
    # (compile wrapper を保存すると state dict が _orig_mod. 付きになる)
    base_model = model
    model = apply_compile_settings(model, cfg["speed"])

    # ---- データ (streaming, メモリに全部載せない) ----
    from src.data.byte_dataset import build_byte_dataloader
    data_cfg = dict(cfg["data"])
    speed_micro_batch = cfg.get("speed", {}).get("micro_batch_size")
    if speed_micro_batch is not None:
        data_micro_batch = data_cfg.get("micro_batch_size")
        if data_micro_batch is not None and int(data_micro_batch) != int(speed_micro_batch):
            print(
                "[train] data.micro_batch_size="
                f"{data_micro_batch} overridden by speed.micro_batch_size={speed_micro_batch}"
            )
        data_cfg["micro_batch_size"] = speed_micro_batch
    else:
        data_cfg.setdefault("micro_batch_size", 4)
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
        # model 形状が違う checkpoint を strict=False で黙って部分ロードする事故を防ぐ.
        # checkpoint には保存時の実効 config が入っているので model 節を突き合わせる.
        resolved = ckpt.resolve(args.resume)
        saved_cfg_file = resolved / "config.yaml" if resolved else None
        if saved_cfg_file is not None and saved_cfg_file.exists():
            saved_model_cfg = yaml.safe_load(saved_cfg_file.read_text()).get("model", {})
            if saved_model_cfg and saved_model_cfg != cfg["model"]:
                diff_keys = sorted(
                    k for k in set(saved_model_cfg) | set(cfg["model"])
                    if saved_model_cfg.get(k) != cfg["model"].get(k)
                )
                msg = (
                    f"checkpoint の model 設定と現在の config が不一致: {diff_keys}. "
                    f"再開するなら `--config {saved_cfg_file}` を使うか、"
                    "意図的なら --allow-config-mismatch を付ける。"
                )
                if not args.allow_config_mismatch:
                    raise SystemExit(f"[train] ERROR: {msg}")
                print(f"[train] WARNING: {msg}")
        meta, dl_state = ckpt.load(args.resume, base_model, optimizer, scheduler, map_location=device)
        global_step = meta.global_step
        best_loss = meta.best_loss
        if dl_state is not None:
            train_loader.load_state_dict(dl_state)
        print(f"[train] resumed from step={global_step}, best_loss={best_loss:.4f}")

    # ---- 学習ループ ----
    stop = StopFlag()
    meter = ThroughputMeter(window=cfg["logging"].get("throughput_window", 50))
    effective_cfg = copy.deepcopy(cfg)
    effective_cfg["data"] = dict(data_cfg)
    effective_cfg["checkpoint"] = dict(ckpt_cfg)
    effective_cfg["checkpoint"]["dir"] = str(ckpt_dir)
    cfg_hash = config_hash(effective_cfg)
    run_info = run_metadata(args, device)
    save_every = ckpt_cfg["save_every_steps"]
    grad_accum = cfg["speed"].get("grad_accum_steps", 1)
    log_every = cfg["logging"].get("log_every_steps", 20)
    total_steps = cfg["optim"]["total_steps"]
    micro_batch = data_cfg.get("micro_batch_size")
    context_length = data_cfg.get("context_length")
    if micro_batch and context_length:
        tokens_per_update = int(micro_batch) * int(context_length) * int(grad_accum)
        print(
            "[train] throughput_meter="
            f"optimizer_step rolling_window={meter.window} log_every={log_every} "
            f"micro_batch={micro_batch} grad_accum={grad_accum} "
            f"context_length={context_length} tokens_per_update={tokens_per_update}"
        )
        if int(micro_batch) < 4:
            print(
                "[train] speed_profile=low_vram "
                "micro_batch<4 lowers GPU occupancy; README steady-state notes assume a larger micro-batch"
            )
    else:
        print(
            "[train] throughput_meter="
            f"optimizer_step rolling_window={meter.window} log_every={log_every} "
            f"grad_accum={grad_accum}"
        )
    print("[train] note=early tok/s includes compile/warmup; use logs after the rolling window fills for steady throughput")

    # checkpoint 保存時のサンプル生成 (任意)。学習を止めないよう失敗は警告に留める.
    sampling_cfg = cfg.get("sampling", {})
    sampling_enabled = bool(sampling_cfg.get("enabled", False))
    if sampling_enabled:
        print(
            "[train] sampling=ON prompts={} max_new_bytes={}".format(
                len(sampling_cfg.get("prompts", [])),
                sampling_cfg.get("max_new_bytes", 100),
            )
        )

    def sample_at_checkpoint(step_dir: Path, step: int) -> None:
        from src.infer.generate import generate_samples

        prompts = sampling_cfg.get("prompts") or ["The ", "日本の"]
        base_model.eval()
        try:
            t0 = time.perf_counter()
            samples = generate_samples(
                base_model,
                prompts,
                max_new_bytes=int(sampling_cfg.get("max_new_bytes", 100)),
                temperature=float(sampling_cfg.get("temperature", 0.8)),
                top_p=float(sampling_cfg.get("top_p", 0.95)),
                max_context=int(context_length) if context_length else 2048,
                seed=int(sampling_cfg.get("seed", 42)),
            )
            lines = [f"# step {step}"]
            for prompt, text in samples:
                print(f"[sample] step={step} prompt={prompt!r} -> {text!r}")
                lines.append(f"\n## prompt: {prompt}\n{text}")
            (step_dir / "samples.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"[sample] wrote {step_dir / 'samples.txt'} in {time.perf_counter() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001 - サンプル生成失敗で学習は止めない
            print(f"[sample] generation failed (continuing training): {type(e).__name__}: {e}")
        finally:
            base_model.train()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss_tensor: torch.Tensor | None = None
    best_loss_tensor = torch.tensor(best_loss, device=device, dtype=torch.float32)
    # best 判定は単発 step ではなく EMA 損失で行う (train loss のノイズで
    # "best" が偶然の低い step に張り付くのを防ぐ。validation best ではない点に注意)
    ema_loss_tensor: torch.Tensor | None = None
    ema_decay = float(cfg["logging"].get("best_ema_decay", 0.98))

    data_iter = iter(train_loader)
    while global_step < total_steps:
        try:
            tokens_this_step = 0
            for micro in range(grad_accum):
                batch = next(data_iter)
                inputs = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                tokens_this_step += inputs.numel()
                amp_context = (
                    torch.autocast(device_type=device.type, dtype=compute_dtype)
                    if use_autocast
                    else nullcontext()
                )
                with amp_context:
                    out = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        out.logits.flatten(0, 1), labels.flatten(), ignore_index=-100
                    ) / grad_accum
                loss.backward()
                detached_loss = loss.detach()
                accum_loss_tensor = (
                    detached_loss
                    if accum_loss_tensor is None
                    else accum_loss_tensor + detached_loss
                )

            if cfg["optim"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["optim"]["grad_clip"], foreach=True
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            meter.step(tokens_this_step)

            loss_for_step = (
                accum_loss_tensor.detach().float()
                if accum_loss_tensor is not None
                else torch.tensor(0.0, device=device)
            )
            ema_loss_tensor = (
                loss_for_step
                if ema_loss_tensor is None
                else ema_decay * ema_loss_tensor + (1.0 - ema_decay) * loss_for_step
            )
            is_best_tensor = ema_loss_tensor < best_loss_tensor
            best_loss_tensor = torch.minimum(best_loss_tensor, ema_loss_tensor)

            should_save = (
                global_step % save_every == 0
                or stop.requested
                or global_step >= total_steps
            )
            need_loss_scalar = global_step % log_every == 0 or should_save or args.dry_run
            cur_loss = float(loss_for_step.cpu()) if need_loss_scalar else None

            if global_step % log_every == 0:
                assert cur_loss is not None
                print(
                    f"step={global_step} loss={cur_loss:.4f} "
                    f"tok/s={meter.tokens_per_sec():.0f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
            accum_loss_tensor = None

            # best はトラッキングのみ。実保存は定期 / 中断 / 最終 step に限定する.
            # 毎 step ベスト更新で save するとディスクを食いつぶすので分離.
            if should_save:
                best_loss = float(best_loss_tensor.cpu())
                is_best = bool(is_best_tensor.cpu())
                meta = CheckpointMeta(
                    global_step=global_step,
                    best_loss=best_loss,
                    config_hash=cfg_hash,
                    git_sha=git_info.get("sha") if isinstance(git_info.get("sha"), str) else None,
                    git_dirty=(
                        git_info.get("dirty") if isinstance(git_info.get("dirty"), bool) else None
                    ),
                    wandb_run_id=os.environ.get("WANDB_RUN_ID"),
                    extra={
                        "git": git_info,
                        "run": run_info,
                    },
                )
                dl_state = train_loader.state_dict() if hasattr(train_loader, "state_dict") else None
                saved_dir = ckpt.save(
                    base_model, optimizer, scheduler, dl_state, meta, config=effective_cfg,
                    is_best=is_best,
                    is_final=global_step >= total_steps,
                )
                print(f"[train] saved checkpoint @ step={global_step}{' (best)' if is_best else ''}")
                if sampling_enabled:
                    sample_at_checkpoint(saved_dir, global_step)

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
