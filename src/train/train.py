"""学習エントリポイント.

実行:
    python -m src.train.train --config configs/arbor.yaml
    python -m src.train.train --config configs/arbor.yaml --resume latest

設計方針:
- データはストリーミング (HF datasets `streaming=True` 等)。全件メモリ展開しない。
- BF16 mixed precision + Flash Attn + torch.compile + 8bit Adam で速度を稼ぐ。
- SIGINT/SIGTERM で次 step 境界に安全保存して終了 (二重押しで強制終了)。
- チェックポイントは外部 dir に safetensors + 状態一式をアトミック保存。
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from contextlib import nullcontext
import hashlib
import itertools
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# torch import / CUDA 初期化より前に効かせる必要がある env (env.sh と二重で保険).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

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


_TIMING_ENABLED = os.environ.get("ARBOR_TIMING", "0") == "1"
_TIMING_T0 = time.perf_counter()
_TIMING_LAST = _TIMING_T0


def timing_mark(label: str, device: torch.device | None = None) -> None:
    """Print coarse wall-clock timing when ARBOR_TIMING=1."""
    global _TIMING_LAST
    if not _TIMING_ENABLED:
        return
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    now = time.perf_counter()
    print(
        f"[timing] {label}: +{now - _TIMING_LAST:.3f}s total={now - _TIMING_T0:.3f}s",
        flush=True,
    )
    _TIMING_LAST = now


# --------------------------------------------------------------- 引数 / 設定
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", default=None, help="'latest' | 'best' | step | path")
    p.add_argument(
        "--init-from", default=None,
        help="checkpoint dir から重みのみ読み込んで step 0 の新規 run を開始する "
             "(optimizer/scheduler/dataloader は初期化。長コンテキスト拡張などの continued pretraining 用)",
    )
    p.add_argument("--dry-run", action="store_true", help="1 step だけ走らせて即終了")
    p.add_argument(
        "--allow-config-mismatch", action="store_true",
        help="resume 時に checkpoint の model 設定と現在の config が違っても続行する",
    )
    p.add_argument(
        "--rebase-lr-on-resume", action="store_true",
        help=(
            "resume 時に optimizer/scheduler state は復元しつつ、LR の基準値だけ "
            "現在の config の optim.lr に差し替える"
        ),
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_entropy_lm_reference(cfg: dict, arbor_config_path: Path) -> dict:
    """entropy mode の ByteLM 構成を entropy_lm.yaml から一元的に取り込む。

    Arbor 側に同じ model 定義を複製しない。inline ``model.entropy_model`` と参照を
    同時指定した場合は、どちらを採用するか曖昧なのでエラーにする。
    """
    resolved = copy.deepcopy(cfg)
    model_cfg = resolved.get("model", {})
    if model_cfg.get("arch", "arbor") != "arbor":
        return resolved
    if model_cfg.get("patching_mode", "static") != "entropy":
        return resolved

    reference = resolved.get("entropy_lm_config")
    if not reference:
        raise ValueError(
            "model.patching_mode=entropy には top-level entropy_lm_config が必要です"
        )
    if model_cfg.get("entropy_model") is not None:
        raise ValueError(
            "entropy_lm_config と model.entropy_model の二重管理は禁止です。"
            "entropy_lm_config だけを指定してください"
        )

    reference_path = Path(reference)
    if not reference_path.is_absolute():
        reference_path = arbor_config_path.resolve().parent / reference_path
    entropy_cfg = load_config(reference_path)
    entropy_model_cfg = copy.deepcopy(entropy_cfg.get("model", {}))
    if entropy_model_cfg.get("arch") != "byte_lm":
        raise ValueError(
            f"entropy_lm_config の model.arch は byte_lm 必須: {reference_path}"
        )
    entropy_model_cfg.pop("arch")
    model_cfg["entropy_model"] = entropy_model_cfg

    if not model_cfg.get("entropy_model_ckpt"):
        checkpoint_dir = entropy_cfg.get("checkpoint", {}).get("dir")
        if not checkpoint_dir:
            raise ValueError(
                f"entropy_lm_config に checkpoint.dir がありません: {reference_path}"
            )
        model_cfg["entropy_model_ckpt"] = str(Path(checkpoint_dir) / "latest")
    print(
        "[train] entropy model config loaded from "
        f"{reference_path} ckpt={model_cfg['entropy_model_ckpt']}"
    )
    return resolved


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


def should_restore_dataloader_state(saved_data_cfg: dict | None, current_data_cfg: dict) -> bool:
    """Return whether checkpoint dataloader state is compatible with current data config."""
    return saved_data_cfg is None or saved_data_cfg == current_data_cfg


def rebase_scheduler_lr(
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    base_lr: float,
) -> list[float]:
    """Keep resume progress but change the scheduler's base LR.

    Loading a checkpoint restores both optimizer param group LR and LambdaLR
    base_lrs from the checkpoint. This helper is for intentional mid-run LR
    changes: Adam moments/scheduler step are preserved, but the peak/base LR is
    replaced by the current config value.
    """
    if base_lr <= 0:
        raise ValueError(f"optim.lr must be positive: {base_lr}")
    groups = optimizer.param_groups
    for group in groups:
        group["initial_lr"] = base_lr

    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [base_lr for _ in groups]

    last_epoch = int(getattr(scheduler, "last_epoch", 0))
    lr_lambdas = getattr(scheduler, "lr_lambdas", None)
    if lr_lambdas is not None:
        lrs = [base_lr * float(fn(last_epoch)) for fn in lr_lambdas]
        if len(lrs) == 1 and len(groups) > 1:
            lrs = lrs * len(groups)
    else:
        lrs = [base_lr for _ in groups]
    if len(lrs) != len(groups):
        raise ValueError(
            f"scheduler LR count mismatch: lrs={len(lrs)} param_groups={len(groups)}"
        )

    for group, lr in zip(groups, lrs):
        group["lr"] = lr
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = list(lrs)
    return lrs


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    loaders: dict[str, object],
    device: torch.device,
    compute_dtype: torch.dtype,
    use_autocast: bool,
    max_batches: int,
    batch_cache: dict[str, list[dict[str, torch.Tensor]]] | None = None,
) -> dict[str, float]:
    """Return domain bits-per-byte plus ``mean_bpb`` for configured loaders.

    ``batch_cache`` を渡すと、各 domain の batch を初回評価時に CPU 側へ保持し
    以降はそれを再利用する。streaming loader は iter() の度に HF Hub から
    ストリームを開き直して skip_samples 分を読み飛ばすため、キャッシュ無しだと
    validation の度に数十秒〜数分のダウンロードが走り、かつ評価データが
    呼び出し毎にぶれる。
    """
    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    total_loss_sum = 0.0
    total_labels = 0
    amp_context = (
        torch.autocast(device_type=device.type, dtype=compute_dtype)
        if use_autocast
        else nullcontext()
    )
    try:
        for domain, loader in loaders.items():
            if batch_cache is not None and domain in batch_cache:
                batches: list[dict[str, torch.Tensor]] = batch_cache[domain]
            else:
                batches = []
                iterator = iter(loader)
                for _ in range(max_batches):
                    try:
                        batches.append(next(iterator))
                    except StopIteration:
                        break
                if hasattr(loader, "shutdown_workers"):
                    loader.shutdown_workers()
                if batch_cache is not None:
                    batch_cache[domain] = batches
            loss_sum = 0.0
            label_count = 0
            for batch in batches:
                inputs = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                valid = labels != -100
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    continue
                with amp_context:
                    out = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        out.logits.flatten(0, 1),
                        labels.flatten(),
                        ignore_index=-100,
                        reduction="sum",
                    )
                loss_sum += float(loss.detach().cpu())
                label_count += n_valid
            if label_count > 0:
                bpb = loss_sum / label_count / torch.log(torch.tensor(2.0)).item()
                results[f"{domain}_bpb"] = bpb
                total_loss_sum += loss_sum
                total_labels += label_count
        if total_labels > 0:
            results["mean_bpb"] = (
                total_loss_sum / total_labels / torch.log(torch.tensor(2.0)).item()
            )
        return results
    finally:
        if was_training:
            model.train()


class CudaBatchPrefetcher:
    """Move the next CPU batch to CUDA on a side stream while the current step runs."""

    def __init__(self, source_iter, device: torch.device, initial_batch: dict | None = None):
        if device.type != "cuda":
            raise ValueError("CudaBatchPrefetcher requires a CUDA device")
        self.source_iter = source_iter
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch: dict[str, torch.Tensor] | None = None
        if initial_batch is None:
            timing_mark("cuda_prefetcher_preload_start")
            self._preload()
            timing_mark("cuda_prefetcher_preload_done", device)
        else:
            timing_mark("cuda_prefetcher_stage_resume_batch_start")
            self._stage(initial_batch)
            timing_mark("cuda_prefetcher_stage_resume_batch_done", device)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        for value in batch.values():
            if torch.is_tensor(value):
                value.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch

    def _preload(self) -> None:
        try:
            batch = next(self.source_iter)
        except StopIteration:
            self.next_batch = None
            return
        self._stage(batch)

    def _stage(self, batch: dict) -> None:
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

    def state_dict(self) -> dict | None:
        if self.next_batch is None:
            return None
        self.stream.synchronize()
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in self.next_batch.items()
        }

    def close(self) -> None:
        self.stream.synchronize()
        self.next_batch = None
        self.source_iter = None


class ThreadedBatchPrefetcher:
    """DataLoader からの batch 取り出し (CPU 側の document packing) を別スレッドへ逃がす。

    num_workers=0 (正確 resume のための単一プロセス) だと packing が学習ループと
    直列になり、実測で step あたり ~235ms (=4%強) GPU を遊ばせていた。worker
    thread が最大 ``depth`` 個先読みして深さ付き queue に積む。

    正確 resume の要件: checkpoint には「loader の state_dict」と「読み出し済みで
    未消費の batch 列」が同一瞬間のペアで入らなければならない (ずれると batch の
    重複/欠落が起きる)。そのため worker の next() と state_dict() を同じ lock で
    排他する。未消費分は ``state_dict()`` が返す pending として checkpoint に保存し、
    resume 時は ``initial_batches`` で先に replay する。
    """

    def __init__(self, loader, depth: int = 3, initial_batches: list[dict] | None = None):
        self.loader = loader
        self.depth = max(1, int(depth))
        self._source = iter(loader)
        self._replay = deque(initial_batches or [])
        self._buf: deque = deque()
        self._cond = threading.Condition()
        self._stop = False
        self._exhausted = False
        self._exc: BaseException | None = None
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="cpu-batch-prefetch"
        )
        self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._cond:
                while len(self._buf) >= self.depth and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                try:
                    batch = next(self._source)
                except StopIteration:
                    self._exhausted = True
                    self._cond.notify_all()
                    return
                except BaseException as exc:  # noqa: BLE001 - 消費側で再送出
                    self._exc = exc
                    self._cond.notify_all()
                    return
                self._buf.append(batch)
                self._cond.notify_all()

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._replay:
            return self._replay.popleft()
        with self._cond:
            while not self._buf and not self._exhausted and self._exc is None:
                self._cond.wait()
            # 先読み済みの正常 batch を先に消費し、例外は発生位置で送出する
            if self._buf:
                batch = self._buf.popleft()
                self._cond.notify_all()
                return batch
            if self._exc is not None:
                raise self._exc
            raise StopIteration

    def state_dict(self) -> tuple[dict | None, list[dict]]:
        """(loader state, 未消費 batch 列) を同一瞬間のペアで返す。"""
        with self._cond:
            state = self.loader.state_dict() if hasattr(self.loader, "state_dict") else None
            pending = list(self._replay) + list(self._buf)
            return state, pending

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        # HF streaming が network 待ちで固まっている場合に永久 join しない。
        # daemon thread なので取り残してもプロセス終了は阻害しない。
        self._thread.join(timeout=10.0)


# ---------------------------------------------------------- グローバル最適化
def apply_speed_settings(speed: dict) -> None:
    """学習開始前に効かせるスループット系の設定をまとめて適用."""
    if speed.get("tf32_matmul", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if speed.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True


def pick_device(requested: str = "auto") -> torch.device:
    """実行 device を選ぶ。明示指定時は利用不能でも別 device へ落とさない。"""
    normalized = requested.lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "speed.device=cuda を指定したが CUDA は利用できません。"
                "CPU/MPS への暗黙フォールバックは行いません"
            )
        return torch.device("cuda")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            reason = (
                "PyTorch が MPS 対応でビルドされていません"
                if not torch.backends.mps.is_built()
                else "この macOS / Apple Silicon 環境で MPS を初期化できません"
            )
            raise RuntimeError(
                f"speed.device=mps を指定したが MPS は利用できません: {reason}。"
                "CPU/CUDA への暗黙フォールバックは行いません"
            )
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized != "auto":
        raise ValueError(f"unknown speed.device: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def adapt_config_for_device(cfg: dict, device: torch.device) -> dict:
    """単一 config を device の実行制約へ、意味を変えずに適合させる。

    MPS では 1B/8K の activation memory を抑えるため checkpointing と
    micro-batch=1 を使い、grad_accum を同率で増やして effective batch を保つ。
    optimizer、state_precision、モデル形状、データ混合は変更しない。
    """
    resolved = copy.deepcopy(cfg)
    model_cfg = resolved.setdefault("model", {})
    speed_cfg = resolved.setdefault("speed", {})
    attn_impl = str(model_cfg.get("global_attn_impl", "sdpa")).lower()
    if attn_impl not in {"sdpa", "flex"}:
        raise ValueError(
            f"unknown model.global_attn_impl: {attn_impl!r} "
            "(choices: sdpa | flex; auto/fallbackは禁止)"
        )
    if attn_impl == "flex" and device.type != "cuda":
        raise ValueError(
            "model.global_attn_impl=flex はCUDA専用です。"
            "暗黙フォールバックは行わないため、sdpaを明示してください"
        )
    if attn_impl == "flex" and not speed_cfg.get("torch_compile", True):
        raise ValueError(
            "model.global_attn_impl=flex には speed.torch_compile=true が必要です。"
            "暗黙フォールバックは行いません"
        )
    fp8_raw = speed_cfg.get("bitlinear_fp8", "off")
    if fp8_raw in (None, False):
        fp8_raw = "off"
    fp8_mode = str(fp8_raw).lower()
    if fp8_mode == "native":
        fp8_mode = "int8"
        speed_cfg["bitlinear_fp8"] = "int8"
    if fp8_mode not in {"off", "bwd", "full", "int8", "ternary"}:
        raise ValueError(
            f"unknown speed.bitlinear_fp8: {fp8_mode!r} "
            "(choices: off | bwd | full | int8 | ternary; auto/fallbackは禁止)"
        )
    if fp8_mode != "off" and device.type != "cuda":
        raise ValueError(
            f"speed.bitlinear_fp8={fp8_mode} はCUDA専用です。"
            "暗黙フォールバックは行わないため、offを明示してください"
        )
    int8_backend = str(speed_cfg.get("bitlinear_int8_backend", "auto")).lower()
    if int8_backend not in {"auto", "int_mm", "triton"}:
        raise ValueError(
            f"unknown speed.bitlinear_int8_backend: {int8_backend!r} "
            "(choices: auto | int_mm | triton)"
        )
    ternary_backend = str(
        speed_cfg.get("bitlinear_ternary_backend", "dot_current")
    ).lower().replace("-", "_")
    if ternary_backend in {"tl_dot", "tensor_core", "optimized"}:
        ternary_backend = "dot"
        speed_cfg["bitlinear_ternary_backend"] = "dot"
    if ternary_backend in {"current", "legacy"}:
        ternary_backend = "dot_current"
        speed_cfg["bitlinear_ternary_backend"] = "dot_current"
    if ternary_backend not in {"dot", "dot_current"}:
        raise ValueError(
            f"unknown speed.bitlinear_ternary_backend: {ternary_backend!r} "
            "(choices: dot | dot_current)"
        )
    ternary_wgrad_backend = str(
        speed_cfg.get("bitlinear_ternary_wgrad_backend", "int8")
    ).lower().replace("-", "_")
    if ternary_wgrad_backend in {"hybrid", "shape_auto"}:
        ternary_wgrad_backend = "auto"
        speed_cfg["bitlinear_ternary_wgrad_backend"] = "auto"
    if ternary_wgrad_backend not in {"int8", "fp8", "auto"}:
        raise ValueError(
            "unknown speed.bitlinear_ternary_wgrad_backend: "
            f"{ternary_wgrad_backend!r} (choices: int8 | fp8 | auto)"
        )
    compile_mode = str(speed_cfg.get("compile_mode", "default"))
    grad_accum = int(speed_cfg.get("grad_accum_steps", 1))
    if (
        device.type == "cuda"
        and speed_cfg.get("torch_compile", True)
        and grad_accum > 1
        and compile_mode in {"reduce-overhead", "max-autotune"}
    ):
        raise ValueError(
            f"speed.compile_mode={compile_mode} は CUDA Graphs を使うため "
            f"grad_accum_steps>1 (={grad_accum}) と併用不可 "
            "(gradient tensor 上書きで実行時クラッシュ)。"
            "compile_mode=default か max-autotune-no-cudagraphs を使うこと"
        )
    if device.type != "mps":
        return resolved

    if not model_cfg.get("gradient_checkpointing", False):
        model_cfg["gradient_checkpointing"] = True
        print("[train] MPS: gradient_checkpointing=ON (model shape/precision unchanged)")

    micro_batch = int(speed_cfg.get("micro_batch_size", 1))
    grad_accum = int(speed_cfg.get("grad_accum_steps", 1))
    if micro_batch > 1:
        speed_cfg["micro_batch_size"] = 1
        speed_cfg["grad_accum_steps"] = grad_accum * micro_batch
        print(
            "[train] MPS: micro_batch_size=1 grad_accum_steps={} "
            "(effective batch preserved: {} sequences)".format(
                speed_cfg["grad_accum_steps"], micro_batch * grad_accum
            )
        )

    validation_cfg = resolved.get("validation")
    if isinstance(validation_cfg, dict):
        validation_cfg["micro_batch_size"] = 1
    return resolved


def resolve_precision(name: str) -> tuple[torch.dtype, bool]:
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16, True
    if normalized in ("fp16", "float16"):
        return torch.float16, True
    if normalized in ("fp32", "float32"):
        return torch.float32, False
    if normalized in ("bf8", "float8_e5m2"):
        # bf8 は 8bit float だが、rms_norm / add / SDPA など forward の主要 op に
        # float8 の eager カーネルが無いため、compute dtype には使えない。黙って
        # 別精度に落とさず、8bit にしたい用途 (optimizer state) を明示案内する。
        raise ValueError(
            "speed.precision=bf8 は非対応です: PyTorch eager に float8 の "
            "rms_norm/elementwise/SDPA カーネルが無く、BitNet forward を計算できません。"
            "計算精度は bf16 | fp16 | fp32 から選び、8bit にしたい場合は "
            "optim.state_precision: bf8 (optimizer state) を使ってください"
        )
    raise ValueError(f"unknown speed.precision: {name} (choices: bf16 | fp16 | fp32)")


def resolve_autocast(speed: dict, default: bool) -> bool:
    """autocast の明示 override を検証する。文字列等を bool 化しない。"""
    if "autocast" not in speed:
        return default
    value = speed["autocast"]
    if not isinstance(value, bool):
        raise TypeError(f"speed.autocast must be bool, got {type(value).__name__}")
    return value


def byte_kind_loss_stats(
    losses: torch.Tensor,
    labels: torch.Tensor,
    *,
    cpu: bool = True,
) -> dict[str, float | torch.Tensor]:
    """Return count/loss sums for UTF-8 byte classes from unreduced CE losses.

    boolean インデックス (`losses[mask]`) や `.any()` は出力形状/値がデータ依存の
    ため device→host 同期を強制する。micro-batch 毎に呼ばれる関数なので、同期が
    入ると CPU の先行が毎回リセットされ GPU に launch gap が積もる (profiler 実測で
    cudaStreamSynchronize 9回/micro の主犯だった)。マスク乗算 + sum のみ:
    cpu=False では同期ゼロで tensor を返す。
    """
    flat_losses = losses.reshape(-1).detach().float()
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    byte_values = flat_labels - 4
    masks = {
        "ascii": valid & (byte_values >= 0) & (byte_values < 0x80),
        "utf8_cont": valid & (byte_values >= 0x80) & (byte_values <= 0xBF),
        "utf8_lead": valid & (byte_values >= 0xC0) & (byte_values <= 0xF7),
        "other": valid & ((byte_values >= 0xF8) | (byte_values < 0)),
    }
    out: dict[str, float | torch.Tensor] = {}
    for name, mask in masks.items():
        count = mask.sum()
        loss_sum = (flat_losses * mask).sum()
        if cpu:
            count_f = float(count.cpu())
            if count_f == 0.0:
                continue
            out[f"{name}_count"] = count_f
            out[f"{name}_loss_sum"] = float(loss_sum.cpu())
        else:
            out[f"{name}_count"] = count.detach()
            out[f"{name}_loss_sum"] = loss_sum.detach()
    return out


def apply_compile_settings(
    model: torch.nn.Module, speed: dict, device: torch.device
) -> torch.nn.Module:
    """Apply torch.compile according to speed config and return the trainable model."""
    # Arbor v2 は静的 patching で形状固定なので compile が素直に効く (既定 ON)
    if not speed.get("torch_compile", True):
        print("[train] torch_compile=OFF")
        return model
    # compile_mode=reduce-overhead / max-autotune は CUDA Graphs を有効にする。
    # gradient accumulation では複数 microbatch の backward で同じ graph を replay し、
    # 「CUDAGraphs が上書きした gradient tensor へアクセスした」実行時クラッシュを
    # 起こす (compile_mode=max-autotune + grad_accum=16 で再現済み)。黙って走らせると
    # accumulation 途中で落ちるので、config 段階で弾き cudagraph 無しの mode を案内する。
    mode = speed.get("compile_mode", "default")
    grad_accum = int(speed.get("grad_accum_steps", 1))
    if grad_accum > 1 and mode in {"reduce-overhead", "max-autotune"}:
        raise ValueError(
            f"speed.compile_mode={mode} は CUDA Graphs を使うため grad_accum_steps>1 "
            f"(={grad_accum}) と併用不可 (gradient tensor 上書きで実行時クラッシュ)。"
            "compile_mode=default か max-autotune-no-cudagraphs を使うか、"
            "grad_accum_steps=1 にすること"
        )
    # torch.compile は Inductor→(CUDA|CPU) 前提。MPS backend は codegen が不安定で
    # 落ちる/遅い。compile は結果を変えない速度最適化なので、非 CUDA では
    # semantic を変えずに OFF にする (別 optimizer/精度への置換とは異なる)。
    if device.type != "cuda":
        print(f"[train] torch_compile=OFF (device={device.type}: CUDA 以外は非対応)")
        return model
    # torch 2.5 では compile × gradient_checkpointing の併用で最初の backward
    # から loss が NaN になる (1B/小モデル・窓/密マスク・モデル全体/層単位
    # compile の全組合せで再現を確認済み)。黙って走らせると run 全体が無駄に
    # なるので起動時に弾く。
    cfg_obj = getattr(model, "cfg", None)
    uses_ckpt = bool(
        cfg_obj.get("gradient_checkpointing", False) if isinstance(cfg_obj, dict)
        else getattr(cfg_obj, "gradient_checkpointing", False)
    )
    if uses_ckpt:
        raise ValueError(
            "speed.torch_compile と model.gradient_checkpointing の併用は不可 "
            "(torch 2.5 で backward が NaN になる実測バグ)。compile を切るか "
            "micro_batch_size を下げて gradient_checkpointing を外すこと"
        )
    print(
        f"[train] torch_compile=ON mode={mode} "
        f"compile_threads={os.environ.get('TORCHINDUCTOR_COMPILE_THREADS')}"
    )
    return torch.compile(model, mode=mode)


def build_validation_model(
    base_model: torch.nn.Module,
    validation_cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Build a validation-only wrapper sharing parameters with ``base_model``.

    Training用compiled wrapperをvalidationへ流用するとtrain/evalで別graphが同じ
    compile policyに混在する。既定は安定性優先のeager。明示した場合だけtrainingと
    独立したcompile modeで別wrapperを作る。
    """
    if not validation_cfg.get("enabled", False):
        return base_model
    if not validation_cfg.get("torch_compile", False):
        print("[val] model=eager (training compiled wrapperとは分離)")
        return base_model
    if device.type != "cuda":
        print(f"[val] torch_compile=OFF (device={device.type}: CUDA 以外は非対応)")
        return base_model
    mode = str(validation_cfg.get("compile_mode", "default"))
    allowed = {
        "default",
        "lite",
        "reduce-overhead",
        "max-autotune-no-cudagraphs",
        "max-autotune",
    }
    if mode not in allowed:
        raise ValueError(
            f"unknown validation.compile_mode: {mode!r} "
            f"(choices: {sorted(allowed)})"
        )
    print(f"[val] model=compiled mode={mode} (training wrapperとは分離)")
    return torch.compile(base_model, mode=mode)


# ------------------------------------------------------------------- main
def main() -> int:
    timing_mark("process_start")
    args = parse_args()
    timing_mark("parse_args")
    cfg = load_config(args.config)
    cfg = resolve_entropy_lm_reference(cfg, args.config)
    timing_mark("load_config")
    git_info = git_metadata(_ROOT)
    timing_mark("git_metadata")

    requested_device = str(cfg.get("speed", {}).get("device", "auto"))
    device = pick_device(requested_device)
    cfg = adapt_config_for_device(cfg, device)
    torch.manual_seed(cfg.get("seed", 42))
    apply_speed_settings(cfg.get("speed", {}))
    timing_mark("seed_and_speed_settings")

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
    compute_dtype, precision_autocast = resolve_precision(
        cfg.get("speed", {}).get("precision", "bf16")
    )
    # autocast は結果に影響する compute 設定だが、MPS では autocast を挟むと
    # 実測で遅くなるため、既定は CUDA のみ ON。明示 speed.autocast で上書き可能。
    default_autocast = precision_autocast and device.type == "cuda"
    use_autocast = resolve_autocast(cfg.get("speed", {}), default_autocast)
    if use_autocast and compute_dtype == torch.float32:
        raise ValueError("speed.autocast=true と speed.precision=fp32 は併用できません")
    print(f"[train] arch={arch} precision={compute_dtype} autocast={use_autocast}")
    print("[train] building model...")
    timing_mark("before_model_build", device)
    t0 = time.perf_counter()
    model = build_model(cfg["model"]).to(device=device, dtype=compute_dtype)
    print(f"[train] model built and moved to {device} in {time.perf_counter() - t0:.1f}s")
    timing_mark("model_build_to_device", device)
    # checkpoint 保存とサンプル生成は compile 前のモデルで行う
    # (compile wrapper を保存すると state dict が _orig_mod. 付きになる)
    base_model = model
    bitnet_cache_info: dict | None = None

    # ---- 重みのみの初期化 (--init-from): 長コンテキスト拡張などの continued pretraining ----
    # RoPE バッファは非永続 (config から再計算) なので max_bytes / rope_theta が
    # 違う checkpoint でも strict ロードできる。step/optimizer/scheduler は新規。
    if args.init_from:
        if args.resume:
            raise SystemExit("[train] ERROR: --init-from と --resume は併用できない")
        from safetensors.torch import load_file as safe_load

        init_path = Path(args.init_from).resolve()
        if init_path.is_dir():
            init_path = init_path / "model.safetensors"
        if not init_path.exists():
            raise SystemExit(f"[train] ERROR: --init-from に model.safetensors が無い: {init_path}")
        t0 = time.perf_counter()
        print(f"[train] loading init weights from {init_path}...")
        weights = safe_load(str(init_path), device=str(device))
        if any(k.startswith("_orig_mod.") for k in weights):
            weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}
        base_model.load_state_dict(weights, strict=True)
        print(
            f"[train] init_from={init_path} loaded in {time.perf_counter() - t0:.1f}s "
            "(weights only; optimizer/scheduler/step は新規)"
        )

    # ---- BitNet 訓練高速化: QKV/gate-up 融合と量子化重みcache ----
    try:
        from src.model.bitlinear import (
            configure_bitlinear_training_cache,
            install_arbor_projection_fusions,
            refresh_bitlinear_training_cache,
            set_bitlinear_fp8_mode,
            set_bitlinear_int8_backend,
            set_bitlinear_ternary_backend,
            set_bitlinear_ternary_wgrad_backend,
        )
    except Exception:  # pragma: no cover - bitnet 無効構成でも学習は継続
        refresh_bitlinear_training_cache = None
    else:
        speed_cfg = cfg.get("speed", {})
        # modeを先に設定し、cache allocatorがINT8/FP8両layoutの実コストを使う。
        install_arbor_projection_fusions(base_model)
        fp8_raw = speed_cfg.get("bitlinear_fp8", "off")
        if fp8_raw in (None, False):
            fp8_raw = "off"
        int8_backend = set_bitlinear_int8_backend(
            str(speed_cfg.get("bitlinear_int8_backend", "auto"))
        )
        ternary_backend = set_bitlinear_ternary_backend(
            str(speed_cfg.get("bitlinear_ternary_backend", "dot"))
        )
        ternary_wgrad_backend = set_bitlinear_ternary_wgrad_backend(
            str(speed_cfg.get("bitlinear_ternary_wgrad_backend", "int8"))
        )
        fp8_info = set_bitlinear_fp8_mode(base_model, str(fp8_raw))
        bitnet_cache_info = configure_bitlinear_training_cache(
            base_model,
            enabled=speed_cfg.get("bitnet_weight_cache", "auto"),
            grad_accum_steps=int(speed_cfg.get("grad_accum_steps", 1)),
            max_cache_gib=speed_cfg.get("bitnet_weight_cache_gib", 1.25),
            min_numel=int(speed_cfg.get("bitnet_weight_cache_min_numel", 65536)),
        )
        print(
            "[train] bitnet_weight_cache="
            f"{bitnet_cache_info['mode']} enabled={bitnet_cache_info['enabled']} "
            f"cached_layers={bitnet_cache_info['cached_layers']}/"
            f"{bitnet_cache_info['eligible_layers']} "
            f"fused_groups={bitnet_cache_info['fused_groups']} "
            f"cache={bitnet_cache_info['cache_gib']:.2f}GiB "
            f"format={bitnet_cache_info['cache_format']} "
            f"qkv_groups={bitnet_cache_info['qkv_groups']} "
            f"gate_up_groups={bitnet_cache_info['gate_up_groups']}"
        )
        print(
            f"[train] bitlinear_fp8={fp8_info['mode']} "
            f"int8_backend={int8_backend} "
            f"ternary_backend={ternary_backend} "
            f"ternary_wgrad_backend={ternary_wgrad_backend} "
            f"layers={fp8_info['layers']}"
        )

    model = apply_compile_settings(model, cfg["speed"], device)
    validation_model = build_validation_model(
        base_model, cfg.get("validation", {}), device
    )
    timing_mark("compile_wrapper_created", device)

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
    data_cfg.setdefault("seed", cfg.get("seed", 42))
    # static patching では document 境界を patch 境界へ align しないと、straddle patch
    # 内で 2 文書が混ざり local 階層の document isolation が壊れる。model.patch_size を
    # dataloader へ渡し、document packing 側で新文書を patch 境界から始めさせる。
    # 動的 patching (utf8/space/entropy) は境界がデータ依存なので align しない。
    model_cfg = cfg.get("model", {})
    if str(model_cfg.get("patching_mode", "static")) == "static":
        data_cfg.setdefault("patch_align", int(model_cfg.get("patch_size", 1)))
    # pinned host memory は CUDA の H2D 転送専用の最適化。非 CUDA では効果が無く
    # DataLoader が警告を出すだけなので、結果を変えない範囲で OFF にする。
    if device.type != "cuda" and data_cfg.get("pin_memory", False):
        print(f"[train] pin_memory=OFF (device={device.type}: CUDA 以外は無効)")
        data_cfg["pin_memory"] = False
    train_loader = build_byte_dataloader(data_cfg, split="train")
    timing_mark("dataloader_object_created", device)

    validation_cfg = cfg.get("validation", {})
    validation_loaders: dict[str, object] = {}
    validation_enabled = bool(validation_cfg.get("enabled", False))
    if validation_enabled:
        domains = validation_cfg.get("domains", {})
        if not domains:
            raise SystemExit("[train] ERROR: validation.enabled=true だが validation.domains が空")
        val_micro_batch = int(validation_cfg.get("micro_batch_size", data_cfg["micro_batch_size"]))
        for domain_name, domain_cfg in domains.items():
            val_data_cfg = dict(domain_cfg)
            val_data_cfg.setdefault("context_length", data_cfg["context_length"])
            # validation domain は素のテキストコーパス (忘却監視用)。学習側が
            # packing: sft でも、それを継承すると _iter_hf_sft が conversations 列の
            # 無い行から SFT サンプルを作れず有効サンプルを探して無限ストリームする。
            # よって sft は必ず text packing に落とす (domain_cfg で明示指定は尊重)。
            base_packing = data_cfg.get("packing", "concat")
            if base_packing == "sft":
                base_packing = "document"
            val_data_cfg.setdefault("packing", base_packing)
            val_data_cfg.setdefault("byte_offset", data_cfg.get("byte_offset", 4))
            val_data_cfg.setdefault("eos_token_id", data_cfg.get("eos_token_id", 2))
            val_data_cfg.setdefault("pad_token_id", data_cfg.get("pad_token_id", 3))
            val_data_cfg.setdefault("shuffle_buffer", 0)
            val_data_cfg.setdefault("num_workers", 0)
            val_data_cfg.setdefault("pin_memory", data_cfg.get("pin_memory", True))
            val_data_cfg.setdefault("micro_batch_size", val_micro_batch)
            val_data_cfg.setdefault("seed", cfg.get("seed", 42) + 10_000)
            if "patch_align" in data_cfg:
                val_data_cfg.setdefault("patch_align", data_cfg["patch_align"])
            validation_loaders[domain_name] = build_byte_dataloader(val_data_cfg, split="validation")
        # 初回 validation で読んだ batch を保持して再利用する (domain 毎 ~17MB)。
        # 2 回目以降はネットワークアクセス無し・毎回同一データで bpb を比較できる。
        validation_batch_cache: dict[str, list[dict[str, torch.Tensor]]] = {}
        print(
            "[train] validation=ON domains={} max_batches={} best_metric=mean_bpb".format(
                ",".join(validation_loaders.keys()),
                int(validation_cfg.get("max_batches", 16)),
            )
        )

    # ---- optimizer / scheduler ----
    optimizer = build_optimizer(model.parameters(), cfg["optim"])
    timing_mark("optimizer_created", device)
    scheduler = build_scheduler(optimizer, cfg["optim"])
    timing_mark("scheduler_created", device)

    # ---- チェックポイント ----
    ckpt_cfg = cfg["checkpoint"]
    ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", ckpt_cfg["dir"]))
    ckpt = CheckpointManager(
        ckpt_dir,
        keep_last_k=ckpt_cfg.get("keep_last_k", 3),
        keep_every_n_steps=ckpt_cfg.get("keep_every_n_steps"),
        async_save=ckpt_cfg.get("async_save", True),
    )
    # loss/ema/lr の時系列 (log_every_steps ごとに 1 行追記)。resume 時は追記継続
    # なので、巻き戻した場合は同じ step が重複しうる (プロット時は後勝ちで dedup)。
    metrics_path = ckpt_dir / "metrics.jsonl"

    # ---- 再開処理 ----
    global_step = 0
    best_loss = float("inf")
    if args.resume:
        # model 形状が違う checkpoint を strict=False で黙って部分ロードする事故を防ぐ.
        # checkpoint には保存時の実効 config が入っているので model 節を突き合わせる.
        resolved = ckpt.resolve(args.resume)
        saved_cfg_file = resolved / "config.yaml" if resolved else None
        saved_data_cfg = None
        saved_optim_cfg = None
        if saved_cfg_file is not None and saved_cfg_file.exists():
            saved_cfg = yaml.safe_load(saved_cfg_file.read_text()) or {}
            saved_model_cfg = saved_cfg.get("model", {})
            saved_data_cfg = saved_cfg.get("data")
            saved_optim_cfg = saved_cfg.get("optim")
            if saved_model_cfg and saved_model_cfg != cfg["model"]:
                diff_keys = sorted(
                    k for k in set(saved_model_cfg) | set(cfg["model"])
                    if saved_model_cfg.get(k) != cfg["model"].get(k)
                )
                compatible_diff_keys = {"max_patches"}
                blocking_diff_keys = [k for k in diff_keys if k not in compatible_diff_keys]
                msg = (
                    f"checkpoint の model 設定と現在の config が不一致: {diff_keys}. "
                    f"再開するなら `--config {saved_cfg_file}` を使うか、"
                    "意図的なら --allow-config-mismatch を付ける。"
                )
                if blocking_diff_keys and not args.allow_config_mismatch:
                    raise SystemExit(f"[train] ERROR: {msg}")
                if blocking_diff_keys:
                    print(f"[train] WARNING: {msg}")
                else:
                    print(
                        "[train] compatible model config diff allowed on resume: "
                        f"{diff_keys}"
                    )
        timing_mark("before_checkpoint_load", device)
        t0 = time.perf_counter()
        print(f"[train] loading checkpoint resume={args.resume}...")
        meta, dl_state = ckpt.load(args.resume, base_model, optimizer, scheduler, map_location=device)
        print(f"[train] checkpoint loaded in {time.perf_counter() - t0:.1f}s")
        timing_mark("checkpoint_loaded", device)
        optim_diff_keys = (
            sorted(
                k for k in set(saved_optim_cfg or {}) | set(cfg["optim"])
                if (saved_optim_cfg or {}).get(k) != cfg["optim"].get(k)
            )
            if saved_optim_cfg
            else []
        )
        config_base_lr = float(cfg["optim"]["lr"])
        loaded_base_lrs = [float(v) for v in getattr(scheduler, "base_lrs", [])]
        loaded_current_lrs = [float(v) for v in scheduler.get_last_lr()]
        lr_base_mismatch = any(
            abs(v - config_base_lr) > max(1e-12, abs(config_base_lr) * 1e-9)
            for v in loaded_base_lrs
        )
        if args.rebase_lr_on_resume:
            lrs = rebase_scheduler_lr(optimizer, scheduler, config_base_lr)
            print(
                "[train] rebase_lr_on_resume=ON "
                f"optim_diff={optim_diff_keys} "
                f"old_base_lrs={[f'{v:.3e}' for v in loaded_base_lrs]} "
                f"old_current_lr={loaded_current_lrs[0]:.3e} "
                f"base_lr={config_base_lr:.3e} current_lr={lrs[0]:.3e}"
            )
        elif optim_diff_keys or lr_base_mismatch:
            print(
                "[train] WARNING: checkpoint の optimizer/scheduler state は復元済み。"
                f" optim_diff={optim_diff_keys} "
                f"checkpoint_base_lrs={[f'{v:.3e}' for v in loaded_base_lrs]} "
                f"config_lr={config_base_lr:.3e} current_lr={loaded_current_lrs[0]:.3e}. "
                "LR だけ変えて続行するなら --rebase-lr-on-resume を付ける。"
            )
        if refresh_bitlinear_training_cache is not None:
            refreshed = refresh_bitlinear_training_cache(base_model)
            if refreshed:
                print(f"[train] bitnet_weight_cache refreshed after resume layers={refreshed}")
        global_step = meta.global_step
        best_loss = meta.best_loss
        pending_prefetch_batch = None
        pending_cpu_batches = None
        if dl_state is not None:
            if isinstance(dl_state, dict):
                pending_prefetch_batch = dl_state.pop("_cuda_prefetch_next_batch", None)
                pending_cpu_batches = dl_state.pop("_cpu_prefetch_pending", None)
                if pending_prefetch_batch is not None or pending_cpu_batches:
                    print(
                        "[train] resume in-flight batches: cuda_staged={} cpu_pending={}".format(
                            int(pending_prefetch_batch is not None),
                            len(pending_cpu_batches or []),
                        )
                    )
            if not should_restore_dataloader_state(saved_data_cfg, data_cfg):
                print(
                    "[train] WARNING: checkpoint の data 設定が現在の config と不一致のため "
                    "dataloader state は復元しない。model/optimizer/scheduler は resume し、"
                    "新しいデータ混合は先頭から開始する。"
                )
                pending_prefetch_batch = None
                pending_cpu_batches = None
            else:
                train_loader.load_state_dict(dl_state)
        print(f"[train] resumed from step={global_step}, best_loss={best_loss:.4f}")
    else:
        pending_prefetch_batch = None
        pending_cpu_batches = None

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
    sync_each_step = bool(cfg["speed"].get("sync_each_step", False))
    cuda_prefetch = bool(cfg["speed"].get("cuda_prefetch", False)) and device.type == "cuda"
    # CPU 側 packing の先読み深さ。num_workers>0 なら DataLoader が既に並列なので無効。
    cpu_prefetch_depth = int(cfg["speed"].get("cpu_prefetch_depth", 3))
    if int(data_cfg.get("num_workers", 0)) != 0:
        cpu_prefetch_depth = 0
    log_every = cfg["logging"].get("log_every_steps", 20)
    byte_kind_metrics = bool(cfg["logging"].get("byte_kind_metrics", False))
    total_steps = cfg["optim"]["total_steps"]
    micro_batch = data_cfg.get("micro_batch_size")
    context_length = data_cfg.get("context_length")
    if micro_batch and context_length:
        bytes_per_update = int(micro_batch) * int(context_length) * int(grad_accum)
        print(
            "[train] throughput_meter="
            f"optimizer_step rolling_window={meter.window} log_every={log_every} "
            f"micro_batch={micro_batch} grad_accum={grad_accum} "
            f"context_length={context_length} bytes_per_update={bytes_per_update}"
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
    steady_after_steps = int(cfg["logging"].get("steady_after_steps", meter.window))
    profile_sections_every = int(cfg["logging"].get("profile_sections_every_steps", 0))
    print(
        "[train] note=early bytes/s includes compile/warmup; "
        "use phase=steady logs for throughput decisions"
    )
    if profile_sections_every > 0:
        print(
            "[train] profile_sections=ON "
            f"every={profile_sections_every} optimizer steps "
            "(one no-grad patching probe; Arbor_ms is estimated from compiled forward time)"
        )

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
        if bool(sampling_cfg.get("use_cache", False)):
            print("[train] sampling.use_cache=ON")

    # checkpoint 保存時の固定 good/bad target probe (任意)。
    probes_cfg = cfg.get("probes", {})
    probes_enabled = bool(probes_cfg.get("enabled", False))
    if probes_enabled:
        probe_items = probes_cfg.get("items", [])
        if not probe_items:
            raise SystemExit("[train] ERROR: probes.enabled=true だが probes.items が空")
        print(f"[train] probes=ON items={len(probe_items)}")

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
                use_cache=bool(sampling_cfg.get("use_cache", False)),
            )
            lines = [f"# step {step}"]
            for prompt, text in samples:
                print(f"[sample] step={step} prompt={prompt!r} -> {text!r}")
                lines.append(f"\n## prompt: {prompt}\n{text}")
            # checkpoint がまだ background thread で書き込み中なら、ここで待つ
            # (step_dir はそれまで存在しない)。待ち時間は generate_samples の
            # GPU 処理と write の重なりぶん相殺されるので、通常はほぼ即座に返る。
            ckpt.wait_for_pending_save()
            (step_dir / "samples.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"[sample] wrote {step_dir / 'samples.txt'} in {time.perf_counter() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001 - サンプル生成失敗で学習は止めない
            print(f"[sample] generation failed (continuing training): {type(e).__name__}: {e}")
        finally:
            base_model.train()

    def probes_at_checkpoint(step_dir: Path, step: int) -> dict[str, dict[str, float]] | None:
        from src.eval.probes import run_text_probes

        try:
            t0 = time.perf_counter()
            results = run_text_probes(
                base_model,
                probes_cfg.get("items", []),
                device=device,
                dtype=compute_dtype,
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step,
                    "probe": results,
                    "time": time.time(),
                }, ensure_ascii=False) + "\n")
            ckpt.wait_for_pending_save()
            (step_dir / "probes.json").write_text(
                json.dumps({"step": step, "probe": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            parts = " ".join(
                f"{name}:good={vals['good_bpb']:.4f} "
                f"bad_min={vals['bad_min_bpb']:.4f} margin={vals['margin']:.4f}"
                for name, vals in results.items()
            )
            print(f"[probe] step={step} {parts} elapsed={time.perf_counter() - t0:.1f}s")
            return results
        except Exception as e:  # noqa: BLE001 - probe 失敗で学習は止めない
            print(f"[probe] failed (continuing training): {type(e).__name__}: {e}")
            return None

    def make_checkpoint_meta(
        *,
        step: int,
        best_value: float,
        validation_results: dict[str, float] | None,
        validation_status: str,
    ) -> CheckpointMeta:
        return CheckpointMeta(
            global_step=step,
            best_loss=best_value,
            config_hash=cfg_hash,
            git_sha=(
                git_info.get("sha")
                if isinstance(git_info.get("sha"), str)
                else None
            ),
            git_dirty=(
                git_info.get("dirty")
                if isinstance(git_info.get("dirty"), bool)
                else None
            ),
            wandb_run_id=os.environ.get("WANDB_RUN_ID"),
            extra={
                "git": git_info,
                "run": run_info,
                "best_metric": (
                    "validation_mean_bpb" if validation_enabled else "train_ema_loss"
                ),
                "validation": validation_results,
                "validation_status": validation_status,
            },
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss_tensor: torch.Tensor | None = None
    best_loss_tensor = torch.tensor(best_loss, device=device, dtype=torch.float32)
    # validation が無効な場合は EMA train loss を best に使う。validation が
    # 有効なら checkpoint 保存時の mean bpb だけで best を更新する。
    ema_loss_tensor: torch.Tensor | None = None
    ema_decay = float(cfg["logging"].get("best_ema_decay", 0.98))
    # 「前回保存以降に best (EMA 最小) が更新されたか」。保存 step 単発の判定だと
    # 保存間に更新があっても symlink が動かない (best が古い step を指し続ける)
    best_improved_tensor = torch.tensor(False, device=device)
    interval_t0 = time.perf_counter()
    interval_bytes = 0
    interval_patches_tensor: torch.Tensor | None = None
    interval_max_patch_tensor: torch.Tensor | None = None
    interval_fill_ratio_tensor: torch.Tensor | None = None
    interval_fill_samples = 0
    interval_byte_kind: dict[str, torch.Tensor] = {}
    interval_cpu_ms = {
        "batch_wait": 0.0,
        "h2d": 0.0,
        "step": 0.0,
    }
    cuda_records: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "forward": [],
        "backward": [],
        "optimizer": [],
    }
    cpu_records_ms = {
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    latest_section_profile: dict[str, float] | None = None
    stop_notice_printed = False
    interval_steps = 0
    logs_emitted = 0

    def start_gpu_section(name: str):
        if device.type != "cuda":
            return time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def end_gpu_section(name: str, start) -> None:
        if device.type != "cuda":
            cpu_records_ms[name] += (time.perf_counter() - start) * 1000.0
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        cuda_records[name].append((start, end))

    def collect_section_ms() -> dict[str, float]:
        if device.type != "cuda":
            values = dict(cpu_records_ms)
            for key in cpu_records_ms:
                cpu_records_ms[key] = 0.0
            return values
        values = {
            name: sum(start.elapsed_time(end) for start, end in records)
            for name, records in cuda_records.items()
        }
        for records in cuda_records.values():
            records.clear()
        return values

    cpu_prefetcher: ThreadedBatchPrefetcher | None = None

    def make_data_iter():
        nonlocal pending_prefetch_batch, pending_cpu_batches, cpu_prefetcher
        timing_mark("make_data_iter_start", device)
        if cpu_prefetcher is not None:
            cpu_prefetcher.close()
            cpu_prefetcher = None
        if cpu_prefetch_depth > 0:
            cpu_prefetcher = ThreadedBatchPrefetcher(
                train_loader, depth=cpu_prefetch_depth, initial_batches=pending_cpu_batches
            )
            source_iter = cpu_prefetcher
        elif pending_cpu_batches:
            # checkpoint に未消費 batch が残っている状態で cpu_prefetch を無効化して
            # resume した場合もデータを落とさない
            source_iter = itertools.chain(iter(pending_cpu_batches), iter(train_loader))
        else:
            source_iter = iter(train_loader)
        pending_cpu_batches = None
        timing_mark("train_loader_iter_created", device)
        if not cuda_prefetch:
            return source_iter
        initial_batch = pending_prefetch_batch
        pending_prefetch_batch = None
        data_iter = CudaBatchPrefetcher(source_iter, device, initial_batch=initial_batch)
        timing_mark("make_data_iter_done", device)
        return data_iter

    if cuda_prefetch:
        print("[train] cuda_prefetch=ON")
    if cpu_prefetch_depth > 0:
        print(f"[train] cpu_prefetch=ON depth={cpu_prefetch_depth}")

    # ---- 単発 torch.profiler (ARBOR_TORCH_PROFILE="wait,active") ----
    # 例: ARBOR_TORCH_PROFILE=25,2 → 25 optimizer step 待って 2 step 分の
    # CPU+CUDA トレースを logs/ に chrome trace として書き出し、以後は通常続行。
    # optimizer 境界の GPU アイドル (dmon で sm が 1 step 毎に落ちる) の調査用。
    prof_spec = os.environ.get("ARBOR_TORCH_PROFILE")
    prof_wait: int | None = None
    prof_active = 0
    if prof_spec:
        try:
            prof_wait, prof_active = (int(x) for x in prof_spec.split(","))
            print(f"[train] torch profiler armed: wait={prof_wait} active={prof_active}")
        except ValueError:
            print(f"[train] WARNING: ARBOR_TORCH_PROFILE='{prof_spec}' は 'wait,active' 形式でないため無視")
            prof_wait = None
    torch_prof = None
    prof_steps_done = 0
    if sync_each_step:
        print("[train] sync_each_step=ON")

    data_iter = make_data_iter()
    while global_step < total_steps:
        try:
            if prof_wait is not None and torch_prof is None and prof_steps_done == prof_wait:
                try:
                    torch.cuda.synchronize()
                    torch_prof = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                    )
                    torch_prof.__enter__()
                    print(f"[train] torch profiler started @ step={global_step}", flush=True)
                except Exception as exc:  # noqa: BLE001 - 調査用機能で学習は止めない
                    print(f"[train] WARNING: torch profiler start failed: {exc}", flush=True)
                    torch_prof = None
                    prof_wait = None
            step_t0 = time.perf_counter()
            bytes_this_step = 0
            for micro in range(grad_accum):
                if global_step == 0:
                    timing_mark(f"step0_micro{micro}_before_next_batch", device)
                t0 = time.perf_counter()
                batch = next(data_iter)
                interval_cpu_ms["batch_wait"] += (time.perf_counter() - t0) * 1000.0
                if global_step == 0:
                    timing_mark(f"step0_micro{micro}_batch_ready", device)
                t0 = time.perf_counter()
                if cuda_prefetch:
                    inputs = batch["input_ids"]
                    labels = batch["labels"]
                else:
                    inputs = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                interval_cpu_ms["h2d"] += (time.perf_counter() - t0) * 1000.0
                if global_step == 0:
                    timing_mark(f"step0_micro{micro}_batch_on_device", device)
                bytes_this_step += inputs.numel()
                interval_bytes += inputs.numel()
                if "fill_ratio" in batch:
                    fill_ratio = batch["fill_ratio"].detach().float()
                    interval_fill_ratio_tensor = (
                        fill_ratio.sum()
                        if interval_fill_ratio_tensor is None
                        else interval_fill_ratio_tensor + fill_ratio.sum()
                    )
                    interval_fill_samples += int(fill_ratio.numel())
                amp_context = (
                    torch.autocast(device_type=device.type, dtype=compute_dtype)
                    if use_autocast
                    else nullcontext()
                )
                do_section_profile = (
                    profile_sections_every > 0
                    and (logs_emitted == 0 or (global_step + 1) % profile_sections_every == 0)
                    and micro == 0
                    and hasattr(base_model, "profile_patching_sections")
                )
                with amp_context:
                    if global_step == 0:
                        timing_mark(f"step0_micro{micro}_before_forward", device)
                    if do_section_profile:
                        try:
                            with torch.no_grad():
                                latest_section_profile = base_model.profile_patching_sections(inputs)
                        except RuntimeError as exc:
                            latest_section_profile = {"profile_error": str(exc)[:200]}
                            print(f"[train] WARNING: section profile failed: {exc}", flush=True)
                    fwd_start = start_gpu_section("forward")
                    out = model(inputs)
                    end_gpu_section("forward", fwd_start)
                    if global_step == 0:
                        timing_mark(f"step0_micro{micro}_forward_done", device)
                    if byte_kind_metrics:
                        flat_losses = torch.nn.functional.cross_entropy(
                            out.logits.flatten(0, 1),
                            labels.flatten(),
                            ignore_index=-100,
                            reduction="none",
                        )
                        # flat_losses[valid].mean() は boolean インデックスで
                        # device→host 同期するため、マスク乗算 + sum で同値を取る
                        valid_f = (labels.flatten() != -100).to(flat_losses.dtype)
                        loss = (flat_losses * valid_f).sum() / (
                            valid_f.sum().clamp_min(1.0) * grad_accum
                        )
                        stats = byte_kind_loss_stats(flat_losses, labels, cpu=False)
                        for key, value in stats.items():
                            if not torch.is_tensor(value):
                                value = torch.tensor(value, device=device)
                            interval_byte_kind[key] = interval_byte_kind.get(key, value.new_zeros(())) + value
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            out.logits.flatten(0, 1), labels.flatten(), ignore_index=-100
                        ) / grad_accum
                if out.patch_count is not None:
                    # compile_mode=reduce-overhead (CUDA Graphs) では次 replay で
                    # グラフ出力バッファが上書きされるため、保持する前に clone が必要
                    pc = out.patch_count.detach().clone()
                    interval_patches_tensor = pc if interval_patches_tensor is None else interval_patches_tensor + pc
                if out.max_patch_count is not None:
                    max_pc = out.max_patch_count.detach().clone()
                    interval_max_patch_tensor = (
                        max_pc
                        if interval_max_patch_tensor is None
                        else torch.maximum(interval_max_patch_tensor, max_pc)
                    )
                if global_step == 0:
                    timing_mark(f"step0_micro{micro}_before_backward", device)
                bwd_start = start_gpu_section("backward")
                loss.backward()
                end_gpu_section("backward", bwd_start)
                if global_step == 0:
                    timing_mark(f"step0_micro{micro}_backward_done", device)
                detached_loss = loss.detach()
                accum_loss_tensor = (
                    detached_loss
                    if accum_loss_tensor is None
                    else accum_loss_tensor + detached_loss
                )
                if stop.requested and not stop_notice_printed:
                    remaining = grad_accum - micro - 1
                    print(
                        "[train] stop requested during accumulation: "
                        f"step={global_step + 1} micro={micro + 1}/{grad_accum}; "
                        f"finishing {remaining} remaining microbatches before optimizer/save. "
                        "Press again to force-exit.",
                        flush=True,
                    )
                    stop_notice_printed = True

            if cfg["optim"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["optim"]["grad_clip"]
                )
            opt_start = start_gpu_section("optimizer")
            optimizer.step()
            if refresh_bitlinear_training_cache is not None:
                refresh_bitlinear_training_cache(base_model)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            end_gpu_section("optimizer", opt_start)

            global_step += 1
            interval_steps += 1
            if device.type == "cuda" and sync_each_step:
                torch.cuda.synchronize()
            meter.step(bytes_this_step)
            interval_cpu_ms["step"] += (time.perf_counter() - step_t0) * 1000.0

            prof_steps_done += 1
            if (
                torch_prof is not None
                and prof_wait is not None
                and prof_steps_done >= prof_wait + prof_active
            ):
                try:
                    torch.cuda.synchronize()
                    torch_prof.__exit__(None, None, None)
                    trace_path = Path("logs") / f"torch_profile_step{global_step}.json"
                    torch_prof.export_chrome_trace(str(trace_path))
                    print(f"[train] torch profiler trace written: {trace_path}", flush=True)
                    print(
                        torch_prof.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=30
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[train] WARNING: torch profiler export failed: {exc}", flush=True)
                finally:
                    torch_prof = None
                    prof_wait = None

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
            if not validation_enabled:
                is_best_tensor = ema_loss_tensor < best_loss_tensor
                best_improved_tensor = best_improved_tensor | is_best_tensor
                best_loss_tensor = torch.minimum(best_loss_tensor, ema_loss_tensor)

            should_save = (
                global_step % save_every == 0
                or stop.requested
                or global_step >= total_steps
            )
            need_loss_scalar = global_step % log_every == 0 or should_save or args.dry_run
            cur_loss = float(loss_for_step.cpu()) if need_loss_scalar else None

            if need_loss_scalar:
                assert cur_loss is not None
                cur_ema = float(ema_loss_tensor.cpu())
                cur_lr = scheduler.get_last_lr()[0]
                section_ms = collect_section_ms()
                interval_dt = max(time.perf_counter() - interval_t0, 1e-9)
                cur_bytes_s = interval_bytes / interval_dt
                patch_count = (
                    float(interval_patches_tensor.cpu())
                    if interval_patches_tensor is not None
                    else 0.0
                )
                cur_patches_s = patch_count / interval_dt if patch_count > 0 else 0.0
                bytes_per_patch = interval_bytes / patch_count if patch_count > 0 else 0.0
                seq_count = (
                    interval_bytes / int(context_length)
                    if context_length
                    else 0.0
                )
                patches_per_seq = patch_count / seq_count if seq_count > 0 else 0.0
                max_patch_per_seq = (
                    float(interval_max_patch_tensor.cpu())
                    if interval_max_patch_tensor is not None
                    else 0.0
                )
                phase = (
                    "steady"
                    if global_step >= steady_after_steps and logs_emitted >= 2
                    else "warmup"
                )
                patch_capacity = 0.0
                max_patches = getattr(base_model, "max_patches", None)
                if max_patches and context_length:
                    patch_capacity = (interval_bytes / int(context_length)) * int(max_patches)
                patch_util = patch_count / patch_capacity if patch_capacity > 0 else 0.0
                max_patch_util = (
                    max_patch_per_seq / int(max_patches)
                    if max_patches
                    else 0.0
                )
                avg_fill_ratio = (
                    float(interval_fill_ratio_tensor.cpu()) / interval_fill_samples
                    if interval_fill_ratio_tensor is not None and interval_fill_samples > 0
                    else 1.0
                )
                patch_headroom = (
                    int(max_patches) - max_patch_per_seq
                    if max_patches
                    else 0.0
                )
                denom_steps = max(interval_steps, 1)
                fwd_ms = section_ms.get("forward", 0.0) / denom_steps
                bwd_ms = section_ms.get("backward", 0.0) / denom_steps
                opt_ms = section_ms.get("optimizer", 0.0) / denom_steps
                batch_ms = interval_cpu_ms["batch_wait"] / denom_steps
                h2d_ms = interval_cpu_ms["h2d"] / denom_steps
                step_ms = interval_cpu_ms["step"] / denom_steps
                profile_text = ""
                if latest_section_profile:
                    forward_micro_ms = fwd_ms / max(int(grad_accum), 1)
                    if "arbor_ms" not in latest_section_profile and "profile_error" not in latest_section_profile:
                        measured_overhead = (
                            latest_section_profile.get("bytelm_ms", 0.0)
                            + latest_section_profile.get("patching_ms", 0.0)
                        )
                        latest_section_profile["arbor_ms"] = max(0.0, forward_micro_ms - measured_overhead)
                    profile_text = (
                        " "
                        f"ByteLM_ms={latest_section_profile.get('bytelm_ms', 0.0):.1f}"
                        f" patching_ms={latest_section_profile.get('patching_ms', 0.0):.1f}"
                        f" Arbor_ms={latest_section_profile.get('arbor_ms', 0.0):.1f}"
                    )
                    if "profile_error" in latest_section_profile:
                        profile_text += " profile_error=1"
                source_stats = (
                    train_loader.source_stats()
                    if hasattr(train_loader, "source_stats")
                    else None
                )
                source_byte_ratios = None
                byte_kind_bpb = None
                if byte_kind_metrics and interval_byte_kind:
                    log2 = torch.log(torch.tensor(2.0)).item()
                    byte_kind_bpb = {}
                    for name in ("ascii", "utf8_lead", "utf8_cont", "other"):
                        count_tensor = interval_byte_kind.get(f"{name}_count")
                        loss_tensor = interval_byte_kind.get(f"{name}_loss_sum")
                        count = float(count_tensor.cpu()) if count_tensor is not None else 0.0
                        loss_sum = float(loss_tensor.cpu()) if loss_tensor is not None else 0.0
                        if count > 0:
                            byte_kind_bpb[f"{name}_bpb"] = round(loss_sum / count / log2, 6)
                            byte_kind_bpb[f"{name}_count"] = int(count)
                if source_stats:
                    emitted = source_stats.get("emitted_source_bytes") or []
                    total_emitted = sum(float(v) for v in emitted)
                    if total_emitted > 0:
                        source_byte_ratios = {
                            str(name): round(float(byte_count) / total_emitted, 6)
                            for name, byte_count in zip(
                                source_stats.get("source_names") or [], emitted
                            )
                        }
                print(
                    f"step={global_step} loss={cur_loss:.4f} ema={cur_ema:.4f} "
                    f"bytes/s={cur_bytes_s:.0f} patches/s={cur_patches_s:.0f} "
                    f"bytes/patch={bytes_per_patch:.2f} patches/seq={patches_per_seq:.0f} "
                    f"max_patch/seq={max_patch_per_seq:.0f} patch_headroom={patch_headroom:.0f} "
                    f"patch_util={patch_util * 100:.1f}% max_patch_util={max_patch_util * 100:.1f}% "
                    f"pack_fill={avg_fill_ratio * 100:.1f}% "
                    f"fwd_ms={fwd_ms:.1f} bwd_ms={bwd_ms:.1f} opt_ms={opt_ms:.1f} "
                    f"batch_ms={batch_ms:.1f} h2d_ms={h2d_ms:.1f} step_ms={step_ms:.1f} "
                    f"phase={phase} lr={cur_lr:.2e}{profile_text}"
                )
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": global_step,
                        "loss": round(cur_loss, 6),
                        "ema": round(cur_ema, 6),
                        "lr": cur_lr,
                        "bytes_s": round(cur_bytes_s),
                        "patches_s": round(cur_patches_s),
                        "bytes_per_patch": round(bytes_per_patch, 6),
                        "patches_per_seq": round(patches_per_seq, 6),
                        "max_patch_per_seq": round(max_patch_per_seq, 6),
                        "patch_headroom": round(patch_headroom, 6),
                        "patch_util": round(patch_util, 6),
                        "max_patch_util": round(max_patch_util, 6),
                        "pack_fill": round(avg_fill_ratio, 6),
                        "patch_capacity": round(patch_capacity, 3),
                        "fwd_ms": round(fwd_ms, 3),
                        "bwd_ms": round(bwd_ms, 3),
                        "opt_ms": round(opt_ms, 3),
                        "batch_ms": round(batch_ms, 3),
                        "h2d_ms": round(h2d_ms, 3),
                        "step_ms": round(step_ms, 3),
                        "phase": phase,
                        "section_profile": latest_section_profile,
                        "byte_kind_bpb": byte_kind_bpb,
                        "source_byte_ratios": source_byte_ratios,
                        "source_stats": source_stats,
                        "time": time.time(),
                    }) + "\n")
                interval_t0 = time.perf_counter()
                interval_bytes = 0
                interval_steps = 0
                interval_patches_tensor = None
                interval_max_patch_tensor = None
                interval_fill_ratio_tensor = None
                interval_fill_samples = 0
                interval_byte_kind = {}
                logs_emitted += 1
                for key in interval_cpu_ms:
                    interval_cpu_ms[key] = 0.0
                latest_section_profile = None
            accum_loss_tensor = None

            # best はトラッキングのみ。実保存は定期 / 中断 / 最終 step に限定する.
            # 毎 step ベスト更新で save するとディスクを食いつぶすので分離.
            if should_save:
                stop_save = stop.requested
                validation_results: dict[str, float] | None = None
                is_best = (
                    bool(best_improved_tensor.cpu())
                    if not validation_enabled
                    else False
                )
                if stop_save and validation_enabled:
                    print(
                        "[train] stop checkpoint: skipping validation for fast shutdown"
                    )
                if cpu_prefetcher is not None:
                    # loader state と未消費 batch を同一瞬間のペアで取る (排他は内部 lock)
                    dl_state, pending_cpu = cpu_prefetcher.state_dict()
                    if dl_state is not None and pending_cpu:
                        dl_state["_cpu_prefetch_pending"] = pending_cpu
                else:
                    dl_state = train_loader.state_dict() if hasattr(train_loader, "state_dict") else None
                if (
                    dl_state is not None
                    and cuda_prefetch
                    and isinstance(data_iter, CudaBatchPrefetcher)
                ):
                    prefetched = data_iter.state_dict()
                    if prefetched is not None:
                        dl_state["_cuda_prefetch_next_batch"] = prefetched

                # P0: validation開始前に、このoptimizer stepのmodel/optimizer/
                # scheduler/dataloader stateをrecovery checkpointとして確定する。
                # CUDA illegal memory access後の救済saveは安全でないため、async設定でも
                # publish完了を待ってからvalidationへ進む。
                best_loss = float(best_loss_tensor.cpu())
                validation_status = (
                    "skipped_stop"
                    if stop_save and validation_enabled
                    else "pending"
                    if validation_enabled
                    else "disabled"
                )
                recovery_meta = make_checkpoint_meta(
                    step=global_step,
                    best_value=best_loss,
                    validation_results=None,
                    validation_status=validation_status,
                )
                t0 = time.perf_counter()
                saved_dir = ckpt.save(
                    base_model,
                    optimizer,
                    scheduler,
                    dl_state,
                    recovery_meta,
                    config=effective_cfg,
                    is_best=is_best,
                    is_final=global_step >= total_steps,
                    force_sync=stop_save,
                )
                ckpt.wait_for_pending_save()
                save_seconds = time.perf_counter() - t0
                print(
                    f"[train] recovery checkpoint durable @ step={global_step}"
                    f"{' (best)' if is_best else ''} in {save_seconds:.1f}s"
                )

                # P1: validationはtraining用compiled wrapperではなく、base_modelを
                # 共有する独立wrapper (既定eager) で実行する。validationがここで
                # 例外終了しても、上のrecovery checkpointは既にresume可能。
                if validation_enabled and not stop_save:
                    val_t0 = time.perf_counter()
                    validation_results = evaluate_validation(
                        validation_model,
                        validation_loaders,
                        device,
                        compute_dtype,
                        use_autocast,
                        int(validation_cfg.get("max_batches", 16)),
                        batch_cache=validation_batch_cache,
                    )
                    mean_bpb = validation_results.get("mean_bpb")
                    if mean_bpb is None:
                        is_best = False
                        print("[val] no valid labels; best not updated")
                    else:
                        val_score = torch.tensor(
                            mean_bpb, device=device, dtype=torch.float32
                        )
                        is_best = bool((val_score < best_loss_tensor).cpu())
                        best_loss_tensor = torch.minimum(best_loss_tensor, val_score)
                        parts = " ".join(
                            f"{k}={v:.4f}"
                            for k, v in sorted(validation_results.items())
                        )
                        print(
                            f"[val] step={global_step} {parts} "
                            f"elapsed={time.perf_counter() - val_t0:.1f}s"
                        )
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "validation": validation_results,
                            "best_metric": "validation_mean_bpb",
                            "time": time.time(),
                        }) + "\n")

                    best_loss = float(best_loss_tensor.cpu())
                    validated_meta = make_checkpoint_meta(
                        step=global_step,
                        best_value=best_loss,
                        validation_results=validation_results,
                        validation_status="complete",
                    )
                    ckpt.update_metadata(
                        saved_dir, validated_meta, is_best=is_best
                    )
                    print(
                        f"[train] checkpoint metadata updated @ step={global_step}"
                        f"{' (best)' if is_best else ''}"
                    )

                best_improved_tensor = torch.tensor(False, device=device)
                if sampling_enabled:
                    sample_at_checkpoint(saved_dir, global_step)
                if probes_enabled:
                    probes_at_checkpoint(saved_dir, global_step)

            if stop.requested:
                print("[train] stop requested, exiting cleanly.")
                break
            if args.dry_run:
                break

        except StopIteration:
            data_iter = make_data_iter()
            continue

    if isinstance(data_iter, CudaBatchPrefetcher):
        data_iter.close()
    if cpu_prefetcher is not None:
        cpu_prefetcher.close()
    data_iter = None
    if hasattr(train_loader, "shutdown_workers"):
        train_loader.shutdown_workers()
    # 最終 prune の daemon スレッドが途中終了して checkpoint を部分削除しないよう待つ
    ckpt._await_thread()
    if device.type == "cuda":
        torch.cuda.synchronize()
        model = None
        validation_model = None
        base_model = None
        optimizer = None
        scheduler = None
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
