"""Atomic, resumable checkpoint manager.

Layout (per spec):
    checkpoints/
      step_0000010000/
        model.safetensors      # BF16 shadow weights
        optimizer.pt           # optimizer state (8-bit Adam etc.)
        scheduler.pt           # LR scheduler
        rng.pt                 # torch / cuda / numpy / python RNG
        dataloader.pt          # iterator position
        meta.json              # global_step, wandb_run_id, config_hash, git_sha
      step_0000010000.tmp/     # in-progress write (renamed on success)
      latest/                  # symlink to most recent finished checkpoint
      best/                    # symlink to lowest-loss checkpoint
      final/                   # symlink to the last planned checkpoint

Saves are staged into ``<step>.tmp/`` first and then renamed into the final
step directory only after all files are written. Existing step directories are
never deleted as part of save; attempting to write the same step twice is an
error. This keeps interrupted saves from destroying the last good checkpoint.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save


def _step_name(step: int) -> str:
    return f"step_{step:010d}"


_STEP_RE = re.compile(r"^step_(\d{10})$")


@dataclass
class CheckpointMeta:
    global_step: int
    epoch: int = 0
    best_loss: float = float("inf")
    wandb_run_id: str | None = None
    config_hash: str | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointMeta":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        unknown = {k: v for k, v in d.items() if k not in known}
        if unknown:
            extra = dict(kwargs.get("extra") or {})
            extra.update(unknown)
            kwargs["extra"] = extra
        return cls(**kwargs)


class CheckpointManager:
    def __init__(
        self,
        root: str | os.PathLike,
        keep_last_k: int = 3,
        keep_every_n_steps: int | None = 10_000,
        async_save: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.keep_every_n_steps = keep_every_n_steps
        self.async_save = async_save
        self._thread: threading.Thread | None = None
        self._thread_exc: list[BaseException] = []
        # Clean up any orphan .tmp dirs from a previous crash.
        for p in self.root.glob("*.tmp"):
            shutil.rmtree(p, ignore_errors=True)

    # ------------------------------------------------------------------ save
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        dataloader_state: dict[str, Any] | None,
        meta: CheckpointMeta,
        config: dict[str, Any] | None = None,
        is_best: bool = False,
        is_final: bool = False,
        force_sync: bool = False,
    ) -> Path:
        """Save a checkpoint and update retention symlinks.

        GPU/optimizer state is snapshotted to CPU synchronously (so training can
        safely mutate live tensors right after this call returns), then the
        actual disk write (safetensors/torch.save/fsync/rename/symlinks/prune)
        runs in a background thread when ``async_save`` is true. ``is_final``
        and ``force_sync`` (used for interrupt/shutdown saves) always write
        synchronously so the process cannot exit before the data is durable.
        Call ``wait_for_pending_save()`` before touching files under the
        returned ``step_dir`` (e.g. writing sample/probe output alongside it).
        """
        # Wait for prior async save to finish so we don't race symlinks/prune,
        # and surface any exception it raised.
        self._await_thread()

        step_dir = self.root / _step_name(meta.global_step)
        tmp_dir = self.root / f"{_step_name(meta.global_step)}.tmp"
        if step_dir.exists() or step_dir.is_symlink():
            raise FileExistsError(f"checkpoint already exists: {step_dir}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        # ---- synchronous snapshot phase: clone every live GPU/optimizer ----
        # tensor to CPU now, before returning control to the caller. Everything
        # captured below is a detached copy, so the background thread below
        # (or the synchronous path) never touches tensors the training loop
        # might mutate in-place afterwards (e.g. optimizer.step()).
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        optimizer_state = _cpu_snapshot(optimizer.state_dict())
        scheduler_state = _cpu_snapshot(scheduler.state_dict()) if scheduler is not None else None
        dataloader_state_cpu = _cpu_snapshot(dataloader_state) if dataloader_state is not None else None
        rng_state = _snapshot_rng()
        meta_json = json.dumps(meta.to_dict(), indent=2)
        config_yaml = (
            yaml.safe_dump(config, sort_keys=True, allow_unicode=True) if config is not None else None
        )

        def write_and_publish() -> None:
            try:
                # 1) weights -> safetensors (handles BF16 cleanly, mmap-friendly).
                safe_save(weights, str(tmp_dir / "model.safetensors"))

                # 2) optimizer / scheduler / dataloader / rng / meta -> torch.save (.pt)
                torch.save(optimizer_state, tmp_dir / "optimizer.pt")
                if scheduler_state is not None:
                    torch.save(scheduler_state, tmp_dir / "scheduler.pt")
                if dataloader_state_cpu is not None:
                    torch.save(dataloader_state_cpu, tmp_dir / "dataloader.pt")
                torch.save(rng_state, tmp_dir / "rng.pt")
                (tmp_dir / "meta.json").write_text(meta_json)
                if config_yaml is not None:
                    (tmp_dir / "config.yaml").write_text(config_yaml)
                _fsync_tree(tmp_dir)

                # 3) publish. ``step_dir`` was checked above; do not remove old data here.
                os.rename(tmp_dir, step_dir)
                _fsync_dir(self.root)

                # 4) update symlinks
                _atomic_symlink(self.root / "latest", step_dir.name)
                if is_best:
                    _atomic_symlink(self.root / "best", step_dir.name)
                if is_final:
                    _atomic_symlink(self.root / "final", step_dir.name)

                # 5) prune old checkpoints after symlinks point at protected targets.
                self._prune()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                self._thread_exc.append(exc)

        if self.async_save and not is_final and not force_sync:
            self._thread = threading.Thread(target=write_and_publish, daemon=False)
            self._thread.start()
        else:
            write_and_publish()
            self._raise_pending_thread_exc()

        return step_dir

    def wait_for_pending_save(self) -> None:
        """Block until any in-flight async save finishes; re-raises its exception if it failed."""
        self._await_thread()

    # ------------------------------------------------------------------ load
    def load(
        self,
        which: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[CheckpointMeta, dict[str, Any] | None]:
        """Restore from a checkpoint.

        ``which`` can be 'latest', 'best', 'final', a step number, or an absolute path.
        Returns (meta, dataloader_state).
        """
        ckpt_dir = self.resolve(which)
        if ckpt_dir is None or not ckpt_dir.exists():
            raise FileNotFoundError(f"checkpoint not found: {which}")

        # weights。strict=True: 形状/キー不一致を黙って通すと部分ロード事故になる.
        # torch.compile 済みモデルで保存された旧 checkpoint の "_orig_mod." prefix は剥がす.
        weights = safe_load(str(ckpt_dir / "model.safetensors"), device=str(map_location))
        if any(k.startswith("_orig_mod.") for k in weights):
            weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True)

        # weights_only=False: RNG/dataloader/optimizer は pickle 由来の Python オブジェクトを含む
        if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(ckpt_dir / "optimizer.pt", map_location=map_location, weights_only=False)
            )
        if scheduler is not None and (ckpt_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(ckpt_dir / "scheduler.pt", map_location=map_location, weights_only=False)
            )

        if (ckpt_dir / "rng.pt").exists():
            _restore_rng(torch.load(ckpt_dir / "rng.pt", map_location="cpu", weights_only=False))

        dataloader_state = None
        if (ckpt_dir / "dataloader.pt").exists():
            dataloader_state = torch.load(
                ckpt_dir / "dataloader.pt", map_location="cpu", weights_only=False
            )

        meta = CheckpointMeta.from_dict(json.loads((ckpt_dir / "meta.json").read_text()))
        return meta, dataloader_state

    # ------------------------------------------------------------ resolve
    def resolve(self, which: str | int) -> Path | None:
        if isinstance(which, int):
            return self.root / _step_name(which)
        if which in ("latest", "best", "final"):
            link = self.root / which
            if link.is_symlink():
                return (self.root / os.readlink(link)).resolve()
            return None
        p = Path(which)
        return p if p.is_absolute() else self.root / which

    # -------------------------------------------------------------- prune
    def _prune(self) -> None:
        steps = []
        for p in self.root.iterdir():
            m = _STEP_RE.match(p.name)
            if p.is_dir() and m:
                steps.append((int(m.group(1)), p))
        steps = [p for _, p in sorted(steps, key=lambda item: item[0])]
        # Always keep latest K and best/latest symlink targets.
        protected: set[Path] = set()
        for link_name in ("latest", "best", "final"):
            link = self.root / link_name
            if link.is_symlink():
                protected.add((self.root / os.readlink(link)).resolve())
        protected.update(p.resolve() for p in steps[-self.keep_last_k :])
        # Long-term keeps: every N steps.
        if self.keep_every_n_steps:
            for p in steps:
                step = int(_STEP_RE.match(p.name).group(1))  # type: ignore[union-attr]
                if step % self.keep_every_n_steps == 0:
                    protected.add(p.resolve())
        for p in steps:
            if p.resolve() not in protected:
                shutil.rmtree(p, ignore_errors=True)

    def _await_thread(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._raise_pending_thread_exc()

    def _raise_pending_thread_exc(self) -> None:
        if self._thread_exc:
            exc = self._thread_exc.pop(0)
            raise RuntimeError("background checkpoint save failed") from exc


def _cpu_snapshot(obj: Any) -> Any:
    """Recursively clone tensors in (possibly nested) state to detached CPU copies.

    Used before handing state to the background save thread: optimizer /
    scheduler / dataloader state dicts can hold GPU tensors (e.g. Adam's
    exp_avg/exp_avg_sq), and the training loop mutates those in-place right
    after this function returns (optimizer.step() on the next micro-batch), so
    the thread must never touch the originals.
    """
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_snapshot(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_snapshot(v) for v in obj)
    return obj


# --------------------------------------------------------------------- helpers
def _atomic_symlink(link: Path, target_name: str) -> None:
    tmp = link.with_name(link.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target_name)
    os.replace(tmp, link)
    _fsync_dir(link.parent)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    for p in path.iterdir():
        if p.is_file():
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    _fsync_dir(path)


def _snapshot_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
