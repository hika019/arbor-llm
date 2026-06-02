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
      best/                    # symlink to lowest-val-loss checkpoint

Saves are atomic: everything is written into ``<dir>.tmp/`` first, then
``os.replace`` swaps it into place. A kill at any point leaves either the
previous checkpoint intact or the .tmp dir (which is cleaned up on resume).

Async-save is supported (off the training thread) so the GPU does not idle
during a heavy disk write.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save


def _step_name(step: int) -> str:
    return f"step_{step:010d}"


@dataclass
class CheckpointMeta:
    global_step: int
    epoch: int = 0
    best_loss: float = float("inf")
    wandb_run_id: str | None = None
    config_hash: str | None = None
    git_sha: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointMeta":
        return cls(**d)


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
        is_best: bool = False,
    ) -> Path:
        """Save synchronously, then (optionally) prune in a background thread."""
        # Wait for prior async-prune to finish so we don't race symlinks.
        self._await_thread()

        step_dir = self.root / _step_name(meta.global_step)
        tmp_dir = self.root / f"{_step_name(meta.global_step)}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        # 1) weights -> safetensors (handles BF16 cleanly, mmap-friendly).
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        safe_save(weights, str(tmp_dir / "model.safetensors"))

        # 2) optimizer / scheduler / dataloader / rng / meta -> torch.save (.pt)
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        if dataloader_state is not None:
            torch.save(dataloader_state, tmp_dir / "dataloader.pt")
        torch.save(_snapshot_rng(), tmp_dir / "rng.pt")
        (tmp_dir / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2))

        # 3) atomic rename
        if step_dir.exists():
            shutil.rmtree(step_dir)
        os.replace(tmp_dir, step_dir)

        # 4) update symlinks
        _atomic_symlink(self.root / "latest", step_dir.name)
        if is_best:
            _atomic_symlink(self.root / "best", step_dir.name)

        # 5) prune old (in background if requested)
        if self.async_save:
            self._thread = threading.Thread(target=self._prune, daemon=True)
            self._thread.start()
        else:
            self._prune()

        return step_dir

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

        ``which`` can be 'latest', 'best', a step number, or an absolute path.
        Returns (meta, dataloader_state).
        """
        ckpt_dir = self.resolve(which)
        if ckpt_dir is None or not ckpt_dir.exists():
            raise FileNotFoundError(f"checkpoint not found: {which}")

        # weights
        weights = safe_load(str(ckpt_dir / "model.safetensors"), device=str(map_location))
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing or unexpected:
            print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")

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
        if which in ("latest", "best"):
            link = self.root / which
            if link.is_symlink():
                return (self.root / os.readlink(link)).resolve()
            return None
        p = Path(which)
        return p if p.is_absolute() else self.root / which

    # -------------------------------------------------------------- prune
    def _prune(self) -> None:
        steps = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("step_")),
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
        # Always keep latest K and best/latest symlink targets.
        protected: set[Path] = set()
        for link_name in ("latest", "best"):
            link = self.root / link_name
            if link.is_symlink():
                protected.add((self.root / os.readlink(link)).resolve())
        protected.update(steps[-self.keep_last_k :])
        # Long-term keeps: every N steps.
        if self.keep_every_n_steps:
            for p in steps:
                step = int(p.name.split("_", 1)[1])
                if step % self.keep_every_n_steps == 0:
                    protected.add(p.resolve())
        for p in steps:
            if p.resolve() not in protected:
                shutil.rmtree(p, ignore_errors=True)

    def _await_thread(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None


# --------------------------------------------------------------------- helpers
def _atomic_symlink(link: Path, target_name: str) -> None:
    tmp = link.with_name(link.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target_name)
    os.replace(tmp, link)


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
