from __future__ import annotations

import pytest
import torch

from src.train.checkpoint import CheckpointManager, CheckpointMeta


def _model() -> torch.nn.Module:
    return torch.nn.Linear(3, 2, bias=False)


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def test_checkpoint_round_trip_and_symlinks(tmp_path):
    model = _model()
    optimizer = _optimizer(model)
    manager = CheckpointManager(tmp_path, keep_last_k=2, keep_every_n_steps=None, async_save=True)

    manager.save(model, optimizer, None, {"offset": 1}, CheckpointMeta(global_step=1), is_best=True)
    manager.save(model, optimizer, None, {"offset": 2}, CheckpointMeta(global_step=2))
    manager.save(model, optimizer, None, {"offset": 3}, CheckpointMeta(global_step=3), is_final=True)

    assert manager.resolve("latest") == (tmp_path / "step_0000000003").resolve()
    assert manager.resolve("best") == (tmp_path / "step_0000000001").resolve()
    assert manager.resolve("final") == (tmp_path / "step_0000000003").resolve()

    restored = _model()
    meta, dataloader_state = manager.load("best", restored, map_location="cpu")
    assert meta.global_step == 1
    assert dataloader_state == {"offset": 1}


def test_checkpoint_refuses_to_overwrite_existing_step(tmp_path):
    model = _model()
    optimizer = _optimizer(model)
    manager = CheckpointManager(tmp_path, keep_last_k=2, keep_every_n_steps=None, async_save=False)

    meta = CheckpointMeta(global_step=1)
    manager.save(model, optimizer, None, None, meta)

    with pytest.raises(FileExistsError):
        manager.save(model, optimizer, None, None, meta)
