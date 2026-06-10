from __future__ import annotations

import json

import pytest
import torch
import yaml

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


def test_checkpoint_writes_config_and_repro_metadata(tmp_path):
    model = _model()
    optimizer = _optimizer(model)
    manager = CheckpointManager(tmp_path, keep_last_k=2, keep_every_n_steps=None, async_save=False)
    cfg = {"run_name": "unit", "speed": {"micro_batch_size": 4}}
    meta = CheckpointMeta(
        global_step=7,
        config_hash="abc123",
        git_sha="deadbeef",
        git_dirty=True,
        extra={"run": {"config_path": "configs/unit.yaml"}},
    )

    manager.save(model, optimizer, None, None, meta, config=cfg)

    step_dir = tmp_path / "step_0000000007"
    assert yaml.safe_load((step_dir / "config.yaml").read_text()) == cfg
    meta_json = json.loads((step_dir / "meta.json").read_text())
    assert meta_json["config_hash"] == "abc123"
    assert meta_json["git_sha"] == "deadbeef"
    assert meta_json["git_dirty"] is True
    assert meta_json["extra"]["run"]["config_path"] == "configs/unit.yaml"


def test_checkpoint_meta_preserves_unknown_fields_in_extra():
    meta = CheckpointMeta.from_dict({"global_step": 1, "future_field": "kept"})

    assert meta.global_step == 1
    assert meta.extra["future_field"] == "kept"
