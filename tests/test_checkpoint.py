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


def test_async_save_snapshots_before_mutation(tmp_path):
    """The background write must use a CPU snapshot taken at save()-call time,
    not whatever the live tensors look like when the thread actually runs."""
    model = _model()
    optimizer = _optimizer(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
    # give AdamW some state so optimizer_state actually has tensors to snapshot
    optimizer.zero_grad()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()

    manager = CheckpointManager(tmp_path, keep_last_k=2, keep_every_n_steps=None, async_save=True)
    weight_at_save_time = model.weight.detach().clone()
    manager.save(model, optimizer, None, None, CheckpointMeta(global_step=1))

    # Mutate live model/optimizer state right after save() returns, exactly
    # like the training loop's next micro-batch would.
    with torch.no_grad():
        model.weight.fill_(999.0)
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                v.fill_(999.0)

    manager.wait_for_pending_save()

    restored = _model()
    meta, _ = manager.load(1, restored, map_location="cpu")
    assert meta.global_step == 1
    assert torch.equal(restored.weight, weight_at_save_time), (
        "checkpoint captured the post-save mutation instead of the save()-time snapshot"
    )


def test_final_save_is_synchronous(tmp_path):
    model = _model()
    optimizer = _optimizer(model)
    manager = CheckpointManager(tmp_path, keep_last_k=2, keep_every_n_steps=None, async_save=True)

    manager.save(model, optimizer, None, None, CheckpointMeta(global_step=1), is_final=True)

    # No wait_for_pending_save() call: files must already exist on disk.
    assert (tmp_path / "step_0000000001" / "model.safetensors").exists()
    assert manager._thread is None


def test_force_sync_save_is_synchronous(tmp_path):
    model = _model()
    optimizer = _optimizer(model)
    manager = CheckpointManager(tmp_path, keep_last_k=2, keep_every_n_steps=None, async_save=True)

    manager.save(model, optimizer, None, None, CheckpointMeta(global_step=1), force_sync=True)

    assert (tmp_path / "step_0000000001" / "model.safetensors").exists()
    assert manager._thread is None


def test_async_save_exception_surfaces_on_wait(tmp_path, monkeypatch):
    import src.train.checkpoint as checkpoint_mod

    def boom(*args, **kwargs):
        raise RuntimeError("disk full (simulated)")

    monkeypatch.setattr(checkpoint_mod, "safe_save", boom)

    model = _model()
    optimizer = _optimizer(model)
    manager = CheckpointManager(tmp_path, keep_last_k=2, keep_every_n_steps=None, async_save=True)
    manager.save(model, optimizer, None, None, CheckpointMeta(global_step=1))

    with pytest.raises(RuntimeError, match="background checkpoint save failed"):
        manager.wait_for_pending_save()
