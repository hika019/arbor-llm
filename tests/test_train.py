from __future__ import annotations

import pytest
import torch

from src.train.train import resolve_precision
from src.train.train import should_restore_dataloader_state


def test_resolve_precision_accepts_supported_modes():
    assert resolve_precision("bf16") == (torch.bfloat16, True)
    assert resolve_precision("fp16") == (torch.float16, True)
    assert resolve_precision("fp32") == (torch.float32, False)


def test_resolve_precision_rejects_unknown_mode():
    with pytest.raises(ValueError, match="speed.precision"):
        resolve_precision("int8")


def test_should_restore_dataloader_state_when_data_config_matches():
    data_cfg = {
        "sources": [{"path": "dataset-a", "weight": 1.0}],
        "context_length": 128,
        "micro_batch_size": 2,
    }

    assert should_restore_dataloader_state(data_cfg, dict(data_cfg))


def test_should_not_restore_dataloader_state_when_data_config_changed():
    saved_data_cfg = {
        "sources": [{"path": "dataset-a", "weight": 1.0}],
        "context_length": 128,
        "micro_batch_size": 2,
    }
    current_data_cfg = {
        "sources": [
            {"path": "dataset-a", "weight": 0.5},
            {"path": "dataset-b", "weight": 0.5},
        ],
        "context_length": 128,
        "micro_batch_size": 2,
    }

    assert not should_restore_dataloader_state(saved_data_cfg, current_data_cfg)
