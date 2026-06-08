from __future__ import annotations

import pytest
import torch

from src.train.train import resolve_precision


def test_resolve_precision_accepts_supported_modes():
    assert resolve_precision("bf16") == (torch.bfloat16, True)
    assert resolve_precision("fp16") == (torch.float16, True)
    assert resolve_precision("fp32") == (torch.float32, False)


def test_resolve_precision_rejects_unknown_mode():
    with pytest.raises(ValueError, match="speed.precision"):
        resolve_precision("int8")
