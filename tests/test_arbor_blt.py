from __future__ import annotations

import pytest

from src.model import arbor_blt


def _cfg(**overrides):
    cfg = {
        "backend": "blt",
        "vocab_size": 260,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
    }
    cfg.update(overrides)
    return cfg


def test_blt_backend_does_not_silently_fallback_to_stub(monkeypatch):
    def fail(_cfg):
        raise ImportError("missing blt")

    monkeypatch.setattr(arbor_blt, "_build_blt", fail)

    with pytest.raises(RuntimeError, match="BLT backend requested"):
        arbor_blt.build_arbor_blt(_cfg())


def test_stub_fallback_must_be_explicit(monkeypatch):
    def fail(_cfg):
        raise ImportError("missing blt")

    monkeypatch.setattr(arbor_blt, "_build_blt", fail)

    model = arbor_blt.build_arbor_blt(_cfg(allow_stub_fallback=True))
    assert isinstance(model, arbor_blt._StubArborBLT)
