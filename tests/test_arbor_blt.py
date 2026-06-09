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


def test_blt_uses_configured_kv_heads():
    model = arbor_blt._build_blt(_cfg(num_key_value_heads=2, max_position_embeddings=64))

    assert model.global_transformer.layers[0].attention.n_kv_heads == 2
    assert model.local_encoder.layers[0].attention.n_kv_heads == 2
    assert model.local_decoder.layers[0].attention.n_kv_heads == 2


def test_blt_can_use_narrower_local_hidden_size():
    model = arbor_blt._build_blt(
        _cfg(
            hidden_size=32,
            local_hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            local_num_attention_heads=2,
            local_num_key_value_heads=1,
            max_position_embeddings=64,
        )
    )

    assert model.global_transformer.dim == 32
    assert model.local_encoder.dim == 16
    assert model.local_decoder.dim == 16
    assert model.global_transformer.token_embedding_projection is not None
    assert model.local_encoder.layers[0].attention.n_heads == 2
    assert model.local_decoder.layers[0].attention.n_kv_heads == 1


def test_blt_uses_configured_relu2_intermediate_size():
    model = arbor_blt.build_arbor_blt(
        _cfg(
            intermediate_size=24,
            max_position_embeddings=64,
            relu2_ffn_in_global=True,
        )
    )

    assert model.blt.global_transformer.layers[0].feed_forward.hidden_dim == 24


def test_blt_rejects_invalid_kv_heads():
    with pytest.raises(ValueError, match="num_attention_heads"):
        arbor_blt._build_blt(_cfg(num_key_value_heads=3))


def test_blt_rejects_invalid_local_heads():
    with pytest.raises(ValueError, match="local_hidden_size"):
        arbor_blt._build_blt(
            _cfg(
                local_hidden_size=18,
                local_num_attention_heads=4,
            )
        )
