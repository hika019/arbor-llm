"""src/infer/generate.py のサンプリング・復号ロジックのテスト (CPU)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.infer.generate import (
    BYTE_OFFSET,
    _sample_next,
    _strip_compile_prefix,
    generate_text,
)
from src.model.arbor import ArborConfig, ArborModel, ArborOutput


class _NextByteModel(nn.Module):
    """最後のバイトから次バイトを表引きで決める決定的な疑似モデル."""

    def __init__(self, mapping: dict[int, int], default: int = 65):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.mapping = mapping
        self.default = default

    def forward(self, x: torch.Tensor) -> ArborOutput:
        b, s = x.shape
        logits = torch.full((b, s, 260), -10.0)
        for i in range(b):
            last_byte = int(x[i, -1]) - BYTE_OFFSET
            nxt = self.mapping.get(last_byte, self.default)
            logits[i, -1, nxt + BYTE_OFFSET] = 10.0
        return ArborOutput(logits=logits)


class _SpecialTokenLover(nn.Module):
    """特殊 ID 0 に最大 logit を出すモデル。マスクの検証用."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> ArborOutput:
        b, s = x.shape
        logits = torch.full((b, s, 260), -10.0)
        logits[:, -1, 0] = 100.0  # <boe>: 必ずマスクされるべき
        logits[:, -1, 65 + BYTE_OFFSET] = 5.0  # 'A'
        return ArborOutput(logits=logits)


def test_greedy_follows_mapping():
    # 'a' -> 'b' -> 'c' -> 'a' の循環
    m = _NextByteModel({ord("a"): ord("b"), ord("b"): ord("c"), ord("c"): ord("a")})
    out = generate_text(m, "a", max_new_bytes=5, temperature=0.0)
    assert out == "bcabc"


def test_utf8_multibyte_incremental_decode():
    # "あ" = E3 81 82 を循環生成。バイト途中で文字化けせず文字単位で出ること
    m = _NextByteModel({0xE3: 0x81, 0x81: 0x82, 0x82: 0xE3})
    out = generate_text(m, "あ", max_new_bytes=6, temperature=0.0)
    assert out == "ああ"


def test_special_ids_are_masked():
    m = _SpecialTokenLover()
    out = generate_text(m, "x", max_new_bytes=3, temperature=0.0)
    assert out == "AAA"


def test_arbor_generate_defaults_to_full_forward(monkeypatch):
    class _FailingGenerator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cache should be opt-in")

    import src.model.arbor as arbor_mod

    monkeypatch.setattr(arbor_mod, "ArborByteGenerator", _FailingGenerator)
    cfg = dict(
        vocab_size=260, patch_size=4, max_bytes=16,
        hidden_size=32, num_heads=2, num_kv_heads=2, intermediate_size=64,
        num_hidden_layers=1,
        local_hidden_size=16, local_num_heads=2, local_num_kv_heads=2,
        local_intermediate_size=32,
        num_local_encoder_layers=1, num_local_decoder_layers=1,
    )
    torch.manual_seed(0)
    m = ArborModel(ArborConfig.from_dict(cfg)).eval()

    generate_text(m, "x", max_new_bytes=1, temperature=0.0)
    with pytest.raises(RuntimeError, match="cache should be opt-in"):
        generate_text(m, "x", max_new_bytes=1, temperature=0.0, use_cache=True)


def test_sample_next_greedy_and_topk():
    logits = torch.tensor([0.0, 1.0, 5.0, 2.0])
    assert _sample_next(logits, temperature=0.0, top_k=0, top_p=1.0) == 2
    # top_k=1 は実質 greedy
    g = torch.Generator().manual_seed(0)
    assert _sample_next(logits, temperature=1.0, top_k=1, top_p=1.0, generator=g) == 2


def test_sample_next_top_p_keeps_top_token():
    logits = torch.tensor([0.0, 10.0, 0.0])
    g = torch.Generator().manual_seed(0)
    # top_p が極小でも最上位 1 トークンは必ず残る
    assert _sample_next(logits, temperature=1.0, top_k=0, top_p=1e-9, generator=g) == 1


def test_sample_next_deterministic_with_seed():
    logits = torch.randn(260)
    a = _sample_next(logits, 1.0, 0, 0.9, torch.Generator().manual_seed(7))
    b = _sample_next(logits, 1.0, 0, 0.9, torch.Generator().manual_seed(7))
    assert a == b


def test_strip_compile_prefix():
    state = {"_orig_mod.blt.w": torch.zeros(1), "_orig_mod.blt.b": torch.zeros(1)}
    out = _strip_compile_prefix(state)
    assert set(out) == {"blt.w", "blt.b"}
    # prefix が無ければそのまま
    plain = {"blt.w": torch.zeros(1)}
    assert _strip_compile_prefix(plain) is plain
