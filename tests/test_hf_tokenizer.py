"""HF 互換バイト tokenizer のテスト."""
from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from src.hf.tokenization_arbor import ArborByteTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return ArborByteTokenizer()


def test_vocab_size(tok):
    assert tok.vocab_size == 260
    assert len(tok.get_vocab()) >= 260


def test_byte_offset_mapping(tok):
    # token = byte + 4 (学習データの byte_offset=4 と一致すること)
    ids = tok("A").input_ids
    assert ids == [ord("A") + 4]


def test_special_token_ids(tok):
    assert tok.convert_tokens_to_ids("<boe>") == 0
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2
    assert tok.pad_token_id == 3


def test_roundtrip_ascii_and_japanese(tok):
    for text in ("hello, world!", "こんにちは世界", "混ぜる mixed 123"):
        assert tok.decode(tok(text).input_ids) == text


def test_decode_skips_special_tokens(tok):
    ids = [1] + tok("ok").input_ids + [2]
    assert tok.decode(ids) == "ok"
