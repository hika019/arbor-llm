from __future__ import annotations

from src.data.byte_dataset import ByteStreamDataset


def test_hf_stream_uses_byte_offset(monkeypatch):
    ds = ByteStreamDataset(source="dummy", context_length=3, byte_offset=4)
    monkeypatch.setattr(ds, "_build_hf_stream", lambda: iter([{"text": "abcde"}]))

    sample = next(ds._iter_hf_stream())

    assert sample["input_ids"].tolist() == [ord("a") + 4, ord("b") + 4, ord("c") + 4]
    assert sample["labels"].tolist() == [ord("b") + 4, ord("c") + 4, ord("d") + 4]
