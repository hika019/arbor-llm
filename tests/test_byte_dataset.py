from __future__ import annotations

import pytest

from src.data.byte_dataset import ByteStreamDataset
from src.data.byte_dataset import build_byte_dataloader


def test_hf_stream_uses_byte_offset(monkeypatch):
    ds = ByteStreamDataset(source="dummy", context_length=3, byte_offset=4)
    monkeypatch.setattr(ds, "_build_hf_stream", lambda: iter([{"text": "abcde"}]))

    sample = next(ds._iter_hf_stream())

    assert sample["input_ids"].tolist() == [ord("a") + 4, ord("b") + 4, ord("c") + 4]
    assert sample["labels"].tolist() == [ord("b") + 4, ord("c") + 4, ord("d") + 4]


def test_resumable_loader_counts_emitted_samples(tmp_path):
    data_file = tmp_path / "bytes.txt"
    data_file.write_text("abcdefghijklmnopqrstuvwxyz")
    loader = build_byte_dataloader(
        {
            "source": f"file:{data_file}",
            "context_length": 3,
            "micro_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        }
    )

    next(iter(loader))
    next(iter(loader))

    assert loader.state_dict()["samples_emitted"] == 4


def test_multi_worker_iterable_requires_explicit_opt_in(tmp_path):
    data_file = tmp_path / "bytes.txt"
    data_file.write_text("abcdefghijklmnopqrstuvwxyz")

    with pytest.raises(ValueError, match="num_workers <= 1"):
        build_byte_dataloader(
            {
                "source": f"file:{data_file}",
                "context_length": 3,
                "micro_batch_size": 2,
                "num_workers": 2,
                "pin_memory": False,
            }
        )
