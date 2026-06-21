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


class _FakeStatefulStream:
    """datasets の IterableDataset (state_dict/load_state_dict 持ち) を模す."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.idx = 0

    def __iter__(self):
        while self.idx < len(self.rows):
            row = self.rows[self.idx]
            self.idx += 1
            yield row

    def state_dict(self):
        return {"idx": self.idx}

    def load_state_dict(self, state):
        self.idx = state["idx"]


def test_hf_stream_exact_resume_via_state_dict(monkeypatch):
    rows = [{"text": "abcdefgh"}, {"text": "ijklmnop"}]
    ds = ByteStreamDataset(source="dummy", context_length=3, byte_offset=0)
    monkeypatch.setattr(ds, "_build_hf_stream", lambda: _FakeStatefulStream(rows))
    it = ds._iter_hf_stream()
    first = next(it)
    state = ds.state_dict()

    ds2 = ByteStreamDataset(source="dummy", context_length=3, byte_offset=0)
    monkeypatch.setattr(ds2, "_build_hf_stream", lambda: _FakeStatefulStream(rows))
    ds2.load_state_dict(state)
    second = next(ds2._iter_hf_stream())

    # 1 サンプル目 "abcd" の続き ("efgh") が、再走査スキップなしで出ること
    assert first["input_ids"].tolist() == [ord("a"), ord("b"), ord("c")]
    assert second["input_ids"].tolist() == [ord("e"), ord("f"), ord("g")]


def test_local_file_resume_uses_byte_offset_without_double_skip(tmp_path):
    data_file = tmp_path / "bytes.txt"
    data_file.write_text("abcdefghijklmnopqrstuvwxyz")
    ds = ByteStreamDataset(source=f"file:{data_file}", context_length=3, byte_offset=0)
    iterator = iter(ds)

    first = next(iterator)
    state = ds.state_dict()

    restored = ByteStreamDataset(source=f"file:{data_file}", context_length=3, byte_offset=0)
    restored.load_state_dict(state)
    second = next(iter(restored))

    # block 単位 stride (重複なし): 1 サンプル目 "abcd" の次は "efgh"
    assert first["input_ids"].tolist() == [ord("a"), ord("b"), ord("c")]
    assert second["input_ids"].tolist() == [ord("e"), ord("f"), ord("g")]


def test_document_packing_masks_cross_document_label(monkeypatch):
    ds = ByteStreamDataset(
        sources=[{"path": "a", "weight_bytes": 1.0}],
        context_length=6,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: ([[{"text": "abc"}, {"text": "de"}]], [{"path": "a", "weight_bytes": 1.0}]),
    )

    sample = next(ds._iter_hf_document_packed())

    assert sample["input_ids"].tolist() == [ord("a"), ord("b"), ord("c"), 2, ord("d"), ord("e")]
    assert sample["labels"].tolist() == [ord("b"), ord("c"), 2, -100, ord("e"), 2]
    assert sample["fill_ratio"].item() == 1.0


def test_document_packing_masks_padding_labels(monkeypatch):
    ds = ByteStreamDataset(
        sources=[{"path": "a", "weight_bytes": 1.0, "max_epochs": 1}],
        context_length=8,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: ([[{"text": "abc"}]], [{"path": "a", "weight_bytes": 1.0, "max_epochs": 1}]),
    )

    sample = next(ds._iter_hf_document_packed())

    assert sample["input_ids"].tolist() == [ord("a"), ord("b"), ord("c"), 2, 3, 3, 3, 3]
    assert sample["labels"].tolist() == [ord("b"), ord("c"), 2, -100, -100, -100, -100, -100]
    assert sample["fill_ratio"].item() == 0.5
