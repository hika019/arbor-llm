from __future__ import annotations

import pytest

from src.data.byte_dataset import ByteStreamDataset
from src.data.byte_dataset import build_byte_dataloader
from src.data.text_filter import evaluate_text_filter


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


def test_document_packing_resume_restores_per_source_stream_states(monkeypatch):
    specs = [{"path": "a", "weight_bytes": 1.0}, {"path": "b", "weight_bytes": 1.0}]
    rows_a = [{"text": "abc"}, {"text": "def"}]
    rows_b = [{"text": "XYZ"}, {"text": "UVW"}]
    ds = ByteStreamDataset(
        sources=specs,
        context_length=3,
        byte_offset=0,
        packing="document",
        seed=1,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: ([_FakeStatefulStream(rows_a), _FakeStatefulStream(rows_b)], specs),
    )

    iterator = ds._iter_hf_document_packed()
    first = next(iterator)
    second = next(iterator)
    state = ds.state_dict()

    restored = ByteStreamDataset(
        sources=specs,
        context_length=3,
        byte_offset=0,
        packing="document",
        seed=1,
    )
    monkeypatch.setattr(
        restored,
        "_build_hf_source_streams",
        lambda: ([_FakeStatefulStream(rows_a), _FakeStatefulStream(rows_b)], specs),
    )
    restored.load_state_dict(state)
    third = next(restored._iter_hf_document_packed())

    assert first["input_ids"].tolist() == [ord("a"), ord("b"), ord("c")]
    assert second["input_ids"].tolist() == [ord("X"), ord("Y"), ord("Z")]
    assert third["input_ids"].tolist() == [ord("d"), ord("e"), ord("f")]


def test_document_packing_resume_restores_in_progress_pack_buffer(monkeypatch):
    specs = [{"path": "a", "weight_bytes": 1.0, "max_epochs": 1}]
    rows = [{"text": "abc"}, {"text": "defgh"}, {"text": "ij"}]
    ds = ByteStreamDataset(
        sources=specs,
        context_length=6,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: ([_FakeStatefulStream(rows)], specs),
    )

    first = next(ds._iter_hf_document_packed())
    state = ds.state_dict()

    restored = ByteStreamDataset(
        sources=specs,
        context_length=6,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
    )
    monkeypatch.setattr(
        restored,
        "_build_hf_source_streams",
        lambda: ([_FakeStatefulStream(rows)], specs),
    )
    restored.load_state_dict(state)
    second = next(restored._iter_hf_document_packed())

    assert first["input_ids"].tolist() == [ord("a"), ord("b"), ord("c"), 2, ord("d"), ord("e")]
    assert first["labels"].tolist() == [ord("b"), ord("c"), 2, -100, ord("e"), ord("f")]
    assert second["input_ids"].tolist() == [ord("g"), ord("h"), 2, ord("i"), ord("j"), 2]
    assert second["labels"].tolist() == [ord("h"), 2, -100, ord("j"), 2, -100]


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


def test_document_packing_aligns_new_document_to_patch_boundary(monkeypatch):
    """patch_align>1 のとき、新 document は必ず patch 境界から始まる (P0 isolation)."""
    ds = ByteStreamDataset(
        sources=[{"path": "a", "weight_bytes": 1.0}],
        context_length=8,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
        patch_align=4,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: ([[{"text": "abc"}, {"text": "de"}]], [{"path": "a", "weight_bytes": 1.0}]),
    )

    sample = next(ds._iter_hf_document_packed())
    ids = sample["input_ids"].tolist()
    # docA(abc)+EOS = 4 byte でちょうど patch 境界。次の PAD 無しで docB が index 4 から。
    assert ids[:4] == [ord("a"), ord("b"), ord("c"), 2]
    # docA+EOS が既に境界に揃うので PAD は挿入されず docB が index 4 (patch 境界) から
    assert ids[4] == ord("d")
    assert ids[5] == ord("e")


def test_document_packing_pads_short_document_to_patch_boundary(monkeypatch):
    """EOS が patch 境界に揃わないとき PAD で境界まで埋め、次文書を境界から開始する."""
    ds = ByteStreamDataset(
        sources=[{"path": "a", "weight_bytes": 1.0}],
        context_length=12,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
        patch_align=4,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: ([[{"text": "ab"}, {"text": "cd"}]], [{"path": "a", "weight_bytes": 1.0}]),
    )

    sample = next(ds._iter_hf_document_packed())
    ids = sample["input_ids"].tolist()
    labels = sample["labels"].tolist()
    # docA "ab"+EOS = 3 byte → patch_align=4 まで PAD 1 個埋め、docB は index 4 から。
    # pack: [a,b,EOS,PAD, c,d,EOS, ...]; context_length+1=13 を 12 で切り出す。
    assert ids[:4] == [ord("a"), ord("b"), 2, 3]
    assert ids[4] == ord("c")           # docB が patch 境界 (index 4) から開始
    assert ids[5] == ord("d")
    # 各 patch(4byte) は 1 文書のみ: patch0=docA+PAD, patch1=docB。境界跨ぎ無し。
    # PAD 位置と、PAD→docB を予測する transition の label は学習対象外 (-100)。
    assert labels[2] == -100            # EOS→PAD
    assert labels[3] == -100            # PAD→docB 先頭 (boundary mask)


def test_ja_web_text_filter_accepts_paragraph_text():
    text = 3 * (
        "日本の四季には春、夏、秋、冬があり、それぞれに異なる気候と風景があります。"
        "春には桜が咲き、夏には海や山の行楽が楽しまれます。"
        "秋には紅葉が色づき、冬には雪景色や温かい料理が人々の暮らしを彩ります。"
        "地域によって季節の移ろい方は少しずつ異なり、同じ日本の中でも多様な文化が育まれてきました。"
        "こうした変化は観光や農業だけでなく、日々の服装や食事にも深く関わっています。"
    )

    result = evaluate_text_filter(text, {"preset": "ja_web_v1"})

    assert result.accepted
    assert result.reasons == ()


def test_ja_web_text_filter_rejects_boilerplate_listing():
    text = """
    ログイン
    会員登録
    お問い合わせ
    サイトマップ
    ランキング
    関連記事
    続きを読む
    Copyright 2025 Example All rights reserved
    https://example.com/item/1
    https://example.com/item/2
    """

    result = evaluate_text_filter(text, {"preset": "ja_web_v1"})

    assert not result.accepted
    assert "boilerplate" in result.reasons


def test_ja_web_text_filter_rejects_suspicious_sequences():
    text = 3 * (
        "保険の選択では、保証内容と毎月の保険??料を比べることが大切です。"
        "資料を読みながら、聞??いておきたい点を整理して相談すると判断しやすくなります。"
        "家族構成や働き方によって必要な保障は変わるため、複数の商品を比較します。"
    )

    result = evaluate_text_filter(text, {"preset": "ja_web_v1"})

    assert not result.accepted
    assert "suspicious_sequences" in result.reasons


def test_document_packing_skips_rejected_documents(monkeypatch):
    accepted = 3 * (
        "日本の四季には春、夏、秋、冬があり、それぞれに異なる気候と風景があります。"
        "春には桜が咲き、夏には海や山の行楽が楽しまれます。"
        "秋には紅葉が色づき、冬には雪景色や温かい料理が人々の暮らしを彩ります。"
        "地域によって季節の移ろい方は少しずつ異なり、同じ日本の中でも多様な文化が育まれてきました。"
        "こうした変化は観光や農業だけでなく、日々の服装や食事にも深く関わっています。"
    )
    rejected = "ログイン\n会員登録\nお問い合わせ\nランキング\n関連記事\nCopyright\nhttps://example.com"
    ds = ByteStreamDataset(
        sources=[{"id": "fineweb2_ja", "path": "a", "weight_bytes": 1.0, "max_epochs": 1, "text_filter": {"preset": "ja_web_v1"}}],
        context_length=16,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: (
            [[{"text": rejected}, {"text": accepted}]],
            [{"id": "fineweb2_ja", "path": "a", "weight_bytes": 1.0, "max_epochs": 1, "text_filter": {"preset": "ja_web_v1"}}],
        ),
    )

    sample = next(ds._iter_hf_document_packed())
    state = ds.state_dict()

    assert sample["input_ids"][0].item() == accepted.encode("utf-8")[0]
    assert state["extra"]["text_filter_stats"]["fineweb2_ja"] == {"accepted": 1, "rejected": 1}


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


def test_document_packing_carries_over_instead_of_padding_when_more_docs_exist(monkeypatch):
    ds = ByteStreamDataset(
        sources=[{"path": "a", "weight_bytes": 1.0, "max_epochs": 1}],
        context_length=6,
        byte_offset=0,
        packing="document",
        eos_token_id=2,
        pad_token_id=3,
    )
    monkeypatch.setattr(
        ds,
        "_build_hf_source_streams",
        lambda: (
            [[{"text": "abc"}, {"text": "defgh"}, {"text": "ij"}]],
            [{"path": "a", "weight_bytes": 1.0, "max_epochs": 1}],
        ),
    )

    iterator = ds._iter_hf_document_packed()
    first = next(iterator)
    second = next(iterator)

    assert first["input_ids"].tolist() == [ord("a"), ord("b"), ord("c"), 2, ord("d"), ord("e")]
    assert first["labels"].tolist() == [ord("b"), ord("c"), 2, -100, ord("e"), ord("f")]
    assert first["fill_ratio"].item() == 1.0
    assert second["input_ids"].tolist() == [ord("g"), ord("h"), 2, ord("i"), ord("j"), 2]
    assert second["labels"].tolist() == [ord("h"), 2, -100, ord("j"), 2, -100]
    assert second["fill_ratio"].item() == 1.0
