from __future__ import annotations

import pytest

import src.data.eval_split as eval_split


def test_collect_hashes_reads_until_bytes_and_skips_empty():
    spec = {"text_column": "text", "bytes": 8}
    rows = [{"text": ""}, {"text": "abc"}, {"text": "defg"}, {"text": "never"}]
    got = eval_split.collect_hashes(spec, rows=iter(rows))
    assert got == [eval_split.doc_hash("abc").hex(), eval_split.doc_hash("defg").hex()]


def test_eval_doc_hashes_are_cached(tmp_path, monkeypatch):
    validation_cfg = {"max_batches": 2, "micro_batch_size": 1,
                      "domains": {"web": {"source": "x", "skip_samples": 5}}}
    data_cfg = {"context_length": 16, "micro_batch_size": 2}
    calls = []

    def fake_collect(spec):
        calls.append(spec)
        return [eval_split.doc_hash("doc").hex()]

    monkeypatch.setattr(eval_split, "collect_hashes", fake_collect)
    first = eval_split.eval_doc_hashes(validation_cfg, data_cfg, cache_dir=tmp_path)
    assert first == frozenset({eval_split.doc_hash("doc")})
    assert calls[0]["bytes"] == 2 * 2 * 1 * 16 and calls[0]["skip_samples"] == 5

    monkeypatch.setattr(eval_split, "collect_hashes", lambda spec: pytest.fail("cache を使っていない"))
    assert eval_split.eval_doc_hashes(validation_cfg, data_cfg, cache_dir=tmp_path) == first
