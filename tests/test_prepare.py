from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data import prepare


def _write_tsv(path: Path, n: int) -> None:
    path.write_text("".join(f"{i:03x}/x.flac\t発話{i}\n" for i in range(n)) + "000/y.flac\t\n", encoding="utf-8")


def test_reazonspeech_builds_docs_and_skips_when_complete(tmp_path, monkeypatch):
    tsv = tmp_path / "reazonspeech_v2_all.tsv"
    _write_tsv(tsv, 70)
    out = tmp_path / "reazonspeech_text"
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: pytest.fail("tsv があるのにダウンロードした"))
    prepare.ensure_prepared("reazonspeech_v2_text", out)
    docs = [d for f in sorted(out.glob("part-*.parquet")) for d in pq.read_table(f).column("text").to_pylist()]
    assert len(docs) == 3                          # 30 + 30 + 10 発話 (空の書き起こしは捨てる)
    assert docs[0].split("\n")[:2] == ["発話0", "発話1"]
    assert (out / prepare.COMPLETE_MARK).exists()
    assert not (tmp_path / "reazonspeech_text.tmp").exists()

    called = []
    monkeypatch.setitem(prepare.PREPARERS, "reazonspeech_v2_text", lambda d: called.append(d))
    prepare.ensure_prepared("reazonspeech_v2_text", out)
    assert called == []                            # 完了印があれば何もしない


def test_incomplete_output_is_rebuilt(tmp_path):
    tsv = tmp_path / "reazonspeech_v2_all.tsv"
    _write_tsv(tsv, 5)
    out = tmp_path / "reazonspeech_text"
    out.mkdir()
    (out / "part-00000.parquet").write_text("broken")   # 完了印の無い中途半端な出力
    prepare.ensure_prepared("reazonspeech_v2_text", out)
    assert (out / prepare.COMPLETE_MARK).exists()
    assert pq.read_table(out / "part-00000.parquet").num_rows == 1


def test_loader_runs_prepare_for_source(tmp_path, monkeypatch):
    from src.data import byte_dataset

    seen = []
    monkeypatch.setattr(prepare, "ensure_prepared", lambda name, d: seen.append((name, d)))

    class _Stop(Exception):
        pass

    def fake_builder(*a, **k):
        raise _Stop

    import datasets
    monkeypatch.setattr(datasets, "load_dataset_builder", fake_builder)
    spec = {"data_files": str(tmp_path / "d" / "part-*.parquet"), "prepare": "reazonspeech_v2_text"}
    with pytest.raises(_Stop):
        byte_dataset._load_hf_streaming("parquet", None, "train", spec)
    assert seen == [("reazonspeech_v2_text", tmp_path / "d")]
