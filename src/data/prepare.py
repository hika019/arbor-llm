"""学習データの事前準備 (ダウンロード・変換)。source spec の ``prepare`` で指定する.

HF から直接 streaming できないデータ (音声コーパスの書き起こしなど) を、学習開始時に
ローカルへ用意する。出力ディレクトリに完了印 ``_COMPLETE`` があれば何もしない。
途中で落ちても中途半端な出力が残らないよう、一時ディレクトリに作ってから入れ替える。

    - path: "parquet"
      data_files: "~/datasets/reazonspeech_text/part-*.parquet"
      prepare: reazonspeech_v2_text      # data_files の親ディレクトリに用意する
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

COMPLETE_MARK = "_COMPLETE"

REAZON_TSV_URL = "https://corpus.reazon-research.org/reazonspeech-v2/tsv/all.tsv"


def build_reazonspeech_v2_text(
    out_dir: Path,
    tsv_path: Path | None = None,
    utts_per_doc: int = 30,
    docs_per_file: int = 20000,
) -> None:
    """ReazonSpeech v2 (TV 放送音声コーパス) の書き起こしを学習用 parquet にする.

    公式配布は書き起こし一覧 (tsv、2.1GB、2,194 万発話、「音声ファイル名<TAB>書き起こし」) が
    音声 (tar) と別ファイルなので tsv だけを取得する (~1 分)。HF 上のコピー
    (japanese-asr/ja_asr.reazon_speech_all) は音声と同じ parquet に入っており、テキストだけを
    範囲読みしても ~8 時間かかったので使わない。

    利用条件: https://huggingface.co/datasets/reazon-research/reazonspeech で利用規約に同意
    (著作権法 30 条の 4 に基づく情報解析目的に限る。テキストの再配布はしないこと)。

    1 発話 (平均 ~80 byte) は短すぎて EOS だらけになるので utts_per_doc 発話ずつ改行でつないで
    1 文書にする。tsv の並びは音声ファイル名 (hash) 順で、隣り合う発話に文脈のつながりは無い。
    """
    tsv_path = tsv_path or out_dir.parent / "reazonspeech_v2_all.tsv"
    if not tsv_path.exists():
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[prepare] downloading {REAZON_TSV_URL} -> {tsv_path}", flush=True)
        tmp = tsv_path.with_name(tsv_path.name + ".tmp")
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(tmp), REAZON_TSV_URL], check=True)
        tmp.rename(tsv_path)

    work = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    n_utt = n_bytes = n_files = 0
    docs: list[str] = []
    utts: list[str] = []

    def flush(final: bool = False) -> None:
        nonlocal docs, n_files
        if docs and (final or len(docs) >= docs_per_file):
            pq.write_table(pa.table({"text": docs}), work / f"part-{n_files:05d}.parquet",
                           compression="zstd")
            n_files += 1
            docs = []

    with tsv_path.open(encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n").partition("\t")[2].strip()
            if not text:
                continue
            utts.append(text)
            n_utt += 1
            n_bytes += len(text.encode()) + 1
            if len(utts) >= utts_per_doc:
                docs.append("\n".join(utts))
                utts = []
                flush()
    if utts:
        docs.append("\n".join(utts))
    flush(final=True)
    (work / COMPLETE_MARK).write_text(f"utterances={n_utt} bytes={n_bytes} files={n_files}\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    work.rename(out_dir)
    print(f"[prepare] reazonspeech_v2_text: utterances={n_utt} text={n_bytes / 1e9:.2f}GB "
          f"files={n_files} -> {out_dir}", flush=True)


PREPARERS: dict[str, Callable[[Path], None]] = {
    "reazonspeech_v2_text": build_reazonspeech_v2_text,
}


def ensure_prepared(name: str, out_dir: Path) -> None:
    """out_dir に完了印があれば何もしない。無ければ name の準備処理で作る."""
    if (out_dir / COMPLETE_MARK).exists():
        return
    if name not in PREPARERS:
        raise ValueError(f"unknown prepare: {name!r} (choices: {', '.join(PREPARERS)})")
    print(f"[prepare] {name}: {out_dir} が未作成なので作る", flush=True)
    PREPARERS[name](out_dir)
