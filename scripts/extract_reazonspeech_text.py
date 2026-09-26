#!/usr/bin/env python3
"""ReazonSpeech v2 (TV 放送音声コーパス) の書き起こしを学習用のローカル parquet にする.

公式配布は書き起こし一覧 (tsv、2.1GB、2,194 万発話、「音声ファイル名<TAB>書き起こし」) が
音声 (tar) と別ファイルなので、tsv だけを取得する (~1 分)。HF 上のコピー
(japanese-asr/ja_asr.reazon_speech_all) は音声と同じ parquet に入っており、テキストだけを
範囲読みしても 5,596 ファイル × ~40 往復で ~8 時間かかったので使わない。

利用条件: https://huggingface.co/datasets/reazon-research/reazonspeech で利用規約に同意
(著作権法 30 条の 4 に基づく情報解析目的に限る。抜き出したテキストの再配布はしないこと)。

出力: <out>/part-NNNNN.parquet (列 "text")。1 発話 (平均 ~80 byte) は短すぎて EOS だらけになる
ので --utts-per-doc 発話ずつ改行でつないで 1 文書にする。tsv の並びは音声ファイル名 (hash) 順で
隣り合う発話に文脈のつながりは無い。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TSV_URL = "https://corpus.reazon-research.org/reazonspeech-v2/tsv/all.tsv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=Path.home() / "datasets" / "reazonspeech_v2_all.tsv")
    ap.add_argument("--out", type=Path, default=Path.home() / "datasets" / "reazonspeech_text")
    ap.add_argument("--utts-per-doc", type=int, default=30)
    ap.add_argument("--docs-per-file", type=int, default=20000)
    args = ap.parse_args()

    if not args.tsv.exists():
        args.tsv.parent.mkdir(parents=True, exist_ok=True)
        print(f"[reazon] downloading {TSV_URL} -> {args.tsv}", flush=True)
        tmp = args.tsv.with_suffix(".tmp")
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(tmp), TSV_URL], check=True)
        tmp.rename(args.tsv)

    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob("part-*.parquet")):
        raise SystemExit(f"{args.out} に既存の出力がある。作り直すなら先に削除すること")

    n_utt = n_bytes = n_files = 0
    docs: list[str] = []
    utts: list[str] = []

    def flush_docs(final: bool = False) -> None:
        nonlocal docs, n_files
        if docs and (final or len(docs) >= args.docs_per_file):
            pq.write_table(pa.table({"text": docs}), args.out / f"part-{n_files:05d}.parquet",
                           compression="zstd")
            n_files += 1
            docs = []

    with args.tsv.open(encoding="utf-8") as f:
        for line in f:
            _, _, text = line.rstrip("\n").partition("\t")
            text = text.strip()
            if not text:
                continue
            utts.append(text)
            n_utt += 1
            n_bytes += len(text.encode()) + 1
            if len(utts) >= args.utts_per_doc:
                docs.append("\n".join(utts))
                utts = []
                flush_docs()
    if utts:
        docs.append("\n".join(utts))
    flush_docs(final=True)
    (args.out / "_COMPLETE").write_text(f"utterances={n_utt} bytes={n_bytes} files={n_files}\n")
    print(f"[reazon] utterances={n_utt} text={n_bytes / 1e9:.2f}GB files={n_files} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
