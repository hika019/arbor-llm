#!/usr/bin/env python3
"""ReazonSpeech (japanese-asr/ja_asr.reazon_speech_all) の書き起こしテキストだけをローカルに抜き出す.

HF 上の parquet は音声と同じファイルに入っており、普通に streaming すると音声ごと読んで
テキスト 1MB あたり ~700MB 通信する (2026-09-26 実測)。pyarrow で transcription 列だけを
範囲読み (HfFileSystem block_size 256KB / cache 無し / pre_buffer 無し) すると通信は
テキストの ~1.8 倍で済むが、row group ごとに往復するので 1 ファイル ~30s かかる。
学習中の逐次読みには遅すぎるため、並列で一度だけ抜き出して学習ではローカル parquet を読む。

出力: <out>/<subset>/<元ファイル名>.parquet (列 "text")。1 発話 (平均 ~80 byte) は短すぎて
EOS だらけになるので、--utts-per-doc 発話ずつ改行でつないで 1 文書にする (発話の並びは
元々シャッフルされており文書内の文脈のつながりは無い)。出力済みのファイルは飛ばすので
途中で止めても再実行で続きから。

ライセンス: ReazonSpeech は著作権法 30 条の 4 に基づく情報解析目的に限る。抜き出したテキストの
再配布はしないこと。
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/japanese-asr/ja_asr.reazon_speech_all"


def extract_one(fs: HfFileSystem, src: str, dst: Path, utts_per_doc: int) -> tuple[int, int]:
    with fs.open(src, "rb", block_size=256 * 1024, cache_type="none") as fh:
        pf = pq.ParquetFile(fh, pre_buffer=False)
        utts = [u for u in pf.read(columns=["transcription"]).column(0).to_pylist() if u and u.strip()]
    docs = ["\n".join(utts[i:i + utts_per_doc]) for i in range(0, len(utts), utts_per_doc)]
    tmp = dst.with_suffix(".tmp")
    pq.write_table(pa.table({"text": docs}), tmp, compression="zstd")
    tmp.rename(dst)
    return len(utts), sum(len(d.encode()) for d in docs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "datasets" / "reazonspeech_text")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--utts-per-doc", type=int, default=30)
    args = ap.parse_args()

    fs = HfFileSystem()
    files = sorted(fs.glob(f"{REPO}/*/**/*.parquet"))
    jobs = []
    for src in files:
        rel = src[len(REPO) + 1:]                      # subset_0/.../xxx.parquet
        dst = args.out / rel.split("/")[0] / Path(rel).name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((src, dst))
    print(f"[reazon] files={len(files)} todo={len(jobs)} out={args.out}", flush=True)

    t0 = time.time()
    done = n_utt = n_bytes = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, fs, s, d, args.utts_per_doc): s for s, d in jobs}
        for fut in as_completed(futs):
            try:
                u, b = fut.result()
                n_utt += u
                n_bytes += b
            except Exception as e:  # noqa: BLE001 - 失敗したファイルは再実行で拾う
                failed += 1
                print(f"[reazon] FAILED {futs[fut]}: {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(f"[reazon] {done}/{len(jobs)} utt={n_utt} text={n_bytes / 1e6:.0f}MB "
                      f"failed={failed} elapsed={el / 60:.1f}min eta={eta / 60:.1f}min", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
