"""バイト直 streaming データローダ.

設計方針:
- 全件メモリ展開はしない。HF datasets の `streaming=True` で逐次取得。
- ローカル shard を使う場合も mmap (numpy memmap) で参照のみ、コピーしない。
- 再開時の連続性のため、(shard_index, byte_offset, samples_emitted) を
  state_dict で保存・復元できる。
- backend=`hf` (HF streaming) と `local` (ローカルファイル mmap) の 2 種類を持つ.
"""
from __future__ import annotations

import mmap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset


@dataclass
class _ResumeState:
    samples_emitted: int = 0
    shard_index: int = 0
    byte_offset: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ByteStreamDataset(IterableDataset):
    """バイト列を context_length に区切って (input_ids, labels) を yield する.

    バックエンドは HuggingFace datasets の streaming iterator。テキスト列を
    UTF-8 バイトに変換し、リングバッファで詰めてから固定長に切り出す。
    全件メモリ展開しない。
    """

    def __init__(self, source: str | None = None, context_length: int = 2048,
                 split: str = "train", shuffle_buffer: int = 0,
                 text_column: str = "text", byte_offset: int = 4,
                 sources: list[dict[str, Any]] | None = None) -> None:
        """byte_offset: バイト値 b を token id (b + offset) に写す.

        BLT は 0..3 を BOE/BOS/EOS/BPE の特殊 ID として使い, 生バイトは
        OFFSET=4 から始まる (vocab_size = OFFSET + 256 = 260). stub model
        ではこの分離が不要なので 0 でも回るが, 実 BLT を使う場合は 4 にする.

        ソース指定は 2 通り:
        - `source`: 単一データセット (str). `file:` prefix でローカル mmap.
        - `sources`: 複数データセットを重み付き混合 (list[dict]). 各要素は
          `{path, name?, weight?, text_column?, split?}`. weight は確率に正規化し
          datasets.interleave_datasets で行レベルに混ぜる (例: 日本語 60% + 英語 40%).
          ローカル file: は混合対象外 (HF streaming のみ).
        """
        super().__init__()
        if sources:
            self.sources = [dict(s) for s in sources]
            self.source = None
        else:
            if not source:
                raise ValueError("source または sources のどちらかが必要")
            self.sources = None
            self.source = source
        self.context_length = context_length
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.text_column = text_column
        self.byte_offset = byte_offset
        self._state = _ResumeState()

    # --- state_dict: 学習ループから保存/復元される -----------------------
    def state_dict(self) -> dict[str, Any]:
        return self._state.__dict__.copy()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._state = _ResumeState(**state)

    # --- iterator: 1 サンプル = context_length+1 バイト ------------------
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # backend は source の prefix で判定: 'file:' で始まればローカル mmap.
        if self.source is not None and self.source.startswith("file:"):
            yield from self._iter_local_file(self.source[len("file:"):])
            return
        yield from self._iter_hf_stream()

    def _build_hf_stream(self):
        """単一/複数ソースを HF streaming の行イテレータに組み立てる.

        複数の場合は各ソースを "text" 列に正規化してから weight を確率に
        正規化し interleave_datasets で混ぜる. seed 固定なので skip ベースの
        resume でも同じ順序を再現できる.
        """
        from datasets import load_dataset

        def _one(path, name, col, split):
            ds = load_dataset(path, name=name, split=split, streaming=True)
            if col != "text":
                ds = ds.rename_column(col, "text")
            ds = ds.select_columns(["text"])  # スキーマ衝突回避 (混合時)
            if self.shuffle_buffer > 0:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=42)
            return ds

        if not self.sources:
            return _one(self.source, None, self.text_column, self.split)

        from datasets import interleave_datasets

        streams, weights = [], []
        for s in self.sources:
            streams.append(_one(
                s["path"], s.get("name"),
                s.get("text_column", self.text_column),
                s.get("split", self.split),
            ))
            weights.append(float(s.get("weight", 1.0)))
        total = sum(weights) or 1.0
        probs = [w / total for w in weights]
        # all_exhausted: 巨大コーパスでは実質無限。比率を保ちつつ枯渇分は再サンプル.
        return interleave_datasets(
            streams, probabilities=probs, seed=42,
            stopping_strategy="all_exhausted",
        )

    def _iter_hf_stream(self) -> Iterator[dict[str, torch.Tensor]]:
        try:
            ds = self._build_hf_stream()
        except ImportError as e:
            raise RuntimeError("`datasets` が必要 (pip install datasets)") from e

        block = self.context_length + 1
        off = self.byte_offset

        # 正確 resume: datasets の state_dict API でストリーム位置を復元する
        # (旧来の「最初から流し直して skip」はデータ量に比例して遅い)。
        # 注意: shuffle バッファの中身は復元されない (datasets の仕様)。
        hf_state = self._state.extra.get("hf_state")
        ring = bytearray(self._state.extra.get("ring", b""))
        skip = 0
        if hf_state is not None and hasattr(ds, "load_state_dict"):
            ds.load_state_dict(hf_state)
        else:
            ring.clear()
            skip = self._state.samples_emitted  # 旧 checkpoint 向け fallback

        for row in ds:
            text = row.get("text")
            if not text:
                continue
            ring.extend(text.encode("utf-8", errors="ignore"))
            ring.extend(b"\n")
            row_state = ds.state_dict() if hasattr(ds, "state_dict") else None
            while len(ring) >= block:
                chunk = bytes(ring[:block])
                del ring[:block]
                if skip > 0:
                    skip -= 1
                    continue
                buf = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long() + off
                ids = buf[:-1]
                tgt = buf[1:]
                self._state.samples_emitted += 1
                if row_state is not None:
                    self._state.extra["hf_state"] = row_state
                    self._state.extra["ring"] = bytes(ring)
                yield {"input_ids": ids, "labels": tgt}

    def _iter_local_file(self, path_str: str) -> Iterator[dict[str, torch.Tensor]]:
        """ローカルファイルを mmap で参照し、context_length バイトずつ切り出す.

        ファイル全体はメモリに読まない (mmap = カーネルのページキャッシュ経由).
        終端まで来たら循環して継続する.
        """
        path = Path(path_str).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        block = self.context_length + 1
        # byte_offset is exact for in-process local iteration. If it is missing
        # (for example a parent DataLoader tracking worker output), fall back to
        # samples_emitted-based skipping.
        skip = self._state.samples_emitted if self._state.byte_offset == 0 else 0

        off = self.byte_offset
        with path.open("rb") as f, mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
            size = len(mm)
            if size < block:
                raise ValueError(f"file too small: {size} < {block}")
            cursor = self._state.byte_offset
            while True:
                if cursor + block > size:
                    cursor = 0  # 循環
                chunk = mm[cursor:cursor + block]
                # block 単位で進める (1 byte stride だと隣接サンプルが 2047 byte 重複する)
                cursor += block
                self._state.byte_offset = cursor
                if skip > 0:
                    skip -= 1
                    continue
                ids = torch.frombuffer(bytearray(chunk[:-1]), dtype=torch.uint8).long() + off
                tgt = torch.frombuffer(bytearray(chunk[1:]), dtype=torch.uint8).long() + off
                self._state.samples_emitted += 1
                yield {"input_ids": ids, "labels": tgt}


class _ResumableLoader(DataLoader):
    """DataLoader に state_dict / load_state_dict を生やしたラッパ."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._samples_emitted = 0

    def __iter__(self):
        for batch in super().__iter__():
            if isinstance(batch, dict) and "input_ids" in batch:
                self._samples_emitted += int(batch["input_ids"].shape[0])
            yield batch

    def state_dict(self) -> dict[str, Any]:
        ds = self.dataset
        state = ds.state_dict() if hasattr(ds, "state_dict") else {}
        state["samples_emitted"] = self._samples_emitted
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ds = self.dataset
        self._samples_emitted = int(state.get("samples_emitted", 0))
        if hasattr(ds, "load_state_dict"):
            ds.load_state_dict(state)


def build_byte_dataloader(cfg: dict, split: str = "train") -> _ResumableLoader:
    ds = ByteStreamDataset(
        source=cfg.get("source"),
        sources=cfg.get("sources"),
        context_length=cfg["context_length"],
        split=split,
        shuffle_buffer=cfg.get("shuffle_buffer", 0),
        text_column=cfg.get("text_column", "text"),
        byte_offset=cfg.get("byte_offset", 4),
    )
    num_workers = cfg.get("num_workers", 4)
    if num_workers > 1 and not cfg.get("allow_multi_worker_iterable", False):
        raise ValueError(
            "ByteStreamDataset is resumable only with num_workers <= 1. "
            "Set allow_multi_worker_iterable=true only for non-resumable throughput experiments."
        )
    kwargs: dict[str, Any] = dict(
        batch_size=cfg.get("micro_batch_size", 4),
        num_workers=num_workers,
        pin_memory=cfg.get("pin_memory", True),
        drop_last=True,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = cfg.get("prefetch_factor", 4)
        kwargs["persistent_workers"] = True
    return _ResumableLoader(ds, **kwargs)
