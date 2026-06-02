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

    def __init__(self, source: str, context_length: int, split: str = "train",
                 shuffle_buffer: int = 0, text_column: str = "text",
                 byte_offset: int = 4) -> None:
        """byte_offset: バイト値 b を token id (b + offset) に写す.

        BLT は 0..3 を BOE/BOS/EOS/BPE の特殊 ID として使い, 生バイトは
        OFFSET=4 から始まる (vocab_size = OFFSET + 256 = 260). stub model
        ではこの分離が不要なので 0 でも回るが, 実 BLT を使う場合は 4 にする.
        """
        super().__init__()
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
        if self.source.startswith("file:"):
            yield from self._iter_local_file(self.source[len("file:"):])
            return
        yield from self._iter_hf_stream()

    def _iter_hf_stream(self) -> Iterator[dict[str, torch.Tensor]]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError("`datasets` が必要 (pip install datasets)") from e

        ds = load_dataset(self.source, split=self.split, streaming=True)
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=42)

        ring = bytearray()
        block = self.context_length + 1
        skip = self._state.samples_emitted
        off = self.byte_offset

        for row in ds:
            text = row.get(self.text_column)
            if not text:
                continue
            ring.extend(text.encode("utf-8", errors="ignore"))
            ring.extend(b"\n")
            while len(ring) >= block:
                chunk = bytes(ring[:block])
                del ring[:block]
                if skip > 0:
                    skip -= 1
                    continue
                ids = torch.tensor([b + off for b in chunk[:-1]], dtype=torch.long)
                tgt = torch.tensor([b + off for b in chunk[1:]], dtype=torch.long)
                self._state.samples_emitted += 1
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
        skip = self._state.samples_emitted

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
                cursor += 1
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

    def state_dict(self) -> dict[str, Any]:
        ds = self.dataset
        return ds.state_dict() if hasattr(ds, "state_dict") else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ds = self.dataset
        if hasattr(ds, "load_state_dict"):
            ds.load_state_dict(state)


def build_byte_dataloader(cfg: dict, split: str = "train") -> _ResumableLoader:
    ds = ByteStreamDataset(
        source=cfg["source"],
        context_length=cfg["context_length"],
        split=split,
        shuffle_buffer=cfg.get("shuffle_buffer", 0),
        text_column=cfg.get("text_column", "text"),
        byte_offset=cfg.get("byte_offset", 4),
    )
    num_workers = cfg.get("num_workers", 4)
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
