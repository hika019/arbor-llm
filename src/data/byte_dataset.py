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
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from src.data.text_filter import evaluate_text_filter
from src.data.text_filter import resolve_text_filter_config


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
                 sources: list[dict[str, Any]] | None = None,
                 packing: str = "concat",
                 eos_token_id: int = 2,
                 pad_token_id: int = 3,
                 seed: int = 42,
                 skip_samples: int = 0,
                 name: str | None = None,
                 revision: str | None = None) -> None:
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
        self.name = name
        self.revision = revision
        self.byte_offset = byte_offset
        self.packing = packing
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed
        self.skip_samples = skip_samples
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
        if self.packing == "document":
            yield from self._iter_hf_document_packed()
            return
        if self.packing != "concat":
            raise ValueError(f"unknown data.packing: {self.packing}")
        yield from self._iter_hf_stream()

    def _build_hf_stream(self):
        """単一/複数ソースを HF streaming の行イテレータに組み立てる.

        複数の場合は各ソースを "text" 列に正規化してから weight を確率に
        正規化し interleave_datasets で混ぜる. seed 固定なので skip ベースの
        resume でも同じ順序を再現できる.
        """
        from datasets import load_dataset

        def _one(path, name, col, split, revision=None, skip_samples=0):
            kwargs = {"split": split, "streaming": True}
            if revision:
                kwargs["revision"] = revision
            ds = load_dataset(path, name=name, **kwargs)
            if col != "text":
                ds = ds.rename_column(col, "text")
            ds = ds.select_columns(["text"])  # スキーマ衝突回避 (混合時)
            if skip_samples:
                ds = ds.skip(int(skip_samples))
            return ds

        if not self.sources:
            ds = _one(
                self.source, self.name, self.text_column, self.split,
                self.revision, self.skip_samples,
            )
            if self.shuffle_buffer > 0:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
            return ds

        from datasets import interleave_datasets

        streams, weights = [], []
        for s in self.sources:
            streams.append(_one(
                s["path"], s.get("name"),
                s.get("text_column", self.text_column),
                s.get("split", self.split),
                s.get("revision"),
                s.get("skip_samples", self.skip_samples),
            ))
            weights.append(float(s.get("weight_bytes", s.get("weight", 1.0))))
        total = sum(weights) or 1.0
        probs = [w / total for w in weights]
        # all_exhausted: 巨大コーパスでは実質無限。比率を保ちつつ枯渇分は再サンプル.
        ds = interleave_datasets(
            streams, probabilities=probs, seed=self.seed,
            stopping_strategy="all_exhausted",
        )
        if self.shuffle_buffer > 0:
            # source ごとに shuffle すると buffer_size * source数 の行を保持し、
            # WSL の小さめの RAM 上限では OOM になりやすい。混合後に一度だけ shuffle する。
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return ds

    def _build_hf_source_streams(self):
        """Return per-source HF streams normalized to a ``text`` column."""
        from datasets import load_dataset

        specs = self.sources or [{
            "path": self.source,
            "name": self.name,
            "revision": self.revision,
            "text_column": self.text_column,
            "split": self.split,
            "weight_bytes": 1.0,
        }]
        streams = []
        for idx, s in enumerate(specs):
            kwargs = {
                "split": s.get("split", self.split),
                "streaming": True,
            }
            if s.get("revision"):
                kwargs["revision"] = s["revision"]
            ds = load_dataset(s["path"], name=s.get("name"), **kwargs)
            col = s.get("text_column", self.text_column)
            if col != "text":
                ds = ds.rename_column(col, "text")
            ds = ds.select_columns(["text"])
            skip_samples = int(s.get("skip_samples", self.skip_samples))
            if skip_samples:
                ds = ds.skip(skip_samples)
            if self.shuffle_buffer > 0:
                # Divide the configured buffer across sources to avoid multiplying memory use.
                per_source_buffer = max(1, self.shuffle_buffer // max(len(specs), 1))
                ds = ds.shuffle(buffer_size=per_source_buffer, seed=self.seed + idx)
            streams.append(ds)
        return streams, [dict(s) for s in specs]

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

    def _iter_hf_document_packed(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield fixed-length samples while preserving document boundaries.

        Short documents are packed as ``doc EOS doc EOS ...``. Labels whose
        target crosses from an EOS token into the next document are masked with
        ``-100``. Until model-side attention reset exists, each sample contains
        documents from only one source.
        """
        try:
            streams, specs = self._build_hf_source_streams()
        except ImportError as e:
            raise RuntimeError("`datasets` が必要 (pip install datasets)") from e

        source_iters = [iter(ds) for ds in streams]
        weights = [float(s.get("weight_bytes", s.get("weight", 1.0))) for s in specs]
        total_weight = sum(weights) or 1.0
        probs = [w / total_weight for w in weights]
        emitted_source_bytes = list(
            self._state.extra.get("emitted_source_bytes", [0 for _ in specs])
        )
        emitted_source_docs = list(
            self._state.extra.get("emitted_source_docs", [0 for _ in specs])
        )
        source_epochs = list(self._state.extra.get("source_epochs", [0 for _ in specs]))
        active = [True for _ in specs]
        rng = random.Random(self.seed + self._state.samples_emitted)
        off = self.byte_offset
        block = self.context_length + 1
        source_names = [
            s.get("id") or s.get("name") or s.get("path") or f"source_{i}"
            for i, s in enumerate(specs)
        ]
        source_generators = [
            self._source_document_samples(
                idx, streams, source_iters, specs, source_epochs,
                emitted_source_docs, block, off,
            )
            for idx in range(len(specs))
        ]

        def choose_source() -> int | None:
            live = [i for i, is_active in enumerate(active) if is_active]
            if not live:
                return None
            if sum(emitted_source_bytes[i] for i in live) == 0:
                return rng.choices(live, weights=[probs[i] for i in live], k=1)[0]
            return min(live, key=lambda i: emitted_source_bytes[i] / max(probs[i], 1e-12))

        while True:
            idx = choose_source()
            if idx is None:
                return
            try:
                sample = next(source_generators[idx])
            except StopIteration:
                active[idx] = False
                continue
            source_bytes = int(sample.pop("_source_bytes"))
            emitted_source_bytes[idx] += source_bytes
            self._state.samples_emitted += 1
            stats = {
                "source_names": source_names,
                "target_byte_probs": probs,
                "emitted_source_bytes": emitted_source_bytes,
                "emitted_source_docs": emitted_source_docs,
                "source_epochs": source_epochs,
            }
            self._state.extra["source_stats"] = stats
            self._state.extra["emitted_source_bytes"] = emitted_source_bytes
            self._state.extra["emitted_source_docs"] = emitted_source_docs
            self._state.extra["source_epochs"] = source_epochs
            yield sample

    def _source_document_samples(
        self,
        source_idx: int,
        streams: list[Any],
        source_iters: list[Iterator],
        specs: list[dict[str, Any]],
        source_epochs: list[int],
        emitted_source_docs: list[int],
        block: int,
        off: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        pack: list[int] = []
        pack_is_raw_byte: list[bool] = []
        boundary_label_positions: list[int] = []
        text_filter = resolve_text_filter_config(specs[source_idx].get("text_filter"))

        def next_doc_bytes() -> bytes | None:
            while True:
                try:
                    row = next(source_iters[source_idx])
                except StopIteration:
                    source_epochs[source_idx] += 1
                    max_epochs = specs[source_idx].get("max_epochs")
                    if max_epochs is not None and source_epochs[source_idx] >= int(max_epochs):
                        return None
                    source_iters[source_idx] = iter(streams[source_idx])
                    continue
                text = row.get("text")
                if not text:
                    continue
                if text_filter is not None:
                    decision = evaluate_text_filter(text, text_filter)
                    stats = self._state.extra.setdefault("text_filter_stats", {})
                    source_key = specs[source_idx].get("id") or str(source_idx)
                    source_stats = stats.setdefault(source_key, {"accepted": 0, "rejected": 0})
                    if not decision.accepted:
                        source_stats["rejected"] += 1
                        continue
                    source_stats["accepted"] += 1
                data = text.encode("utf-8", errors="ignore")
                if not data:
                    continue
                emitted_source_docs[source_idx] += 1
                return data

        def make_sample(*, pad_final: bool = False) -> dict[str, torch.Tensor]:
            nonlocal pack, pack_is_raw_byte, boundary_label_positions
            seq = list(pack[:block])
            raw_flags = list(pack_is_raw_byte[:block])
            labels_mask = torch.zeros(self.context_length, dtype=torch.bool)
            fill_tokens = min(sum(tok != self.pad_token_id for tok in seq[:self.context_length]), self.context_length)
            if len(seq) < block:
                if not pad_final:
                    raise RuntimeError("internal error: attempted to emit a short non-final pack")
                seq.extend([self.pad_token_id] * (block - len(seq)))
                raw_flags.extend([False] * (block - len(raw_flags)))
            ids = torch.tensor(seq[:-1], dtype=torch.long)
            labels = torch.tensor(seq[1:], dtype=torch.long)
            for pos in boundary_label_positions:
                if 0 <= pos < self.context_length:
                    labels_mask[pos] = True
            labels_mask |= labels == self.pad_token_id
            labels[labels_mask] = -100
            source_bytes = sum(raw_flags[:self.context_length])
            fill_ratio = fill_tokens / max(self.context_length, 1)
            pack = pack[block:]
            pack_is_raw_byte = pack_is_raw_byte[block:]
            boundary_label_positions = [pos - block for pos in boundary_label_positions if pos >= block]
            return {
                "input_ids": ids,
                "labels": labels,
                "source_id": torch.tensor(source_idx, dtype=torch.long),
                "fill_ratio": torch.tensor(fill_ratio, dtype=torch.float32),
                "_source_bytes": source_bytes,
            }

        while True:
            raw = next_doc_bytes()
            if raw is None:
                if pack:
                    yield make_sample(pad_final=True)
                return
            tokens = [b + off for b in raw]
            doc_seq = tokens + [self.eos_token_id]
            if pack:
                boundary_label_positions.append(len(pack) - 1)
            pack.extend(doc_seq)
            pack_is_raw_byte.extend([True] * len(tokens) + [False])
            while len(pack) >= block:
                yield make_sample()

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

    def shutdown_workers(self) -> None:
        """Stop persistent DataLoader workers before process teardown."""
        iterator = getattr(self, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        self._iterator = None

    def source_stats(self) -> dict[str, Any] | None:
        ds = self.dataset
        if not hasattr(ds, "state_dict"):
            return None
        extra = ds.state_dict().get("extra", {})
        stats = extra.get("source_stats")
        return dict(stats) if isinstance(stats, dict) else None


def build_byte_dataloader(cfg: dict, split: str = "train") -> _ResumableLoader:
    ds = ByteStreamDataset(
        source=cfg.get("source"),
        sources=cfg.get("sources"),
        context_length=cfg["context_length"],
        split=cfg.get("split", split),
        shuffle_buffer=cfg.get("shuffle_buffer", 0),
        text_column=cfg.get("text_column", "text"),
        name=cfg.get("name"),
        revision=cfg.get("revision"),
        byte_offset=cfg.get("byte_offset", 4),
        packing=cfg.get("packing", "concat"),
        eos_token_id=cfg.get("eos_token_id", 2),
        pad_token_id=cfg.get("pad_token_id", 3),
        seed=cfg.get("seed", 42),
        skip_samples=cfg.get("skip_samples", 0),
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
