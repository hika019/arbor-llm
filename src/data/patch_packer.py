"""学習系列を patch 数で固定して詰める (BLT の packing).

dataloader の chunk を source ごとの lane につないで区切り、1 lane の seq_patches 個の patch を
1 系列にする。max_bytes の枠に入らない patch は次の系列に回す。chunk が届くたびに未確定の
最後の patch の先頭から walk し直す。labels は EOS の次 (新文書の先頭 byte) を -100 にする。
"""
from __future__ import annotations

import threading
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

from src.model.arbor import (
    ENTROPY_MODES,
    ArborConfig,
    patch_candidates,
    patch_len_bounds,
    soft_patch_len,
    walk_patch_starts,
)

# 未確定 patch の先頭より前に残す byte 数 (entropy_char は 1 つ前の文字とその直前を見る)
_LOOKBACK = 8

# (chunk (B, T), 各行の実長, 各行の lane の状態) → (エントロピー (B, T), char_rest | None, 新しい状態)
EntropyFn = Callable[[torch.Tensor, list, list], "tuple[torch.Tensor, torch.Tensor | None, list]"]


@dataclass
class _Lane:
    pending: torch.Tensor                      # まだ系列にしていない byte (int16、CPU)
    lens: list[int] = field(default_factory=list)  # pending の先頭から並ぶ確定 patch の長さ
    open_len: int = 0                          # pending 末尾の未確定 patch の長さ
    lm_state: Any = None                       # entropy_fn の流れの状態 (ByteLM は直前の窓の各層 K/V)
    tail_ids: torch.Tensor | None = None       # 未確定 patch と直前 _LOOKBACK byte
    tail_ent: torch.Tensor | None = None
    tail_rest: torch.Tensor | None = None

    def ready(self, seq_patches: int, max_bytes: int, start: int = 0) -> int:
        """lens[start:] から系列にできる patch 数 (0 = まだ足りない)。枠に収まらない次の patch が
        確定していれば枠で切る."""
        total = 0
        for n, length in enumerate(self.lens[start:start + seq_patches]):
            if total + length > max_bytes:
                return n
            total += length
        return seq_patches if len(self.lens) - start >= seq_patches else 0


class PatchPacker:
    """dataloader の batch (input_ids (B, C) の chunk、source_id (B,)) から学習系列を作る.

    出力 (CPU): input_ids (B, max_bytes) (余りは PAD)、patch_starts (B, max_bytes) bool、
    labels (B, max_bytes)、n_bytes (B,)、n_patches (B,)。
    """

    def __init__(
        self,
        cfg: ArborConfig,
        entropy_model: torch.nn.Module | None,
        *,
        chunk_len: int,
        device: torch.device,
        compute_dtype: torch.dtype = torch.bfloat16,
        use_autocast: bool = True,
        entropy_fn: EntropyFn | None = None,
        compile_entropy: bool = False,
        threshold: float | None = None,
        seq_patches: int | None = None,
        max_bytes: int | None = None,
        byte_buckets: list[int] | None = None,
    ) -> None:
        self.mode = cfg.patching_mode
        self.threshold = float(cfg.entropy_threshold if threshold is None else threshold)
        self.seq_patches = int(seq_patches or cfg.seq_patches)
        self.max_bytes = int(max_bytes or cfg.max_bytes)
        # 系列の byte 長の枠。batch ごとに最長の行が入る最小の枠へ PAD する (最後は max_bytes)
        self.byte_buckets = sorted(int(b) for b in (byte_buckets or [self.max_bytes]))
        if self.byte_buckets[-1] != self.max_bytes:
            raise ValueError(f"byte_buckets の最大 {self.byte_buckets[-1]} は max_bytes {self.max_bytes} と一致させる")
        self.min_len, max_len = patch_len_bounds(cfg)
        self.soft_len = soft_patch_len(self.mode, max_len)
        # 枠より長い patch は系列に入らない
        self.max_len = min(max_len or self.max_bytes, self.max_bytes)
        self.eos, self.pad = int(cfg.eos_token_id), int(cfg.pad_token_id)
        self.chunk_len = int(chunk_len)
        self.device = device
        self.entropy_fn: EntropyFn | None = None
        if self.mode in ENTROPY_MODES:
            if entropy_fn is None and entropy_model is None:
                raise ValueError(f"patching_mode={self.mode} には entropy_model が必要")
            self.entropy_fn = entropy_fn or ByteLMStream(
                entropy_model, device, compute_dtype, use_autocast, compile_entropy,
            )
        self.lanes: dict[int, _Lane] = {}
        self.ready_queue: deque[int] = deque()
        self.flushed = False
        self.source: Iterator[dict[str, Any]] | None = None

    # ------------------------------------------------------------ input
    def set_source(self, source: Iterator[dict[str, Any]]) -> None:
        self.source = source

    def feed(self, batch: dict[str, Any]) -> None:
        """chunk の batch を lane に足して区切る (host 同期 1 回).

        同じ source の chunk が batch に複数あれば、並び順に続けた 1 本として区切る。
        """
        ids = batch["input_ids"]
        source_ids = batch.get("source_id")
        rows: list[tuple[int, torch.Tensor]] = []
        for r in range(ids.size(0)):
            chunk = ids[r][ids[r] != self.pad].cpu()
            if chunk.numel():
                rows.append((int(source_ids[r]) if source_ids is not None else 0, chunk))
        if not rows:
            return
        if any(c.numel() > self.chunk_len for _, c in rows):
            raise ValueError(f"chunk が chunk_len={self.chunk_len} より長い")
        groups: dict[int, list[int]] = {}
        for r, (key, _) in enumerate(rows):
            groups.setdefault(key, []).append(r)
            self.lanes.setdefault(key, _Lane(pending=torch.empty(0, dtype=torch.int16)))

        x = torch.full((len(rows), self.chunk_len), self.pad, dtype=torch.long)
        for r, (_, chunk) in enumerate(rows):
            x[r, :chunk.numel()] = chunk
        x = x.to(self.device, non_blocking=True)
        ent = rest = None
        if self.entropy_fn is not None:
            ent, rest = self._entropy(x, rows, groups)

        starts_dev, walked, seqs = [], [], []
        for key, members in groups.items():
            lane = self.lanes[key]
            tail = 0 if lane.tail_ids is None else lane.tail_ids.numel()
            spans = [(r, rows[r][1].numel()) for r in members]
            seq = torch.cat(([lane.tail_ids] if tail else []) + [x[r, :n] for r, n in spans])[None]
            seq_ent = seq_rest = None
            if ent is not None:
                seq_ent = _cat_tail(lane.tail_ent, torch.cat([ent[r, :n] for r, n in spans])[None])
                if rest is not None:
                    seq_rest = _cat_tail(lane.tail_rest, torch.cat([rest[r, :n] for r, n in spans])[None])
            raw, force, char_start = patch_candidates(
                seq, self.mode, seq_ent, seq_rest, self.threshold, self.eos,
            )
            o = tail - lane.open_len
            starts_dev.append(walk_patch_starts(
                raw[:, o:], force[:, o:], char_start[:, o:], self.min_len, self.max_len, self.soft_len,
            )[0])
            walked.append(seq.size(1) - o)
            seqs.append((seq[0], seq_ent, seq_rest))
        starts_all = torch.cat(starts_dev).cpu()

        offset = 0
        for (key, members), length, (seq, seq_ent, seq_rest) in zip(groups.items(), walked, seqs):
            lane = self.lanes[key]
            pos = torch.nonzero(starts_all[offset:offset + length]).flatten().tolist()
            offset += length
            lane.lens.extend(b - a for a, b in zip(pos, pos[1:]))
            lane.open_len = length - pos[-1]
            keep = lane.open_len + _LOOKBACK
            lane.tail_ids = _last(seq, keep)
            lane.tail_ent = None if seq_ent is None else seq_ent[:, -keep:]
            lane.tail_rest = None if seq_rest is None else seq_rest[:, -keep:]
            new = torch.cat([rows[r][1] for r in members])
            lane.pending = torch.cat((lane.pending, new.to(torch.int16)))
            self._mark_ready(key)

    def _entropy(
        self, x: torch.Tensor, rows: list, groups: dict[int, list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """同じ lane の chunk は前の chunk の状態が要るので、lane ごとの k 番目の chunk をまとめて k 回に分ける."""
        ent = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        rest = None
        for wave in range(max(len(m) for m in groups.values())):
            keys = [key for key, m in groups.items() if len(m) > wave]
            idx = [groups[key][wave] for key in keys]
            e, r, states = self.entropy_fn(
                x[idx], [rows[i][1].numel() for i in idx], [self.lanes[key].lm_state for key in keys],
            )
            ent[idx] = e
            if r is not None:
                rest = torch.zeros_like(ent) if rest is None else rest
                rest[idx] = r
            for key, state in zip(keys, states):
                self.lanes[key].lm_state = state
        return ent, rest

    def flush(self) -> None:
        """流れの終わり: 未確定の最後の patch を確定させる (以後その lane に続きは来ない)."""
        self.flushed = True
        for key, lane in self.lanes.items():
            if lane.open_len:
                lane.lens.append(lane.open_len)
                lane.open_len = 0
            self._mark_ready(key)

    # ------------------------------------------------------------ output
    def next_batch(self, rows: int) -> dict[str, torch.Tensor]:
        """rows 系列の batch。足りなければ source から chunk を読む (尽きたら StopIteration)."""
        while self._rows_ready(rows) < rows:
            if self.source is None:
                raise StopIteration
            self.feed(next(self.source))
        return self._collate([self._take_row() for _ in range(rows)])

    def drain(self, rows: int) -> Iterator[dict[str, torch.Tensor]]:
        """flush 後に残りの系列を全部出す (最後の batch は rows 未満のことがある)."""
        out = []
        while self.ready_queue:
            out.append(self._take_row())
            if len(out) == rows:
                yield self._collate(out)
                out = []
        if out:
            yield self._collate(out)

    def _rows_ready(self, limit: int) -> int:
        """今すぐ出せる系列数 (limit で数えるのをやめる)."""
        count = 0
        for key in self.ready_queue:
            lane, start = self.lanes[key], 0
            while count < limit and (n := lane.ready(self.seq_patches, self.max_bytes, start)):
                count += 1
                start += n
            if count >= limit:
                break
        return count

    def _mark_ready(self, key: int) -> None:
        lane = self.lanes[key]
        n = len(lane.lens) if self.flushed else lane.ready(self.seq_patches, self.max_bytes)
        if n and key not in self.ready_queue:
            self.ready_queue.append(key)

    def _take_row(self) -> dict[str, torch.Tensor | int]:
        key = self.ready_queue.popleft()
        lane = self.lanes[key]
        n = lane.ready(self.seq_patches, self.max_bytes)
        if n == 0:  # flush 後の端数
            n = len(lane.lens)
            while sum(lane.lens[:n]) > self.max_bytes:
                n -= 1
        lens = lane.lens[:n]
        nb = sum(lens)
        ids = lane.pending[:nb].long()
        nxt = lane.pending[nb:nb + 1].long()
        labels = torch.cat((ids[1:], nxt if nxt.numel() else ids.new_full((1,), -100)))
        labels[ids == self.eos] = -100
        starts = torch.zeros(nb, dtype=torch.bool)
        starts[torch.tensor([0] + lens[:-1]).cumsum(0)] = True
        lane.pending = lane.pending[nb:]
        lane.lens = lane.lens[n:]
        if lane.lens and (self.flushed or lane.ready(self.seq_patches, self.max_bytes)):
            self.ready_queue.append(key)
        return {"input_ids": ids, "labels": labels, "patch_starts": starts, "n_patches": n}

    def _collate(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(row["input_ids"].numel() for row in rows)
        b, t = len(rows), next(n for n in self.byte_buckets if n >= longest)
        ids = torch.full((b, t), self.pad, dtype=torch.long)
        labels = torch.full((b, t), -100, dtype=torch.long)
        starts = torch.zeros((b, t), dtype=torch.bool)
        for r, row in enumerate(rows):
            nb = row["input_ids"].numel()
            ids[r, :nb] = row["input_ids"]
            labels[r, :nb] = row["labels"]
            starts[r, :nb] = row["patch_starts"]
        batch = {
            "input_ids": ids, "labels": labels, "patch_starts": starts,
            "n_bytes": torch.tensor([row["input_ids"].numel() for row in rows]),
            "n_patches": torch.tensor([row["n_patches"] for row in rows]),
        }
        if self.device.type == "cuda":
            batch = {k: v.pin_memory() for k, v in batch.items()}
        return batch

    # ------------------------------------------------------------ resume
    def state_dict(self) -> dict[str, Any]:
        def cpu(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.detach().cpu()

        return {
            "lanes": {
                key: {
                    "pending": lane.pending.clone(), "lens": list(lane.lens), "open_len": lane.open_len,
                    "lm_state": _map_tensors(lane.lm_state, cpu), "tail_ids": cpu(lane.tail_ids),
                    "tail_ent": cpu(lane.tail_ent), "tail_rest": cpu(lane.tail_rest),
                }
                for key, lane in self.lanes.items()
            },
            "ready_queue": list(self.ready_queue),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        def dev(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(self.device)

        self.lanes = {
            int(key): _Lane(
                pending=s["pending"], lens=list(s["lens"]), open_len=int(s["open_len"]),
                lm_state=_map_tensors(s["lm_state"], dev), tail_ids=dev(s["tail_ids"]),
                tail_ent=dev(s["tail_ent"]), tail_rest=dev(s["tail_rest"]),
            )
            for key, s in state["lanes"].items()
        }
        self.ready_queue = deque(int(k) for k in state["ready_queue"])


class PrefetchedPacker:
    """packer.next_batch を別スレッド・別 CUDA stream で depth 個先まで作る (ByteLM と packing を学習と重ねる)."""

    def __init__(
        self, packer: PatchPacker, rows: int, depth: int = 3,
        initial_batches: list[dict] | None = None, device: torch.device | None = None,
    ) -> None:
        self.packer, self.rows, self.depth = packer, int(rows), max(1, int(depth))
        self.stream = torch.cuda.Stream(device) if device is not None and device.type == "cuda" else None
        self._replay = deque(initial_batches or [])
        self._buf: deque = deque()
        self._cond = threading.Condition()
        self._pack_lock = threading.Lock()  # packing 中は state_dict を取らない
        self._stop = self._exhausted = False
        self._exc: BaseException | None = None
        if self.stream is not None:
            # load_state_dict で現 stream 上に置いた lane の状態を、別 stream で使い・解放する
            self.stream.wait_stream(torch.cuda.current_stream(device))
            for lane in packer.lanes.values():
                for t in (lane.tail_ids, lane.tail_ent, lane.tail_rest, *(lane.lm_state or ())):
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        t.record_stream(self.stream)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="patch-packer")
        self._thread.start()

    def _worker(self) -> None:
        stream_ctx = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with stream_ctx:
            while True:
                with self._cond:
                    while len(self._buf) >= self.depth and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        return
                with self._pack_lock:
                    try:
                        batch = self.packer.next_batch(self.rows)
                    except StopIteration:
                        batch, done = None, True
                    except BaseException as exc:  # noqa: BLE001 - 消費側で再送出
                        with self._cond:
                            self._exc = exc
                            self._cond.notify_all()
                        return
                    else:
                        done = False
                    with self._cond:
                        if done:
                            self._exhausted = True
                        else:
                            self._buf.append(batch)
                        self._cond.notify_all()
                if done:
                    return

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._replay:
            return self._replay.popleft()
        with self._cond:
            while not self._buf and not self._exhausted and self._exc is None:
                self._cond.wait()
            if self._buf:
                batch = self._buf.popleft()
                self._cond.notify_all()
                return batch
            if self._exc is not None:
                raise self._exc
            raise StopIteration

    def state_dict(self, source_state: Callable[[], Any] | None = None) -> tuple[Any, dict, list[dict]]:
        """(source の状態, packer の状態, 未消費の系列) を同じ瞬間でそろえて返す."""
        with self._pack_lock:
            if self.stream is not None:
                self.stream.synchronize()
            source = source_state() if source_state is not None else None
            packer_state = self.packer.state_dict()
            with self._cond:
                pending = list(self._replay) + list(self._buf)
        return source, packer_state, pending

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=30.0)


def _cat_tail(prev: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    if prev is None:
        return new.float()
    return torch.cat((prev, new.float()), dim=1)


def _last(t: torch.Tensor, n: int) -> torch.Tensor:
    return t[max(t.numel() - n, 0):]


def _map_tensors(state: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    if isinstance(state, torch.Tensor):
        return fn(state)
    if isinstance(state, (tuple, list)):
        return type(state)(_map_tensors(v, fn) for v in state)
    return state


class ByteLMStream:
    """lane ごとに ByteLM の直前 W 位置の各層 K/V (状態 = (K, V, 有効長)) を持ち、chunk をそれに続けて計算する."""

    def __init__(
        self, lm: torch.nn.Module, device: torch.device, compute_dtype: torch.dtype,
        use_autocast: bool, compile_entropy: bool,
    ) -> None:
        if lm.attention_window is None:
            raise ValueError("ByteLM の attention_window が必要")
        attn = lm.layers[0].attn
        self.w = int(lm.attention_window)
        self.ctx_shape = (len(lm.layers), attn.n_kv_heads, self.w, attn.head_dim)
        self.device = device
        self.amp = (lambda: torch.autocast(device_type=device.type, dtype=compute_dtype)) if use_autocast \
            else nullcontext
        self.dtype = compute_dtype if use_autocast else next(lm.parameters()).dtype
        self.score = torch.compile(lm.boundary_entropy_stream, dynamic=False) if compile_entropy \
            else lm.boundary_entropy_stream

    def __call__(self, x: torch.Tensor, lengths: list[int], states: list) -> tuple:
        empty = torch.zeros(self.ctx_shape, dtype=self.dtype, device=self.device)
        k = torch.stack([empty if s is None else s[0] for s in states], dim=1)
        v = torch.stack([empty if s is None else s[1] for s in states], dim=1)
        n_valid = torch.tensor([0 if s is None else s[2] for s in states], device=self.device)
        valid = torch.arange(self.w, device=self.device) >= (self.w - n_valid).unsqueeze(1)
        with torch.no_grad(), self.amp():
            ent, rest, k_keep, v_keep = self.score(
                x, k, v, valid, torch.tensor(lengths, device=self.device),
            )
        new_states = [
            (k_keep[:, r].to(self.dtype), v_keep[:, r].to(self.dtype),
             min((0 if states[r] is None else states[r][2]) + n, self.w))
            for r, n in enumerate(lengths)
        ]
        return ent.float(), None if rest is None else rest.float(), new_states
