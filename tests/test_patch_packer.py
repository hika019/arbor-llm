"""PatchPacker: source の流れを続けて区切り、patch 数固定の系列に詰める."""
from __future__ import annotations

import pytest
import torch

from src.data.patch_packer import PatchPacker
from src.model.arbor import ArborConfig, ByteLM, compute_patch_starts, patch_len_bounds

ALL_MODES = ["static", "utf8", "space", "entropy", "entropy_char"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
PAD, EOS = 3, 2


def _cfg(mode: str, **over) -> ArborConfig:
    cfg = dict(patching_mode=mode, patch_size=4, min_patch_len=1, max_patch_len=8, max_bytes=96,
               seq_patches=12, entropy_threshold=0.3)
    cfg.update(over)
    return ArborConfig.from_dict(cfg)


def _fake_entropy(x: torch.Tensor):
    """因果的で直前 2 byte だけで決まるエントロピー (文脈をつなげば流れ全体と一致する).

    lane の先頭は文脈の代わりに PAD が並ぶので、PAD は流れの先頭の外 (0) と同じに扱う。
    """
    xf = x.float()
    prev = torch.nn.functional.pad(xf[:, :-1], (1, 0))
    prev = torch.where(prev == PAD, 0.0, prev)
    return ((xf * 7 + prev * 3) % 11) / 4, None


def _stream(seed: int, reps: int = 20, text: str = "日本語のテキスト。 Hello world, this is a test. 漢字とかな😀 ") -> torch.Tensor:
    text = text.encode() * reps
    s = torch.tensor(list(text)) + 4
    g = torch.Generator().manual_seed(seed)
    s[torch.rand(len(s), generator=g) < 0.01] = EOS
    return s


def _batches(streams: dict[int, torch.Tensor], chunk: int, rows: int, order: list[int]):
    """source の chunk を order の順に rows 個ずつ batch にする (最後の chunk は PAD 埋め)."""
    pos = {k: 0 for k in streams}
    items = []
    for key in order:
        s = streams[key]
        if pos[key] >= len(s):
            continue
        items.append((key, s[pos[key]:pos[key] + chunk]))
        pos[key] += chunk
    for i in range(0, len(items), rows):
        part = items[i:i + rows]
        ids = torch.full((len(part), chunk), PAD)
        for r, (_, c) in enumerate(part):
            ids[r, :len(c)] = c
        yield {"input_ids": ids, "source_id": torch.tensor([k for k, _ in part])}


def _run(packer: PatchPacker, batches, rows: int = 2) -> list[dict]:
    packer.set_source(iter(batches))
    out = []
    try:
        while True:
            out.append(packer.next_batch(rows))
    except StopIteration:
        pass
    packer.flush()
    return out + list(packer.drain(rows))


def _split_rows(out: list[dict]):
    for b in out:
        for r in range(b["input_ids"].size(0)):
            n = int(b["n_bytes"][r])
            yield b["input_ids"][r, :n], b["patch_starts"][r, :n], b["labels"][r, :n], b, r


def _whole_stream_starts(cfg: ArborConfig, stream: torch.Tensor) -> torch.Tensor:
    lo, hi = patch_len_bounds(cfg)
    ent = _fake_entropy(stream[None])[0] if cfg.patching_mode.startswith("entropy") else None
    return compute_patch_starts(stream[None], cfg.patching_mode, lo, hi, entropy_values=ent,
                                threshold=cfg.entropy_threshold, eos_token_id=EOS)[0]


def _want_labels(stream: torch.Tensor) -> torch.Tensor:
    labels = torch.cat((stream[1:], torch.tensor([-100])))
    labels[stream == EOS] = -100
    return labels


def _packer(cfg: ArborConfig, device: str, chunk: int = 40, **kw) -> PatchPacker:
    fn = _fake_entropy if cfg.patching_mode.startswith("entropy") else None
    lm = ByteLM(dict(hidden_size=16, num_heads=2, num_kv_heads=2, intermediate_size=32,
                     num_hidden_layers=1, max_bytes=256))
    return PatchPacker(cfg, lm, chunk_len=chunk, device=torch.device(device), entropy_fn=fn,
                       use_autocast=False, **kw)


@pytest.mark.parametrize("max_len", [8, None])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_chunked_packing_matches_whole_stream(mode, device, max_len):
    """chunk ごとに区切っても、流れ全体を一度に区切ったのと同じ区切り・byte・labels になる."""
    cfg = _cfg(mode, max_patch_len=max_len)
    stream = _stream(0)
    out = _run(_packer(cfg, device), _batches({0: stream}, 40, 2, [0] * 100))
    ids, starts, labels = (torch.cat(t) for t in list(zip(*_split_rows(out)))[:3])
    assert torch.equal(ids, stream)
    assert torch.equal(starts, _whole_stream_starts(cfg, stream))
    assert torch.equal(labels, _want_labels(stream))
    for b in out:
        assert b["input_ids"].size(1) == cfg.max_bytes
        assert (b["patch_starts"].sum(1) == b["n_patches"]).all()
        assert (b["n_patches"] <= cfg.seq_patches).all()
        for r in range(b["input_ids"].size(0)):
            assert (b["input_ids"][r, int(b["n_bytes"][r]):] == PAD).all()


@pytest.mark.parametrize("mode", ["space", "entropy_char"])
def test_sources_are_packed_in_separate_lanes(mode):
    """source が混ざった batch (同じ source が 1 batch に 2 つも) でも、系列は 1 source の流れ."""
    cfg = _cfg(mode)
    streams = {0: _stream(1), 5: _stream(2, reps=40, text="31 4159 26535 ")}
    order = [0, 0, 5, 0, 5, 5, 0] * 40
    out = _run(_packer(cfg, "cpu"), _batches(streams, 40, 3, order), rows=2)
    got = {k: [] for k in streams}
    for ids, starts, labels, _, _ in _split_rows(out):
        digits = torch.isin(ids - 4, torch.tensor(list(b"0123456789")))
        key = 5 if bool(digits.any()) else 0  # source 0 に数字は無い
        got[key].append((ids, starts, labels))
    for key, stream in streams.items():
        ids, starts, labels = (torch.cat(t) for t in zip(*got[key]))
        assert torch.equal(ids, stream)
        assert torch.equal(starts, _whole_stream_starts(cfg, stream))
        assert torch.equal(labels, _want_labels(stream))


def test_rows_are_cut_by_max_bytes_without_losing_bytes():
    """seq_patches 個が max_bytes に収まらない系列は枠で切り、残りの patch は次の系列に回る."""
    cfg = _cfg("static", patch_size=8, seq_patches=12, max_bytes=40)
    stream = _stream(3)
    out = _run(_packer(cfg, "cpu"), _batches({0: stream}, 40, 2, [0] * 100))
    rows = list(_split_rows(out))
    assert torch.equal(torch.cat([r[0] for r in rows]), stream)
    assert all(r[0].numel() <= 40 for r in rows)
    assert any(int(b["n_patches"][i]) < 12 for _, _, _, b, i in rows)


@pytest.mark.parametrize("max_len", [8, None])
@pytest.mark.parametrize("mode", ["space", "entropy_char"])
def test_resume_from_state_dict_is_exact(mode, max_len):
    cfg = _cfg(mode, max_patch_len=max_len)
    streams = {0: _stream(4), 1: _stream(5, reps=9)}
    batches = list(_batches(streams, 40, 2, [0, 1, 0, 0, 1] * 40))
    want = _run(_packer(cfg, "cpu"), batches)

    first = _packer(cfg, "cpu")
    first.set_source(iter(batches[:7]))
    head = [first.next_batch(2) for _ in range(3)]
    consumed = 7 - sum(1 for _ in first.source)  # 読み残しは再開後の loader が出す
    resumed = _packer(cfg, "cpu")
    resumed.load_state_dict(first.state_dict())
    got = head + _run(resumed, batches[consumed:])
    assert len(got) == len(want)
    for a, b in zip(got, want):
        for k in a:
            assert torch.equal(a[k], b[k]), k


def test_uncapped_patch_longer_than_chunk_matches_whole_stream():
    cfg = _cfg("space", max_patch_len=None, max_bytes=256, seq_patches=6)
    stream = _stream(7, reps=6, text="a" * 150 + " bb cc " + "d" * 70 + " ")
    out = _run(_packer(cfg, "cpu"), _batches({0: stream}, 40, 2, [0] * 100))
    ids, starts = (torch.cat(t) for t in list(zip(*_split_rows(out)))[:2])
    assert torch.equal(ids, stream)
    assert torch.equal(starts, _whole_stream_starts(cfg, stream))
    pos = starts.nonzero().flatten().tolist()
    assert max(b - a for a, b in zip(pos, pos[1:])) > 100


def test_uncapped_patch_is_cut_by_max_bytes():
    cfg = _cfg("space", max_patch_len=None, max_bytes=96, seq_patches=6)
    stream = _stream(8, reps=3, text="z" * 300 + " ")
    out = _run(_packer(cfg, "cpu"), _batches({0: stream}, 40, 2, [0] * 100))
    rows = list(_split_rows(out))
    assert torch.equal(torch.cat([r[0] for r in rows]), stream)
    lens = []
    for _, st, _, _, _ in rows:
        pos = st.nonzero().flatten().tolist()
        lens += [b - a for a, b in zip(pos, pos[1:] + [st.numel()])]
    assert max(lens) == 96


def test_bytelm_context_makes_entropy_match_whole_stream():
    """窓付き ByteLM は直前の窓ぶんの byte を文脈にすれば、chunk ごとでも流れ全体と同じ区切り."""
    torch.manual_seed(0)
    lm = ByteLM(dict(hidden_size=32, num_heads=2, num_kv_heads=2, intermediate_size=64,
                     num_hidden_layers=1, max_bytes=2048, attention_window=16)).eval()
    cfg = _cfg("entropy_char", entropy_threshold=0.05)
    stream = _stream(6)
    packer = PatchPacker(cfg, lm, chunk_len=64, device=torch.device("cpu"), use_autocast=False)
    out = _run(packer, _batches({0: stream}, 64, 2, [0] * 100))
    starts = torch.cat([r[1] for r in _split_rows(out)])
    with torch.no_grad():
        ent, _ = lm.boundary_entropy(stream[None])
    lo, hi = patch_len_bounds(cfg)
    want = compute_patch_starts(stream[None], "entropy_char", lo, hi, entropy_values=ent,
                                threshold=0.05, eos_token_id=EOS)[0]
    assert int(want.sum()) > len(stream) // 8  # 閾値が実際に効く状態で比べる
    assert torch.equal(starts, want)
