from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from src.model.arbor import compute_patch_starts


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "show_entropy_threshold_stats.py"
    spec = importlib.util.spec_from_file_location("show_entropy_threshold_stats", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stats_counts_patch_lengths_caps_and_row_bytes():
    module = _module()
    starts = torch.zeros(2, 16, dtype=torch.bool)
    starts[0, [0, 4, 8]] = True          # 3 patch / 12 byte (系列 byte の対象)
    starts[1, [0, 2, 3, 4, 5, 6]] = True  # 6 patch / 12 byte (seq_patches 未満なので系列 byte に入れない)
    batch = {"input_ids": torch.zeros(2, 16), "patch_starts": starts,
             "n_bytes": torch.tensor([12, 12]), "n_patches": torch.tensor([3, 6])}
    st = module.stats([batch], seq_patches=3, soft_len=6, frame=10)
    assert abs(st["bytes_per_patch"] - 24 / 9) < 1e-6
    assert abs(st["capped_patches"] - 1 / 9) < 1e-6      # 長さ 6 の 1 patch
    assert abs(st["capped_bytes"] - 6 / 24) < 1e-6
    assert st["rows"] == 1 and st["row_max"] == 12
    assert st["over_frame"] == 1.0 and st["frame_pad"] == 0.0
    assert st["len_max"] == 6
    uncapped = module.stats([batch], seq_patches=3, soft_len=None, frame=10)
    assert uncapped["capped_patches"] == 0.0 and uncapped["capped_bytes"] == 0.0


def test_patch_starts_begin_at_position_zero():
    g = torch.Generator().manual_seed(0)
    ent = torch.rand(4, 128, generator=g) * 5.0
    ids = torch.randint(4, 260, (4, 128), generator=g)
    starts = compute_patch_starts(ids, "entropy_char", 1, 32, entropy_values=ent, threshold=0.0)
    assert bool(starts[:, 0].all())
