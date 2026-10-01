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


def test_patch_stats_covers_every_byte_and_counts_caps_and_budget():
    module = _module()
    starts = torch.zeros(2, 12, dtype=torch.bool)
    starts[0, [0, 4, 8]] = True
    starts[1, [0, 2, 3, 4, 5, 6]] = True
    st = module.patch_stats(starts, soft_len=6, budget=4)
    assert st["patches_mean"] == 4.5 and st["patches_max"] == 6
    assert abs(st["bytes_per_patch"] - 24 / 9) < 1e-6
    assert abs(st["capped_ratio"] - 1 / 9) < 1e-6
    assert st["over_budget_ratio"] == 0.5


def test_patch_starts_begin_at_position_zero():
    g = torch.Generator().manual_seed(0)
    ent = torch.rand(4, 128, generator=g) * 5.0
    ids = torch.randint(4, 260, (4, 128), generator=g)
    starts = compute_patch_starts(ids, "entropy_char", 1, 32, entropy_values=ent, threshold=0.0)
    assert bool(starts[:, 0].all())
