from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from src.model.arbor import compute_patch_starts


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_entropy_threshold.py"
    spec = importlib.util.spec_from_file_location("calibrate_entropy_threshold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_search_threshold_hits_target_fill_within_budget():
    module = _module()
    g = torch.Generator().manual_seed(0)
    ent = torch.rand(8, 256, generator=g) * 5.0
    ids = torch.randint(4, 260, (8, 256), generator=g)
    budget, max_len = 64, 16

    def counts_at(threshold):
        starts = compute_patch_starts(ids, "entropy", 1, max_len, entropy_values=ent,
                                      threshold=threshold, budget=budget, horizon=256)
        return starts.sum(1).float()

    thr = module.search_threshold(counts_at, 0.9 * budget, -1.0, 6.0, 24)
    counts = counts_at(thr)
    assert abs(float(counts.mean()) - 0.9 * budget) <= 2.0
    assert int(counts.max()) <= budget
