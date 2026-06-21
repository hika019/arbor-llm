from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_entropy_threshold.py"
    spec = importlib.util.spec_from_file_location("calibrate_entropy_threshold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_count_respects_min_and_max():
    module = _module()
    entropy = np.full(32, 10.0, dtype=np.float32)
    assert module.patch_count(entropy, 0.0, min_len=4, max_len=16) == 8
    entropy.fill(0.0)
    assert module.patch_count(entropy, 10.0, min_len=4, max_len=8) == 4


def test_threshold_monotonically_increases_bytes_per_patch():
    module = _module()
    rng = np.random.default_rng(0)
    samples = [rng.uniform(0.0, 5.0, 128).astype(np.float32) for _ in range(8)]
    low, _ = module.evaluate(samples, 0.5, min_len=3, max_len=16)
    high, counts = module.evaluate(samples, 4.5, min_len=3, max_len=16)
    assert low <= high
    capacity = module.recommend_capacity(counts, context=128, min_len=3)
    assert int(counts.max()) <= capacity <= 43
