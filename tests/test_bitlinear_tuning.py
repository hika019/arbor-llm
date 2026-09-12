"""Packed ternary runtime tuning policy tests (CPU-only)."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest
import torch

from src.model.bitlinear_tuning import (
    DeviceInfo,
    PACKED_TERNARY_KERNEL_VERSION,
    PackedLaunchConfig,
    PackedTernaryTuner,
    PackedTuningOptions,
    PersistentTuneCache,
    SoftwareInfo,
    TuneFingerprint,
    TuneKey,
    conservative_packed_launch_config,
    load_packed_tuning_plan,
    packed_tune_entry,
    packed_launch_candidates,
    parse_packed_launch_config,
)


def _device(name: str = "Test GPU") -> DeviceInfo:
    return DeviceInfo(
        device_index=0,
        name=name,
        compute_capability=(9, 9),
        total_memory=24 << 30,
        multi_processor_count=128,
        max_threads_per_multi_processor=1536,
        shared_memory_per_block=64 << 10,
        warp_size=32,
    )


def _software(torch_version: str = "test-torch") -> SoftwareInfo:
    return SoftwareInfo(
        torch_version=torch_version,
        cuda_version="test-cuda",
        triton_version="test-triton",
    )


def _key(m: int = 1024) -> TuneKey:
    return TuneKey(
        backend="kmajor_single_dot",
        m=m,
        k=2048,
        n=11264,
        dtype="bfloat16",
        scale_per_output=True,
    )


def _gloo_preflight_worker(
    rank: int,
    world_size: int,
    rendezvous_file: str,
    cache_path: str,
    result_dir: str,
    fail_on_rank0: bool,
) -> None:
    """Exercise distributed preflight in a fresh CPU-only process."""
    import json
    import os
    from pathlib import Path

    import torch
    import src.model.bitlinear_tuning as btl

    # CI containers may not have a hostname resolvable to a local interface.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        device = DeviceInfo(
            device_index=0,
            name="Gloo test device",
            compute_capability=(9, 0),
            total_memory=24 << 30,
            multi_processor_count=128,
            max_threads_per_multi_processor=1536,
            shared_memory_per_block=64 << 10,
            warp_size=32,
        )
        software = SoftwareInfo(
            torch_version="gloo-test-torch",
            cuda_version="gloo-test-cuda",
            triton_version="gloo-test-triton",
        )
        key = TuneKey(
            backend="kmajor_single_dot",
            m=64,
            k=64,
            n=64,
            dtype="bfloat16",
            scale_per_output=True,
        )
        btl._GLOBAL_TUNER.configure(
            PackedTuningOptions(
                mode="auto",
                cache_enabled=True,
                cache_path=Path(cache_path),
                warmup=0,
                iterations=1,
                coarse_iterations=1,
                precise_candidates=1,
            )
        )
        measure_calls = 0

        def measure(*args, **kwargs):
            del args, kwargs
            nonlocal measure_calls
            measure_calls += 1
            if fail_on_rank0:
                raise RuntimeError("intentional rank-0 tune failure")
            # A deterministic CPU stand-in for CUDA event timing.  The real
            # resolver/cache path remains intact while this keeps CI CPU-only.
            return float(measure_calls)

        btl._GLOBAL_TUNER._measure = measure
        result: dict[str, object] = {"rank": rank, "measure_calls": 0}
        try:
            btl.packed_ternary_preflight(
                [(key, lambda config: config)], device=device, software=software
            )
            selection = btl._GLOBAL_TUNER.resolve(
                key=key,
                device=device,
                launcher=lambda config: config,
                software=software,
            )
            result.update(
                outcome="ok",
                source=selection.source,
                winner=selection.config.to_dict(),
            )
        except RuntimeError as exc:
            result.update(outcome="error", error=str(exc))
        finally:
            result["measure_calls"] = measure_calls
            (Path(result_dir) / f"rank-{rank}.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
    finally:
        torch.distributed.destroy_process_group()


def test_tune_key_equality_and_hash_include_shape_and_semantics():
    assert _key() == _key()
    assert hash(_key()) == hash(_key())
    assert _key(2048) != _key()
    assert _key() != TuneKey(
        backend="kmajor_single_dot",
        m=1024,
        k=2048,
        n=11264,
        dtype="float16",
        scale_per_output=True,
    )


def test_observed_tune_key_diagnostics_records_lazy_resolution():
    import src.model.bitlinear_tuning as btl

    btl.clear_observed_packed_tune_keys()
    tuner = PackedTernaryTuner(PackedTuningOptions(mode="off", cache_enabled=False))
    key = _key()
    tuner.resolve(key=key, device=_device(), software=_software(), launcher=lambda _: None)

    assert btl.observed_packed_tune_keys() == frozenset({key})
    btl.clear_observed_packed_tune_keys()
    assert btl.observed_packed_tune_keys() == frozenset()


def test_candidate_generator_is_shape_table_free_for_representative_m_values():
    device = _device()
    candidate_sets = {
        packed_launch_candidates(
            m=m,
            k=2048,
            n=11264,
            backend="kmajor_single_dot",
            device=device,
        )
        for m in (1024, 2048, 4096, 8192)
    }
    assert len(candidate_sets) == 1
    candidates = next(iter(candidate_sets))
    assert len(candidates) == 54
    assert PackedLaunchConfig(128, 64, 64, 4, 2) in candidates
    assert PackedLaunchConfig(128, 128, 64, 4, 2) in candidates
    assert PackedLaunchConfig(128, 128, 64, 8, 2) in candidates
    assert {config.num_stages for config in candidates} == {2, 3, 4, 5}
    assert {config.num_warps for config in candidates} == {2, 4, 8}


def test_candidate_sources_include_triton_official_and_dedupe():
    device = _device()
    from src.model.bitlinear_tuning import (
        _CANDIDATE_SOURCE_ARBOR,
        _CANDIDATE_SOURCE_TRITON_MATMUL,
        _CANDIDATE_SOURCE_TRITON_PERSISTENT,
        packed_launch_candidates_with_sources,
    )

    sources = packed_launch_candidates_with_sources(
        m=1024,
        n=11264,
        k=2048,
        backend="kmajor_single_dot",
        device=device,
    )
    assert len(sources) == 54
    # Official configs with no Arbor overlap:
    assert PackedLaunchConfig(256, 128, 128, 8, 3) in sources
    assert (
        _CANDIDATE_SOURCE_TRITON_MATMUL
        in sources[PackedLaunchConfig(256, 128, 128, 8, 3)]
    )
    assert PackedLaunchConfig(128, 128, 32, 4, 4) in sources
    assert (
        _CANDIDATE_SOURCE_TRITON_MATMUL
        in sources[PackedLaunchConfig(128, 128, 32, 4, 4)]
    )
    # Config shared by matmul and persistent sources must merge the tags:
    overlap = PackedLaunchConfig(128, 256, 64, 8, 3)
    assert overlap in sources
    assert sources[overlap] == frozenset(
        {_CANDIDATE_SOURCE_TRITON_MATMUL, _CANDIDATE_SOURCE_TRITON_PERSISTENT}
    )
    # Arbor-only config retained with its own tag:
    assert PackedLaunchConfig(64, 64, 32, 4, 2) in sources
    assert (
        sources[PackedLaunchConfig(64, 64, 32, 4, 2)]
        == frozenset({_CANDIDATE_SOURCE_ARBOR})
    )
    # Arbor/persistent overlap merges both tags:
    assert sources[PackedLaunchConfig(128, 128, 64, 8, 2)] == frozenset(
        {_CANDIDATE_SOURCE_ARBOR, _CANDIDATE_SOURCE_TRITON_PERSISTENT}
    )


def test_candidate_pruning_removes_extreme_boundary_tiles():
    candidates = packed_launch_candidates(
        m=7,
        k=33,
        n=17,
        backend="kmajor_single_dot",
        device=_device(),
    )
    assert candidates
    assert all(config.block_m <= 64 for config in candidates)
    assert all(config.block_n <= 64 for config in candidates)
    assert all(config.block_k <= 32 for config in candidates)


def test_fixed_tile_parser_and_conservative_defaults():
    assert parse_packed_launch_config("128,64,64,4") == PackedLaunchConfig(
        128, 64, 64, 4, 2
    )
    assert parse_packed_launch_config([64, 128, 32, 8, 3]) == PackedLaunchConfig(
        64, 128, 32, 8, 3
    )
    assert conservative_packed_launch_config("dot") == PackedLaunchConfig(
        32, 64, 128, 4, 3
    )
    with pytest.raises(ValueError, match="BM,BN,BK"):
        parse_packed_launch_config("64,64")
    with pytest.raises(ValueError, match="powers of two"):
        parse_packed_launch_config("48,64,32,4")
    with pytest.raises(ValueError, match="num_warps"):
        parse_packed_launch_config("64,64,32,3")
    with pytest.raises(ValueError, match="unknown packed ternary backend"):
        TuneKey("unknown", 1, 1, 1, "bfloat16", True)
    with pytest.raises(ValueError, match="requires integer"):
        parse_packed_launch_config({"block_m": 64})


def test_persistent_cache_round_trip_and_fingerprint_invalidation(tmp_path):
    cache = PersistentTuneCache(tmp_path / "tune.json")
    fingerprint = TuneFingerprint(_key(), _device(), _software())
    selection_config = PackedLaunchConfig(128, 64, 64, 4, 3)
    from src.model.bitlinear_tuning import TuneSelection

    cache.put(
        fingerprint,
        TuneSelection(selection_config, source="measured", median_ms=0.25),
    )
    loaded = PersistentTuneCache(cache.path).get(fingerprint)
    assert loaded is not None
    assert loaded.config == selection_config
    assert loaded.median_ms == pytest.approx(0.25)

    assert cache.get(
        TuneFingerprint(
            _key(),
            _device(),
            _software(),
            kernel_version=PACKED_TERNARY_KERNEL_VERSION + "_changed",
        )
    ) is None
    assert cache.get(TuneFingerprint(_key(), _device(), _software("other"))) is None
    assert cache.get(TuneFingerprint(_key(), _device("Other GPU"), _software())) is None


def test_load_packed_tuning_plan_filters_for_current_fingerprint(tmp_path):
    path = tmp_path / "tune.json"
    cache = PersistentTuneCache(path)
    config = PackedLaunchConfig(128, 64, 64, 4, 3)
    from src.model.bitlinear_tuning import TuneSelection

    cache.put(
        TuneFingerprint(_key(), _device(), _software()),
        TuneSelection(config, source="measured", median_ms=0.25),
    )
    cache.put(
        TuneFingerprint(_key(2048), _device(), _software("stale")),
        TuneSelection(config, source="measured", median_ms=0.25),
    )

    assert load_packed_tuning_plan(
        path, device=_device(), software=_software()
    ) == {_key(): config}


@pytest.mark.parametrize("payload", ["{not-json", "[]"])
def test_malformed_cache_is_ignored(tmp_path, payload):
    path = tmp_path / "broken.json"
    path.write_text(payload, encoding="utf-8")
    cache = PersistentTuneCache(path)
    with pytest.warns(RuntimeWarning, match="malformed"):
        assert cache.get(TuneFingerprint(_key(), _device(), _software())) is None


def test_non_object_cache_entry_is_ignored(tmp_path):
    path = tmp_path / "broken-entry.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "entries": {
                    TuneFingerprint(_key(), _device(), _software()).cache_id: []
                },
            }
        ),
        encoding="utf-8",
    )
    cache = PersistentTuneCache(path)
    fingerprint = TuneFingerprint(_key(), _device(), _software())
    assert cache.get(fingerprint) is None
    assert load_packed_tuning_plan(
        path, device=_device(), software=_software()
    ) == {}


def test_fixed_and_off_modes_do_not_benchmark():
    called = False

    def launcher(config):
        del config
        nonlocal called
        called = True

    fixed = PackedLaunchConfig(64, 64, 32, 4, 3)
    tuner = PackedTernaryTuner(
        PackedTuningOptions(mode="fixed", fixed_config=fixed, cache_enabled=False)
    )
    assert tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=launcher
    ).config == fixed
    assert not called

    tuner.configure(PackedTuningOptions(mode="off", cache_enabled=False))
    assert tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=launcher
    ).source == "off"
    assert not called


def test_auto_mode_uses_memory_then_persistent_cache(tmp_path, monkeypatch):
    path = tmp_path / "tune.json"
    options = PackedTuningOptions(
        mode="auto",
        cache_enabled=True,
        cache_path=path,
        warmup=0,
        iterations=1,
    )
    tuner = PackedTernaryTuner(options)
    measurements = 0

    def fake_measure(launcher, candidate, device_index, **kwargs):
        del launcher, device_index, kwargs
        nonlocal measurements
        measurements += 1
        return float(candidate.block_m + candidate.block_n + candidate.block_k)

    monkeypatch.setattr(tuner, "_measure", fake_measure)
    first = tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )
    measured_count = measurements
    assert first.source == "measured"
    assert measured_count > 1

    second = tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )
    assert second.source == "memory"
    assert measurements == measured_count

    reloaded = PackedTernaryTuner(options)
    disk = reloaded.resolve(
        key=_key(),
        device=_device(),
        software=_software(),
        launcher=lambda config: pytest.fail(f"unexpected benchmark: {config}"),
    )
    assert disk.source == "cache"
    assert disk.config == first.config
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == 2
    entry = list(payload["entries"].values())[0]
    assert entry["sources"]


def test_checkpoint_tune_entries_reuse_compatible_and_skip_benchmark(monkeypatch):
    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
        )
    )
    measurements = 0

    def fake_measure(launcher, candidate, device_index, **kwargs):
        del launcher, device_index, kwargs
        nonlocal measurements
        measurements += 1
        return float(candidate.block_m)

    monkeypatch.setattr(tuner, "_measure", fake_measure)
    first = tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )
    assert first.source == "measured"
    assert measurements > 0

    entries = list(tuner.snapshot_entries(device=_device(), software=_software()))
    assert len(entries) == 1
    assert entries[0]["fingerprint"]["key"] == _key().to_dict()

    restored = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
        )
    )
    assert (
        restored.preload_checkpoint_entries(
            entries, device=_device(), software=_software()
        )
        == 1
    )

    def must_not_benchmark(*args, **kwargs):
        del args, kwargs
        pytest.fail("compatible checkpoint tune entry must not benchmark")

    monkeypatch.setattr(restored, "_measure", must_not_benchmark)
    reused = restored.resolve(
        key=_key(),
        device=_device(),
        software=_software(),
        launcher=lambda config: pytest.fail("compatible entry must not launch"),
    )
    assert reused.source == "checkpoint"
    assert reused.config == first.config
    assert reused.median_ms == first.median_ms


def test_checkpoint_tune_entries_incompatible_falls_back_to_benchmark(monkeypatch):
    from src.model.bitlinear_tuning import TuneSelection

    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
        )
    )
    entry = packed_tune_entry(
        TuneFingerprint(_key(), _device(), _software()),
        TuneSelection(PackedLaunchConfig(128, 64, 64, 4, 3), source="measured"),
    )
    assert (
        tuner.preload_checkpoint_entries(
            [entry], device=_device("Another GPU"), software=_software()
        )
        == 0
    )

    measurements = 0

    def fake_measure(launcher, candidate, device_index, **kwargs):
        del launcher, device_index, kwargs
        nonlocal measurements
        measurements += 1
        return float(candidate.block_m)

    monkeypatch.setattr(tuner, "_measure", fake_measure)
    selected = tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )
    assert selected.source == "measured"
    assert measurements > 0


def test_checkpoint_tune_entries_shape_change_falls_back(monkeypatch):
    from src.model.bitlinear_tuning import TuneSelection

    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
        )
    )
    entry = packed_tune_entry(
        TuneFingerprint(_key(1024), _device(), _software()),
        TuneSelection(PackedLaunchConfig(128, 64, 64, 4, 3), source="measured"),
    )
    assert (
        tuner.preload_checkpoint_entries(
            [entry], device=_device(), software=_software()
        )
        == 1
    )

    measurements = 0

    def fake_measure(launcher, candidate, device_index, **kwargs):
        del launcher, device_index, kwargs
        nonlocal measurements
        measurements += 1
        return float(candidate.block_m)

    monkeypatch.setattr(tuner, "_measure", fake_measure)
    selected = tuner.resolve(
        key=_key(2048),
        device=_device(),
        software=_software(),
        launcher=lambda config: config,
    )
    assert selected.source == "measured"
    assert measurements > 0


def test_checkpoint_tune_entries_missing_falls_back(monkeypatch):
    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
        )
    )
    assert (
        tuner.preload_checkpoint_entries(
            [], device=_device(), software=_software()
        )
        == 0
    )
    measurements = 0

    def fake_measure(launcher, candidate, device_index, **kwargs):
        del launcher, device_index, kwargs
        nonlocal measurements
        measurements += 1
        return float(candidate.block_m)

    monkeypatch.setattr(tuner, "_measure", fake_measure)
    selected = tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )
    assert selected.source == "measured"
    assert measurements > 0


def test_auto_mode_ignores_cache_config_outside_current_candidates(
    tmp_path, monkeypatch
):
    path = tmp_path / "tune.json"
    options = PackedTuningOptions(
        mode="auto", cache_enabled=True, cache_path=path, warmup=0, iterations=1
    )
    fingerprint = TuneFingerprint(_key(), _device(), _software())
    from src.model.bitlinear_tuning import TuneSelection

    PersistentTuneCache(path).put(
        fingerprint,
        TuneSelection(PackedLaunchConfig(16, 16, 16, 1), source="measured"),
    )
    tuner = PackedTernaryTuner(options)
    measurements = 0

    def fake_measure(launcher, candidate, device_index, **kwargs):
        del launcher, device_index, kwargs
        nonlocal measurements
        measurements += 1
        return float(candidate.block_m)

    monkeypatch.setattr(tuner, "_measure", fake_measure)
    selected = tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )
    assert selected.source == "measured"
    assert selected.config != PackedLaunchConfig(16, 16, 16, 1)
    assert measurements > 0


def test_cache_disabled_does_not_create_file(tmp_path, monkeypatch):
    path = tmp_path / "disabled.json"
    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            cache_path=path,
            warmup=0,
            iterations=1,
        )
    )
    monkeypatch.setattr(
        tuner,
        "_measure",
        lambda launcher, candidate, device_index, **kwargs: float(candidate.block_m),
    )
    tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )
    assert not path.exists()


def test_all_candidate_failures_are_reported(monkeypatch):
    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
        )
    )

    def fail_measure(launcher, candidate, device_index, **kwargs):
        del launcher, candidate, device_index, kwargs
        raise RuntimeError("compile rejected")

    monkeypatch.setattr(tuner, "_measure", fail_measure)
    with pytest.raises(
        RuntimeError,
        match=r"(?s)all packed ternary autotune candidates failed.*"
        r"backend=kmajor_single_dot.*shape=1024x2048x11264",
    ):
        tuner.resolve(
            key=_key(),
            device=_device(),
            software=_software(),
            launcher=lambda config: config,
        )


def test_auto_mode_uses_two_stage_measurement(monkeypatch):
    from src.model.bitlinear_tuning import packed_launch_candidates

    candidate_count = len(
        packed_launch_candidates(
            m=1024,
            n=11264,
            k=2048,
            backend="kmajor_single_dot",
            device=_device(),
        )
    )
    precise_candidates = 3
    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
            coarse_iterations=1,
            precise_candidates=precise_candidates,
        )
    )
    calls: list[PackedLaunchConfig] = []

    def fake_measure(launcher, candidate, device_index, **kwargs):
        del launcher, device_index, kwargs
        calls.append(candidate)
        return float(candidate.block_m)

    monkeypatch.setattr(tuner, "_measure", fake_measure)
    tuner.resolve(
        key=_key(), device=_device(), software=_software(), launcher=lambda config: config
    )

    coarse = calls[:candidate_count]
    precise = calls[candidate_count:]
    assert len(set(coarse)) == candidate_count
    assert len(precise) == precise_candidates
    assert set(precise) <= set(coarse)


def test_render_packed_tuning_report_builds_markdown(tmp_path):
    from src.model.bitlinear_tuning import (
        CandidateFailure,
        CandidateTiming,
        TuneRecord,
        TuneSelection,
        render_packed_tuning_report,
    )

    path = tmp_path / "tune.json"
    fingerprint = TuneFingerprint(_key(), _device(), _software())
    winner = PackedLaunchConfig(128, 64, 64, 4, 3)
    runner_up = PackedLaunchConfig(128, 64, 64, 4, 2)
    cache = PersistentTuneCache(path)
    cache.put(
        fingerprint,
        TuneSelection(
            winner,
            source="measured",
            median_ms=0.2,
            candidate_sources=frozenset({"arbor_base"}),
        ),
        record=TuneRecord(
            candidate_count=54,
            timings=(
                CandidateTiming(winner, 0.2, frozenset({"arbor_base"})),
                CandidateTiming(runner_up, 0.3, frozenset({"arbor_base"})),
            ),
            failures=(
                CandidateFailure(PackedLaunchConfig(256, 128, 128, 8, 3), "oom"),
            ),
            boundary=False,
        ),
    )

    report = render_packed_tuning_report(
        path, device=_device(), software=_software()
    )
    assert "shape=1024x2048x11264" in report
    assert "winner: 128x64x64 warps=4 stages=3" in report
    assert "winner_sources: arbor_base" in report
    assert "candidate_count: 54" in report
    assert "failed_count: 1" in report
    assert "candidate ranking (coarse median, ascending)" in report
    assert "**(winner)**" in report
    assert "| 2 | 128x64x64 | 4 | 2 | 0.3 | - | arbor_base |" in report


def test_nonzero_rank_does_not_measure_and_raises_on_miss(monkeypatch):
    import src.model.bitlinear_tuning as btl

    tuner = PackedTernaryTuner(
        PackedTuningOptions(
            mode="auto",
            cache_enabled=False,
            warmup=0,
            iterations=1,
        )
    )
    monkeypatch.setattr(btl, "_distributed_rank", lambda: 1)
    measured = False

    def fail_measure(*args, **kwargs):
        del args, kwargs
        nonlocal measured
        measured = True
        return 0.0

    monkeypatch.setattr(tuner, "_measure", fail_measure)
    with pytest.raises(RuntimeError, match="cache miss on non-zero rank"):
        tuner.resolve(
            key=_key(),
            device=_device(),
            software=_software(),
            launcher=lambda config: config,
        )
    assert not measured


def test_preflight_rank0_tunes_and_reloads(monkeypatch):
    import src.model.bitlinear_tuning as btl

    keys = [(_key(), lambda config: config)]
    resolve_calls: list[dict] = []
    reload_calls: list[bool] = []

    monkeypatch.setattr(btl, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(
        btl._GLOBAL_TUNER,
        "resolve",
        lambda **kwargs: resolve_calls.append(kwargs),
    )
    monkeypatch.setattr(
        btl._GLOBAL_TUNER,
        "reload_persistent_cache",
        lambda: reload_calls.append(True),
    )

    broadcast = {"called": False}

    def fake_broadcast(obj_list, src=0):
        del obj_list
        assert src == 0
        broadcast["called"] = True

    monkeypatch.setattr(
        btl.torch.distributed, "broadcast_object_list", fake_broadcast
    )
    monkeypatch.setattr(btl.torch.distributed, "barrier", lambda: None)

    btl.packed_ternary_preflight(keys, device=_device(), software=_software())
    assert len(resolve_calls) == 1
    assert reload_calls == [True]
    assert broadcast["called"]


def test_preflight_nonzero_rank_does_not_resolve(monkeypatch):
    import src.model.bitlinear_tuning as btl

    reload_calls: list[bool] = []
    monkeypatch.setattr(btl, "_distributed_rank", lambda: 1)

    def should_not_resolve(**kwargs):
        del kwargs
        raise AssertionError("non-zero rank must not run autotune")

    monkeypatch.setattr(btl._GLOBAL_TUNER, "resolve", should_not_resolve)
    monkeypatch.setattr(
        btl._GLOBAL_TUNER,
        "reload_persistent_cache",
        lambda: reload_calls.append(True),
    )
    monkeypatch.setattr(
        btl.torch.distributed,
        "broadcast_object_list",
        lambda obj_list, src=0: None,
    )
    monkeypatch.setattr(btl.torch.distributed, "barrier", lambda: None)

    btl.packed_ternary_preflight(
        [(_key(), lambda config: config)], device=_device(), software=_software()
    )
    assert reload_calls == [True]


def test_preflight_rank0_failure_is_broadcast(monkeypatch):
    import src.model.bitlinear_tuning as btl

    monkeypatch.setattr(btl, "_distributed_rank", lambda: 0)

    def fail_resolve(**kwargs):
        del kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(btl._GLOBAL_TUNER, "resolve", fail_resolve)
    monkeypatch.setattr(
        btl._GLOBAL_TUNER, "reload_persistent_cache", lambda: None
    )

    broadcast_status: dict[str, object] = {}

    def fake_broadcast(obj_list, src=0):
        del src
        broadcast_status["status"] = obj_list[0]

    monkeypatch.setattr(
        btl.torch.distributed, "broadcast_object_list", fake_broadcast
    )
    monkeypatch.setattr(btl.torch.distributed, "barrier", lambda: None)

    with pytest.raises(RuntimeError, match="failed on global rank 0"):
        btl.packed_ternary_preflight(
            [(_key(), lambda config: config)], device=_device(), software=_software()
        )
    assert broadcast_status["status"]["ok"] is False


def test_preflight_callable_rank0_only_tunes(monkeypatch):
    import src.model.bitlinear_tuning as btl

    tuned: list[bool] = []
    reloaded: list[bool] = []
    monkeypatch.setattr(btl, "_distributed_rank", lambda: 0)

    def tune_all():
        tuned.append(True)

    monkeypatch.setattr(btl._GLOBAL_TUNER, "reload_persistent_cache", lambda: reloaded.append(True))
    monkeypatch.setattr(btl.torch.distributed, "broadcast_object_list", lambda obj, src=0: None)
    monkeypatch.setattr(btl.torch.distributed, "barrier", lambda: None)

    btl.packed_ternary_preflight_callable(tune_all)
    assert tuned == [True]
    assert reloaded == [True]


def test_preflight_callable_nonzero_does_not_tune(monkeypatch):
    import src.model.bitlinear_tuning as btl

    reloaded: list[bool] = []
    monkeypatch.setattr(btl, "_distributed_rank", lambda: 1)

    def must_not_tune():
        raise AssertionError("non-zero rank must not tune")

    monkeypatch.setattr(btl._GLOBAL_TUNER, "reload_persistent_cache", lambda: reloaded.append(True))
    monkeypatch.setattr(btl.torch.distributed, "broadcast_object_list", lambda obj, src=0: None)
    monkeypatch.setattr(btl.torch.distributed, "barrier", lambda: None)

    btl.packed_ternary_preflight_callable(must_not_tune)
    assert reloaded == [True]


def test_preflight_gloo_multiprocess_rank0_cache_and_failure_propagation(tmp_path):
    """Real CPU/Gloo coverage for rank-0-only tune coordination.

    The worker replaces CUDA event timing only; cache persistence and all Gloo
    collectives execute in independent spawned processes.
    """
    world_size = 2
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    cache_path = tmp_path / "packed-tuning-cache.json"

    torch.multiprocessing.spawn(
        _gloo_preflight_worker,
        args=(
            world_size,
            str(tmp_path / "success-rendezvous"),
            str(cache_path),
            str(result_dir),
            False,
        ),
        nprocs=world_size,
        join=True,
    )

    success = [
        json.loads((result_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    assert cache_path.is_file()
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(cache_payload["entries"]) == 1
    assert [item["outcome"] for item in success] == ["ok", "ok"]
    assert success[0]["measure_calls"] > 0
    assert success[1]["measure_calls"] == 0
    assert success[0]["source"] == success[1]["source"] == "cache"
    assert success[0]["winner"] == success[1]["winner"]

    failure_dir = tmp_path / "failure-results"
    failure_dir.mkdir()
    torch.multiprocessing.spawn(
        _gloo_preflight_worker,
        args=(
            world_size,
            str(tmp_path / "failure-rendezvous"),
            str(tmp_path / "failure-cache.json"),
            str(failure_dir),
            True,
        ),
        nprocs=world_size,
        join=True,
    )
    failures = [
        json.loads((failure_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    assert [item["outcome"] for item in failures] == ["error", "error"]
    assert all("failed on global rank 0" in item["error"] for item in failures)
    assert all("intentional rank-0 tune failure" in item["error"] for item in failures)
