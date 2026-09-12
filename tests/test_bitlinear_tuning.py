"""Packed ternary runtime tuning policy tests (CPU-only)."""
from __future__ import annotations

import json

import pytest

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
