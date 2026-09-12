"""Runtime launch-config tuning for packed ternary BitLinear kernels.

This module owns hardware/runtime policy only.  The Triton kernel and its
numerical semantics stay in :mod:`src.model.bitlinear`; callers provide a
launcher that executes one candidate against the real tensors.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
import threading
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

try:
    import triton
except Exception:  # pragma: no cover - CUDA stack dependent
    triton = None


PACKED_TUNE_CACHE_SCHEMA = 2
PACKED_TERNARY_KERNEL_VERSION = "packed_ternary_v3_runtime_tuning"
PACKED_BACKENDS = ("dot", "dot_current", "kmajor_current", "kmajor_single_dot")
_TUNING_MODES = ("auto", "fixed", "off")

_CANDIDATE_SOURCE_ARBOR = "arbor_base"
_CANDIDATE_SOURCE_TRITON_MATMUL = "triton_matmul"
_CANDIDATE_SOURCE_TRITON_PERSISTENT = "triton_persistent_matmul"
_CANDIDATE_SOURCE_EXPERIMENTAL = "experimental"

_PACKED_TUNE_SOURCE_TAGS = (
    _CANDIDATE_SOURCE_ARBOR,
    _CANDIDATE_SOURCE_TRITON_MATMUL,
    _CANDIDATE_SOURCE_TRITON_PERSISTENT,
    _CANDIDATE_SOURCE_EXPERIMENTAL,
)

# Structural Arbor-only samples.  These cover M/N/K tile tradeoffs while
# limiting first-use JIT latency; no template encodes a winner for a model
# shape or GPU SKU.  num_stages is expanded to {2, 3} at generation time.
_ARBOR_BASE_TEMPLATES: tuple[tuple[int, int, int, int], ...] = (
    (64, 64, 32, 4),
    (64, 64, 64, 4),
    (64, 128, 32, 4),
    (64, 128, 64, 4),
    (128, 64, 32, 4),
    (128, 64, 64, 4),
    (128, 128, 32, 4),
    (128, 128, 64, 4),
    (128, 128, 32, 8),
    (128, 128, 64, 8),
)

@dataclass(frozen=True, order=True)
class PackedLaunchConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 2

    def __post_init__(self) -> None:
        values = (
            self.block_m,
            self.block_n,
            self.block_k,
            self.num_warps,
            self.num_stages,
        )
        if min(values) <= 0:
            raise ValueError(f"packed launch config values must be positive: {values}")
        if any(value & (value - 1) for value in values[:3]):
            raise ValueError(
                "packed launch block sizes must be powers of two: "
                f"{values[:3]}"
            )
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "packed launch num_warps must be one of 1, 2, 4, 8: "
                f"{self.num_warps}"
            )

    @property
    def tile_string(self) -> str:
        return f"{self.block_m}x{self.block_n}x{self.block_k}"

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PackedLaunchConfig:
        try:
            return cls(
                block_m=int(value["block_m"]),
                block_n=int(value["block_n"]),
                block_k=int(value["block_k"]),
                num_warps=int(value["num_warps"]),
                num_stages=int(value.get("num_stages", 2)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "packed launch config requires integer block_m, block_n, "
                "block_k, num_warps, and optional num_stages"
            ) from exc


@dataclass(frozen=True)
class DeviceInfo:
    device_index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory: int
    multi_processor_count: int
    max_threads_per_multi_processor: int
    shared_memory_per_block: int
    warp_size: int

    @classmethod
    def from_cuda_device(cls, device: torch.device | int | str) -> DeviceInfo:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"packed ternary tuning requires CUDA, got {resolved}")
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        return _device_info_for_index(index)

    def fingerprint_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "compute_capability": list(self.compute_capability),
            "total_memory": self.total_memory,
            "multi_processor_count": self.multi_processor_count,
            "warp_size": self.warp_size,
        }


@dataclass(frozen=True)
class SoftwareInfo:
    torch_version: str
    cuda_version: str | None
    triton_version: str | None

    @classmethod
    def current(cls) -> SoftwareInfo:
        return cls(
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            triton_version=getattr(triton, "__version__", None),
        )


@dataclass(frozen=True)
class TuneKey:
    backend: str
    m: int
    k: int
    n: int
    dtype: str
    scale_per_output: bool

    def __post_init__(self) -> None:
        if self.backend not in PACKED_BACKENDS:
            raise ValueError(
                f"unknown packed ternary backend: {self.backend!r} "
                f"(choices: {PACKED_BACKENDS})"
            )
        if min(self.m, self.k, self.n) <= 0:
            raise ValueError(f"packed tune shape must be positive: {(self.m, self.k, self.n)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuneFingerprint:
    key: TuneKey
    device: DeviceInfo
    software: SoftwareInfo
    kernel_version: str = PACKED_TERNARY_KERNEL_VERSION
    schema: int = PACKED_TUNE_CACHE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kernel_version": self.kernel_version,
            "key": self.key.to_dict(),
            "gpu": self.device.fingerprint_dict(),
            "software": asdict(self.software),
        }

    @property
    def cache_id(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TuneSelection:
    config: PackedLaunchConfig
    source: str
    median_ms: float | None = None
    candidate_sources: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CandidateTiming:
    config: PackedLaunchConfig
    median_ms: float
    sources: frozenset[str] = frozenset()
    precise_median_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "median_ms": self.median_ms,
            "sources": sorted(self.sources),
            "precise_median_ms": self.precise_median_ms,
        }


@dataclass(frozen=True)
class CandidateFailure:
    config: PackedLaunchConfig
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "reason": self.reason}


@dataclass(frozen=True)
class TuneRecord:
    candidate_count: int
    timings: tuple[CandidateTiming, ...]
    failures: tuple[CandidateFailure, ...]
    boundary: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "timings": [timing.to_dict() for timing in self.timings],
            "failures": [failure.to_dict() for failure in self.failures],
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class PackedTuningOptions:
    mode: str = "auto"
    cache_enabled: bool = True
    cache_path: Path | None = None
    fixed_config: PackedLaunchConfig | None = None
    warmup: int = 10
    iterations: int = 30
    coarse_iterations: int = 8
    precise_candidates: int = 4
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _TUNING_MODES:
            raise ValueError(
                f"unknown packed ternary tuning mode: {self.mode!r} "
                f"(choices: {_TUNING_MODES})"
            )
        if self.mode == "fixed" and self.fixed_config is None:
            raise ValueError("packed ternary tuning=fixed requires fixed_config")
        if self.warmup < 0 or self.iterations <= 0:
            raise ValueError("packed tuning warmup must be >=0 and iterations must be >0")
        if self.coarse_iterations <= 0 or self.precise_candidates <= 0:
            raise ValueError(
                "packed tuning coarse_iterations and precise_candidates must be >0"
            )


def default_packed_tune_cache_path() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "arbor" / "packed_ternary_autotune.json"


def parse_packed_launch_config(
    value: PackedLaunchConfig | str | Sequence[int] | dict[str, Any] | None,
) -> PackedLaunchConfig | None:
    if value is None:
        return None
    if isinstance(value, PackedLaunchConfig):
        return value
    if isinstance(value, dict):
        return PackedLaunchConfig.from_dict(value)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) not in (4, 5):
            raise ValueError(
                "fixed packed tile must be BM,BN,BK,WARPS[,STAGES], "
                f"got {value!r}"
            )
        parsed = tuple(int(part) for part in parts)
    else:
        parsed = tuple(int(part) for part in value)
        if len(parsed) not in (4, 5):
            raise ValueError(
                "fixed packed tile must contain BM,BN,BK,WARPS[,STAGES], "
                f"got {parsed!r}"
            )
    if len(parsed) == 4:
        parsed = (*parsed, 2)
    return PackedLaunchConfig(*parsed)


def conservative_packed_launch_config(
    backend: str,
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
) -> PackedLaunchConfig:
    """Return a broad, compile-friendly default, not a hardware winner."""
    if backend not in PACKED_BACKENDS:
        raise ValueError(
            f"unknown packed ternary backend: {backend!r} "
            f"(choices: {PACKED_BACKENDS})"
        )
    del k
    if backend == "dot":
        return PackedLaunchConfig(32, 64, 128, 4, 3)
    if m is not None and n is not None and (m < 32 or n < 64):
        return PackedLaunchConfig(16, 32, 32, 4, 3)
    return PackedLaunchConfig(32, 64, 32, 4, 3)


def _candidate_is_legal(
    config: PackedLaunchConfig,
    *,
    m: int,
    n: int,
    k: int,
    backend: str,
    device: DeviceInfo,
) -> bool:
    if config.block_k % 4 != 0:
        return False
    if backend == "kmajor_single_dot" and config.block_k < 16:
        return False
    if config.num_warps * device.warp_size > 1024:
        return False
    # Avoid extreme boundary-only tiles.  This is shape/resource pruning, not
    # a shape-to-winner mapping.
    if m < 64 and config.block_m > 64:
        return False
    if n < 64 and config.block_n > 64:
        return False
    if k < 64 and config.block_k > 32:
        return False
    return True


# Triton official CUDA matmul autotune configs as of triton 3.6.0, taken from
# python/tutorials/03-matrix-multiplication.py `get_cuda_autotune_config()`.
# GROUP_SIZE_M is a launch-ordering hint for the tutorial's 1D grid and has no
# Arbor analog, so only the (BM, BN, BK, warps, stages) tile is retained.
# HIP-only configs are intentionally excluded.
_TRITON_OFFICIAL_MATMUL_CONFIGS: tuple[PackedLaunchConfig, ...] = (
    PackedLaunchConfig(128, 256, 64, 8, 3),
    PackedLaunchConfig(64, 256, 32, 4, 4),
    PackedLaunchConfig(128, 128, 32, 4, 4),
    PackedLaunchConfig(128, 64, 32, 4, 4),
    PackedLaunchConfig(64, 128, 32, 4, 4),
    PackedLaunchConfig(128, 32, 32, 4, 4),
    PackedLaunchConfig(64, 32, 32, 2, 5),
    PackedLaunchConfig(32, 64, 32, 2, 5),
    PackedLaunchConfig(128, 256, 128, 8, 3),
    PackedLaunchConfig(256, 128, 128, 8, 3),
    PackedLaunchConfig(256, 64, 128, 4, 4),
    PackedLaunchConfig(64, 256, 128, 4, 4),
    PackedLaunchConfig(128, 128, 128, 4, 4),
    PackedLaunchConfig(128, 64, 64, 4, 4),
    PackedLaunchConfig(64, 128, 64, 4, 4),
    PackedLaunchConfig(128, 32, 64, 4, 4),
)

# Triton official persistent matmul CUDA configs as of triton 3.6.0, from
# python/tutorials/09-persistent-matmul.py `matmul_get_configs()`.
# `matmul_tma_persistent_get_configs()` adds EPILOGUE_SUBTILE/WS choices that
# expand the TMA kernel shape, not the tile; its (BM, BN, BK, warps, stages)
# set is identical, so it adds no new tile here.
_TRITON_OFFICIAL_PERSISTENT_MATMUL_CONFIGS: tuple[PackedLaunchConfig, ...] = tuple(
    PackedLaunchConfig(bm, bn, bk, warps, stages)
    for bm in (128,)
    for bn in (128, 256)
    for bk in (64, 128)
    for stages in (2, 3, 4)
    for warps in (4, 8)
)


def _arbor_base_configs(
    *, m: int, n: int, k: int, backend: str
) -> tuple[PackedLaunchConfig, ...]:
    configs = [
        PackedLaunchConfig(bm, bn, bk, warps, stages)
        for bm, bn, bk, warps in _ARBOR_BASE_TEMPLATES
        for stages in (2, 3)
    ]
    configs.append(conservative_packed_launch_config(backend, m=m, n=n, k=k))
    return tuple(configs)


def packed_launch_candidates_with_sources(
    *,
    m: int,
    n: int,
    k: int,
    backend: str,
    device: DeviceInfo,
) -> dict[PackedLaunchConfig, frozenset[str]]:
    """Return the deduplicated candidate space keyed by source tags.

    A config shared by multiple sources retains the union of those tags.  This
    is the authoritative mapping used by both the tuner and the tuning report;
    :func:`packed_launch_candidates` is a thin projection to the later's keys.
    """
    if min(m, n, k) <= 0:
        raise ValueError(f"packed candidate shape must be positive: {(m, k, n)}")

    merged: dict[PackedLaunchConfig, set[str]] = {}

    def add(configs: tuple[PackedLaunchConfig, ...], source: str) -> None:
        for config in configs:
            if _candidate_is_legal(
                config, m=m, n=n, k=k, backend=backend, device=device
            ):
                merged.setdefault(config, set()).add(source)

    add(
        _arbor_base_configs(m=m, n=n, k=k, backend=backend),
        _CANDIDATE_SOURCE_ARBOR,
    )
    add(_TRITON_OFFICIAL_MATMUL_CONFIGS, _CANDIDATE_SOURCE_TRITON_MATMUL)
    add(
        _TRITON_OFFICIAL_PERSISTENT_MATMUL_CONFIGS,
        _CANDIDATE_SOURCE_TRITON_PERSISTENT,
    )
    return {
        config: frozenset(tags)
        for config, tags in sorted(merged.items())
    }


def packed_launch_candidates(
    *,
    m: int,
    n: int,
    k: int,
    backend: str,
    device: DeviceInfo,
) -> tuple[PackedLaunchConfig, ...]:
    """Generate a bounded, GPU-SKU-independent launch search space."""
    return tuple(
        packed_launch_candidates_with_sources(
            m=m, n=n, k=k, backend=backend, device=device
        ).keys()
    )


def _sources_from_json(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    valid = frozenset(_PACKED_TUNE_SOURCE_TAGS)
    return frozenset(str(item) for item in value if str(item) in valid)


def _winner_at_search_boundary(
    winner: PackedLaunchConfig,
    candidates: tuple[PackedLaunchConfig, ...],
) -> bool:
    """Return True when the winner sits at >=2 axes of the search boundary."""
    maxima = (
        max(config.block_m for config in candidates),
        max(config.block_n for config in candidates),
        max(config.block_k for config in candidates),
        max(config.num_warps for config in candidates),
        max(config.num_stages for config in candidates),
    )
    winner_values = (
        winner.block_m,
        winner.block_n,
        winner.block_k,
        winner.num_warps,
        winner.num_stages,
    )
    return sum(a == b for a, b in zip(winner_values, maxima)) >= 2


def _distributed_rank() -> int | None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return torch.distributed.get_rank()


def _distributed_world_size() -> int | None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return torch.distributed.get_world_size()


class PersistentTuneCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._loaded = False
        self._entries: dict[str, Any] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("cache root must be an object")
            if payload.get("schema") != PACKED_TUNE_CACHE_SCHEMA:
                return
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                raise ValueError("entries must be an object")
            self._entries = entries
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"ignoring malformed packed ternary tune cache {self.path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._entries = {}

    def get(self, fingerprint: TuneFingerprint) -> TuneSelection | None:
        self._load()
        raw = self._entries.get(fingerprint.cache_id)
        if not isinstance(raw, dict):
            return None
        if raw.get("fingerprint") != fingerprint.to_dict():
            return None
        try:
            return TuneSelection(
                config=PackedLaunchConfig.from_dict(raw["config"]),
                source="cache",
                median_ms=(
                    None if raw.get("median_ms") is None else float(raw["median_ms"])
                ),
                candidate_sources=_sources_from_json(raw.get("sources")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def entries(self) -> tuple[tuple[str, Any], ...]:
        """Return loaded cache records for frozen-plan construction."""
        self._load()
        return tuple(self._entries.items())

    def put(
        self,
        fingerprint: TuneFingerprint,
        selection: TuneSelection,
        record: TuneRecord | None = None,
    ) -> None:
        self._load()
        entry: dict[str, Any] = {
            "fingerprint": fingerprint.to_dict(),
            "config": selection.config.to_dict(),
            "median_ms": selection.median_ms,
            "sources": sorted(selection.candidate_sources),
        }
        if record is not None:
            entry.update(record.to_dict())
        self._entries[fingerprint.cache_id] = entry
        payload = {
            "schema": PACKED_TUNE_CACHE_SCHEMA,
            "entries": self._entries,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                dir=str(self.path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except OSError as exc:
            warnings.warn(
                f"could not write packed ternary tune cache {self.path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def clear(self) -> None:
        self._entries = {}
        self._loaded = True
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"could not clear packed ternary tune cache {self.path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


class PackedTernaryTuner:
    def __init__(self, options: PackedTuningOptions | None = None) -> None:
        self._options = options or PackedTuningOptions(
            cache_path=default_packed_tune_cache_path()
        )
        self._memory: dict[tuple[Any, ...], TuneSelection] = {}
        self._persistent: PersistentTuneCache | None = None
        self._logged_static: set[tuple[Any, ...]] = set()
        self._lock = threading.RLock()

    @property
    def options(self) -> PackedTuningOptions:
        return self._options

    def configure(self, options: PackedTuningOptions) -> None:
        with self._lock:
            self._options = options
            self._memory.clear()
            self._persistent = None
            self._logged_static.clear()

    def clear(self, *, persistent: bool = False) -> None:
        with self._lock:
            self._memory.clear()
            self._logged_static.clear()
            if persistent:
                cache = self._persistent_cache()
                if cache is not None:
                    cache.clear()

    def _persistent_cache(self) -> PersistentTuneCache | None:
        if not self._options.cache_enabled or self._options.cache_path is None:
            return None
        if self._persistent is None:
            self._persistent = PersistentTuneCache(self._options.cache_path)
        return self._persistent

    def reload_persistent_cache(self) -> None:
        """Discard in-memory state so the next hit re-reads the cache file.

        Distributed preflight calls this after a barrier so every rank sees the
        entries written by global rank 0.
        """
        with self._lock:
            self._memory.clear()
            self._persistent = None

    def resolve(
        self,
        *,
        key: TuneKey,
        device: DeviceInfo,
        launcher: Callable[[PackedLaunchConfig], Any],
        software: SoftwareInfo | None = None,
    ) -> TuneSelection:
        options = self._options
        if options.mode == "fixed":
            assert options.fixed_config is not None
            selection = TuneSelection(options.fixed_config, source="fixed")
            self._log_static_once(key, device, selection)
            return selection
        if options.mode == "off":
            selection = TuneSelection(
                conservative_packed_launch_config(
                    key.backend, m=key.m, n=key.n, k=key.k
                ),
                source="off",
            )
            self._log_static_once(key, device, selection)
            return selection

        fingerprint = TuneFingerprint(
            key=key,
            device=device,
            software=software or SoftwareInfo.current(),
        )
        memory_key = (
            key,
            device.name,
            device.compute_capability,
            device.total_memory,
            device.multi_processor_count,
            fingerprint.software,
            fingerprint.kernel_version,
            fingerprint.schema,
        )
        memory_hit = self._memory.get(memory_key)
        if memory_hit is not None:
            return TuneSelection(
                memory_hit.config,
                source="memory",
                median_ms=memory_hit.median_ms,
                candidate_sources=memory_hit.candidate_sources,
            )
        with self._lock:
            memory_hit = self._memory.get(memory_key)
            if memory_hit is not None:
                return TuneSelection(
                    memory_hit.config,
                    source="memory",
                    median_ms=memory_hit.median_ms,
                    candidate_sources=memory_hit.candidate_sources,
                )

            candidate_map = packed_launch_candidates_with_sources(
                m=key.m,
                n=key.n,
                k=key.k,
                backend=key.backend,
                device=device,
            )
            candidates = tuple(candidate_map.keys())
            persistent = self._persistent_cache()
            disk_hit = persistent.get(fingerprint) if persistent is not None else None
            # A syntactically valid but manually edited/stale cache entry must
            # not bypass the current launch policy.
            if disk_hit is not None and disk_hit.config not in candidates:
                disk_hit = None
            if disk_hit is not None:
                sources = disk_hit.candidate_sources or candidate_map[disk_hit.config]
                resolved = TuneSelection(
                    disk_hit.config,
                    source="cache",
                    median_ms=disk_hit.median_ms,
                    candidate_sources=sources,
                )
                self._memory[memory_key] = resolved
                self._log_selection(key, device, resolved)
                return resolved

            rank = _distributed_rank()
            if rank is not None and rank != 0:
                raise RuntimeError(
                    "packed ternary autotune cache miss on non-zero rank; "
                    "distributed preflight must tune and write the cache on "
                    "global rank 0 before entering the hot path "
                    f"(backend={key.backend} shape={key.m}x{key.k}x{key.n} "
                    f"dtype={key.dtype})"
                )

            # Stage 1: compile + coarse-measure every legal candidate.  This
            # pass JIT-compiles each config and gathers a cheap first ranking
            # so that the expensive precise pass only targets the most
            # promising few configs (two-stage measurement, section 13).
            coarse: list[tuple[float, PackedLaunchConfig]] = []
            failures: list[tuple[PackedLaunchConfig, str]] = []
            for candidate in candidates:
                try:
                    median_ms = self._measure(
                        launcher,
                        candidate,
                        device.device_index,
                        warmup=options.warmup,
                        iterations=options.coarse_iterations,
                    )
                    coarse.append((median_ms, candidate))
                    if options.verbose:
                        print(
                            "[bitlinear-tune] coarse "
                            f"shape={key.m}x{key.k}x{key.n} "
                            f"backend={key.backend} BM={candidate.block_m} "
                            f"BN={candidate.block_n} BK={candidate.block_k} "
                            f"warps={candidate.num_warps} stages={candidate.num_stages} "
                            f"median={median_ms:.4f}ms"
                        )
                except Exception as exc:  # candidate compile/runtime failure
                    failures.append((candidate, f"{type(exc).__name__}: {exc}"))
                    if options.verbose:
                        print(
                            "[bitlinear-tune] rejected "
                            f"shape={key.m}x{key.k}x{key.n} backend={key.backend} "
                            f"config={candidate} reason={failures[-1][1]}"
                        )

            if not coarse:
                attempted = "\n".join(
                    f"  {config}: {reason}" for config, reason in failures
                )
                raise RuntimeError(
                    "all packed ternary autotune candidates failed\n"
                    f"backend={key.backend} shape={key.m}x{key.k}x{key.n} "
                    f"dtype={key.dtype} gpu={device.name!r} "
                    f"cc={device.compute_capability}\n{attempted}"
                )

            coarse.sort(key=lambda item: item[0])
            precise_targets = [
                candidate for _, candidate in coarse[: options.precise_candidates]
            ]

            # Stage 2: precise re-measurement of the coarse top-K only.
            timings: list[tuple[float, PackedLaunchConfig]] = []
            for candidate in precise_targets:
                try:
                    median_ms = self._measure(
                        launcher,
                        candidate,
                        device.device_index,
                        warmup=options.warmup,
                        iterations=options.iterations,
                    )
                    timings.append((median_ms, candidate))
                    if options.verbose:
                        print(
                            "[bitlinear-tune] precise "
                            f"shape={key.m}x{key.k}x{key.n} "
                            f"backend={key.backend} BM={candidate.block_m} "
                            f"BN={candidate.block_n} BK={candidate.block_k} "
                            f"warps={candidate.num_warps} stages={candidate.num_stages} "
                            f"median={median_ms:.4f}ms"
                        )
                except Exception as exc:
                    failures.append((candidate, f"{type(exc).__name__}: {exc}"))
                    if options.verbose:
                        print(
                            "[bitlinear-tune] rejected-final "
                            f"shape={key.m}x{key.k}x{key.n} backend={key.backend} "
                            f"config={candidate} reason={failures[-1][1]}"
                        )
            if not timings:
                attempted = "\n".join(
                    f"  {config}: {reason}" for config, reason in failures
                )
                raise RuntimeError(
                    "all packed ternary autotune candidates failed final measurement\n"
                    f"backend={key.backend} shape={key.m}x{key.k}x{key.n} "
                    f"dtype={key.dtype} gpu={device.name!r} "
                    f"cc={device.compute_capability}\n{attempted}"
                )

            timings.sort(key=lambda item: item[0])
            median_ms, winner = timings[0]
            precise_by_config = {config: ms for ms, config in timings}
            record_timings = tuple(
                CandidateTiming(
                    config=config,
                    median_ms=coarse_ms,
                    sources=candidate_map[config],
                    precise_median_ms=precise_by_config.get(config),
                )
                for coarse_ms, config in coarse
            )
            record_failures = tuple(
                CandidateFailure(config=config, reason=reason)
                for config, reason in failures
            )
            boundary = _winner_at_search_boundary(winner, candidates)
            selection = TuneSelection(
                config=winner,
                source="measured",
                median_ms=median_ms,
                candidate_sources=candidate_map[winner],
            )
            self._memory[memory_key] = selection
            if persistent is not None:
                persistent.put(
                    fingerprint,
                    selection,
                    record=TuneRecord(
                        candidate_count=len(candidates),
                        timings=record_timings,
                        failures=record_failures,
                        boundary=boundary,
                    ),
                )
            if boundary:
                warnings.warn(
                    "packed ternary winner is at >=2 search-space boundary axes; "
                    "consider extending the candidate space",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._log_selection(
                key, device, selection, candidate_count=len(candidates)
            )
            return selection

    def _log_static_once(
        self,
        key: TuneKey,
        device: DeviceInfo,
        selection: TuneSelection,
    ) -> None:
        log_key = (
            key,
            device.device_index,
            selection.config,
            selection.source,
        )
        with self._lock:
            if log_key in self._logged_static:
                return
            self._logged_static.add(log_key)
        self._log_selection(key, device, selection)

    def _measure(
        self,
        launcher: Callable[[PackedLaunchConfig], Any],
        candidate: PackedLaunchConfig,
        device_index: int,
        *,
        warmup: int,
        iterations: int,
    ) -> float:
        with torch.cuda.device(device_index):
            for _ in range(warmup):
                launcher(candidate)
            torch.cuda.synchronize(device_index)
            starts = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(iterations)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(iterations)
            ]
            for start, end in zip(starts, ends):
                start.record()
                launcher(candidate)
                end.record()
            ends[-1].synchronize()
            return statistics.median(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )

    @staticmethod
    def _log_selection(
        key: TuneKey,
        device: DeviceInfo,
        selection: TuneSelection,
        *,
        candidate_count: int | None = None,
    ) -> None:
        config = selection.config
        details = (
            f"[bitlinear-tune] backend={key.backend} "
            f"shape={key.m}x{key.k}x{key.n} "
            f"dtype={key.dtype} "
            f"scale={'per_output' if key.scale_per_output else 'tensorwise'} "
            f"tile={config.tile_string} warps={config.num_warps} "
            f"stages={config.num_stages} source={selection.source}"
        )
        if candidate_count is not None:
            details += f" candidates={candidate_count}"
        if selection.median_ms is not None:
            details += f" median={selection.median_ms:.4f}ms"
        details += (
            f" gpu={device.name!r} "
            f"cc={device.compute_capability[0]}.{device.compute_capability[1]}"
        )
        print(details)


_DEVICE_INFO_CACHE: dict[int, DeviceInfo] = {}


def _device_info_for_index(index: int) -> DeviceInfo:
    cached = _DEVICE_INFO_CACHE.get(index)
    if cached is not None:
        return cached
    props = torch.cuda.get_device_properties(index)
    info = DeviceInfo(
        device_index=index,
        name=props.name,
        compute_capability=(props.major, props.minor),
        total_memory=props.total_memory,
        multi_processor_count=props.multi_processor_count,
        max_threads_per_multi_processor=props.max_threads_per_multi_processor,
        shared_memory_per_block=props.shared_memory_per_block,
        warp_size=props.warp_size,
    )
    _DEVICE_INFO_CACHE[index] = info
    return info


_GLOBAL_TUNER = PackedTernaryTuner(PackedTuningOptions(mode="off"))


def configure_packed_ternary_tuning(
    *,
    mode: str = "auto",
    cache_enabled: bool = True,
    cache_path: str | os.PathLike[str] | None = None,
    fixed_config: PackedLaunchConfig | str | Sequence[int] | dict[str, Any] | None = None,
    warmup: int = 10,
    iterations: int = 30,
    verbose: bool = False,
) -> PackedTuningOptions:
    normalized_mode = str(mode).lower().replace("-", "_")
    parsed_fixed = parse_packed_launch_config(fixed_config)
    resolved_path = (
        default_packed_tune_cache_path()
        if cache_path is None or str(cache_path).lower() == "auto"
        else Path(cache_path).expanduser()
    )
    options = PackedTuningOptions(
        mode=normalized_mode,
        cache_enabled=bool(cache_enabled),
        cache_path=resolved_path,
        fixed_config=parsed_fixed,
        warmup=int(warmup),
        iterations=int(iterations),
        verbose=bool(verbose),
    )
    _GLOBAL_TUNER.configure(options)
    return options


def resolve_packed_launch_config(
    *,
    key: TuneKey,
    device: DeviceInfo | torch.device | int | str,
    launcher: Callable[[PackedLaunchConfig], Any],
) -> TuneSelection:
    device_info = (
        device if isinstance(device, DeviceInfo) else DeviceInfo.from_cuda_device(device)
    )
    return _GLOBAL_TUNER.resolve(key=key, device=device_info, launcher=launcher)


def clear_packed_tuning_cache(*, persistent: bool = False) -> None:
    _GLOBAL_TUNER.clear(persistent=persistent)


def packed_ternary_preflight(
    keys_and_launchers: Sequence[
        tuple[TuneKey, Callable[[PackedLaunchConfig], Any]]
    ],
    *,
    device: DeviceInfo | torch.device | int | str = "cuda",
    software: SoftwareInfo | None = None,
) -> None:
    """Tune all provided keys on rank 0, then synchronize all ranks.

    In a homogeneous distributed job only global rank 0 performs measurement
    and persistent-cache writes.  Non-zero ranks wait, receive the shared
    status, and reload the cache so they see rank-0's results.  A rank-0
    failure is broadcast instead of silently hanging the other ranks.
    """
    device_info = (
        device if isinstance(device, DeviceInfo) else DeviceInfo.from_cuda_device(device)
    )
    software_info = software or SoftwareInfo.current()
    items = list(keys_and_launchers)
    rank = _distributed_rank()

    if rank is None:
        for key, launcher in items:
            _GLOBAL_TUNER.resolve(
                key=key, device=device_info, launcher=launcher, software=software_info
            )
        return

    status: dict[str, Any] = {"ok": True, "error": None}
    if rank == 0:
        try:
            for key, launcher in items:
                _GLOBAL_TUNER.resolve(
                    key=key,
                    device=device_info,
                    launcher=launcher,
                    software=software_info,
                )
        except Exception as exc:
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            status = {"ok": True, "error": None}

    status_list: list[dict[str, Any]] = [status]
    torch.distributed.broadcast_object_list(status_list, src=0)
    torch.distributed.barrier()

    if not status_list[0]["ok"]:
        raise RuntimeError(
            "packed ternary autotune failed on global rank 0: "
            f"{status_list[0]['error']}"
        )
    _GLOBAL_TUNER.reload_persistent_cache()


def packed_ternary_preflight_callable(tune_all: Callable[[], None]) -> None:
    """Run ``tune_all`` on rank 0 only, then synchronize all ranks.

    Unlike :func:`packed_ternary_preflight`, this accepts a side-effecting
    callable (for example a dummy forward/backward that triggers the lazy
    ``resolve`` autotune path) instead of an explicit key/launcher map.  That
    is the integration point most convenient for training, where the actual
    ``TuneKey`` set is discovered only when the packed ops first execute.
    """
    rank = _distributed_rank()
    if rank is None:
        tune_all()
        return

    status: dict[str, Any] = {"ok": True, "error": None}
    if rank == 0:
        try:
            tune_all()
        except Exception as exc:
            status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            status = {"ok": True, "error": None}

    status_list: list[dict[str, Any]] = [status]
    torch.distributed.broadcast_object_list(status_list, src=0)
    torch.distributed.barrier()

    if not status_list[0]["ok"]:
        raise RuntimeError(
            "packed ternary autotune failed on global rank 0: "
            f"{status_list[0]['error']}"
        )
    _GLOBAL_TUNER.reload_persistent_cache()


def load_packed_tuning_plan(
    cache_path: str | os.PathLike[str] | None = None,
    *,
    device: DeviceInfo | torch.device | int | str = "cuda",
    software: SoftwareInfo | None = None,
) -> dict[TuneKey, PackedLaunchConfig]:
    """Load fingerprint-matching cache entries for traceable raw execution."""
    resolved_path = (
        default_packed_tune_cache_path()
        if cache_path is None or str(cache_path).lower() == "auto"
        else Path(cache_path).expanduser()
    )
    device_info = (
        device if isinstance(device, DeviceInfo) else DeviceInfo.from_cuda_device(device)
    )
    software_info = software or SoftwareInfo.current()
    cache = PersistentTuneCache(resolved_path)
    plan: dict[TuneKey, PackedLaunchConfig] = {}
    for cache_id, raw in cache.entries():
        if not isinstance(raw, dict):
            continue
        fingerprint = raw.get("fingerprint")
        if not isinstance(fingerprint, dict):
            continue
        try:
            key = TuneKey(**fingerprint["key"])
            expected = TuneFingerprint(key, device_info, software_info)
            if cache_id != expected.cache_id or fingerprint != expected.to_dict():
                continue
            config = PackedLaunchConfig.from_dict(raw["config"])
        except (KeyError, TypeError, ValueError):
            continue
        if config not in packed_launch_candidates(
            m=key.m,
            n=key.n,
            k=key.k,
            backend=key.backend,
            device=device_info,
        ):
            continue
        plan[key] = config
    return plan


def render_packed_tuning_report(
    cache_path: str | os.PathLike[str] | None = None,
    *,
    device: DeviceInfo | torch.device | int | str = "cuda",
    software: SoftwareInfo | None = None,
) -> str:
    """Render a Markdown report for current-GPU cache entries."""
    resolved_path = (
        default_packed_tune_cache_path()
        if cache_path is None or str(cache_path).lower() == "auto"
        else Path(cache_path).expanduser()
    )
    device_info = (
        device if isinstance(device, DeviceInfo) else DeviceInfo.from_cuda_device(device)
    )
    software_info = software or SoftwareInfo.current()
    cache = PersistentTuneCache(resolved_path)
    gpu_fingerprint = device_info.fingerprint_dict()
    software_fingerprint = asdict(software_info)

    lines = ["# Packed ternary autotune report", ""]
    matched = 0
    for cache_id, raw in sorted(cache.entries(), key=lambda item: item[0]):
        if not isinstance(raw, dict):
            continue
        fingerprint = raw.get("fingerprint")
        if not isinstance(fingerprint, dict):
            continue
        if fingerprint.get("gpu") != gpu_fingerprint:
            continue
        if fingerprint.get("software") != software_fingerprint:
            continue
        try:
            key = TuneKey(**fingerprint["key"])
            winner = PackedLaunchConfig.from_dict(raw["config"])
        except (KeyError, TypeError, ValueError):
            continue
        matched += 1
        sources = _sources_from_json(raw.get("sources"))
        timings = raw.get("timings")
        failures = raw.get("failures")
        candidate_count = raw.get("candidate_count")
        boundary = raw.get("boundary")
        median_ms = raw.get("median_ms")

        lines.append(f"## backend={key.backend} shape={key.m}x{key.k}x{key.n}")
        lines.append("")
        lines.append(f"- dtype: {key.dtype}")
        lines.append(
            f"- scale: {'per_output' if key.scale_per_output else 'tensorwise'}"
        )
        lines.append(
            f"- candidate_count: {candidate_count if candidate_count is not None else '?'}"
        )
        lines.append(
            f"- winner: {winner.tile_string} warps={winner.num_warps} "
            f"stages={winner.num_stages}"
        )
        lines.append(
            f"- winner_sources: {', '.join(sorted(sources)) or '(none)'}"
        )
        lines.append(
            f"- winner_median_ms: "
            f"{median_ms if isinstance(median_ms, (int, float)) else '?'}"
        )
        lines.append(f"- boundary_winner: {bool(boundary)}")
        if isinstance(timings, list) and timings:
            lines.append("- candidate ranking (coarse median, ascending):")
            lines.append(
                "| rank | tile | warps | stages | coarse_ms | precise_ms | sources |"
            )
            lines.append("|---|---:|---|---:|---:|---:|---|")
            for position, item in enumerate(timings, start=1):
                if not isinstance(item, dict):
                    continue
                try:
                    config = PackedLaunchConfig.from_dict(item["config"])
                except (KeyError, TypeError, ValueError):
                    continue
                median = item.get("median_ms")
                precise = item.get("precise_median_ms")
                item_sources = _sources_from_json(item.get("sources"))
                winner_mark = " **(winner)**" if config == winner else ""
                lines.append(
                    f"| {position} | {config.tile_string} | {config.num_warps} | "
                    f"{config.num_stages} | "
                    f"{median if isinstance(median, (int, float)) else '?'} | "
                    f"{precise if isinstance(precise, (int, float)) else '-'} | "
                    f"{','.join(sorted(item_sources)) or '-'}{winner_mark} |"
                )
        if isinstance(failures, list) and failures:
            lines.append(f"- failed_count: {len(failures)}")
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                try:
                    config = PackedLaunchConfig.from_dict(failure["config"])
                except (KeyError, TypeError, ValueError):
                    continue
                lines.append(
                    f"  - {config.tile_string} warps={config.num_warps} "
                    f"stages={config.num_stages}: {failure.get('reason', 'failed')}"
                )
        lines.append("")

    if matched == 0:
        lines.append("(no matching entries for current GPU/software fingerprint)")
        lines.append("")
    return "\n".join(lines)


def current_packed_tuning_options() -> PackedTuningOptions:
    return _GLOBAL_TUNER.options
