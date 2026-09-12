"""Post-process Nsight Systems CUDA traces for launch-bound analysis.

This implements the report generation side of Phase 1 (gap histogram and long
gap classification) and Phase 2 (launch-count ranking) of the packed-ternary
launch-bound plan.  It consumes the CSVs exported by Nsight Systems rather
than launching a GPU profile itself:

    nsys stats --report cuda_gpu_trace --format csv \
        --output <dir> --force-export=true <capture>.nsys-rep
    nsys stats --report cuda_api_trace --format csv \
        --output <dir> --force-export=true <capture>.nsys-rep

The CUDA API trace is optional; without it the classifier falls back to the
surrounding GPU entries only.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_GAP_THRESHOLDS_US = (10, 50, 100, 500, 1000)
_TOP_GAPS = 20
_TOP_OPS = 30


@dataclass(frozen=True)
class GapAnalysis:
    total_wall_us: float
    busy_us: float
    idle_us: float
    histogram: dict[int, int]
    top_gaps: tuple[dict[str, Any], ...]
    causes: Counter[str]


def _strip_units(header: str) -> str:
    return header.strip().lower()


def _parse_ns(value: str) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV with Nsight's header row, keyed by lowercased headers."""
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        for raw in reader:
            rows.append(
                {
                    _strip_units(key): (value or "").strip()
                    for key, value in raw.items()
                    if key is not None
                }
            )
    return rows


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _start_ns(row: dict[str, str]) -> float:
    value = _field(row, "start (ns)", "start", "begin", "timestamp (ns)")
    return _parse_ns(value)


def _end_ns(row: dict[str, str], start_ns: float) -> float:
    value = _field(row, "end (ns)", "end", "finish")
    end = _parse_ns(value)
    if end == end and end >= start_ns:
        return end
    return start_ns + _duration_ns(row)


def _duration_ns(row: dict[str, str]) -> float:
    value = _field(row, "duration (ns)", "duration", "duration(ns)")
    return max(0.0, _parse_ns(value))


def _calls_and_time(
    rows: Sequence[dict[str, str]],
) -> tuple[Counter[str], dict[str, list[float]]]:
    calls: Counter[str] = Counter()
    durations: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = _field(row, "name", "kernel", "operation", "op")
        if not name:
            continue
        calls[name] += 1
        durations[name].append(_duration_ns(row) / 1000.0)
    return calls, durations


def _classify_family(name: str) -> str:
    lowered = name.lower()
    rules = (
        ("packed GEMM", ("packed", "bitlinear", "gemm", "tl.dot", "dot_")),
        ("A8 quant", ("quant", "a8", "absmax")),
        ("RMSNorm", ("rms", "layernorm", "norm")),
        ("gate/up", ("gate", "up_proj", "up_projection", "gate_proj")),
        ("activation", ("relu", "silu", "gelu", "act")),
        ("scale/dequant", ("scale", "dequant", "requant")),
        ("copy", ("copy", "d2d", "memcpy", "contiguous")),
        ("optimizer", ("adam", "optim", "step_")),
        ("CUDAGraph", ("graph",)),
    )
    for family, needles in rules:
        if any(needle in lowered for needle in needles):
            return family
    return "other"


def _fusion_candidate(family: str) -> str:
    if family in {"RMSNorm"}:
        return "RMSNorm+A8 quant"
    if family in {"A8 quant"}:
        return "A8 quant+packed GEMM (high effort)"
    if family in {"packed GEMM"}:
        return "fused A8 quant+packed GEMM (high effort)"
    if family in {"gate/up", "activation"}:
        return "gate/up+activation"
    return ""


def launch_count_ranking(
    rows: Sequence[dict[str, str]],
    *,
    active_steps: int = 1,
    top: int = _TOP_OPS,
) -> tuple[dict[str, Any], ...]:
    """Return top-30 ops by both launch count and total GPU time."""
    calls, durations = _calls_and_time(rows)
    if active_steps <= 0:
        active_steps = 1

    def build(name: str, count: int) -> dict[str, Any]:
        times = durations[name]
        median_us = statistics.median(times) if times else 0.0
        total_ms = sum(times) / 1000.0
        family = _classify_family(name)
        return {
            "name": name,
            "calls_step": count / active_steps,
            "median_us": median_us,
            "total_ms_step": total_ms / active_steps,
            "graph": "unknown",
            "family": family,
            "fusion_candidate": _fusion_candidate(family),
        }

    by_calls = sorted(calls.items(), key=lambda item: (-item[1], item[0]))[:top]
    by_time = sorted(
        calls.items(),
        key=lambda item: (-sum(durations[item[0]]), item[0]),
    )[:top]
    names = sorted({name for name, _ in by_calls + by_time})
    return tuple(build(name, calls[name]) for name in names)


def _sorted_gpu_entries(
    rows: Sequence[dict[str, str]],
) -> list[tuple[float, float, dict[str, str]]]:
    entries: list[tuple[float, float, dict[str, str]]] = []
    for row in rows:
        start = _start_ns(row)
        if start != start or start < 0:
            continue
        end = _end_ns(row, start)
        entries.append((start, max(start, end), row))
    entries.sort(key=lambda item: item[0])
    return entries


def _classify_gap(
    before: dict[str, str] | None,
    after: dict[str, str] | None,
    api_rows: Sequence[dict[str, str]],
) -> str:
    del api_rows
    before_name = _field(before or {}, "name", "kernel", "operation", "op").lower()
    after_name = _field(after or {}, "name", "kernel", "operation", "op").lower()
    if "memcpy" in before_name or "memcpy" in after_name:
        return "D. H2D / data loader wait"
    if "graph" in before_name or "graph" in after_name:
        return "G. graph boundary"
    if "sync" in before_name or "sync" in after_name:
        return "B. explicit synchronize"
    if "norm" in before_name or "norm" in after_name:
        return "F. optimizer-side gap"
    return "L. unknown"


def analyze_gaps(
    rows: Sequence[dict[str, str]],
    *,
    api_rows: Sequence[dict[str, str]] = (),
) -> GapAnalysis:
    """Compute busy/idle, gap histogram, top gaps, and cause buckets."""
    entries = _sorted_gpu_entries(rows)
    if not entries:
        return GapAnalysis(0.0, 0.0, 0.0, {}, (), Counter())

    total_busy_ns = sum(max(0.0, end - start) for start, end, _ in entries)
    start_ns = entries[0][0]
    end_ns = max(end for _, end, _ in entries)
    total_wall_ns = max(1.0, end_ns - start_ns)
    idle_ns = max(0.0, total_wall_ns - total_busy_ns)

    hist: Counter[int] = Counter({threshold: 0 for threshold in _GAP_THRESHOLDS_US})
    gaps: list[tuple[float, dict[str, str], dict[str, str]]] = []
    for (_, prev_end, prev_row), (cur_start, _, cur_row) in zip(
        entries, entries[1:]
    ):
        gap_us = (cur_start - prev_end) / 1000.0
        if gap_us <= 0:
            continue
        for threshold in _GAP_THRESHOLDS_US:
            if gap_us > threshold:
                hist[threshold] += 1
        gaps.append((gap_us, prev_row, cur_row))

    top_gaps = sorted(gaps, key=lambda item: item[0], reverse=True)[: _TOP_GAPS]
    top_gap_rows = tuple(
        {
            "gap_us": gap_us,
            "before": _field(before, "name", "kernel", "operation", "op"),
            "after": _field(after, "name", "kernel", "operation", "op"),
            "cause": _classify_gap(before, after, api_rows),
        }
        for gap_us, before, after in top_gaps
    )
    causes = Counter(item["cause"] for item in top_gap_rows)
    return GapAnalysis(
        total_wall_us=total_wall_ns / 1000.0,
        busy_us=total_busy_ns / 1000.0,
        idle_us=idle_ns / 1000.0,
        histogram=dict(sorted(hist.items())),
        top_gaps=top_gap_rows,
        causes=causes,
    )


def render_markdown(
    analysis: GapAnalysis,
    ranking: Sequence[dict[str, Any]],
    *,
    gpu_name: str = "unknown",
) -> str:
    """Render the Phase 1 + Phase 2 report as Markdown."""
    lines = [
        "# Arbor launch-bound analysis",
        "",
        f"GPU: {gpu_name}",
        "",
        "## Phase 1: gap histogram",
        "",
        f"- total_wall_ms: {analysis.total_wall_us / 1000.0:.3f}",
        f"- gpu_busy_ms: {analysis.busy_us / 1000.0:.3f}",
        f"- gpu_idle_ms: {analysis.idle_us / 1000.0:.3f}",
    ]
    if analysis.total_wall_us > 0:
        lines.append(
            f"- gpu_idle_ratio: {analysis.idle_us / analysis.total_wall_us:.1%}"
        )
    lines.append("")
    lines.append("| gap > threshold_us | count |")
    lines.append("|---|---:|")
    for threshold in _GAP_THRESHOLDS_US:
        lines.append(f"| {threshold} | {analysis.histogram.get(threshold, 0)} |")
    lines.append("")
    lines.append("## Phase 1: top gaps")
    lines.append("")
    lines.append("| gap_us | before | after | suspected cause |")
    lines.append("|---|---|---|---|")
    for item in analysis.top_gaps:
        lines.append(
            f"| {item['gap_us']:.2f} | {item['before'] or '-'} | "
            f"{item['after'] or '-'} | {item['cause']} |"
        )
    lines.append("")
    if analysis.causes:
        lines.append("### cause buckets")
        lines.append("")
        for cause, count in analysis.causes.most_common():
            lines.append(f"- {cause}: {count}")
        lines.append("")
    lines.append("## Phase 2: launch count ranking")
    lines.append("")
    lines.append(
        "| kernel/op | calls/step | median_us | total_ms/step | "
        "graph | fusion candidate |"
    )
    lines.append("|---|---:|---:|---:|---|---|")
    for item in ranking:
        lines.append(
            f"| {item['name']} | {item['calls_step']:.1f} | "
            f"{item['median_us']:.3f} | {item['total_ms_step']:.3f} | "
            f"{item['graph']} | {item['fusion_candidate'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-trace-csv", required=True, help="Nsight cuda_gpu_trace CSV export"
    )
    parser.add_argument(
        "--api-trace-csv", help="Optional Nsight cuda_api_trace CSV export"
    )
    parser.add_argument(
        "--active-steps", type=int, default=1, help="Optimizer steps captured"
    )
    parser.add_argument("--gpu-name", default="unknown")
    parser.add_argument("--out", default="arbor_launch_gap_analysis.md")
    args = parser.parse_args()

    gpu_rows = read_csv(args.gpu_trace_csv)
    api_rows = read_csv(args.api_trace_csv) if args.api_trace_csv else []
    analysis = analyze_gaps(gpu_rows, api_rows=api_rows)
    ranking = launch_count_ranking(
        gpu_rows, active_steps=args.active_steps
    )
    markdown = render_markdown(analysis, ranking, gpu_name=args.gpu_name)
    Path(args.out).write_text(markdown + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
