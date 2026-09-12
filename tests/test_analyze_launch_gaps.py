"""CPU-only tests for the Nsight launch-gap post-processor."""
from __future__ import annotations

from scripts.analyze_launch_gaps import (
    analyze_gaps,
    launch_count_ranking,
    read_csv,
    render_markdown,
)


def _rows() -> list[dict[str, str]]:
    return [
        {
            "name": "packed_gemm",
            "start (ns)": "0",
            "end (ns)": "1000",
            "duration (ns)": "1000",
        },
        {
            "name": "a8_quant",
            "start (ns)": "20000",
            "end (ns)": "21000",
            "duration (ns)": "1000",
        },
        {
            "name": "rmsnorm",
            "start (ns)": "160000",
            "end (ns)": "161000",
            "duration (ns)": "1000",
        },
    ]


def test_read_csv_normalizes_headers(tmp_path):
    path = tmp_path / "gpu.csv"
    path.write_text(
        '"Start (ns)","End (ns)","Duration (ns)","Name"\n'
        "0,1000,1000,packed_gemm\n",
        encoding="utf-8",
    )
    assert read_csv(path) == [
        {
            "start (ns)": "0",
            "end (ns)": "1000",
            "duration (ns)": "1000",
            "name": "packed_gemm",
        }
    ]


def test_analyze_gaps_histogram_and_top_gaps():
    analysis = analyze_gaps(_rows())
    assert analysis.total_wall_us == 161.0
    assert analysis.busy_us == 3.0
    assert analysis.idle_us == 158.0
    assert analysis.histogram[10] == 2
    assert analysis.histogram[50] == 1
    assert analysis.histogram[100] == 1
    assert analysis.histogram[500] == 0
    assert analysis.histogram[1000] == 0
    assert len(analysis.top_gaps) == 2
    assert analysis.top_gaps[0]["gap_us"] == 139.0
    assert analysis.top_gaps[0]["cause"] == "F. optimizer-side gap"
    assert analysis.top_gaps[1]["gap_us"] == 19.0
    assert analysis.top_gaps[1]["cause"] == "L. unknown"


def test_launch_count_ranking_classifies_families():
    ranking = launch_count_ranking(_rows(), active_steps=2)
    by_name = {item["name"]: item for item in ranking}
    assert set(by_name) == {"packed_gemm", "a8_quant", "rmsnorm"}
    assert by_name["packed_gemm"]["calls_step"] == 0.5
    assert by_name["packed_gemm"]["family"] == "packed GEMM"
    assert by_name["a8_quant"]["family"] == "A8 quant"
    assert by_name["a8_quant"]["fusion_candidate"] == "A8 quant+packed GEMM (high effort)"
    assert by_name["rmsnorm"]["fusion_candidate"] == "RMSNorm+A8 quant"


def test_render_markdown_contains_report_sections():
    analysis = analyze_gaps(_rows())
    ranking = launch_count_ranking(_rows())
    report = render_markdown(analysis, ranking, gpu_name="RTX 4090")
    assert "GPU: RTX 4090" in report
    assert "total_wall_ms" in report
    assert "gap > threshold_us" in report
    assert "launch count ranking" in report
    assert "packed_gemm" in report
