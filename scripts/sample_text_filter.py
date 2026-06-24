from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.text_filter import evaluate_text_filter
from src.data.text_filter import resolve_text_filter_config


def _find_source(config: dict[str, Any], source_id: str) -> dict[str, Any]:
    for source in config["data"]["sources"]:
        if source.get("id") == source_id:
            return source
    raise SystemExit(f"source id not found: {source_id}")


def _format_sample(kind: str, index: int, text: str, decision) -> str:
    metrics = decision.metrics
    header = (
        f"## {kind} {index}\n\n"
        f"- reasons: {', '.join(decision.reasons) if decision.reasons else 'none'}\n"
        f"- chars: {metrics.get('chars', 0):.0f}\n"
        f"- ja_char_ratio: {metrics.get('ja_char_ratio', 0):.3f}\n"
        f"- kana_ratio: {metrics.get('kana_ratio', 0):.3f}\n"
        f"- url_count: {metrics.get('url_count', 0):.0f}\n"
        f"- boilerplate_hits: {metrics.get('boilerplate_hits', 0):.0f}\n"
        f"- suspicious_sequence_count: {metrics.get('suspicious_sequence_count', 0):.0f}\n"
        f"- short_line_ratio: {metrics.get('short_line_ratio', 0):.3f}\n\n"
    )
    excerpt = shorten(" ".join(text.split()), width=1400, placeholder=" ...")
    return header + "```text\n" + excerpt + "\n```\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample accepted/rejected documents for a text_filter source.")
    parser.add_argument("--config", default="configs/arbor_1b_8k_utf8.yaml")
    parser.add_argument("--source-id", default="fineweb2_ja")
    parser.add_argument("--accepted", type=int, default=5)
    parser.add_argument("--rejected", type=int, default=5)
    parser.add_argument("--scan-limit", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = _find_source(cfg, args.source_id)
    text_filter = resolve_text_filter_config(source.get("text_filter"))
    if text_filter is None:
        raise SystemExit(f"source has no text_filter: {args.source_id}")

    kwargs: dict[str, Any] = {
        "split": source.get("split", cfg["data"].get("split", "train")),
        "streaming": True,
    }
    if source.get("revision"):
        kwargs["revision"] = source["revision"]
    ds = load_dataset(source["path"], name=source.get("name"), **kwargs)
    col = source.get("text_column", cfg["data"].get("text_column", "text"))

    accepted: list[str] = []
    rejected: list[str] = []
    scanned = 0
    rejected_by_reason: dict[str, int] = {}
    for row in ds:
        scanned += 1
        text = row.get(col)
        if not text:
            continue
        decision = evaluate_text_filter(text, text_filter)
        if decision.accepted and len(accepted) < args.accepted:
            accepted.append(_format_sample("ACCEPTED", len(accepted) + 1, text, decision))
        elif not decision.accepted:
            for reason in decision.reasons:
                rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
            if len(rejected) < args.rejected:
                rejected.append(_format_sample("REJECTED", len(rejected) + 1, text, decision))
        if (len(accepted) >= args.accepted and len(rejected) >= args.rejected) or scanned >= args.scan_limit:
            break

    lines = [
        f"# Text Filter Samples: {args.source_id}",
        "",
        f"- config: `{args.config}`",
        f"- scanned: {scanned}",
        f"- accepted_samples: {len(accepted)}",
        f"- rejected_samples: {len(rejected)}",
        f"- rejected_by_reason: {rejected_by_reason}",
        "",
        "# Accepted",
        "",
        *accepted,
        "# Rejected",
        "",
        *rejected,
    ]
    output = "\n".join(lines)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
