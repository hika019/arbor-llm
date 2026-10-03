"""validation の各 domain が読む文書の内容ハッシュ (学習側でこれと一致する文書を除く)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path("data/eval_split")


def doc_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=16).digest()


def domain_spec(domain_cfg: dict[str, Any], validation_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, Any]:
    """domain が読む文書の範囲。bytes は validation が読む byte 数の 2 倍 (末尾の文書の途中までを含める余裕)."""
    batches = int(validation_cfg.get("max_batches", 16))
    micro = int(validation_cfg.get("micro_batch_size") or data_cfg["micro_batch_size"])
    context = int(domain_cfg.get("context_length", data_cfg["context_length"]))
    return {
        "source": domain_cfg["source"],
        "name": domain_cfg.get("name"),
        "revision": domain_cfg.get("revision"),
        "split": domain_cfg.get("split", "train"),
        "text_column": domain_cfg.get("text_column", "text"),
        "skip_samples": int(domain_cfg.get("skip_samples", 0)),
        "bytes": 2 * batches * micro * context,
    }


def eval_doc_hashes(
    validation_cfg: dict[str, Any], data_cfg: dict[str, Any], cache_dir: Path = CACHE_DIR,
) -> frozenset[bytes]:
    hashes: set[bytes] = set()
    for domain, domain_cfg in validation_cfg.get("domains", {}).items():
        spec = domain_spec(domain_cfg, validation_cfg, data_cfg)
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"{domain}_{key}.json"
        if path.exists():
            listed = json.loads(path.read_text(encoding="utf-8"))["hashes"]
        else:
            print(f"[eval_split] {domain}: 評価文書のハッシュを作成中 ({spec['source']} skip={spec['skip_samples']})",
                  flush=True)
            listed = collect_hashes(spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"spec": spec, "hashes": listed}, ensure_ascii=False), encoding="utf-8")
        hashes.update(bytes.fromhex(h) for h in listed)
    return frozenset(hashes)


def collect_hashes(spec: dict[str, Any], rows=None) -> list[str]:
    """validation と同じ順に文書を読み、spec["bytes"] に達するまでのハッシュを返す (rows は試験用)."""
    if rows is None:
        from src.data.byte_dataset import _load_hf_streaming

        ds = _load_hf_streaming(spec["source"], spec["name"], spec["split"],
                                {"revision": spec["revision"], "text_column": spec["text_column"]})
        if spec["skip_samples"]:
            ds = ds.skip(spec["skip_samples"])
        rows = iter(ds)
    out, total = [], 0
    for row in rows:
        text = row.get(spec["text_column"])
        data = text.encode("utf-8", errors="ignore") if text else b""
        if not data:
            continue
        out.append(doc_hash(text).hex())
        total += len(data) + 1
        if total >= spec["bytes"]:
            break
    return out
