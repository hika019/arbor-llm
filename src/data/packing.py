"""シーケンスパッキング.

context_length の境界をまたいで連結 → 切り出し → padding 不要にする。
byte_dataset 側で既にリングバッファで詰めているので、ここではバッチ整形に
追加処理が必要なケース (文書境界マスク等) を入れる場所として残す。
現時点では薄いラッパのみ。
"""
from __future__ import annotations

import torch


def collate_packed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }
    for key in ("source_id", "fill_ratio"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
