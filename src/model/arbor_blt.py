"""ArborBLT: Local Enc + Global Latent (BitNet 化) + Local Dec を組み立てる.

BLT 公式コード (third_party/blt/bytelatent) を import して構築し、Global
部だけ BitLinear に置換する。BLT 取り込み前の現状ではダミー forward を返す
スケルトンとして動作する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.model.global_latent import swap_linear_to_bitlinear


@dataclass
class ArborOutput:
    logits: torch.Tensor


class _StubArborBLT(nn.Module):
    """BLT 未取り込み時の代替。embed → 1 層 Transformer → head。動作確認用."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        v = cfg["vocab_size"]
        h = cfg["hidden_size"]
        self.embed = nn.Embedding(v, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg["num_attention_heads"],
            dim_feedforward=cfg["intermediate_size"],
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(h, v, bias=False)

    def forward(self, input_ids: torch.Tensor) -> ArborOutput:
        x = self.embed(input_ids)
        x = self.body(x)
        return ArborOutput(logits=self.head(x))


def build_arbor_blt(cfg: dict[str, Any]) -> nn.Module:
    """設定からモデルを構築。BLT 取り込み後はここを差し替える."""
    try:
        # 取り込み後はこちらが活きる
        from bytelatent.model import build_blt_model  # type: ignore
        model = build_blt_model(cfg)
        if cfg.get("bitlinear_in_global", True):
            # Global Latent Transformer 配下のみ置換 (Local Enc/Dec は除外)
            global_block = getattr(model, "global_latent", None)
            if global_block is None:
                raise RuntimeError("BLT モデルから global_latent を特定できない")
            n = swap_linear_to_bitlinear(global_block, skip_names=("lm_head", "embed"))
            print(f"[arbor_blt] BitLinear に置換: {n} 層")
        return model
    except ImportError:
        # BLT 未取り込み時は stub で smoke test 可能にする
        print("[arbor_blt] BLT 未取り込み: stub モデルで起動")
        model = _StubArborBLT(cfg)
        if cfg.get("bitlinear_in_global", False):
            n = swap_linear_to_bitlinear(model.body, skip_names=("embed",))
            print(f"[arbor_blt] stub の body 内 nn.Linear を BitLinear に置換: {n} 層")
        return model
