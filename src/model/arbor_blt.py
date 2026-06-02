"""ArborBLT: BLT 本体 + Global Transformer への BitLinear 置換.

BLT 公式 (third_party/blt/bytelatent) を import して構築し、
`global_transformer` 配下の nn.Linear のみ BitLinear (W1.58A8) に差し替える.
Local Encoder/Decoder は FP 維持.

`patch_in_forward=True` を前提に、トークンだけ渡せば内部で patch を切る.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.model.global_latent import swap_linear_to_bitlinear

_BLT_PATH = Path(__file__).resolve().parents[2] / "third_party" / "blt"


@dataclass
class ArborOutput:
    logits: torch.Tensor


class _StubArborBLT(nn.Module):
    """BLT が利用不可な環境 (CUDA 無し等) で smoke 用に立てる代替モデル."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        v = cfg["vocab_size"]
        h = cfg["hidden_size"]
        self.embed = nn.Embedding(v, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=cfg["num_attention_heads"],
            dim_feedforward=cfg["intermediate_size"],
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(h, v, bias=False)

    def forward(self, input_ids: torch.Tensor) -> ArborOutput:
        x = self.embed(input_ids)
        x = self.body(x)
        return ArborOutput(logits=self.head(x))


class _BLTWrapper(nn.Module):
    """BLT 出力 (logits Tensor) を ArborOutput に揃えるラッパ."""

    def __init__(self, blt: nn.Module) -> None:
        super().__init__()
        self.blt = blt

    def forward(self, input_ids: torch.Tensor) -> ArborOutput:
        out = self.blt(input_ids)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        return ArborOutput(logits=logits)


def _build_blt(cfg: dict[str, Any]) -> nn.Module:
    """BLT 本体をビルド. cross-attention は無効化 (FlexAttention 経路を回避)."""
    # third_party/blt を import path に追加
    if str(_BLT_PATH) not in sys.path:
        sys.path.insert(0, str(_BLT_PATH))
    os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")

    from bytelatent.model.blt import ByteLatentTransformer, ByteLatentTransformerArgs

    h = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_layers = cfg["num_hidden_layers"]
    max_seq = cfg.get("max_position_embeddings", 256)
    patch_size = cfg.get("patch_size", 4)

    args = ByteLatentTransformerArgs(
        vocab_size=cfg["vocab_size"],
        dim=h, dim_global=h, dim_token=h,
        dim_local_encoder=h, dim_local_decoder=h,
        n_layers=n_layers,
        n_layers_global=n_layers,
        n_layers_local_encoder=cfg.get("num_local_layers", 1),
        n_layers_local_decoder=cfg.get("num_local_layers", 1),
        n_heads=n_heads, n_heads_global=n_heads,
        n_heads_local_encoder=n_heads, n_heads_local_decoder=n_heads,
        patch_size=patch_size, patching_mode=cfg.get("patching_mode", "space"),
        max_encoder_seq_length=max_seq, max_seqlen=max_seq, max_length=max_seq // patch_size,
        use_local_encoder_transformer=True,
        # cross-attention は FlexAttention 必須で Triton assert に当たるため一旦無効化
        cross_attn_encoder=False, cross_attn_decoder=False,
        cross_attn_all_layers_decoder=False, cross_attn_all_layers_encoder=False,
        cross_attn_init_by_pooling=True, cross_attn_use_flex_attention=False,
        attn_impl="sdpa", attn_bias_type="causal",
        non_linearity="swiglu", use_rope=True,
        pad_to_max_length=False, downsampling_by_pooling="max",
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=50002,
        encoder_hash_byte_group_nb_functions=3,
        patch_in_forward=True,
    )
    return ByteLatentTransformer(args)


def build_arbor_blt(cfg: dict[str, Any]) -> nn.Module:
    """設定からモデルを構築. BLT が import 可なら BLT、不可なら stub."""
    backend = cfg.get("backend", "blt")
    if backend == "stub":
        model = _StubArborBLT(cfg)
        if cfg.get("bitlinear_in_global", False):
            n = swap_linear_to_bitlinear(model.body, skip_names=("embed",))
            print(f"[arbor_blt] stub body の Linear を BitLinear に置換: {n} 層")
        return model

    try:
        blt = _build_blt(cfg)
    except Exception as e:
        print(f"[arbor_blt] BLT 構築に失敗 ({type(e).__name__}: {e}). stub にフォールバック.")
        return build_arbor_blt({**cfg, "backend": "stub"})

    # BitNet 2B4T 整合: SwiGLU を ReLU² に差し替え (Global のみ)
    if cfg.get("relu2_ffn_in_global", True):
        from src.model.ffn import swap_swiglu_to_relu2
        n_ffn = swap_swiglu_to_relu2(blt.global_transformer)
        print(f"[arbor_blt] BLT global の SwiGLU を ReLU² FFN に置換: {n_ffn} 層")

    if cfg.get("bitlinear_in_global", False):
        gt = blt.global_transformer
        n = swap_linear_to_bitlinear(gt, skip_names=("output", "embed", "tok_embeddings"))
        print(f"[arbor_blt] BLT global の Linear を BitLinear に置換: {n} 層")
    return _BLTWrapper(blt)
