"""Arbor v2 (static patching x BitNet b1.58) の MLX 実装.

src/model/arbor.py の PyTorch/MPS 版に対応する Apple Silicon 向け port。
本走 config と同じ階層構造 (Local Encoder -> static patch -> Global -> Local
Decoder) と BitNet b1.58 (W1.58 ternary / A8 int8 / detach-STE / SubLN /
ReLU^2 gated FFN) を再現する。文書境界 block-diagonal マスク (#2) も含む。

MLX 固有の注意:
- 遅延評価。実測時は mx.eval() で計算を確定させる。
- STE は x + mx.stop_gradient(quant(x) - x)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

BYTE_OFFSET = 4


@dataclass
class ArborConfig:
    vocab_size: int = 260
    max_bytes: int = 2048
    patch_size: int = 8
    local_hidden_size: int = 768
    local_num_heads: int = 12
    local_num_kv_heads: int = 12
    local_intermediate_size: int = 2048
    num_local_encoder_layers: int = 2
    num_local_decoder_layers: int = 4
    hidden_size: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_size: int = 5632
    num_hidden_layers: int = 20
    rope_theta: float = 500000.0
    rope_theta_global: float | None = None
    rope_theta_local: float | None = None
    norm_eps: float = 1e-5
    bitnet: bool = True
    activation_precision: str = "int8"  # int8 | bf16
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArborConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _weight_quant_ste(w: mx.array) -> mx.array:
    """absmean ternary {-1,0,+1} (dequant 済み) を detach-STE で返す."""
    scale = 1.0 / mx.clip(mx.mean(mx.abs(w)), 1e-5, None)
    q = mx.clip(mx.round(w * scale), -1, 1) / scale
    return w + mx.stop_gradient(q - w)


def _act_quant_int8_ste(x: mx.array) -> mx.array:
    """per-token absmax int8 fake-quant を detach-STE で返す."""
    scale = 127.0 / mx.clip(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5, None)
    q = mx.clip(mx.round(x * scale), -128, 127) / scale
    return x + mx.stop_gradient(q - x)


class BitLinear(nn.Module):
    """BitNet b1.58 の Linear (bias 無し)。SubLN は呼び出し側の RMSNorm が担う."""

    def __init__(self, in_features: int, out_features: int, activation_precision: str = "int8"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_precision = activation_precision
        scale = 1.0 / math.sqrt(in_features)
        self.weight = mx.random.uniform(-scale, scale, (out_features, in_features))

    def __call__(self, x: mx.array) -> mx.array:
        w = _weight_quant_ste(self.weight)
        if self.activation_precision == "int8":
            x = _act_quant_int8_ste(x)
        return x @ w.T


def _make_linear(in_f: int, out_f: int, bitnet: bool, ap: str) -> nn.Module:
    if bitnet:
        return BitLinear(in_f, out_f, ap)
    return nn.Linear(in_f, out_f, bias=False)


def _rope_cos_sin(head_dim: int, max_pos: int, theta: float) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    t = mx.arange(max_pos).astype(mx.float32)
    freqs = mx.outer(t, inv_freq)  # (max_pos, head_dim/2)
    return mx.cos(freqs), mx.sin(freqs)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: (B, H, L, D)。even/odd interleave (torch _apply_rope と同じ並び)
    b, h, l, d = x.shape
    xr = x.reshape(b, h, l, d // 2, 2)
    x1 = xr[..., 0]
    x2 = xr[..., 1]
    cos = cos[:l][None, None]  # (1,1,L,D/2)
    sin = sin[:l][None, None]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return mx.stack([o1, o2], axis=-1).reshape(b, h, l, d)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, cos, sin, bitnet, eps, causal, ap):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.scale = self.head_dim ** -0.5
        self._cos = cos
        self._sin = sin
        self.wq = _make_linear(dim, n_heads * self.head_dim, bitnet, ap)
        self.wk = _make_linear(dim, n_kv_heads * self.head_dim, bitnet, ap)
        self.wv = _make_linear(dim, n_kv_heads * self.head_dim, bitnet, ap)
        self.wo = _make_linear(n_heads * self.head_dim, dim, bitnet, ap)
        self.attn_sub_norm = nn.RMSNorm(n_heads * self.head_dim, eps)

    def __call__(self, x, mask=None):
        b, t, _ = x.shape
        q = self.wq(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = _apply_rope(q, self._cos, self._sin)
        k = _apply_rope(k, self._cos, self._sin)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        sdpa_mask = mask
        if mask is None and self.causal:
            sdpa_mask = "causal"
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=sdpa_mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.wo(self.attn_sub_norm(out))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, bitnet, eps, ap):
        super().__init__()
        self.gate = _make_linear(dim, hidden, bitnet, ap)
        self.up = _make_linear(dim, hidden, bitnet, ap)
        self.down = _make_linear(hidden, dim, bitnet, ap)
        self.ffn_sub_norm = nn.RMSNorm(hidden, eps)

    def __call__(self, x):
        a = nn.relu(self.gate(x))
        return self.down(self.ffn_sub_norm(a * a * self.up(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ffn_hidden, cos, sin, bitnet, eps, causal, ap):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps)
        self.attn = Attention(dim, n_heads, n_kv_heads, cos, sin, bitnet, eps, causal, ap)
        self.ffn_norm = nn.RMSNorm(dim, eps)
        self.ffn = FeedForward(dim, ffn_hidden, bitnet, eps, ap)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.ffn(self.ffn_norm(x))


class ArborMLX(nn.Module):
    """static patching のみ対応の Arbor MLX 実装."""

    def __init__(self, cfg: ArborConfig):
        super().__init__()
        self.cfg = cfg
        p, dl, dg = cfg.patch_size, cfg.local_hidden_size, cfg.hidden_size
        ap = cfg.activation_precision
        theta_g = cfg.rope_theta_global if cfg.rope_theta_global is not None else cfg.rope_theta
        theta_l = cfg.rope_theta_local if cfg.rope_theta_local is not None else cfg.rope_theta
        self.max_patches = (cfg.max_bytes + p - 1) // p

        lcos, lsin = _rope_cos_sin(dl // cfg.local_num_heads, p, theta_l)
        gcos, gsin = _rope_cos_sin(dg // cfg.num_heads, self.max_patches, theta_g)
        self._lcos, self._lsin = lcos, lsin
        self._gcos, self._gsin = gcos, gsin

        self.byte_emb = nn.Embedding(cfg.vocab_size, dl)
        self.encoder_layers = [
            Block(dl, cfg.local_num_heads, cfg.local_num_kv_heads, cfg.local_intermediate_size,
                  lcos, lsin, cfg.bitnet, cfg.norm_eps, False, ap)
            for _ in range(cfg.num_local_encoder_layers)
        ]
        self.patch_proj = nn.Linear(p * dl, dg, bias=False)
        self.global_bos = mx.random.normal((dg,)) * 0.02
        self.global_layers = [
            Block(dg, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size,
                  gcos, gsin, cfg.bitnet, cfg.norm_eps, True, ap)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.global_norm = nn.RMSNorm(dg, cfg.norm_eps)
        self.global_to_local = nn.Linear(dg, dl, bias=False)
        self.decoder_layers = [
            Block(dl, cfg.local_num_heads, cfg.local_num_kv_heads, cfg.local_intermediate_size,
                  lcos, lsin, cfg.bitnet, cfg.norm_eps, True, ap)
            for _ in range(cfg.num_local_decoder_layers)
        ]
        self.head_norm = nn.RMSNorm(dl, cfg.norm_eps)
        self.head = nn.Linear(dl, cfg.vocab_size, bias=False)

    def _global_doc_mask(self, patch_doc: mx.array) -> mx.array:
        # patch_doc: (B, K) int。causal + same-document の additive mask (B,1,K,K)
        b, k = patch_doc.shape
        idx = mx.arange(k)
        causal = idx[None, :] <= idx[:, None]  # (K,K) key<=query
        key_doc = mx.concatenate([mx.full((b, 1), -1, dtype=patch_doc.dtype), patch_doc[:, :-1]], axis=1)
        same = patch_doc[:, :, None] == key_doc[:, None, :]  # (B,K,K)
        bos = (idx == 0)[None, None, :]
        allow = causal[None] & (same | bos)
        return mx.where(allow, 0.0, -1e9)[:, None]  # (B,1,K,K)

    def __call__(self, input_ids: mx.array) -> mx.array:
        cfg = self.cfg
        b, t = input_ids.shape
        p = cfg.patch_size
        k = t // p  # 呼び出し側で t を p の倍数にする
        dl = cfg.local_hidden_size

        x = self.byte_emb(input_ids)                 # (B, T, dl)
        h = x.reshape(b * k, p, dl)
        for layer in self.encoder_layers:
            h = layer(h)
        patches = self.patch_proj(h.reshape(b, k, p * dl))   # (B, K, dg)

        # doc id (EOS 区切り) -> patch doc (先頭バイト)
        prev_eos = mx.concatenate(
            [mx.zeros((b, 1), dtype=mx.bool_), input_ids[:, :-1] == cfg.eos_token_id], axis=1
        )
        doc_id = mx.cumsum(prev_eos.astype(mx.int32), axis=1)
        patch_doc = doc_id[:, ::p]                    # (B, K)
        gmask = self._global_doc_mask(patch_doc)

        # global: 1 patch 右シフト + causal + doc mask
        bos = mx.broadcast_to(self.global_bos[None, None], (b, 1, patches.shape[-1]))
        g = mx.concatenate([bos, patches[:, :-1]], axis=1)   # (B, K, dg)
        for layer in self.global_layers:
            g = layer(g, gmask)
        g = self.global_to_local(self.global_norm(g))        # (B, K, dl)

        # local decoder: byte_emb + broadcast(global) を patch 内 causal
        d = x.reshape(b, k, p, dl) + g[:, :, None, :]
        d = d.reshape(b * k, p, dl)
        for layer in self.decoder_layers:
            d = layer(d)  # causal (patch 内)
        logits = self.head(self.head_norm(d)).reshape(b, k * p, cfg.vocab_size)
        return logits


def build_arbor_mlx(cfg_dict: dict) -> ArborMLX:
    cfg = ArborConfig.from_dict(cfg_dict)
    model = ArborMLX(cfg)
    mx.eval(model.parameters())
    n = sum(v.size for _, v in _flatten(model.parameters()))
    print(f"[arbor-mlx] params={n/1e6:.1f}M bitnet={'ON' if cfg.bitnet else 'OFF'} "
          f"activations={cfg.activation_precision} patch_size={cfg.patch_size} "
          f"global_layers={cfg.num_hidden_layers} d={cfg.hidden_size}")
    return model


def _flatten(tree, prefix=""):
    out = []
    if isinstance(tree, dict):
        for kk, vv in tree.items():
            out += _flatten(vv, f"{prefix}.{kk}")
    elif isinstance(tree, list):
        for i, vv in enumerate(tree):
            out += _flatten(vv, f"{prefix}.{i}")
    else:
        out.append((prefix, tree))
    return out
