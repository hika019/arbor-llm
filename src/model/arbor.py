"""Arbor v2: バイトレベル階層 Transformer × BitNet b1.58 (自己完結実装).

構造 (MegaByte 流の静的 patching):

    bytes (B, T)
      └ byte embedding (FP, d_local)
      └ Local Encoder: patch 内 bidirectional attention (n_enc 層)
      └ patch 化: (B, K, P*d_local) -> Linear -> (B, K, d_global)   [K = T/P]
      └ Global Transformer: 1 patch 右シフト + causal (n_global 層) -> h_t
            h_t は「patch t より前の全バイト」だけを見る
      └ Local Decoder: 入力 = byte_emb[i] + proj(h_t) を patch 内 causal で処理
      └ head (FP): logits[i] は bytes[0..i] のみから次バイトを予測

因果性: logits[t*P + j] が参照できるのは
  - h_t        … bytes[< t*P]   (global 入力を 1 patch シフトしているため)
  - local attn … bytes[t*P .. t*P+j] (patch 内 causal)
で、合わせて bytes[0..i]。dataset 側が labels を 1 byte シフト済みなので
追加のシフトは不要。

BitNet b1.58 2B4T 準拠:
  - Attention / FFN の全 Linear が BitLinear (W1.58A8, bias 無し)
  - SubLN: q/k/v <- input_norm, o <- attn_sub_norm,
           gate/up <- ffn_norm, down <- ffn_sub_norm
  - FFN は ReLU² gated: down(subln(relu(gate(x))^2 * up(x)))
  - Embedding / patch 射影 / 出力 head / RMSNorm は FP 維持
  - GQA + RoPE

形状は完全に静的 (T 固定なら 1 グラフ) なので torch.compile がそのまま効く。
依存は torch のみ (third_party 不要、CPU でも動く)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ArborOutput:
    logits: torch.Tensor


@dataclass
class ArborConfig:
    vocab_size: int = 260          # 256 bytes + 特殊 4 (BOE/BOS/EOS/PAD)
    patch_size: int = 4
    max_bytes: int = 2048          # 学習 context (bytes)。T はこの約数である必要は無い
    # ---- local (byte 階層) ----
    local_hidden_size: int = 512
    local_num_heads: int = 8
    local_num_kv_heads: int = 8
    local_intermediate_size: int = 1280
    num_local_encoder_layers: int = 1
    num_local_decoder_layers: int = 3
    # ---- global (patch 階層) ----
    hidden_size: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_size: int = 4608
    num_hidden_layers: int = 16
    # ---- 共通 ----
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    bitnet: bool = True            # False で全 Linear を nn.Linear に (debug 用)
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArborConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ------------------------------------------------------------------ modules
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """cos/sin を非永続バッファに前計算。HF export 時は reset_parameters() で再生成."""

    def __init__(self, head_dim: int, max_pos: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_pos = max_pos
        self.theta = theta
        cos, sin = self._compute()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        t = torch.arange(self.max_pos).float()
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        return freqs.cos(), freqs.sin()

    def reset_parameters(self) -> None:
        cos, sin = self._compute()
        self.cos = cos.to(self.cos.device, self.cos.dtype)
        self.sin = sin.to(self.sin.device, self.sin.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: (B, n_heads, T, head_dim)"""
        t = q.size(-2)
        cos = self.cos[:t].to(q.dtype)  # (T, head_dim/2)
        sin = self.sin[:t].to(q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


def _make_linear(in_f: int, out_f: int, bitnet: bool) -> nn.Module:
    if bitnet:
        from src.model.bitlinear import BitLinear

        return BitLinear(in_f, out_f)
    lin = nn.Linear(in_f, out_f, bias=False)
    nn.init.trunc_normal_(lin.weight, std=0.02, a=-0.06, b=0.06)
    return lin


class Attention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, rope: RotaryEmbedding,
        bitnet: bool, norm_eps: float, causal: bool,
    ):
        super().__init__()
        if dim % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError(f"invalid head config: {dim=} {n_heads=} {n_kv_heads=}")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.rope = rope
        self.wq = _make_linear(dim, n_heads * self.head_dim, bitnet)
        self.wk = _make_linear(dim, n_kv_heads * self.head_dim, bitnet)
        self.wv = _make_linear(dim, n_kv_heads * self.head_dim, bitnet)
        self.wo = _make_linear(n_heads * self.head_dim, dim, bitnet)
        # SubLN: 出力射影の前に正規化 (BitNet 2B4T の attn_sub_norm)
        self.attn_sub_norm = RMSNorm(n_heads * self.head_dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        # GQA は KV head を明示的に複製してから SDPA に渡す。
        # torch 2.5 の enable_gqa=True は flash backward が NaN を返すことがある
        # (非 contiguous レイアウト時)。複製は compile で融合されるためコストは小さい。
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.wo(self.attn_sub_norm(out))


class FeedForward(nn.Module):
    """ReLU² gated FFN (BitNet 2B4T): down(subln(relu(gate(x))^2 * up(x)))"""

    def __init__(self, dim: int, hidden: int, bitnet: bool, norm_eps: float):
        super().__init__()
        self.gate = _make_linear(dim, hidden, bitnet)
        self.up = _make_linear(dim, hidden, bitnet)
        self.down = _make_linear(hidden, dim, bitnet)
        self.ffn_sub_norm = RMSNorm(hidden, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.gate(x))
        return self.down(self.ffn_sub_norm(a * a * self.up(x)))


class Block(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, ffn_hidden: int,
        rope: RotaryEmbedding, bitnet: bool, norm_eps: float, causal: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, norm_eps)
        self.attn = Attention(dim, n_heads, n_kv_heads, rope, bitnet, norm_eps, causal)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.ffn = FeedForward(dim, ffn_hidden, bitnet, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


# -------------------------------------------------------------------- model
class ArborModel(nn.Module):
    def __init__(self, cfg: ArborConfig):
        super().__init__()
        self.cfg = cfg
        p, dl, dg = cfg.patch_size, cfg.local_hidden_size, cfg.hidden_size
        max_patches = (cfg.max_bytes + p - 1) // p

        self.byte_emb = nn.Embedding(cfg.vocab_size, dl)
        nn.init.trunc_normal_(self.byte_emb.weight, std=0.02, a=-0.06, b=0.06)

        local_rope = RotaryEmbedding(dl // cfg.local_num_heads, p, cfg.rope_theta)
        global_rope = RotaryEmbedding(dg // cfg.num_heads, max_patches, cfg.rope_theta)

        # Local Encoder: patch 内 bidirectional (patch 表現は次 patch 以降でしか使わないため)
        self.encoder_layers = nn.ModuleList(
            Block(dl, cfg.local_num_heads, cfg.local_num_kv_heads,
                  cfg.local_intermediate_size, local_rope, cfg.bitnet, cfg.norm_eps,
                  causal=False)
            for _ in range(cfg.num_local_encoder_layers)
        )
        self.patch_proj = nn.Linear(p * dl, dg, bias=False)   # FP (embedding 側)
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02, a=-0.06, b=0.06)
        # 右シフトの先頭 patch。ゼロ初期化禁止: 厳密ゼロ行は全層で 0 のまま伝播し、
        # RMSNorm backward の 1/sqrt(eps) 増幅が全層で複利になって勾配が overflow する
        self.global_bos = nn.Parameter(torch.empty(dg))
        nn.init.trunc_normal_(self.global_bos, std=0.02, a=-0.06, b=0.06)

        self.global_layers = nn.ModuleList(
            Block(dg, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size,
                  global_rope, cfg.bitnet, cfg.norm_eps, causal=True)
            for _ in range(cfg.num_hidden_layers)
        )
        self.global_norm = RMSNorm(dg, cfg.norm_eps)
        self.global_to_local = nn.Linear(dg, dl, bias=False)  # FP
        nn.init.trunc_normal_(self.global_to_local.weight, std=0.02, a=-0.06, b=0.06)

        self.decoder_layers = nn.ModuleList(
            Block(dl, cfg.local_num_heads, cfg.local_num_kv_heads,
                  cfg.local_intermediate_size, local_rope, cfg.bitnet, cfg.norm_eps,
                  causal=True)
            for _ in range(cfg.num_local_decoder_layers)
        )
        self.head_norm = RMSNorm(dl, cfg.norm_eps)
        self.head = nn.Linear(dl, cfg.vocab_size, bias=False)  # FP
        nn.init.trunc_normal_(self.head.weight, std=0.02, a=-0.06, b=0.06)

        self._scale_residual_projections()

    def _scale_residual_projections(self) -> None:
        """残差に入る出力射影 (wo / down) を GPT-2 流に 1/sqrt(2L) へ縮小."""
        n_layers = (
            self.cfg.num_hidden_layers
            + self.cfg.num_local_encoder_layers
            + self.cfg.num_local_decoder_layers
        )
        scale = (2 * n_layers) ** -0.5
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (Attention, FeedForward)):
                    out_proj = m.wo if isinstance(m, Attention) else m.down
                    out_proj.weight.mul_(scale)

    # ------------------------------------------------------------- forward
    def forward(self, input_ids: torch.Tensor) -> ArborOutput:
        b, t = input_ids.shape
        p = self.cfg.patch_size
        pad = (p - t % p) % p
        if pad:
            # 右 pad (PAD=3)。patch 内 causal + global 右シフトにより
            # pad が位置 < t の logits に影響することはない (生成時の端数用)
            input_ids = F.pad(input_ids, (0, pad), value=3)
        k = input_ids.size(1) // p

        x = self.byte_emb(input_ids)                       # (B, T', dl)
        h = x.view(b * k, p, -1)                           # patch 内 encoder
        for layer in self.encoder_layers:
            h = self._maybe_ckpt(layer, h)
        patches = self.patch_proj(h.view(b, k, -1))        # (B, K, dg)

        # 1 patch 右シフト: g_in[t] = patches[t-1], g_in[0] = BOS
        g = torch.cat(
            (self.global_bos.to(patches.dtype).expand(b, 1, -1), patches[:, :-1]), dim=1
        )
        for layer in self.global_layers:
            g = self._maybe_ckpt(layer, g)
        g = self.global_to_local(self.global_norm(g))      # (B, K, dl)

        # Local Decoder: byte_emb[i] + h_patch(i) を patch 内 causal で
        d = x.view(b, k, p, -1) + g.unsqueeze(2)
        d = d.view(b * k, p, -1)
        for layer in self.decoder_layers:
            d = self._maybe_ckpt(layer, d)
        logits = self.head(self.head_norm(d)).view(b, k * p, -1)
        if pad:
            logits = logits[:, :t]
        return ArborOutput(logits=logits)

    def _maybe_ckpt(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    # ------------------------------------------------------------ utility
    def num_parameters(self) -> dict[str, int]:
        def count(mod: nn.Module) -> int:
            return sum(par.numel() for par in mod.parameters())

        return {
            "total": count(self),
            "global": sum(count(m) for m in self.global_layers),
            "local_encoder": sum(count(m) for m in self.encoder_layers),
            "local_decoder": sum(count(m) for m in self.decoder_layers),
            "embedding_head": count(self.byte_emb) + count(self.head)
            + count(self.patch_proj) + count(self.global_to_local),
        }


def build_arbor(model_cfg: dict[str, Any]) -> ArborModel:
    """config dict (configs/*.yaml の model 節) から構築する."""
    cfg = ArborConfig.from_dict(model_cfg)
    model = ArborModel(cfg)
    counts = model.num_parameters()
    from src.model.bitlinear import BitLinear

    n_bit = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    print(
        f"[arbor] params={counts['total'] / 1e6:.1f}M "
        f"(global={counts['global'] / 1e6:.1f}M local_enc={counts['local_encoder'] / 1e6:.1f}M "
        f"local_dec={counts['local_decoder'] / 1e6:.1f}M emb/head={counts['embedding_head'] / 1e6:.1f}M) "
        f"bitnet={'ON' if cfg.bitnet else 'OFF'} bitlinear_layers={n_bit} "
        "weights=W1.58(absmean ternary) activations=A8(absmax per-token) "
        "subln=ON backward=STE(detach)"
    )
    return model
