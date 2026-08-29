"""Arbor v2: バイトレベル階層 Transformer × BitNet b1.58 (自己完結実装).

構造:

    bytes (B, T)
      └ byte embedding (FP, d_local)
      └ Local Encoder: patch 内 attention (n_enc 層)
      └ patch 化 (下記)
      └ Global Transformer: 1 patch 右シフト + causal (n_global 層) -> h_t
            h_t は「patch t より前の全バイト」だけを見る
      └ Local Decoder: 入力 = byte_emb[i] + proj(h_patch(i)) を patch 内 causal で処理
      └ head (FP): logits[i] は bytes[0..i] のみから次バイトを予測

patching_mode:
  static   固定長 patch_size バイトで機械的に区切る (MegaByte 方式)。
           形状が完全に固定なので torch.compile がフルに効く。既定・本走用。
  utf8     UTF-8 の文字先頭 byte だけを境界候補にする char-aware patching。
           min/max_patch_len で長さを制限する dynamic mode。
  space    空白・改行の直後で区切る (BLT の space patching)。日本語は句読点・
           改行頼みで patch が長くなりがち。min/max_patch_len で長さを制限。
  entropy  小型バイト LM (entropy_model) の次バイト予測エントロピーが
           threshold を超えた位置で区切る (BLT 本命方式)。entropy_model は
           凍結サブモジュールとして本体に内蔵され checkpoint にも一緒に入る。

動的モード (utf8/space/entropy) の実装方式:
  patch 数を max_patches (既定は max_bytes / min_patch_len の worst-case) に固定 pad し、
  encoder/decoder は flat (B,T) のまま「同一 patch 内のみ許す」block 対角
  attention mask で処理する。これにより動的モードでも tensor 形状は固定。
  T が _WINDOW_CHUNK の倍数のときは T×T 密マスクの代わりに窓マスク
  (WindowMask: 1 patch ≤ max_patch_len を利用し q の前後 w バイトだけ見る)
  で計算する。密マスクは SDPA が math 経路に落ちて T² のスコアを実体化し、
  T=8192 では局所層だけで VRAM ~20GB / 計算 ~50× を浪費するため。
  境界判定だけは逐次処理なので @torch.compiler.disable で compile 対象外。
  pad された patch 行は decoder から一切 gather されないため勾配が流れず、
  ゼロ行の RMSNorm backward 増幅問題 (next.md 参照) も起こさない。

因果性:
  - 境界判定は因果的 (utf8: 現在 byte / space: 直前バイト / entropy: causal LM)
  - patch t の表現は global の 1 patch 右シフトにより bytes[< patch t 開始] に
    しか影響しない。encoder が patch 内 bidirectional でも漏れない
  - tests/test_arbor.py が全モードで「未来バイト変更が過去 logits に漏れない」
    ことを検証する

BitNet b1.58 準拠 (公式レシピ): absmean ternary W / absmax int8 A / detach STE /
SubLN / ReLU² gated FFN / bias 無し。Embedding・射影・head・Norm は FP。
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_OFFSET = 4  # 生バイト b は token id (b + 4)

# 動的 patching の local attention を窓化する際の chunk 長 (T はこの倍数のとき窓経路)
_WINDOW_CHUNK = 128


def _is_block_mask(m: object) -> bool:
    """flex_attention の BlockMask かどうか (torch 未対応環境でも壊れないよう名前で判定)."""
    return type(m).__name__ == "BlockMask"


@dataclass
class WindowMask:
    """patch 内 attention 用の窓マスク (chunk × (chunk+2w) のみ実体化).

    1 patch の長さは max_patch_len (= w) 以下なので、同一 patch の kv は
    q の前後 w バイト以内に必ず収まる。これを利用して T×T の密マスクの
    代わりに chunk ごとの窓だけを見る (メモリ O(T·窓)、計算 ~T/窓 分の 1)。
    """

    mask: torch.Tensor  # (B, n_chunk, chunk, chunk + 2w) bool
    chunk: int
    w: int


def _windowed_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, wm: WindowMask
) -> torch.Tensor:
    """q/k/v: (B, H, T, d), T = n * chunk。戻り値も (B, H, T, d)。"""
    b, h, t, d = q.shape
    c, w = wm.chunk, wm.w
    n = t // c
    win = c + 2 * w
    qc = q.view(b, h, n, c, d).permute(0, 2, 1, 3, 4).reshape(b * n, h, c, d)
    # kv は両側 w を pad してから chunk 幅 c でスライドして窓を切り出す
    kw = F.pad(k, (0, 0, w, w)).unfold(2, win, c).permute(0, 2, 1, 4, 3).reshape(b * n, h, win, d)
    vw = F.pad(v, (0, 0, w, w)).unfold(2, win, c).permute(0, 2, 1, 4, 3).reshape(b * n, h, win, d)
    out = F.scaled_dot_product_attention(qc, kw, vw, attn_mask=wm.mask.reshape(b * n, 1, c, win))
    return out.view(b, n, h, c, d).permute(0, 2, 1, 3, 4).reshape(b, h, t, d)


@dataclass
class ArborOutput:
    logits: torch.Tensor
    patch_count: torch.Tensor | None = None
    max_patch_count: torch.Tensor | None = None


@dataclass
class ArborConfig:
    vocab_size: int = 260          # 256 bytes + 特殊 4 (BOE/BOS/EOS/PAD)
    max_bytes: int = 2048          # 学習 context (bytes)
    # ---- patching ----
    patching_mode: str = "static"  # choices: static | utf8 | space | entropy
    patch_size: int = 4            # static 用: 1 patch のバイト数
    min_patch_len: int = 2         # 動的用: これ未満では区切らない
    max_patch_len: int = 16        # 動的用: これに達したら強制的に区切る
    max_patches: int | None = None # 動的用: 固定 pad する patch 数。None なら worst-case
    entropy_threshold: float = 1.5 # entropy 用: 次バイト H (nats) がこれを超えたら区切る
    entropy_model: dict | None = None       # entropy 用: ByteLM の構成 (inline dict)
    entropy_model_ckpt: str | None = None   # entropy 用: 初回構築時に重みを読む checkpoint dir
    attention_window: int | None = None     # ByteLM 用: causal attention を直近 N byte に制限
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
    # RoPE theta を階層別に上書きする (None なら rope_theta を使う)。
    #   global は max_patches (static 8k/patch8 = 1024) 位置しか見ないため、
    #   128k 長文脈向けの大きな theta は位置分解能を潰す (#1)。系列長相応に下げる。
    #   local は patch 内 (static) / flat バイト列 (dynamic) を見る。
    rope_theta_global: float | None = None
    rope_theta_local: float | None = None
    norm_eps: float = 1e-5
    bitnet: bool = True            # False で全 Linear を nn.Linear に (debug 用)
    activation_precision: str = "int8"  # BitLinear の活性量子化: int8 | bf8 | bf16
    gradient_checkpointing: bool = False
    # packing='document' の EOS 区切り。global attention を文書内に閉じる
    # (block-diagonal) ためのバイト ID。data.eos_token_id と一致させること。
    eos_token_id: int = 2
    # global (patch 階層) attention の実装。
    #   sdpa … 既定。文書マスク時は密 (B,1,K,K) マスク + SDPA (flash 非対応経路)。
    #   flex … CUDA 向け。文書境界を BlockMask にして flex_attention で fuse し、
    #          GQA も native (KV repeat_interleave 不要)。torch>=2.5 / 主に CUDA 用。
    #   auto … CUDA training は flex、eval/他 device は sdpa。
    global_attn_impl: str = "sdpa"  # choices: sdpa | flex | auto

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
        # F.rms_norm は内部 fp32 計算の fused カーネル (手書き 6 カーネル比 ~8 倍速)
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


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

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: (B, n_heads, T, head_dim)。pos_offset は逐次生成時の絶対位置."""
        t = q.size(-2)
        cos = self.cos[pos_offset:pos_offset + t].to(q.dtype)
        sin = self.sin[pos_offset:pos_offset + t].to(q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


def _make_linear(in_f: int, out_f: int, bitnet: bool, activation_precision: str = "int8") -> nn.Module:
    if bitnet:
        from src.model.bitlinear import BitLinear

        return BitLinear(in_f, out_f, activation_precision=activation_precision)
    lin = nn.Linear(in_f, out_f, bias=False)
    nn.init.trunc_normal_(lin.weight, std=0.02, a=-0.06, b=0.06)
    return lin


def _activation_desc(precision: str) -> str:
    return {
        "int8": "A8(absmax per-token int8)",
        "bf8": "bf8(float8_e5m2)",
        "bf16": "bf16(no activation quant)",
    }.get(precision, precision)


class Attention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, rope: RotaryEmbedding,
        bitnet: bool, norm_eps: float, causal: bool, activation_precision: str = "int8",
    ):
        super().__init__()
        if dim % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError(f"invalid head config: {dim=} {n_heads=} {n_kv_heads=}")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.rope = rope
        self.wq = _make_linear(dim, n_heads * self.head_dim, bitnet, activation_precision)
        self.wk = _make_linear(dim, n_kv_heads * self.head_dim, bitnet, activation_precision)
        self.wv = _make_linear(dim, n_kv_heads * self.head_dim, bitnet, activation_precision)
        self.wo = _make_linear(n_heads * self.head_dim, dim, bitnet, activation_precision)
        # SubLN: 出力射影の前に正規化 (BitNet 2B4T の attn_sub_norm)
        self.attn_sub_norm = RMSNorm(n_heads * self.head_dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: "torch.Tensor | WindowMask | None" = None,
        kv_cache: "_LayerKVCache | None" = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv_group = getattr(self, "_fast_qkv_group", None)
        if self.training and qkv_group is not None and qkv_group.training_weight_cache_enabled:
            q_width = self.n_heads * self.head_dim
            kv_width = self.n_kv_heads * self.head_dim
            q_raw, k_raw, v_raw = qkv_group(x).split((q_width, kv_width, kv_width), dim=-1)
            q = q_raw.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = k_raw.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v_raw.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, pos_offset)
        is_incremental = False
        if kv_cache is not None:
            # cache には複製前の KV を入れる (メモリ節約)。cache 済み分は全て過去
            # なので、新規トークンが 1 個ならマスク無しで全 attend が causal と等価
            is_incremental = kv_cache.size() > 0
            k, v = kv_cache.append(k, v)
        # 素の causal path (本走の static patching で使う経路) は native enable_gqa を使い
        # K/V の repeat_interleave (帯域 4 倍) を避ける。torch 2.5 の enable_gqa=True は
        # 過去に compile 併用で flash backward が壊れる報告があったため長らく避けていたが、
        # 本構成 (bf16 / SDPA flash / static shape) では 200 step soak (bench, NaN無し・
        # +9% throughput) で確認できたので採用する。WindowMask/密マスク/kv_cache 経路は
        # 未検証のため従来通り repeat_interleave のままにする。
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
        else:
            n_rep = 1
        native_gqa = n_rep > 1 and attn_mask is None and not is_incremental
        use_flex = _is_block_mask(attn_mask)
        if n_rep > 1 and not native_gqa and not use_flex:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        if use_flex:
            # CUDA 向け fused 経路。文書境界 BlockMask で causal+doc を課し、GQA も
            # native (KV は複製しない)。torch.compile 併用で fused kernel になる。
            from torch.nn.attention.flex_attention import flex_attention

            out = flex_attention(q, k, v, block_mask=attn_mask, enable_gqa=n_rep > 1)
        elif isinstance(attn_mask, WindowMask):
            # 動的 patching 用 (窓経路): T×T を実体化しない
            out = _windowed_sdpa(q, k, v, attn_mask)
        elif attn_mask is not None:
            # 動的 patching 用 (密マスク fallback): causal 制約はマスク側に織り込み済み
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        elif is_incremental:
            if t != 1:
                raise ValueError("KV cache への追記は 1 トークンずつ行うこと")
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, enable_gqa=native_gqa)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.wo(self.attn_sub_norm(out))


class FeedForward(nn.Module):
    """ReLU² gated FFN (BitNet 2B4T): down(subln(relu(gate(x))^2 * up(x)))"""

    def __init__(self, dim: int, hidden: int, bitnet: bool, norm_eps: float,
                 activation_precision: str = "int8"):
        super().__init__()
        self.gate = _make_linear(dim, hidden, bitnet, activation_precision)
        self.up = _make_linear(dim, hidden, bitnet, activation_precision)
        self.down = _make_linear(hidden, dim, bitnet, activation_precision)
        self.ffn_sub_norm = RMSNorm(hidden, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group = getattr(self, "_fast_gate_up_group", None)
        if self.training and group is not None and group.training_weight_cache_enabled:
            gate, up = group(x).split((self.gate.out_features, self.up.out_features), dim=-1)
        else:
            gate, up = self.gate(x), self.up(x)
        a = F.relu(gate)
        return self.down(self.ffn_sub_norm(a * a * up))


class Block(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, ffn_hidden: int,
        rope: RotaryEmbedding, bitnet: bool, norm_eps: float, causal: bool,
        activation_precision: str = "int8",
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, norm_eps)
        self.attn = Attention(dim, n_heads, n_kv_heads, rope, bitnet, norm_eps, causal,
                              activation_precision)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.ffn = FeedForward(dim, ffn_hidden, bitnet, norm_eps, activation_precision)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: "torch.Tensor | WindowMask | None" = None,
        kv_cache: "_LayerKVCache | None" = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask, kv_cache, pos_offset)
        return x + self.ffn(self.ffn_norm(x))


def _scale_residual_projections(layer_lists: list[nn.ModuleList]) -> None:
    """残差に入る出力射影 (wo / down) を GPT-2 流に 1/sqrt(2L) へ縮小."""
    n_layers = sum(len(layers) for layers in layer_lists)
    scale = (2 * max(n_layers, 1)) ** -0.5
    with torch.no_grad():
        for layers in layer_lists:
            for block in layers:
                block.attn.wo.weight.mul_(scale)
                block.ffn.down.weight.mul_(scale)


# ---------------------------------------------------------------- byte LM
class ByteLM(nn.Module):
    """entropy patching の境界判定に使う小型 causal バイト LM.

    `arch: byte_lm` で train.py から単体学習もできる (checkpoint/サンプル生成共通)。
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.vocab_size = cfg.get("vocab_size", 260)
        h = cfg["hidden_size"]
        n_heads = cfg.get("num_heads", 8)
        n_kv = cfg.get("num_kv_heads", n_heads)
        ffn = cfg.get("intermediate_size", 4 * h)
        n_layers = cfg.get("num_hidden_layers", 4)
        max_bytes = cfg.get("max_bytes", 2048)
        bitnet = cfg.get("bitnet", False)
        norm_eps = cfg.get("norm_eps", 1e-5)
        activation_precision = cfg.get("activation_precision", "int8")
        attention_window = cfg.get("attention_window")
        self.attention_window = None if attention_window is None else int(attention_window)
        if self.attention_window is not None and self.attention_window <= 0:
            raise ValueError("attention_window must be positive")
        rope = RotaryEmbedding(h // n_heads, max_bytes, cfg.get("rope_theta", 500000.0))

        self.embed = nn.Embedding(self.vocab_size, h)
        nn.init.trunc_normal_(self.embed.weight, std=0.02, a=-0.06, b=0.06)
        self.layers = nn.ModuleList(
            Block(h, n_heads, n_kv, ffn, rope, bitnet, norm_eps, causal=True,
                  activation_precision=activation_precision)
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(h, norm_eps)
        self.head = nn.Linear(h, self.vocab_size, bias=False)
        nn.init.trunc_normal_(self.head.weight, std=0.02, a=-0.06, b=0.06)
        self.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))
        _scale_residual_projections([self.layers])

    def forward(self, input_ids: torch.Tensor) -> ArborOutput:
        x = self.embed(input_ids)
        attn_mask = self._attention_mask(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(layer, x, attn_mask, use_reentrant=False)
            else:
                x = layer(x, attn_mask)
        return ArborOutput(logits=self.head(self.norm(x)))

    def _attention_mask(self, input_ids: torch.Tensor) -> "torch.Tensor | WindowMask | None":
        if self.attention_window is None:
            return None
        b, t = input_ids.shape
        w = min(int(self.attention_window), t)
        c = _WINDOW_CHUNK
        if t % c == 0 and t >= c:
            ar_q = torch.arange(c, device=input_ids.device).view(1, 1, c, 1)
            ar_k = torch.arange(c + 2 * w, device=input_ids.device).view(1, 1, 1, c + 2 * w)
            chunk_start = (torch.arange(t // c, device=input_ids.device) * c).view(1, -1, 1, 1)
            q_pos = chunk_start + ar_q
            k_pos = chunk_start - w + ar_k
            mask = (k_pos >= 0) & (k_pos <= q_pos) & (q_pos - k_pos < w)
            return WindowMask(mask.expand(b, -1, -1, -1), c, w)
        q = torch.arange(t, device=input_ids.device).unsqueeze(1)
        k = torch.arange(t, device=input_ids.device).unsqueeze(0)
        return ((k <= q) & (q - k < w)).view(1, 1, t, t)

    @torch.no_grad()
    def next_byte_entropy(self, input_ids: torch.Tensor) -> torch.Tensor:
        """各位置の「次バイト分布のエントロピー (nats)」(B, T) を返す.

        戻り値の [.., t] は p(x_{t+1} | x_{<=t}) のエントロピー。
        """
        logits = self.forward(input_ids).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        return -(logp.exp() * logp).sum(-1)


def build_byte_lm(model_cfg: dict[str, Any]) -> ByteLM:
    model = ByteLM(dict(model_cfg))
    n = sum(p.numel() for p in model.parameters())
    print(f"[byte_lm] params={n / 1e6:.1f}M bitnet={'ON' if model_cfg.get('bitnet', False) else 'OFF'}")
    return model


# ----------------------------------------------------------- patch 境界判定
_SPACE_BYTES = (0x20, 0x09, 0x0A, 0x0D)  # space, tab, LF, CR


def _is_utf8_char_start_byte(byte: int) -> bool:
    """Return whether byte can start a UTF-8 code point."""
    return byte < 0x80 or 0xC2 <= byte <= 0xF4


@torch.compiler.disable
def compute_patch_starts(
    input_ids: torch.Tensor,
    mode: str,
    min_len: int,
    max_len: int,
    entropy_model: ByteLM | None = None,
    threshold: float = 1.5,
    entropy_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """patch 開始位置の bool tensor (B, T) を返す。判定は過去バイトのみに依存 (causal).

    - utf8:    現在バイトが UTF-8 文字先頭なら新 patch を開始
    - space:   直前バイトが空白系なら新 patch を開始
    - entropy: 直前位置での次バイト予測エントロピーが threshold 超なら開始
    その後 min_len (それ未満では区切らない) / max_len (達したら強制区切り) を適用。

    min/max 制約は「境界 s の次の境界 = min(s+min_len 以降で最初の候補, s+max_len)」
    というジャンプ過程なので、CUDA では raw -> starts の境界 walk を extension
    に渡して GPU 上で完結させる。CPU ではテスト用の torch 実装を使う。
    """
    if min_len <= 0:
        raise ValueError("min_patch_len must be positive")
    if max_len < min_len:
        raise ValueError("max_patch_len must be >= min_patch_len")

    b, t = input_ids.shape
    if mode == "utf8":
        cur = input_ids - BYTE_OFFSET
        raw = ((cur < 0x80) | ((cur >= 0xC2) & (cur <= 0xF4))).to(torch.bool)
        raw[:, 0] = False
    elif mode == "space":
        prev = input_ids[:, :-1] - BYTE_OFFSET
        is_space = torch.zeros_like(prev, dtype=torch.bool)
        for sb in _SPACE_BYTES:
            is_space |= prev == sb
        raw = torch.zeros(b, t, dtype=torch.bool, device=input_ids.device)
        raw[:, 1:] = is_space
    elif mode == "entropy":
        if entropy_values is None:
            if entropy_model is None:
                raise ValueError("patching_mode=entropy には entropy_model が必要")
            ent = entropy_model.next_byte_entropy(input_ids)  # (B, T), no_grad in ByteLM
        else:
            ent = entropy_values
        raw = torch.zeros(b, t, dtype=torch.bool, device=input_ids.device)
        raw[:, 1:] = ent[:, :-1] > threshold
    else:
        raise ValueError(f"unknown dynamic patching mode: {mode}")

    if raw.is_cuda:
        from src.model.patch_starts_cuda import patch_starts_cuda

        return patch_starts_cuda(raw, min_len, max_len)

    starts = torch.zeros_like(raw)
    for r in range(b):
        i = 0
        while i < t:
            starts[r, i] = True
            lo = i + min_len
            if lo >= t:
                break
            hi = min(i + max_len, t)
            candidates = raw[r, lo:hi].nonzero(as_tuple=False)
            i = lo + int(candidates[0, 0]) if candidates.numel() else hi
    return starts


# ------------------------------------------------------------------ KV cache
class _LayerKVCache:
    """1 層分の KV cache (複製前の n_kv_heads で保持)."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def size(self) -> int:
        return 0 if self.k is None else self.k.size(2)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v

    def trim(self, max_tokens: int) -> None:
        if self.k is None or self.k.size(2) <= max_tokens:
            return
        self.k = self.k[:, :, -max_tokens:].contiguous()
        self.v = self.v[:, :, -max_tokens:].contiguous()


# -------------------------------------------------------------------- model
class ArborModel(nn.Module):
    def __init__(self, cfg: ArborConfig):
        super().__init__()
        if cfg.patching_mode not in ("static", "utf8", "space", "entropy"):
            raise ValueError(f"unknown patching_mode: {cfg.patching_mode}")
        from src.model.bitlinear import check_activation_precision

        check_activation_precision(cfg.activation_precision)
        self.cfg = cfg
        self.dynamic = cfg.patching_mode != "static"
        self.profile_sections = False
        self._last_profile: dict[str, float] | None = None
        p, dl, dg = cfg.patch_size, cfg.local_hidden_size, cfg.hidden_size
        if self.dynamic:
            worst_case_patches = math.ceil(cfg.max_bytes / cfg.min_patch_len)
            self.max_patches = cfg.max_patches or worst_case_patches
            if self.max_patches <= 0 or self.max_patches > worst_case_patches:
                raise ValueError(
                    f"max_patches must be in [1, {worst_case_patches}], got {self.max_patches}"
                )
        else:
            self.max_patches = (cfg.max_bytes + p - 1) // p

        self.byte_emb = nn.Embedding(cfg.vocab_size, dl)
        nn.init.trunc_normal_(self.byte_emb.weight, std=0.02, a=-0.06, b=0.06)

        # 動的モードの local 層は flat (B,T) で動くので RoPE は絶対バイト位置
        theta_global = cfg.rope_theta_global if cfg.rope_theta_global is not None else cfg.rope_theta
        theta_local = cfg.rope_theta_local if cfg.rope_theta_local is not None else cfg.rope_theta
        local_rope = RotaryEmbedding(
            dl // cfg.local_num_heads,
            cfg.max_bytes if self.dynamic else p,
            theta_local,
        )
        global_rope = RotaryEmbedding(dg // cfg.num_heads, self.max_patches, theta_global)

        # Local Encoder: patch 内 bidirectional (patch 表現は次 patch 以降でしか使わない)
        self.encoder_layers = nn.ModuleList(
            Block(dl, cfg.local_num_heads, cfg.local_num_kv_heads,
                  cfg.local_intermediate_size, local_rope, cfg.bitnet, cfg.norm_eps,
                  causal=False, activation_precision=cfg.activation_precision)
            for _ in range(cfg.num_local_encoder_layers)
        )
        # patch 表現: static は concat 射影、動的は max-pool 後に射影 (FP)
        self.patch_proj = nn.Linear(dl if self.dynamic else p * dl, dg, bias=False)
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02, a=-0.06, b=0.06)
        # 右シフトの先頭 patch。ゼロ初期化禁止: 厳密ゼロ行は全層で 0 のまま伝播し、
        # RMSNorm backward の 1/sqrt(eps) 増幅が全層で複利になって勾配が overflow する
        self.global_bos = nn.Parameter(torch.empty(dg))
        nn.init.trunc_normal_(self.global_bos, std=0.02, a=-0.06, b=0.06)

        self.global_layers = nn.ModuleList(
            Block(dg, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size,
                  global_rope, cfg.bitnet, cfg.norm_eps, causal=True,
                  activation_precision=cfg.activation_precision)
            for _ in range(cfg.num_hidden_layers)
        )
        self.global_norm = RMSNorm(dg, cfg.norm_eps)
        self.global_to_local = nn.Linear(dg, dl, bias=False)  # FP
        nn.init.trunc_normal_(self.global_to_local.weight, std=0.02, a=-0.06, b=0.06)

        self.decoder_layers = nn.ModuleList(
            Block(dl, cfg.local_num_heads, cfg.local_num_kv_heads,
                  cfg.local_intermediate_size, local_rope, cfg.bitnet, cfg.norm_eps,
                  causal=True, activation_precision=cfg.activation_precision)
            for _ in range(cfg.num_local_decoder_layers)
        )
        self.head_norm = RMSNorm(dl, cfg.norm_eps)
        self.head = nn.Linear(dl, cfg.vocab_size, bias=False)  # FP
        nn.init.trunc_normal_(self.head.weight, std=0.02, a=-0.06, b=0.06)

        _scale_residual_projections(
            [self.encoder_layers, self.global_layers, self.decoder_layers]
        )

        # entropy 用の凍結 ByteLM (checkpoint に同梱される)
        if cfg.patching_mode == "entropy":
            if not cfg.entropy_model:
                raise ValueError(
                    "patching_mode=entropy には model.entropy_model (ByteLM 構成) が必要"
                )
            em_cfg = dict(cfg.entropy_model)
            em_cfg.setdefault("max_bytes", cfg.max_bytes)
            self.entropy_model = ByteLM(em_cfg)
            self.entropy_model.requires_grad_(False)
        else:
            self.entropy_model = None

    # ------------------------------------------------------------- forward
    def forward(self, input_ids: torch.Tensor) -> ArborOutput:
        if self.dynamic:
            return self._forward_dynamic(input_ids)
        return self._forward_static(input_ids)

    # ------------------------------------------------ document boundary mask
    def _byte_doc_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """各バイトが属する文書番号 (B, T) を返す.

        packing='document' は文書を EOS 区切りで連結する。EOS の「次」の
        バイトから文書番号が 1 増える (EOS 自身は直前の文書に属す)。判定は
        過去バイトのみに依存するので causal (未来を見ない)。生バイトは +4
        offset されており EOS/PAD と衝突しないため == 判定で一意に取れる。
        """
        prev_is_eos = F.pad(
            input_ids == self.cfg.eos_token_id, (1, 0), value=False
        )[:, :-1]
        return prev_is_eos.to(torch.long).cumsum(dim=1)

    def _global_doc_mask(self, patch_doc: torch.Tensor) -> torch.Tensor:
        """global (patch 階層) 用の block-diagonal + causal マスク (B,1,K,K).

        global は 1 patch 右シフト後の causal。出力位置 j (= patch j の文脈) が
        key 位置 j' に attend できるのは:
          - j'=0 (先頭の学習可能 BOS。文書非依存で常に許可。全マスク行を防ぐ)
          - それ以外は key s[j'] = patch j'-1 が patch j と同一文書のときだけ。
        これにより文書 B の patch が無関係な文書 A の patch へ attend しない。
        新しい文書の先頭 patch は BOS のみを文脈に持つ (文脈リセット)。
        """
        b, k = patch_doc.shape
        device = patch_doc.device
        causal = torch.tril(torch.ones(k, k, dtype=torch.bool, device=device))
        # key 側 doc: s[0]=BOS(=-1 番兵), s[j']=patch j'-1 の doc
        key_doc = F.pad(patch_doc[:, :-1], (1, 0), value=-1)
        same_doc = patch_doc.unsqueeze(2) == key_doc.unsqueeze(1)  # (B,K,K)
        bos_col = torch.zeros(k, dtype=torch.bool, device=device)
        bos_col[0] = True
        allow = causal.unsqueeze(0) & (same_doc | bos_col.view(1, 1, k))
        return allow.unsqueeze(1)  # (B,1,K,K)

    def _global_mask(self, patch_doc: torch.Tensor):
        """cfg.global_attn_impl に応じて global 用マスクを返す (sdpa=密, flex=BlockMask).

        auto は CUDA training なら flex、それ以外は sdpa を選ぶ。eval 時に未コンパイル
        の base model を使うサンプル生成では、flex の unfused 経路を避ける。
        """
        impl = self.cfg.global_attn_impl
        if impl == "auto":
            impl = "flex" if patch_doc.is_cuda and self.training else "sdpa"
        if impl == "flex":
            return self._global_flex_block_mask(patch_doc)
        return self._global_doc_mask(patch_doc)

    def _global_flex_block_mask(self, patch_doc: torch.Tensor):
        """_global_doc_mask と同じ許可規則を flex_attention の BlockMask で表す.

        密 (B,1,K,K) マスクを実体化せず、CUDA では fused kernel になる。
        許可規則: causal (kv<=q) かつ (kv==0 の BOS or 同一文書)。key 位置 kv は
        右シフト後の global 入力位置なので、その doc は patch_doc[kv-1] (kv>=1)。
        """
        from torch.nn.attention.flex_attention import create_block_mask

        pd = patch_doc
        k = pd.shape[1]

        def mask_mod(b, h, q_idx, kv_idx):
            causal = kv_idx <= q_idx
            is_bos = kv_idx == 0
            key_doc = pd[b, torch.clamp(kv_idx - 1, min=0)]
            same = pd[b, q_idx] == key_doc
            return causal & (is_bos | same)

        return create_block_mask(
            mask_mod, B=pd.shape[0], H=None, Q_LEN=k, KV_LEN=k, device=pd.device
        )

    @torch.compiler.disable
    def _debug_context_contribution(
        self,
        byte_emb: torch.Tensor,
        global_context: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> None:
        """Print context-vs-byte RMS once when ARBOR_DEBUG_CONTEXT=1."""
        if getattr(self, "_debug_context_printed", False):
            return
        self._debug_context_printed = True
        with torch.no_grad():
            byte_rms = byte_emb.float().pow(2).mean().sqrt()
            global_rms = global_context.float().pow(2).mean().sqrt()
            decoder_rms = decoder_input.float().pow(2).mean().sqrt()
            print(
                "[ctx] "
                f"byte_emb_rms={byte_rms.item():.6f} "
                f"global_context_rms={global_rms.item():.6f} "
                f"decoder_input_rms={decoder_rms.item():.6f} "
                f"global_byte_ratio={(global_rms / byte_rms.clamp_min(1e-8)).item():.6f}",
                flush=True,
            )

    def _forward_static(self, input_ids: torch.Tensor) -> ArborOutput:
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

        # patch の doc 番号 = patch 先頭バイトの doc。global を文書内に閉じる。
        patch_doc = self._byte_doc_ids(input_ids)[:, ::p]  # (B, K)
        g = self._run_global(patches, self._global_mask(patch_doc))  # (B, K, dl)

        # Local Decoder: byte_emb[i] + h_patch(i) を patch 内 causal で
        d = x.view(b, k, p, -1) + g.unsqueeze(2)
        if os.environ.get("ARBOR_DEBUG_CONTEXT", "0") == "1":
            self._debug_context_contribution(x, g, d)
        d = d.view(b * k, p, -1)
        for layer in self.decoder_layers:
            d = self._maybe_ckpt(layer, d)
        logits = self.head(self.head_norm(d)).view(b, k * p, -1)
        if pad:
            logits = logits[:, :t]
        patch_count = torch.full((), b * k, dtype=torch.float32, device=logits.device)
        max_patch_count = torch.full((), k, dtype=torch.float32, device=logits.device)
        return ArborOutput(logits=logits, patch_count=patch_count, max_patch_count=max_patch_count)

    def _forward_dynamic(self, input_ids: torch.Tensor) -> ArborOutput:
        cfg = self.cfg
        b, t = input_ids.shape
        profile_sections = bool(getattr(self, "profile_sections", False))
        section_ms: dict[str, float] = {}

        def timed_section(name: str, fn):
            if not profile_sections:
                return fn()
            if input_ids.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn()
                end.record()
                end.synchronize()
                section_ms[name] = section_ms.get(name, 0.0) + start.elapsed_time(end)
                return result
            start_t = time.perf_counter()
            result = fn()
            section_ms[name] = section_ms.get(name, 0.0) + (time.perf_counter() - start_t) * 1000.0
            return result

        entropy_values = None
        if cfg.patching_mode == "entropy":
            if self.entropy_model is None:
                raise ValueError("patching_mode=entropy には entropy_model が必要")
            # Keep only the data-dependent boundary walk out of torch.compile.
            # The frozen ByteLM itself is dense tensor work and benefits from compile.
            entropy_values = timed_section(
                "bytelm_ms",
                lambda: self.entropy_model.next_byte_entropy(input_ids),
            )

        def build_patch_ids() -> tuple[torch.Tensor, torch.Tensor]:
            starts_local = compute_patch_starts(
                input_ids, cfg.patching_mode, cfg.min_patch_len, cfg.max_patch_len,
                self.entropy_model, cfg.entropy_threshold, entropy_values,
            )
            # (B, T) 各バイトの patch 番号
            return starts_local.long().cumsum(1) - 1, starts_local.sum(1).to(torch.float32)

        patch_id, patch_counts = timed_section("patching_ms", build_patch_ids)
        patch_count = patch_counts.sum()
        max_patch_count = patch_counts.max()
        k = self.max_patches
        torch._assert(
            (patch_id < k).all(),
            f"dynamic patch count exceeded max_patches={k}; increase model.max_patches",
        )

        def run_arbor_body() -> torch.Tensor:
            x = self.byte_emb(input_ids)                   # (B, T, dl)

            # patch 内 attention マスク: T が chunk の倍数なら窓経路 (T×T を実体化
            # しない)、端数 (生成 prefill 等) は従来の密マスク fallback
            c, w = _WINDOW_CHUNK, cfg.max_patch_len
            if t % c == 0 and t >= c:
                # kv 側は両側 w を pad。pad 位置は patch_id=-1 で不一致を保証
                qpid = patch_id.view(b, t // c, c)
                kpid = F.pad(patch_id, (w, w), value=-1).unfold(1, c + 2 * w, c)
                same_win = qpid.unsqueeze(3) == kpid.unsqueeze(2)  # (B, n, c, c+2w)
                # 絶対位置: q = i*c + qi, kv = i*c - w + ki なので causal ⇔ qi + w >= ki
                ar_q = torch.arange(c, device=x.device).unsqueeze(1)
                ar_k = torch.arange(c + 2 * w, device=x.device).unsqueeze(0)
                enc_mask: torch.Tensor | WindowMask = WindowMask(same_win, c, w)
                dec_mask: torch.Tensor | WindowMask = WindowMask(same_win & (ar_q + w >= ar_k), c, w)
            else:
                same = patch_id.unsqueeze(2) == patch_id.unsqueeze(1)  # (B, T, T)
                causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
                enc_mask = same.unsqueeze(1)
                dec_mask = (same & causal).unsqueeze(1)

            # Local Encoder: patch 内 bidirectional (block 対角マスク)
            h = x
            for layer in self.encoder_layers:
                h = self._maybe_ckpt(layer, h, enc_mask)

            # patch ごとに max-pool して (B, K, dl) へ。pad patch は 0 埋め
            # (decoder から一切 gather されないため勾配は流れない)
            idx = patch_id.unsqueeze(-1).expand(-1, -1, h.size(-1))
            pooled = h.new_full((b, k, h.size(-1)), float("-inf"))
            pooled.scatter_reduce_(1, idx, h, reduce="amax", include_self=True)
            pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
            patches = self.patch_proj(pooled)              # (B, K, dg)

            # patch の doc 番号 = patch 内バイトの最小 doc (= 先頭バイトの doc)。
            # pad patch (バイト無し) は番兵の大きな値のまま残り、どの実文書とも
            # 一致しないので key/query として実文書へ漏れない。
            byte_doc = self._byte_doc_ids(input_ids)       # (B, T)
            patch_doc = torch.full((b, k), 1 << 30, dtype=torch.long, device=x.device)
            patch_doc.scatter_reduce_(1, patch_id, byte_doc, reduce="amin", include_self=True)
            # global を文書内に閉じる (block-diagonal + causal)。右シフト後、位置 j は
            # patch j-1 を保持し、pad patch は必ず j > t 側に落ちる
            g = self._run_global(patches, self._global_mask(patch_doc))  # (B, K, dl)
            h_byte = g.gather(1, patch_id.unsqueeze(-1).expand(-1, -1, g.size(-1)))

            # Local Decoder: patch 内 causal (block 対角 ∧ 下三角)
            d = x + h_byte
            if os.environ.get("ARBOR_DEBUG_CONTEXT", "0") == "1":
                self._debug_context_contribution(x, h_byte, d)
            for layer in self.decoder_layers:
                d = self._maybe_ckpt(layer, d, dec_mask)
            return self.head(self.head_norm(d))

        logits = timed_section("arbor_ms", run_arbor_body)
        if profile_sections:
            self._last_profile = section_ms
        return ArborOutput(
            logits=logits,
            patch_count=patch_count,
            max_patch_count=max_patch_count,
        )

    @torch.no_grad()
    def profile_patching_sections(self, input_ids: torch.Tensor) -> dict[str, float]:
        """Measure entropy scoring and boundary construction without running Arbor body."""
        if not self.dynamic:
            return {}
        cfg = self.cfg
        section_ms: dict[str, float] = {}

        def timed_section(name: str, fn):
            if input_ids.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn()
                end.record()
                end.synchronize()
                section_ms[name] = start.elapsed_time(end)
                return result
            start_t = time.perf_counter()
            result = fn()
            section_ms[name] = (time.perf_counter() - start_t) * 1000.0
            return result

        entropy_values = None
        if cfg.patching_mode == "entropy":
            if self.entropy_model is None:
                raise ValueError("patching_mode=entropy には entropy_model が必要")
            entropy_values = timed_section(
                "bytelm_ms",
                lambda: self.entropy_model.next_byte_entropy(input_ids),
            )

        def build_patch_ids() -> tuple[torch.Tensor, torch.Tensor]:
            starts_local = compute_patch_starts(
                input_ids, cfg.patching_mode, cfg.min_patch_len, cfg.max_patch_len,
                self.entropy_model, cfg.entropy_threshold, entropy_values,
            )
            return starts_local.long().cumsum(1) - 1, starts_local.sum(1).to(torch.float32)

        patch_id, patch_counts = timed_section("patching_ms", build_patch_ids)
        section_ms["patches_per_seq"] = float(patch_counts.float().mean().cpu())
        section_ms["max_patch_per_seq"] = float(patch_counts.max().cpu())
        section_ms["patch_id_max"] = float(patch_id.max().cpu() + 1)
        return section_ms

    def _run_global(
        self, patches: torch.Tensor, attn_mask: "torch.Tensor | None" = None
    ) -> torch.Tensor:
        """1 patch 右シフト + causal global を回し、local 次元へ射影して返す.

        attn_mask を渡すと causal に加えて文書境界 (block-diagonal) を課す。
        None のときは従来どおり plain causal (native GQA fast-path)。
        """
        b = patches.size(0)
        g = torch.cat(
            (self.global_bos.to(patches.dtype).expand(b, 1, -1), patches[:, :-1]), dim=1
        )
        for layer in self.global_layers:
            g = self._maybe_ckpt(layer, g, attn_mask)
        return self.global_to_local(self.global_norm(g))

    def _maybe_ckpt(
        self, layer: nn.Module, x: torch.Tensor,
        attn_mask: "torch.Tensor | WindowMask | None" = None,
    ) -> torch.Tensor:
        if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(layer, x, attn_mask, use_reentrant=False)
        return layer(x, attn_mask)

    # ------------------------------------------------------------ utility
    def num_parameters(self) -> dict[str, int]:
        def count(mod: nn.Module | None) -> int:
            return 0 if mod is None else sum(par.numel() for par in mod.parameters())

        return {
            "total": count(self),
            "global": sum(count(m) for m in self.global_layers),
            "local_encoder": sum(count(m) for m in self.encoder_layers),
            "local_decoder": sum(count(m) for m in self.decoder_layers),
            "embedding_head": count(self.byte_emb) + count(self.head)
            + count(self.patch_proj) + count(self.global_to_local),
            "entropy_model": count(self.entropy_model),
        }


class ArborByteGenerator:
    """2 階層 KV cache 付きの逐次バイト生成器 (batch=1).

    フルフォワード方式 (1 バイトごとに全系列再計算) と論理的に同一の logits を、
    増分計算だけで返す:
      - global: patch が確定するたびに 1 トークンだけ KV cache に追記
      - local decoder: 現在 patch のプレフィックス (<= max_patch_len トークン) を
        毎バイト再計算 (極小なのでキャッシュ不要)
      - entropy モードは境界判定用 ByteLM にも KV cache を持つ

    使い方:
        gen = ArborByteGenerator(model)
        logits = gen.prefill(prompt_ids)   # 最終位置の next-byte logits (vocab,)
        logits = gen.push(next_id)         # 1 バイト進める

    context が max_bytes に達したら後半半分を残して内部で自動的に作り直す。
    """

    def __init__(self, model: ArborModel):
        if not isinstance(model, ArborModel):
            raise TypeError("ArborByteGenerator は ArborModel 専用")
        self.m = model.eval()
        self.cfg = model.cfg
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.reset()

    def reset(self) -> None:
        self.byte_ids: list[int] = []
        self.cur_patch: list[int] = []
        self.cur_patch_start = 0
        self.n_global = 0
        self.g_caches = [_LayerKVCache() for _ in self.m.global_layers]
        self.h_cur: torch.Tensor | None = None
        if self.cfg.patching_mode == "entropy":
            self.lm_caches = [_LayerKVCache() for _ in self.m.entropy_model.layers]
            self.prev_entropy = 0.0
        self._push_global(self.m.global_bos.view(1, 1, -1))

    @torch.inference_mode()
    def prefill(self, ids: list[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        logits = None
        for byte_id in ids:
            logits = self.push(int(byte_id))
        if logits is None:
            raise ValueError("prefill には 1 バイト以上必要")
        return logits

    @torch.inference_mode()
    def push(self, byte_id: int) -> torch.Tensor:
        if len(self.byte_ids) >= self.cfg.max_bytes:
            self._rebuild(keep=self.cfg.max_bytes // 2)
        if self._starts_new_patch(byte_id):
            self._commit_patch()
        self.cur_patch.append(byte_id)
        self.byte_ids.append(byte_id)
        if self.cfg.patching_mode == "entropy":
            self._advance_entropy_lm(byte_id)
        return self._decode_current()

    # ----------------------------------------------------------- internal
    def _starts_new_patch(self, next_byte_id: int | None = None) -> bool:
        """次に push されるバイトが新しい patch を始めるか (compute_patch_starts と同条件)."""
        run = len(self.cur_patch)
        if run == 0:
            return False
        cfg = self.cfg
        if cfg.patching_mode == "static":
            return run >= cfg.patch_size
        if run >= cfg.max_patch_len:
            return True
        if run < cfg.min_patch_len:
            return False
        if cfg.patching_mode == "utf8":
            if next_byte_id is None:
                return False
            return _is_utf8_char_start_byte(next_byte_id - BYTE_OFFSET)
        if cfg.patching_mode == "space":
            return (self.byte_ids[-1] - BYTE_OFFSET) in _SPACE_BYTES
        return self.prev_entropy > cfg.entropy_threshold

    def _push_global(self, g_in: torch.Tensor) -> None:
        g = g_in.to(self.dtype)
        for layer, cache in zip(self.m.global_layers, self.g_caches):
            g = layer(g, kv_cache=cache, pos_offset=self.n_global)
        self.n_global += 1
        self.h_cur = self.m.global_to_local(self.m.global_norm(g))  # (1, 1, dl)

    def _commit_patch(self) -> None:
        ids = torch.tensor([self.cur_patch], dtype=torch.long, device=self.device)
        x = self.m.byte_emb(ids)
        pos = 0 if not self.m.dynamic else self.cur_patch_start
        for layer in self.m.encoder_layers:
            x = layer(x, pos_offset=pos)
        if self.m.dynamic:
            patch_emb = self.m.patch_proj(x.amax(dim=1))     # max-pool (1, dg)
        else:
            patch_emb = self.m.patch_proj(x.reshape(1, -1))  # concat (1, dg)
        self._push_global(patch_emb.view(1, 1, -1))
        self.cur_patch_start += len(self.cur_patch)
        self.cur_patch = []

    def _decode_current(self) -> torch.Tensor:
        ids = torch.tensor([self.cur_patch], dtype=torch.long, device=self.device)
        d = self.m.byte_emb(ids) + self.h_cur
        pos = 0 if not self.m.dynamic else self.cur_patch_start
        for layer in self.m.decoder_layers:
            d = layer(d, pos_offset=pos)  # causal (<= max_patch_len トークン)
        return self.m.head(self.m.head_norm(d[:, -1]))[0]  # (vocab,)

    def _advance_entropy_lm(self, byte_id: int) -> None:
        lm = self.m.entropy_model
        pos = len(self.byte_ids) - 1
        x = lm.embed(torch.tensor([[byte_id]], dtype=torch.long, device=self.device))
        x = x.to(self.dtype)
        for layer, cache in zip(lm.layers, self.lm_caches):
            if lm.attention_window is not None:
                cache.trim(max(0, lm.attention_window - 1))
            x = layer(x, kv_cache=cache, pos_offset=pos)
        logp = F.log_softmax(lm.head(lm.norm(x)).float()[0, -1], dim=-1)
        self.prev_entropy = float(-(logp.exp() * logp).sum())

    def _rebuild(self, keep: int) -> None:
        tail = self.byte_ids[-keep:]
        self.reset()
        for byte_id in tail:
            if self._starts_new_patch(byte_id):
                self._commit_patch()
            self.cur_patch.append(byte_id)
            self.byte_ids.append(byte_id)
            if self.cfg.patching_mode == "entropy":
                self._advance_entropy_lm(byte_id)
        # 次の push/_decode_current から通常運転


def build_arbor(model_cfg: dict[str, Any]) -> ArborModel:
    """config dict (configs/*.yaml の model 節) から構築する.

    patching_mode=entropy で entropy_model_ckpt が指定され、かつ存在する場合は
    そこから凍結 ByteLM の重みを読む (arbor 自体の checkpoint から resume する
    場合は、その後の strict ロードで上書きされるので二重指定でも安全)。
    """
    cfg = ArborConfig.from_dict(model_cfg)
    model = ArborModel(cfg)

    if cfg.patching_mode == "entropy" and cfg.entropy_model_ckpt:
        from pathlib import Path

        from safetensors.torch import load_file as safe_load

        ckpt = Path(cfg.entropy_model_ckpt)
        weights_file = ckpt / "model.safetensors"
        if weights_file.exists():
            size_mb = weights_file.stat().st_size / 2**20
            t0 = time.perf_counter()
            print(f"[arbor] loading entropy_model weights from {weights_file} ({size_mb:.1f}MiB)...")
            state = safe_load(str(weights_file), device="cpu")
            state = {key.removeprefix("_orig_mod."): v for key, v in state.items()}
            model.entropy_model.load_state_dict(state, strict=True)
            print(
                f"[arbor] entropy_model weights loaded from {ckpt} "
                f"in {time.perf_counter() - t0:.1f}s"
            )
        else:
            print(
                f"[arbor] WARNING: entropy_model_ckpt={ckpt} に model.safetensors が無い。"
                "重みは未初期化 (arbor checkpoint から resume するなら問題ない)"
            )

    counts = model.num_parameters()
    from src.model.bitlinear import BitLinear

    n_bit = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    print(
        f"[arbor] params={counts['total'] / 1e6:.1f}M "
        f"(global={counts['global'] / 1e6:.1f}M local_enc={counts['local_encoder'] / 1e6:.1f}M "
        f"local_dec={counts['local_decoder'] / 1e6:.1f}M emb/head={counts['embedding_head'] / 1e6:.1f}M "
        f"entropy_lm={counts['entropy_model'] / 1e6:.1f}M) "
        f"patching={cfg.patching_mode} bitnet={'ON' if cfg.bitnet else 'OFF'} "
        f"bitlinear_layers={n_bit} "
        "weights=W1.58(absmean ternary) "
        f"activations={_activation_desc(cfg.activation_precision)} "
        "subln=ON backward=STE(detach)"
    )
    return model
