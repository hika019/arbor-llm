"""Arbor v2: バイトレベル階層 Transformer × BitNet b1.58 (自己完結実装).

構造 (BLT, Pagnoni et al. 2024 の local 構造。patch 以降の処理は区切り方に依存しない):

    bytes (B, T)
      └ byte embedding + hash n-gram 埋め込み (n = hash_ngram_sizes)
      └ Local Encoder: 文書内 causal の byte 層 (n_enc 層)。各層の後に patch ごとの
          cross_attn_k 個の query が自 patch の byte へ cross-attention (初回 query は
          patch 内 max-pool の射影)。k 個を連結したものが patch 表現
      └ Global Transformer: 1 patch 右シフト + causal (n_global 層)。出力 j は
          「patch j より前の全バイト」だけを見る
      └ Local Decoder: 入力 = encoder の byte 表現。各層の前に、byte が自 patch の global
          出力 (k 個に分けたもの) へ cross-attention し、文書内 causal の byte 層を通す
      └ head (FP): logits[i] は bytes[0..i] のみから次バイトを予測

patching_mode (区切り方だけが違い、以降の処理は共通):
  static        patch_size バイトごと (と文書先頭) で区切る
  utf8          UTF-8 の文字先頭 byte を境界候補にする
  space         空白・改行の直後で区切る (BLT の space patching)
  entropy       凍結 ByteLM の次バイト予測エントロピーが threshold を超えた位置で区切る
  entropy_char  文字単位のエントロピーの上昇で区切る

学習の 1 系列は seq_patches 個の patch (BLT の seq_len と同じく patch 数で固定) で、byte 数は
区切った結果で決まる (max_bytes の枠に詰め、余りは PAD)。区切りと詰め込みは学習ループ側
(src/data/patch_packer.py) が文書の流れを続けて行い、forward には patch_starts で渡す。
patch_starts を省くと入力の各行をその場で区切る (推論・評価用)。境界判定は causal。encoder の
byte 層は causal なので patch 内の未来 byte を見ず、patch 表現は global の右シフトで次 patch
以降にしか届かない。

BitNet b1.58 準拠 (公式レシピ): absmean ternary W / absmax int8 A / detach STE /
SubLN / ReLU² gated FFN / bias 無し。Embedding・hash n-gram・射影・head・Norm は FP。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_OFFSET = 4  # 生バイト b は token id (b + 4)

# ByteLM の窓付き attention (WindowMask) の chunk 長 (T はこの倍数のとき窓経路)
_WINDOW_CHUNK = 128

def _is_block_mask(m: object) -> bool:
    """flex_attention の BlockMask かどうか (torch 未対応環境でも壊れないよう名前で判定)."""
    return type(m).__name__ == "BlockMask"


@dataclass
class WindowMask:
    """patch 内 attention 用の窓マスク (chunk × (chunk+2w)、causal は chunk × (chunk+w) のみ実体化).

    1 patch の長さは max_patch_len (= w) 以下なので、同一 patch の kv は
    q の前後 w バイト (causal なら前 w バイト) 以内に必ず収まる。これを利用して T×T の密マスクの
    代わりに chunk ごとの窓だけを見る (メモリ O(T·窓)、計算 ~T/窓 分の 1)。
    """

    mask: torch.Tensor  # (B, n_chunk, chunk, chunk + 2w) bool、causal は (B, n_chunk, chunk, chunk + w)
    chunk: int
    w: int
    causal: bool = False


def _windowed_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, wm: WindowMask
) -> torch.Tensor:
    """q: (B, H, T, d), T = n * chunk。k/v は (B, H, T, d) か、左に w 位置の文脈を足した (B, H, w + T, d)."""
    b, h, t, d = q.shape
    c, w = wm.chunk, wm.w
    right = 0 if wm.causal else w
    left = w - (k.size(2) - t)
    n = t // c
    win = c + w + right
    qc = q.view(b, h, n, c, d).permute(0, 2, 1, 3, 4).reshape(b * n, h, c, d)
    # kv は左 w (causal でなければ右も w) を pad してから chunk 幅 c でスライドして窓を切り出す
    kw = F.pad(k, (0, 0, left, right)).unfold(2, win, c).permute(0, 2, 1, 4, 3).reshape(b * n, h, win, d)
    vw = F.pad(v, (0, 0, left, right)).unfold(2, win, c).permute(0, 2, 1, 4, 3).reshape(b * n, h, win, d)
    out = F.scaled_dot_product_attention(qc, kw, vw, attn_mask=wm.mask.reshape(b * n, 1, c, win))
    return out.view(b, n, h, c, d).permute(0, 2, 1, 3, 4).reshape(b, h, t, d)


@dataclass
class ArborOutput:
    logits: torch.Tensor


@dataclass
class ArborConfig:
    vocab_size: int = 260          # 256 bytes + 特殊 4 (BOE/BOS/EOS/PAD)
    max_bytes: int = 2048          # 1 系列の byte の枠 (BLT の max_encoder_seq_length)。推論の文脈長の上限
    seq_patches: int = 128         # 学習の 1 系列の patch 数 (= global の系列長、BLT の seq_len)
    # ---- patching ----
    patching_mode: str = "static"  # choices: static | utf8 | space | entropy | entropy_char
    patch_size: int = 16           # static: 1 patch のバイト数 (= patch の最大長)
    min_patch_len: int = 2         # static 以外: これ未満では区切らない (文書先頭は除く)
    max_patch_len: int | None = 16  # static 以外: これに達したら区切る。None は上限なし
    # entropy: 次バイト H (nats) がこれを超えたら区切る。
    # entropy_char: 文字の H が 1 つ前の文字より これ を超えて上がったら区切る
    entropy_threshold: float = 1.5
    entropy_model: dict | None = None       # entropy 用: ByteLM の構成 (inline dict)
    entropy_model_ckpt: str | None = None   # entropy 用: 初回構築時に重みを読む checkpoint dir
    # entropy_char: 3〜4 byte 文字の H に「3 byte 目以降の予測エントロピー」の推定を足す。
    # 区切りは byte p+1 を予測する時点で決める必要があり、3 byte 目以降のエントロピーは
    # まだ計算できない (ひらがな・絵文字はどの文字かが主に 3〜4 byte 目で決まる)。ByteLM に
    # 1 byte 目までの hidden からその和を回帰する head を足して補う。重みは
    # <entropy_model_ckpt>/char_rest_head.safetensors (scripts/train_char_rest_head.py)
    entropy_char_rest: bool = False
    attention_window: int | None = None     # ByteLM 用: causal attention を直近 N byte に制限
    # ---- local (byte 階層、BLT の local encoder / decoder) ----
    local_hidden_size: int = 768
    local_num_heads: int = 12
    local_num_kv_heads: int = 12
    local_intermediate_size: int = 2048
    num_local_encoder_layers: int = 1
    num_local_decoder_layers: int = 2
    local_attn_window: int | None = None  # encoder / decoder の byte 層の causal 窓。None = 文書内の全文脈
    # local 層の Linear を BitLinear にするか。None は bitnet に従う
    local_bitnet: bool | None = None
    # ---- cross-attention (patch <-> byte) ----
    cross_attn_k: int = 2                 # patch あたりの query 数。patch 表現は k * local_hidden_size 次元
    cross_attn_heads: int | None = None   # None は local_num_heads
    decoder_cross_attn_all_layers: bool = True  # False なら decoder の最初の層の前だけ
    # ---- hash n-gram 埋め込み (直前 n byte の rolling polynomial hash を表引きして byte 埋め込みに足す) ----
    hash_ngram_sizes: tuple[int, ...] = (3, 4, 5, 6, 7, 8)  # 空なら無効
    hash_ngram_vocab: int = 50000         # n ごとの表の行数
    # ---- global (patch 階層) ----
    hidden_size: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_size: int = 4608
    num_hidden_layers: int = 16
    # ---- global 層の mixer (docs/global_seq_recurrent_experiment.md) ----
    # 層ごとに attention (A) か 系列方向の線形再帰 SSD (S) かを文字列パターンで指定し、層数分
    # 巡回する。None (既定) は全層 attention で従来と同一。例: "S" 全層再帰、"AS" 交互 (Jamba 型)。
    global_layer_pattern: str | None = None
    # 数値で書く版: N 層に 1 回 attention、残りは SSD (= "S"*(N-1) + "A" のパターン)。
    # 例: 4 → SSSA (attention 1/4)、8 → SSSSSSSA (1/8)。global_layer_pattern と同時指定は不可。
    global_attention_every: int | None = None
    ssd_conv_width: int = 4          # SSD 入力側の patch 方向 depthwise 因果 conv 幅
    ssd_chunk: int = 64              # SSD 並列 scan の chunk 長
    ssd_output_gate: bool = True     # SSD 出力ゲート (+1·d² / 層)。False で attention 層とパラメータ同等
    # scan の実装。fla = flash-linear-attention の Triton kernel (chunk_simple_gla、CUDA 専用、
    # torch 実装の ~11 倍速)。torch = 純 PyTorch の chunk scan (fp32、参照実装)。auto = CUDA なら fla。
    ssd_backend: str = "auto"        # choices: auto | fla | torch
    # ---- 共通 ----
    rope_theta: float = 500000.0
    # RoPE theta を階層別に上書きする (None なら rope_theta を使う)。
    #   global は seq_patches (例: 512) 位置しか見ないため、
    #   128k 長文脈向けの大きな theta は位置分解能を潰す (#1)。系列長相応に下げる。
    rope_theta_global: float | None = None
    rope_theta_local: float | None = None
    norm_eps: float = 1e-5
    bitnet: bool = True            # False で全 Linear を nn.Linear に (debug 用)
    activation_precision: str = "int8"  # BitLinear の活性量子化: int8 | bf8 | bf16
    gradient_checkpointing: bool = False
    # packing='document' の EOS 区切り。global attention を文書内に閉じる
    # (block-diagonal) ためのバイト ID。data.eos_token_id と一致させること。
    eos_token_id: int = 2
    pad_token_id: int = 3          # max_bytes の枠の余り。どの patch にも属さない
    # global (patch 階層) attention の実装。
    #   sdpa … 既定。文書マスク時は密 (B,1,K,K) マスク + SDPA (flash 非対応経路)。
    #   flex … CUDA 向け。文書境界を BlockMask にして flex_attention で fuse し、
    #          GQA も native (KV repeat_interleave 不要)。torch>=2.5 / 主に CUDA 用。
    global_attn_impl: str = "sdpa"  # choices: sdpa | flex

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArborConfig":
        removed = sorted(set(d) & set(_REMOVED_CONFIG_KEYS))
        if removed:
            raise ValueError("廃止した model 設定: " + ", ".join(
                f"{k} ({_REMOVED_CONFIG_KEYS[k]})" for k in removed))
        known = {f for f in cls.__dataclass_fields__}
        cfg = {k: v for k, v in d.items() if k in known}
        if "hash_ngram_sizes" in cfg:
            cfg["hash_ngram_sizes"] = tuple(cfg["hash_ngram_sizes"] or ())
        return cls(**cfg)


_LOCAL_REBUILT = "local 構造は BLT 方式に一本化した"
_REMOVED_CONFIG_KEYS = {
    "patch_pooling": _LOCAL_REBUILT, "patch_xattn_queries": _LOCAL_REBUILT,
    "num_byte_layers": _LOCAL_REBUILT, "byte_attn_window": _LOCAL_REBUILT,
    "max_patches": "系列を patch 数で固定するようにした。seq_patches と max_bytes で指定する",
}


# ------------------------------------------------------------------ modules
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm は内部 fp32 計算の fused カーネル (手書き 6 カーネル比 ~8 倍速)。
        # torch 2.14 から autocast 下の rms_norm は fp32 に昇格して fp32 を返す
        # (2.11 までは bf16)。そのまま下流へ流すと A8 量子化・FP8 GEMM・pointwise が
        # 全て fp32 経路になり学習が 25% 遅くなった (2026-09-15 実測 233k→173k bytes/s)。
        # autocast を外し、入出力 dtype を x に固定する (内部計算は変わらず fp32)。
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), self.eps)


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
        sub_norm: bool = True,
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
        self.attn_sub_norm = RMSNorm(n_heads * self.head_dim, norm_eps) if sub_norm else None

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
        # 素の causal path (マスク無しの global) は native enable_gqa を使い
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
            # ByteLM の窓付き causal: T×T を実体化しない
            out = _windowed_sdpa(q, k, v, attn_mask)
        elif attn_mask is not None:
            # 密 bool マスク (eager / CPU): causal 制約はマスク側に織り込み済み
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        elif is_incremental:
            if t != 1:
                raise ValueError("KV cache への追記は 1 トークンずつ行うこと")
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, enable_gqa=native_gqa)
        out = out.transpose(1, 2).reshape(b, t, -1)
        if self.attn_sub_norm is not None:
            out = self.attn_sub_norm(out)
        return self.wo(out)


    def forward_stream(
        self, x: torch.Tensor, k_ctx: torch.Tensor, v_ctx: torch.Tensor, attn_mask, keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, T, dim) を、直前 W 位置の K/V (RoPE 前、(B, Hkv, W, hd)) に続けて attend する.

        戻り値は (出力, 次の文脈の K, V)。次の文脈は [文脈, x] の keep (B, W) 位置の RoPE 前の K/V。
        """
        b, t, _ = x.shape
        w = k_ctx.size(2)
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = torch.cat((k_ctx.to(k_new.dtype), k_new), dim=2)
        v = torch.cat((v_ctx.to(v_new.dtype), v_new), dim=2)
        idx = keep[:, None, :, None].expand(-1, self.n_kv_heads, -1, self.head_dim)
        k_keep, v_keep = k.gather(2, idx), v.gather(2, idx)
        cos, sin = self.rope.cos[:w + t].to(q.dtype), self.rope.sin[:w + t].to(q.dtype)
        q, k = _apply_rope(q, cos[w:], sin[w:]), _apply_rope(k, cos, sin)
        n_rep = self.n_heads // self.n_kv_heads
        if _is_block_mask(attn_mask):
            from torch.nn.attention.flex_attention import flex_attention

            out = flex_attention(q, k, v, block_mask=attn_mask, enable_gqa=n_rep > 1)
        else:
            if n_rep > 1:
                k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
            if isinstance(attn_mask, WindowMask):
                out = _windowed_sdpa(q, k, v, attn_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, t, -1)
        if self.attn_sub_norm is not None:
            out = self.attn_sub_norm(out)
        return self.wo(out), k_keep, v_keep


class FeedForward(nn.Module):
    """gated FFN。relu2: down(subln(relu(gate(x))^2 * up(x))) (BitNet 2B4T)、swiglu: down(silu(gate(x)) * up(x))"""

    def __init__(self, dim: int, hidden: int, bitnet: bool, norm_eps: float,
                 activation_precision: str = "int8", sub_norm: bool = True,
                 activation: str = "relu2"):
        super().__init__()
        if activation not in ("relu2", "swiglu"):
            raise ValueError(f"unknown ffn activation: {activation!r} (choices: relu2 | swiglu)")
        self.activation = activation
        self.gate = _make_linear(dim, hidden, bitnet, activation_precision)
        self.up = _make_linear(dim, hidden, bitnet, activation_precision)
        self.down = _make_linear(hidden, dim, bitnet, activation_precision)
        self.ffn_sub_norm = RMSNorm(hidden, norm_eps) if sub_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group = getattr(self, "_fast_gate_up_group", None)
        if self.training and group is not None and group.training_weight_cache_enabled:
            gate, up = group(x).split((self.gate.out_features, self.up.out_features), dim=-1)
        else:
            gate, up = self.gate(x), self.up(x)
        if self.activation == "swiglu":
            h = F.silu(gate) * up
        else:
            a = F.relu(gate)
            h = a * a * up
        if self.ffn_sub_norm is not None:
            h = self.ffn_sub_norm(h)
        return self.down(h)


# flash-linear-attention の chunk_simple_gla (head ごとスカラー減衰の chunk scan、Triton) を
# torch.library.custom_op で包む。fla の autograd.Function をそのまま呼ぶと dynamo が層ごとに graph
# break して周囲の融合が壊れ、torch 実装より遅くなる (実測 69 vs 58 ms)。custom_op なら不透明な
# 1 op として compile に乗る。登録は SSDMixer の構築時に済ませる: forward 内で遅延登録すると custom_op の
# infer_schema が dynamo の skip 対象で graph break し、1B では CUDA graph が 1,600 個/step に割れて
# forward が 2 倍遅くなった (2026-09-19 プロファイル)。import 時にしないのは、fla の import が
# Triton 経由で CUDA を初期化し、SSD を使わない処理 (CPU 実行を含む) でも VRAM を取るため。
def _import_fla_without_package_init() -> None:
    """fla の親パッケージ __init__ を実行せずに fla.ops.simple_gla.chunk を読めるようにする.

    ``fla/__init__.py`` と ``fla/ops/__init__.py`` は全 layer / model / 全 op と
    transformers まで import し、環境が /mnt/d (9p) 上だと stat 待ちで ~88s かかる
    (実際に使うのは chunk_simple_gla だけ)。空の package module を先に登録して
    サブモジュールだけを読む: 88s → 12s (2026-09-25 実測)。fla が既に import 済みなら何もしない。
    """
    import importlib.util
    import re
    import sys
    import types
    from pathlib import Path

    if "fla" in sys.modules:
        return
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.submodule_search_locations:
        return
    base = Path(list(spec.submodule_search_locations)[0])
    for name, sub in (("fla", base), ("fla.ops", base / "ops"),
                      ("fla.ops.simple_gla", base / "ops" / "simple_gla")):
        mod = types.ModuleType(name)
        mod.__path__ = [str(sub)]
        mod.__package__ = name
        sys.modules[name] = mod
    # fla.utils が `from .. import __version__` するので本物の値を入れておく
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", (base / "__init__.py").read_text())
    sys.modules["fla"].__version__ = m.group(1) if m else "0"


def _register_fla_scan_op() -> bool:
    try:
        _import_fla_without_package_init()
        from fla.ops.simple_gla.chunk import (
            RCP_LN2, chunk_local_cumsum, chunk_simple_gla_bwd, chunk_simple_gla_fwd,
        )
    except ImportError:
        return False

    @torch.library.custom_op("arbor::ssd_scan", mutates_args=())
    def ssd_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_a: torch.Tensor) -> torch.Tensor:
        g = chunk_local_cumsum(log_a, chunk_size=64, scale=RCP_LN2)
        o, _ = chunk_simple_gla_fwd(q=q, k=k, v=v, g=g, scale=1.0, chunk_size=64)
        return o.to(q.dtype)

    @ssd_scan.register_fake
    def _(q, k, v, log_a):
        return torch.empty_like(q)

    # backward も不透明な custom_op にする (register_autograd の関数は AOTAutograd にトレースされ、
    # Triton kernel の中に入って FakeTensor エラーになる)
    @torch.library.custom_op("arbor::ssd_scan_bwd", mutates_args=())
    def ssd_scan_bwd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_a: torch.Tensor, do: torch.Tensor,
    ) -> list[torch.Tensor]:
        g = chunk_local_cumsum(log_a, chunk_size=64, scale=RCP_LN2)
        dq, dk, dv, dg, _ = chunk_simple_gla_bwd(
            q=q, k=k, v=v, g=g, g_gamma=None, initial_state=None, do=do.contiguous(), dht=None,
            scale=1.0, chunk_size=64,
        )
        dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True).to(log_a.dtype)
        return [dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg]

    @ssd_scan_bwd.register_fake
    def _(q, k, v, log_a, do):
        return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(log_a)]

    def _backward(ctx, do):
        q, k, v, log_a = ctx.saved_tensors
        dq, dk, dv, dg = torch.ops.arbor.ssd_scan_bwd(q, k, v, log_a, do)
        return dq, dk, dv, dg

    def _setup(ctx, inputs, output):
        q, k, v, log_a = inputs
        ctx.save_for_backward(q, k, v, log_a)

    ssd_scan.register_autograd(_backward, setup_context=_setup)
    return True


_FLA_SCAN_OP_AVAILABLE: bool | None = None


def _ensure_fla_scan_op() -> None:
    global _FLA_SCAN_OP_AVAILABLE
    if _FLA_SCAN_OP_AVAILABLE is None:
        _FLA_SCAN_OP_AVAILABLE = _register_fla_scan_op()


def _fla_chunk_simple_gla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_a: torch.Tensor) -> torch.Tensor:
    if not _FLA_SCAN_OP_AVAILABLE:  # 暗黙フォールバックはしない (速度が桁で違うので黙って落ちると困る)
        raise RuntimeError(
            "ssd_backend=fla/auto(CUDA) には flash-linear-attention が必要: "
            "pip install flash-linear-attention (無ければ ssd_backend: torch を明示)"
        )
    return torch.ops.arbor.ssd_scan(q.contiguous(), k.contiguous(), v.contiguous(), log_a.contiguous())


class SSDMixer(nn.Module):
    """系列方向の線形再帰 (Mamba-2 SSD / GLA と同型)。global 層で attention の代わりに使う.

    head ごとに dh×dh の状態 S を持ち、patch が 1 つ来るごとに
        S_t = a_t · S_{t-1} + k_tᵀ v_t,   out_t = q_t · S_t
    で更新する (a_t ∈ (0,1) は入力依存の忘却ゲート、head ごとのスカラー)。文書境界 (seg が変わる位置)
    で a_t = 0 にして状態をリセットする。学習時は chunk 単位の並列 scan (行列積だけ) で計算し、
    state の受け渡しだけ chunk 数 (512/64 = 8) の逐次ループ。
    入力側に patch 方向の depthwise 因果 conv (幅 conv_width、文書境界マスク) と SiLU、
    出力側に sub-norm と出力ゲートを付ける (線形 RNN の定番構成)。
    q/k/v/gate/out 射影は BitLinear (k, v は n_kv_heads で共有 = GQA と同じ節約)。
    """

    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, bitnet: bool, norm_eps: float,
        activation_precision: str = "int8", conv_width: int = 4, chunk: int = 64,
        output_gate: bool = True, backend: str = "auto",
    ):
        super().__init__()
        if backend not in ("auto", "fla", "torch"):
            raise ValueError(f"unknown ssd_backend: {backend!r} (choices: auto | fla | torch)")
        self.backend = backend
        if backend == "fla" or (backend == "auto" and torch.cuda.is_available()):
            _ensure_fla_scan_op()
        if dim % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError(f"invalid head config: {dim=} {n_heads=} {n_kv_heads=}")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.conv_width, self.chunk = conv_width, chunk
        kv_width = n_kv_heads * self.head_dim
        self.wq = _make_linear(dim, dim, bitnet, activation_precision)
        self.wk = _make_linear(dim, kv_width, bitnet, activation_precision)
        self.wv = _make_linear(dim, kv_width, bitnet, activation_precision)
        self.wg = _make_linear(dim, dim, bitnet, activation_precision) if output_gate else None  # 出力ゲート
        self.wo = _make_linear(dim, dim, bitnet, activation_precision)
        # 忘却ゲート logit (FP)。bias を head ごとに 1..5 に散らし、記憶長 ~4..150 patch で初期化
        self.wa = nn.Linear(dim, n_heads, bias=True)
        nn.init.trunc_normal_(self.wa.weight, std=0.02, a=-0.06, b=0.06)
        with torch.no_grad():
            self.wa.bias.copy_(torch.linspace(1.0, 5.0, n_heads))
        # depthwise 因果 conv: tap 0 (現在) を 1、過去 tap を小さく初期化
        self.conv_weight = nn.Parameter(torch.empty(conv_width, dim))
        nn.init.trunc_normal_(self.conv_weight, std=0.02, a=-0.06, b=0.06)
        with torch.no_grad():
            self.conv_weight[0].fill_(1.0)
        self.sub_norm = RMSNorm(dim, norm_eps)

    def _causal_conv(self, x: torch.Tensor, seg: torch.Tensor | None) -> torch.Tensor:
        # x (B,K,d)。過去 tap j は seg[t-j] == seg[t] のときだけ使う (文書を跨がない)
        y = x * self.conv_weight[0].to(x.dtype)
        for j in range(1, min(self.conv_width, x.size(1))):   # 系列が tap 数より短い (生成の序盤) なら無い tap は飛ばす
            xs = F.pad(x[:, :-j], (0, 0, j, 0))
            if seg is not None:
                same = F.pad(seg[:, :-j] == seg[:, j:], (j, 0), value=False)
                xs = xs * same.unsqueeze(-1).to(x.dtype)
            y = y + xs * self.conv_weight[j].to(x.dtype)
        return y

    def forward(self, x: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        b, k, d = x.shape
        h, hkv, dh, c = self.n_heads, self.n_kv_heads, self.head_dim, self.chunk
        xc = F.silu(self._causal_conv(x, seg))
        qkv_group = getattr(self, "_fast_qkv_group", None)   # Attention と同じ融合 QKV 経路 (低ビット学習)
        if self.training and qkv_group is not None and qkv_group.training_weight_cache_enabled:
            kv_width = hkv * dh
            q, kk, v = qkv_group(xc).split((h * dh, kv_width, kv_width), dim=-1)
            q, kk, v = q.view(b, k, h, dh), kk.view(b, k, hkv, dh), v.view(b, k, hkv, dh)
        else:
            q = self.wq(xc).view(b, k, h, dh)
            kk = self.wk(xc).view(b, k, hkv, dh)
            v = self.wv(xc).view(b, k, hkv, dh)
        if hkv != h:
            kk = kk.repeat_interleave(h // hkv, dim=2)
            v = v.repeat_interleave(h // hkv, dim=2)
        # 忘却ゲートの logit は計算 dtype で、減衰の累積 (log_a) は fp32 (累積積は精度に敏感)
        gate_logit = F.linear(x, self.wa.weight.to(x.dtype), self.wa.bias.to(x.dtype))
        with torch.autocast(device_type=x.device.type, enabled=False):
            log_a = F.logsigmoid(gate_logit.float())                       # (B,K,H) ≤ 0
            if seg is not None:
                reset = F.pad(seg[:, 1:] != seg[:, :-1], (1, 0), value=True)  # 文書先頭で状態を切る
                log_a = torch.where(reset.unsqueeze(-1), torch.full_like(log_a, -1e4), log_a)
            use_fla = self.backend == "fla" or (self.backend == "auto" and x.is_cuda)
            if use_fla:
                y = _fla_chunk_simple_gla(q * dh ** -0.5, kk, v, log_a)         # bf16 入出力、内部 fp32 累積
            else:
                y = self._chunked_scan(q.float() * dh ** -0.5, kk.float(), v.float(), log_a)
        y = self.sub_norm(y.to(x.dtype).reshape(b, k, d))
        if self.wg is not None:
            y = y * F.silu(self.wg(x))
        return self.wo(y)

    def _chunked_scan(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_a: torch.Tensor,
    ) -> torch.Tensor:
        b, kk, h, dh = q.shape
        c = self.chunk
        pad = (c - kk % c) % c
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
            log_a = F.pad(log_a, (0, 0, 0, pad))
        n = q.size(1) // c
        q, k, v = (t.view(b, n, c, h, dh) for t in (q, k, v))
        cum = log_a.view(b, n, c, h).cumsum(dim=2)                             # chunk 内累積
        # chunk 内: L[j,i] = exp(cum[j] - cum[i]) (i ≤ j)、Y = (Q Kᵀ ⊙ L) V
        diff = cum.unsqueeze(3) - cum.unsqueeze(2)                              # (B,N,Cj,Ci,H)
        causal = torch.ones(c, c, dtype=torch.bool, device=q.device).tril().view(1, 1, c, c, 1)
        L = torch.where(causal, diff, torch.full_like(diff, -1e4)).exp()
        scores = torch.einsum("bnjhd,bnihd->bnjih", q, k) * L
        y = torch.einsum("bnjih,bnihd->bnjhd", scores, v)
        # chunk 末尾の状態: S_n(local) = Σ_i exp(cum[C-1] - cum[i]) k_iᵀ v_i
        decay_to_end = (cum[:, :, -1:, :] - cum).exp()                         # (B,N,C,H)
        s_local = torch.einsum("bnchd,bnche,bnch->bnhde", k, v, decay_to_end)   # (B,N,H,dh,dh)
        chunk_decay = cum[:, :, -1, :].exp()                                    # (B,N,H)
        # chunk 間: S_n = decay_n · S_{n-1} + S_n(local)、前 chunk 状態からの寄与 y += exp(cum) q S_{n-1}
        state = torch.zeros(b, h, dh, dh, dtype=q.dtype, device=q.device)
        in_decay = cum.exp()                                                    # (B,N,C,H)
        outs = []
        for i in range(n):
            outs.append(torch.einsum("bchd,bhde,bch->bche", q[:, i], state, in_decay[:, i]))
            state = chunk_decay[:, i].view(b, h, 1, 1) * state + s_local[:, i]
        y = y + torch.stack(outs, dim=1)
        y = y.reshape(b, n * c, h, dh)
        return y[:, :kk] if pad else y


class Block(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, ffn_hidden: int,
        rope: RotaryEmbedding, bitnet: bool, norm_eps: float, causal: bool,
        activation_precision: str = "int8", mixer: str = "attention",
        ssd_conv_width: int = 4, ssd_chunk: int = 64, ssd_output_gate: bool = True,
        ssd_backend: str = "auto", sub_norm: bool = True, ffn_activation: str = "relu2",
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, norm_eps)
        if mixer == "attention":
            self.attn = Attention(dim, n_heads, n_kv_heads, rope, bitnet, norm_eps, causal,
                                  activation_precision, sub_norm=sub_norm)
            self.mixer = None
        elif mixer == "ssd":
            if not causal:
                raise ValueError("SSDMixer は causal 専用")
            self.attn = None
            self.mixer = SSDMixer(dim, n_heads, n_kv_heads, bitnet, norm_eps, activation_precision,
                                  conv_width=ssd_conv_width, chunk=ssd_chunk, output_gate=ssd_output_gate,
                                  backend=ssd_backend)
        else:
            raise ValueError(f"unknown mixer: {mixer!r} (choices: attention | ssd)")
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.ffn = FeedForward(dim, ffn_hidden, bitnet, norm_eps, activation_precision,
                               sub_norm=sub_norm, activation=ffn_activation)

    @property
    def out_proj_owner(self) -> nn.Module:
        return self.attn if self.attn is not None else self.mixer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: "torch.Tensor | WindowMask | None" = None,
        kv_cache: "_LayerKVCache | None" = None,
        pos_offset: int = 0,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attn is not None:
            x = x + self.attn(self.attn_norm(x), attn_mask, kv_cache, pos_offset)
        else:
            if kv_cache is not None:
                raise NotImplementedError("SSDMixer は KV cache (逐次生成) 未対応")
            x = x + self.mixer(self.attn_norm(x), seg)
        return x + self.ffn(self.ffn_norm(x))


def _scale_residual_projections(layer_lists: list[nn.ModuleList]) -> None:
    """残差に入る出力射影 (wo / down) を GPT-2 流に 1/sqrt(2L) へ縮小."""
    n_layers = sum(len(layers) for layers in layer_lists)
    scale = (2 * max(n_layers, 1)) ** -0.5
    with torch.no_grad():
        for layers in layer_lists:
            for block in layers:
                block.out_proj_owner.wo.weight.mul_(scale)
                block.ffn.down.weight.mul_(scale)


class CrossAttention(nn.Module):
    """BLT の cross-attention (q / kv それぞれ RMSNorm → 射影、RoPE・位置埋め込みなし)。残差は呼び出し側で足す.

    射影は他の attention と同じく BitLinear + SubLN (bitnet=False なら nn.Linear)。
    """

    def __init__(self, dim: int, n_heads: int, norm_eps: float, bitnet: bool,
                 activation_precision: str, out_scale: float):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"cross-attention の次元 {dim} は heads {n_heads} で割り切れる必要がある")
        # n_heads という名前にしない: install_arbor_projection_fusions が wq/wk/wv を 1 本の GEMM に
        # 融合する対象 (同じ入力を射影する self-attention) と誤認する。q と kv は入力が違う
        self.heads, self.head_dim = n_heads, dim // n_heads
        self.norm_q = RMSNorm(dim, norm_eps)
        self.norm_kv = RMSNorm(dim, norm_eps)
        self.wq, self.wk, self.wv, self.wo = (
            _make_linear(dim, dim, bitnet, activation_precision) for _ in range(4)
        )
        self.attn_sub_norm = RMSNorm(dim, norm_eps)
        with torch.no_grad():
            self.wo.weight.mul_(out_scale)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.heads, self.head_dim))

    def _out(self, out: torch.Tensor) -> torch.Tensor:
        return self.wo(self.attn_sub_norm(out))

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """x: (N, Q, dim) が kv: (N, L, dim) の全位置へ attend する (生成器の 1 patch 用)."""
        q = self._heads(self.wq(self.norm_q(x))).transpose(1, 2)
        kvn = self.norm_kv(kv)
        k = self._heads(self.wk(kvn)).transpose(1, 2)
        v = self._heads(self.wv(kvn)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self._out(out.transpose(1, 2).flatten(2))

    def forward_segments(
        self, x: torch.Tensor, kv: torch.Tensor, patch_id: torch.Tensor, is_byte: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, K, k, dim) が kv: (B, T, dim) の自 patch の byte へ attend する。byte の無い patch は 0."""
        b, n_patch, kq, _ = x.shape
        q = self._heads(self.wq(self.norm_q(x)))                          # (B, K, k, H, hd)
        kvn = self.norm_kv(kv)
        k = self._heads(self.wk(kvn))                                     # (B, T, H, hd)
        v = self._heads(self.wv(kvn))
        idx = patch_id[:, :, None, None, None].expand(-1, -1, kq, self.heads, self.head_dim)
        q_byte = q.gather(1, idx)                                         # (B, T, k, H, hd)
        scores = torch.einsum("btjhd,bthd->btjh", q_byte.float(), k.float()) * self.head_dim ** -0.5
        scores = scores.masked_fill(~is_byte[:, :, None, None], -1e9)
        seg = patch_id[:, :, None, None].expand_as(scores)
        peak = scores.new_full((b, n_patch, kq, self.heads), -1e9).scatter_reduce(
            1, seg, scores, reduce="amax", include_self=True,
        ).detach()
        w = (scores - peak.gather(1, seg)).exp() * is_byte[:, :, None, None]
        denom = w.new_zeros((b, n_patch, kq, self.heads)).scatter_add(1, seg, w)
        num = torch.einsum("btjh,bthd->btjhd", w, v.float())
        out = q.new_zeros((b, n_patch, kq, self.heads, self.head_dim), dtype=torch.float32).scatter_add(
            1, idx, num,
        ) / denom.clamp_min(1e-30).unsqueeze(-1)
        return self._out(out.to(v.dtype).flatten(-2))

    def forward_per_byte(self, x: torch.Tensor, kv_patch: torch.Tensor, patch_id: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim) の各 byte が、自 patch の kv_patch: (B, K, k, dim) の k 個へ attend する.

        k/v の射影は patch 単位で行ってから byte へ gather する (byte 単位で射影すると T/K 倍の計算)。
        """
        b, t, _ = x.shape
        n_patch, kq = kv_patch.shape[1], kv_patch.shape[2]
        q = self._heads(self.wq(self.norm_q(x)))                          # (B, T, H, hd)
        kvn = self.norm_kv(kv_patch).flatten(1, 2)                        # (B, K*k, dim)
        idx = patch_id[:, :, None, None, None].expand(-1, -1, kq, self.heads, self.head_dim)
        k = self._heads(self.wk(kvn)).unflatten(1, (n_patch, kq)).gather(1, idx)  # (B, T, k, H, hd)
        v = self._heads(self.wv(kvn)).unflatten(1, (n_patch, kq)).gather(1, idx)
        scores = torch.einsum("bthd,btjhd->bthj", q.float(), k.float()) * self.head_dim ** -0.5
        out = torch.einsum("bthj,btjhd->bthd", scores.softmax(dim=-1).to(v.dtype), v)
        return self._out(out.reshape(b, t, -1))


def _wrap_int64(v: int) -> int:
    """int64 の積が overflow して wrap した値 (BLT は prime**j を int64 tensor で計算している)."""
    v %= 2**64
    return v - 2**64 if v >= 2**63 else v


class HashNgramEmbedding(nn.Module):
    """BLT の hash n-gram 埋め込み: 直前 n byte (自身を含む) の rolling polynomial hash を表引きする.

    系列先頭・文書先頭より前の byte は 0 として hash する (文書をまたいで情報が漏れない)。
    """

    _PRIME = 1000000007  # BLT の hash 関数 0 番の素数

    def __init__(self, sizes: tuple[int, ...], vocab: int, dim: int):
        super().__init__()
        self.sizes, self.vocab = tuple(int(n) for n in sizes), int(vocab)
        self.tables = nn.ModuleList(nn.Embedding(self.vocab, dim) for _ in self.sizes)
        std = dim ** -0.5
        for table in self.tables:
            nn.init.trunc_normal_(table.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, ids: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
        """ids, doc: (B, T)。各 n の埋め込みの和 (B, T, dim) を返す."""
        out = None
        for n, table in zip(self.sizes, self.tables):
            win = F.pad(ids, (n - 1, 0)).unfold(1, n, 1)                  # (B, T, n): b_{i-n+1..i}
            win_doc = F.pad(doc, (n - 1, 0), value=-1).unfold(1, n, 1)
            win = torch.where(win_doc == doc.unsqueeze(-1), win, 0).long()
            powers = torch.tensor([_wrap_int64(self._PRIME ** j) for j in range(n)], device=ids.device)
            emb = table((win * powers).sum(-1) % self.vocab)
            out = emb if out is None else out + emb
        return out


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
                  activation_precision=activation_precision,
                  sub_norm=bool(cfg.get("sub_norm", False)),
                  ffn_activation=cfg.get("ffn_activation", "swiglu"))
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(h, norm_eps)
        self.head = nn.Linear(h, self.vocab_size, bias=False)
        nn.init.trunc_normal_(self.head.weight, std=0.02, a=-0.06, b=0.06)
        self.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))
        _scale_residual_projections([self.layers])
        # 多バイト文字の 1 byte 目の位置で、その文字の 3 byte 目以降の予測エントロピーの和を
        # 回帰する (ArborConfig.entropy_char_rest)。本体は凍結したまま head だけを学習する。
        # Linear 1 本より 2 層の方が回帰の R² が 0.54 → 0.65 と良い (中間 1024 でも 0.66)
        self.char_rest = (
            nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Linear(256, 1))
            if cfg.get("char_rest_head", False) else None
        )

    def forward(self, input_ids: torch.Tensor) -> ArborOutput:
        return ArborOutput(logits=self.head(self.norm(self._hidden(input_ids))))

    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        if torch.is_autocast_enabled(x.device.type):
            # fp32 parameter でも残差の流れは計算 dtype にする (fp32 のままだと正規化・残差加算の読み書きが倍)
            x = x.to(torch.get_autocast_dtype(x.device.type))
        attn_mask = self._attention_mask(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(layer, x, attn_mask, use_reentrant=False)
            else:
                x = layer(x, attn_mask)
        return x

    def _attention_mask(self, input_ids: torch.Tensor) -> "torch.Tensor | WindowMask | object | None":
        if self.attention_window is None:
            return None
        b, t = input_ids.shape
        w = min(int(self.attention_window), t)
        if input_ids.is_cuda and torch.compiler.is_compiling():
            # compile 下では sliding window を flex_attention の BlockMask で表す。窓外の
            # block を丸ごと飛ばす fused kernel になり、計算は ~T·w (WindowMask の masked
            # SDPA は chunk + 2w 幅を全部計算する)。eager の flex は T×T を実体化するので
            # compile 時だけ使い、eager / CPU は下の WindowMask 経路で同じマスクを課す。
            from torch.nn.attention.flex_attention import create_block_mask

            def mask_mod(b_idx, h_idx, q_idx, kv_idx):
                return (kv_idx <= q_idx) & (q_idx - kv_idx < w)

            return create_block_mask(mask_mod, B=None, H=None, Q_LEN=t, KV_LEN=t,
                                     device=input_ids.device)
        c = _WINDOW_CHUNK
        if t % c == 0 and t >= c:
            ar_q = torch.arange(c, device=input_ids.device).view(1, 1, c, 1)
            ar_k = torch.arange(c + w, device=input_ids.device).view(1, 1, 1, c + w)
            chunk_start = (torch.arange(t // c, device=input_ids.device) * c).view(1, -1, 1, 1)
            q_pos = chunk_start + ar_q
            k_pos = chunk_start - w + ar_k
            mask = (k_pos >= 0) & (k_pos <= q_pos) & (q_pos - k_pos < w)
            return WindowMask(mask.expand(b, -1, -1, -1), c, w, causal=True)
        q = torch.arange(t, device=input_ids.device).unsqueeze(1)
        k = torch.arange(t, device=input_ids.device).unsqueeze(0)
        return ((k <= q) & (q - k < w)).view(1, 1, t, t)

    def _stream_mask(self, ctx_valid: torch.Tensor, t: int):
        """chunk (長さ t) の query と、直前 W 位置 (有効は ctx_valid (B, W)) + chunk の key の窓付き causal マスク."""
        b, w = ctx_valid.shape
        win = int(self.attention_window)
        dev = ctx_valid.device
        if ctx_valid.is_cuda and torch.compiler.is_compiling():
            from torch.nn.attention.flex_attention import create_block_mask

            def mask_mod(b_idx, h_idx, q_idx, kv_idx):
                kv_pos = kv_idx - w
                valid = (kv_idx >= w) | ctx_valid[b_idx, torch.clamp(kv_idx, max=w - 1)]
                return (kv_pos <= q_idx) & (q_idx - kv_pos < win) & valid

            return create_block_mask(mask_mod, B=b, H=None, Q_LEN=t, KV_LEN=w + t, device=dev)
        c = _WINDOW_CHUNK
        if w == win and t % c == 0:
            ar_q = torch.arange(c, device=dev).view(1, 1, c, 1)
            ar_k = torch.arange(c + w, device=dev).view(1, 1, 1, c + w)
            chunk_start = (torch.arange(t // c, device=dev) * c).view(1, -1, 1, 1)
            q_pos, k_pos = chunk_start + ar_q, chunk_start - w + ar_k
            ctx_ok = ctx_valid[:, (w + k_pos).clamp(0, w - 1).flatten()].view(b, -1, 1, c + w)
            mask = ((k_pos >= 0) | ctx_ok) & (k_pos <= q_pos) & (q_pos - k_pos < win)
            return WindowMask(mask, c, w, causal=True)
        q_pos = torch.arange(t, device=dev).view(1, t, 1)
        k_pos = torch.arange(w + t, device=dev).view(1, 1, w + t) - w
        ok = torch.cat((ctx_valid, torch.ones(b, t, dtype=torch.bool, device=dev)), dim=1).unsqueeze(1)
        return ((k_pos <= q_pos) & (q_pos - k_pos < win) & ok).unsqueeze(1)

    @torch.no_grad()
    def boundary_entropy_stream(
        self, input_ids: torch.Tensor, k_ctx: torch.Tensor, v_ctx: torch.Tensor, ctx_valid: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """boundary_entropy を、直前 W 位置の各層 K/V (RoPE 前、(L, B, Hkv, W, hd)) に続けて計算する.

        流れ全体を一度に通したのと同じ値になる。後ろの 2 つは各行の実長 lengths (B,) までを読んだ後の
        直前 W 位置の各層 K/V (L, B, Hkv, W, hd)。
        """
        if self.attention_window is None:
            raise ValueError("boundary_entropy_stream は attention_window が必要")
        x = self.embed(input_ids)
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        mask = self._stream_mask(ctx_valid, input_ids.size(1))
        w = ctx_valid.size(1)
        keep = lengths.unsqueeze(1) + torch.arange(w, device=lengths.device)
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            h, k, v = layer.attn.forward_stream(layer.attn_norm(x), k_ctx[i], v_ctx[i], mask, keep)
            x = x + h
            x = x + layer.ffn(layer.ffn_norm(x))
            ks.append(k)
            vs.append(v)
        x = self.norm(x)
        logp = F.log_softmax(self.head(x).float(), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        return ent, self.char_rest_from_hidden(x), torch.stack(ks), torch.stack(vs)

    @torch.no_grad()
    def next_byte_entropy(self, input_ids: torch.Tensor) -> torch.Tensor:
        """各位置の「次バイト分布のエントロピー (nats)」(B, T) を返す.

        戻り値の [.., t] は p(x_{t+1} | x_{<=t}) のエントロピー。
        """
        return self.boundary_entropy(input_ids)[0]

    @torch.no_grad()
    def boundary_entropy(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(次バイトのエントロピー, char_rest head の予測) を 1 回の forward で返す.

        head の [.., t] は x_t が 3〜4 byte 文字の 1 byte 目のとき、その文字の 3 byte 目以降の
        予測エントロピーの和の推定 (それ以外の位置の値は使わない)。head が無ければ None。
        """
        x = self.norm(self._hidden(input_ids))
        logp = F.log_softmax(self.head(x).float(), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        return ent, self.char_rest_from_hidden(x)

    def char_rest_from_hidden(self, normed_hidden: torch.Tensor) -> torch.Tensor | None:
        if self.char_rest is None:
            return None
        w_dtype = self.char_rest[0].weight.dtype
        return F.softplus(self.char_rest(normed_hidden.to(w_dtype)).float().squeeze(-1))


def build_byte_lm(model_cfg: dict[str, Any]) -> ByteLM:
    model = ByteLM(dict(model_cfg))
    n = sum(p.numel() for p in model.parameters())
    print(f"[byte_lm] params={n / 1e6:.1f}M bitnet={'ON' if model_cfg.get('bitnet', False) else 'OFF'}")
    return model


# ----------------------------------------------------------- patch 境界判定
_SPACE_BYTES = (0x20, 0x09, 0x0A, 0x0D)  # space, tab, LF, CR
ENTROPY_MODES = ("entropy", "entropy_char")
CHAR_REST_HEAD_FILE = "char_rest_head.safetensors"  # ByteLM の step dir に置く char_rest head の重み
_UTF8_MAX_CONT = 3  # UTF-8 の 1 文字の継続 byte の最大数


def _is_utf8_char_start_byte(byte: int) -> bool:
    """Return whether byte can start a UTF-8 code point."""
    return byte < 0x80 or 0xC2 <= byte <= 0xF4


def _patch_starts_reference(
    raw: torch.Tensor, force: torch.Tensor, min_len: int, max_len: int,
    char_start: torch.Tensor | None = None, soft_len: int = 0,
) -> torch.Tensor:
    """境界 walk の CPU 参照実装。CUDA kernel (csrc/patch_starts.cu) と生成器
    (ArborByteGenerator._starts_new_patch) はこれと同じ規則で実装する.

    - 位置 0 は必ず patch 先頭。patch は最長 max_len (到達したら強制的に区切る)
    - force[p] (文書先頭) は min_len に関係なく区切り候補
    - raw[p] は現 patch が min_len 以上のときだけ候補
    - soft_len > 0 (entropy_char) なら run >= soft_len の文字先頭 (char_start) も候補にし、
      max_len 到達前に文字の境目で区切る
    判定は位置 p までの byte だけで決まる (因果的) ので、続きの byte が届いたら最後の
    patch 先頭から walk し直せば、まとめて walk したのと同じ区切りになる。
    """
    b, t = raw.shape
    starts = torch.zeros(b, t, dtype=torch.bool)
    raw_l, force_l = raw.cpu().tolist(), force.cpu().tolist()
    cs_l = char_start.cpu().tolist() if char_start is not None else None
    for r in range(b):
        i = 0
        while i < t:
            starts[r, i] = True
            hi = min(i + max_len, t)
            lo = i + min_len
            nxt = hi
            for p in range(i + 1, hi):
                soft = soft_len > 0 and p - i >= soft_len and (cs_l is None or cs_l[r][p])
                if force_l[r][p] or (p >= lo and (raw_l[r][p] or soft)):
                    nxt = p
                    break
            i = nxt
    return starts.to(raw.device)


# 境界 walk は逐次処理だが出力形状は入力と同じ (B, T) bool なので、custom_op + fake 実装に
# しておけば torch.compile は不透明な 1 op として扱い、graph break も CUDA graph の分断も
# 起きない (旧実装は @torch.compiler.disable で graph が割れ、step あたり ~37ms の CPU
# 待ちが出ていた: 2026-09-26 プロファイル)。
@torch.library.custom_op("arbor::patch_starts", mutates_args=())
def _patch_starts_op(
    raw: torch.Tensor, force: torch.Tensor, char_start: torch.Tensor,
    min_len: int, max_len: int, soft_len: int,
) -> torch.Tensor:
    if raw.is_cuda:
        from src.model.patch_starts_cuda import patch_starts_cuda

        return patch_starts_cuda(raw, force, min_len, max_len, char_start, soft_len)
    return _patch_starts_reference(raw, force, min_len, max_len, char_start, soft_len)


@_patch_starts_op.register_fake
def _(raw, force, char_start, min_len, max_len, soft_len):
    return torch.empty_like(raw)


def patch_len_bounds(cfg: ArborConfig) -> tuple[int, int | None]:
    """(min_patch_len, max_patch_len)。static は候補なしで patch_size ごとに区切る."""
    if cfg.patching_mode == "static":
        return 1, cfg.patch_size
    return cfg.min_patch_len, cfg.max_patch_len


def soft_patch_len(mode: str, max_len: int | None) -> int:
    """entropy_char で max_len 手前から文字先頭で区切り始める長さ (0 = 無効)."""
    return max_len - _UTF8_MAX_CONT if mode == "entropy_char" and max_len is not None else 0


def patch_candidates(
    input_ids: torch.Tensor,
    mode: str,
    entropy_values: torch.Tensor | None = None,
    rest_values: torch.Tensor | None = None,
    threshold: float = 1.5,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """境界候補 (raw, force, char_start) を返す (いずれも (B, T) bool)。位置 p の値は byte p 以前だけで決まる.

    - utf8:    現在バイトが UTF-8 文字先頭なら候補
    - space:   直前バイトが空白系なら候補
    - entropy: 直前位置での次バイト予測エントロピーが threshold 超なら候補
    - static:  候補なし (max_len ごとと文書先頭だけで区切る)
    - entropy_char: 文字単位のエントロピー上昇 (Harris の successor variety をエントロピーにした
      Jin & Tanaka-Ishii 2006 の増加基準)。多バイト文字は 1 byte 目 (範囲) より 2 byte 目 (どの文字か)
      の予測が難しく、byte の entropy では文字の途中で区切られるので、文字の H = その文字の
      1 byte 目 + (多バイトなら) 2 byte 目の予測エントロピーとし (2 byte 目の予測は文字先頭までで
      決まる)、1 つ前の文字の H より threshold を超えて上がった文字の先頭を候補にする。絶対値の
      閾値と違い文字種ごとの H の水準差 (漢字 ≫ 英字) に左右されない。rest_values (ByteLM の
      char_rest head) があれば 3〜4 byte 文字の H に 3 byte 目以降の予測エントロピーの推定を足す。
    force は文書先頭 (eos_token_id の直後)。系列の先頭は前の byte が無いので候補にしない
    (walk が必ず区切る)。
    """
    cur = input_ids - BYTE_OFFSET
    char_start = (cur < 0x80) | ((cur >= 0xC2) & (cur <= 0xF4))
    if mode == "static":
        raw = torch.zeros_like(char_start)
    elif mode == "utf8":
        raw = char_start.clone()
        raw[:, 0] = False
    elif mode == "space":
        prev = input_ids[:, :-1] - BYTE_OFFSET
        is_space = torch.zeros_like(prev, dtype=torch.bool)
        for sb in _SPACE_BYTES:
            is_space |= prev == sb
        raw = F.pad(is_space, (1, 0), value=False)
    elif mode in ENTROPY_MODES:
        if entropy_values is None:
            raise ValueError(f"patching_mode={mode} には entropy_values が必要")
        ent = entropy_values.float()
        if mode == "entropy":
            raw = F.pad(ent[:, :-1] > threshold, (1, 0), value=False)
        else:
            multi_lead = (cur >= 0xC2) & (cur <= 0xF4)
            h = F.pad(ent[:, :-1], (1, 0), value=0.0) + torch.where(multi_lead, ent, 0.0)
            if rest_values is not None:
                h = h + torch.where((cur >= 0xE0) & (cur <= 0xF4), rest_values.float(), 0.0)
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(char_start)
            last_cs = torch.where(char_start, pos, -1).cummax(dim=1).values
            prev_cs = F.pad(last_cs[:, :-1], (1, 0), value=-1)  # 1 つ前の文字先頭
            h_prev = h.gather(1, prev_cs.clamp(min=0))
            # 系列先頭の文字は直前の予測エントロピーが無く H が過小なので、比較の基準にしない
            raw = char_start & (prev_cs > 0) & (h - h_prev > threshold)
    else:
        raise ValueError(f"unknown dynamic patching mode: {mode}")
    if eos_token_id is None:
        force = torch.zeros_like(raw)
    else:
        force = F.pad(input_ids[:, :-1] == eos_token_id, (1, 0), value=False)
    return raw, force, char_start


def walk_patch_starts(
    raw: torch.Tensor, force: torch.Tensor, char_start: torch.Tensor,
    min_len: int, max_len: int | None, soft_len: int = 0,
) -> torch.Tensor:
    """候補から min_len / max_len / 文書先頭 (min_len 無視) の規則で patch 先頭 (B, T) bool を決める."""
    if min_len <= 0:
        raise ValueError("min_patch_len must be positive")
    if max_len is None:
        max_len = max(raw.shape[1], min_len)
    if max_len < min_len:
        raise ValueError("max_patch_len must be >= min_patch_len")
    if soft_len and soft_len < min_len:
        raise ValueError(f"entropy_char は max_patch_len - {_UTF8_MAX_CONT} >= min_patch_len が必要")
    return torch.ops.arbor.patch_starts(
        raw.contiguous(), force.contiguous(), char_start.contiguous(),
        int(min_len), int(max_len), int(soft_len),
    )


def compute_patch_starts(
    input_ids: torch.Tensor,
    mode: str,
    min_len: int,
    max_len: int | None,
    entropy_model: ByteLM | None = None,
    threshold: float = 1.5,
    entropy_values: torch.Tensor | None = None,
    *,
    rest_values: torch.Tensor | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """各行を独立に区切った patch 開始位置の bool tensor (B, T) (patch_candidates + walk_patch_starts).

    学習の系列は src/data/patch_packer.py が文書の流れを続けて区切って作る。ここは推論・評価で
    任意の byte 列をそのまま区切る用 (行頭は ByteLM の文脈が無い)。
    """
    if mode in ENTROPY_MODES and entropy_values is None:
        if entropy_model is None:
            raise ValueError(f"patching_mode={mode} には entropy_model が必要")
        entropy_values, rest_values = entropy_model.boundary_entropy(input_ids)
    raw, force, char_start = patch_candidates(
        input_ids, mode, entropy_values, rest_values, threshold, eos_token_id,
    )
    return walk_patch_starts(raw, force, char_start, min_len, max_len, soft_patch_len(mode, max_len))


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
        if cfg.patching_mode not in ("static", "utf8", "space", *ENTROPY_MODES):
            raise ValueError(f"unknown patching_mode: {cfg.patching_mode}")
        if cfg.global_attn_impl not in ("sdpa", "flex"):
            raise ValueError(
                f"unknown global_attn_impl: {cfg.global_attn_impl!r} "
                "(choices: sdpa | flex; 暗黙フォールバックは禁止)"
            )
        if cfg.cross_attn_k < 1:
            raise ValueError(f"cross_attn_k must be >= 1, got {cfg.cross_attn_k}")
        if cfg.local_attn_window is not None and cfg.local_attn_window <= 0:
            raise ValueError(f"local_attn_window must be positive or None, got {cfg.local_attn_window}")
        if cfg.num_local_encoder_layers < 1 or cfg.num_local_decoder_layers < 1:
            raise ValueError("num_local_encoder_layers / num_local_decoder_layers は 1 以上")
        if cfg.seq_patches < 1:
            raise ValueError(f"seq_patches must be >= 1, got {cfg.seq_patches}")
        from src.model.bitlinear import check_activation_precision

        check_activation_precision(cfg.activation_precision)
        self.cfg = cfg
        dl, dg, kq = cfg.local_hidden_size, cfg.hidden_size, cfg.cross_attn_k

        self.min_patch_len, self.max_patch_len = patch_len_bounds(cfg)
        if self.min_patch_len < 1 or (self.max_patch_len is not None and self.max_patch_len < self.min_patch_len):
            raise ValueError("1 <= min_patch_len <= max_patch_len が必要")
        self.soft_patch_len = soft_patch_len(cfg.patching_mode, self.max_patch_len)
        if self.soft_patch_len and self.soft_patch_len < self.min_patch_len:
            raise ValueError(f"entropy_char は max_patch_len - {_UTF8_MAX_CONT} >= min_patch_len が必要")

        self.byte_emb = nn.Embedding(cfg.vocab_size, dl)
        nn.init.trunc_normal_(self.byte_emb.weight, std=0.02, a=-0.06, b=0.06)
        self.hash_emb = (
            HashNgramEmbedding(cfg.hash_ngram_sizes, cfg.hash_ngram_vocab, dl)
            if cfg.hash_ngram_sizes else None
        )

        theta_global = cfg.rope_theta_global if cfg.rope_theta_global is not None else cfg.rope_theta
        theta_local = cfg.rope_theta_local if cfg.rope_theta_local is not None else cfg.rope_theta
        local_rope = RotaryEmbedding(dl // cfg.local_num_heads, cfg.max_bytes, theta_local)
        # 推論で patch_starts を省くと patch 数は入力次第 (最大 max_bytes)
        global_rope = RotaryEmbedding(dg // cfg.num_heads, max(cfg.max_bytes, cfg.seq_patches), theta_global)
        local_bitnet = cfg.bitnet if cfg.local_bitnet is None else bool(cfg.local_bitnet)

        def local_block() -> Block:
            return Block(dl, cfg.local_num_heads, cfg.local_num_kv_heads,
                         cfg.local_intermediate_size, local_rope, local_bitnet, cfg.norm_eps,
                         causal=True, activation_precision=cfg.activation_precision)

        n_local = cfg.num_local_encoder_layers + cfg.num_local_decoder_layers

        def cross_attn() -> CrossAttention:
            return CrossAttention(dl, cfg.cross_attn_heads or cfg.local_num_heads, cfg.norm_eps, local_bitnet,
                                  cfg.activation_precision, out_scale=(2 * n_local) ** -0.5)

        # Local Encoder: 文書内 causal の byte 層。各層の後に patch query が自 patch の byte へ cross-attention
        self.encoder_layers = nn.ModuleList(local_block() for _ in range(cfg.num_local_encoder_layers))
        self.encoder_cross_attn = nn.ModuleList(cross_attn() for _ in range(cfg.num_local_encoder_layers))
        # patch 内 max-pool (dl) → k 個の query (k * dl)
        self.patch_query_proj = nn.Linear(dl, kq * dl, bias=False)
        std = dl ** -0.5
        nn.init.trunc_normal_(self.patch_query_proj.weight, std=std, a=-3 * std, b=3 * std)
        # k * dl が global 次元と違うときだけ射影する (BLT 1B は 2 * 1024 = 2048 で不要)
        if kq * dl != dg:
            self.patch_proj: nn.Linear | None = nn.Linear(kq * dl, dg, bias=False)
            nn.init.trunc_normal_(self.patch_proj.weight, std=0.02, a=-0.06, b=0.06)
        else:
            self.patch_proj = None
        # 右シフトの先頭 patch。ゼロ初期化禁止: 厳密ゼロ行は全層で 0 のまま伝播し、
        # RMSNorm backward の 1/sqrt(eps) 増幅が全層で複利になって勾配が overflow する
        self.global_bos = nn.Parameter(torch.empty(dg))
        nn.init.trunc_normal_(self.global_bos, std=0.02, a=-0.06, b=0.06)

        if cfg.global_attention_every is not None:
            if cfg.global_layer_pattern is not None:
                raise ValueError("global_attention_every と global_layer_pattern は同時に指定できない")
            if cfg.global_attention_every < 1:
                raise ValueError(f"global_attention_every must be >= 1, got {cfg.global_attention_every}")
            pattern = "S" * (cfg.global_attention_every - 1) + "A"
        else:
            pattern = (cfg.global_layer_pattern or "A").upper()
        if any(ch not in "AS" for ch in pattern):
            raise ValueError(f"global_layer_pattern は A/S の文字列 (got {cfg.global_layer_pattern!r})")
        self.global_layers = nn.ModuleList(
            Block(dg, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size,
                  global_rope, cfg.bitnet, cfg.norm_eps, causal=True,
                  activation_precision=cfg.activation_precision,
                  mixer="ssd" if pattern[i % len(pattern)] == "S" else "attention",
                  ssd_conv_width=cfg.ssd_conv_width, ssd_chunk=cfg.ssd_chunk,
                  ssd_output_gate=cfg.ssd_output_gate, ssd_backend=cfg.ssd_backend)
            for i in range(cfg.num_hidden_layers)
        )
        self.has_ssd = any(block.mixer is not None for block in self.global_layers)
        self.global_norm = RMSNorm(dg, cfg.norm_eps)
        # global 出力 (dg) → decoder の cross-attention の k 個の key/value (k * dl)
        self.global_to_local = nn.Linear(dg, kq * dl, bias=False)
        nn.init.trunc_normal_(self.global_to_local.weight, std=dg ** -0.5, a=-3 * dg ** -0.5, b=3 * dg ** -0.5)

        # Local Decoder: 入力は encoder の byte 表現。各層の前に自 patch の global 出力へ cross-attention
        self.decoder_layers = nn.ModuleList(local_block() for _ in range(cfg.num_local_decoder_layers))
        n_dec_xattn = cfg.num_local_decoder_layers if cfg.decoder_cross_attn_all_layers else 1
        self.decoder_cross_attn = nn.ModuleList(cross_attn() for _ in range(n_dec_xattn))
        self.head_norm = RMSNorm(dl, cfg.norm_eps)
        self.head = nn.Linear(dl, cfg.vocab_size, bias=False)  # FP
        nn.init.trunc_normal_(self.head.weight, std=0.02, a=-0.06, b=0.06)

        _scale_residual_projections([self.encoder_layers, self.decoder_layers])
        _scale_residual_projections([self.global_layers])

        # entropy 用の凍結 ByteLM (checkpoint に同梱される)
        if cfg.patching_mode in ENTROPY_MODES:
            if not cfg.entropy_model:
                raise ValueError(
                    f"patching_mode={cfg.patching_mode} には model.entropy_model (ByteLM 構成) が必要"
                )
            em_cfg = dict(cfg.entropy_model)
            em_cfg.setdefault("max_bytes", cfg.max_bytes)
            if cfg.entropy_char_rest:
                if cfg.patching_mode != "entropy_char":
                    raise ValueError("entropy_char_rest は patching_mode=entropy_char 専用")
                em_cfg["char_rest_head"] = True
            self.entropy_model = ByteLM(em_cfg)
            self.entropy_model.requires_grad_(False)
        else:
            self.entropy_model = None

    # ------------------------------------------------------------- pieces
    def _byte_doc_ids(self, input_ids: torch.Tensor, is_byte: torch.Tensor | None = None) -> torch.Tensor:
        """各バイトが属する文書番号 (B, T) を返す.

        packing='document' は文書を EOS 区切りで連結する。EOS の「次」の
        バイトから文書番号が 1 増える (EOS 自身は直前の文書に属す)。判定は
        過去バイトのみに依存するので causal (未来を見ない)。PAD (is_byte が偽) は
        1 つずつ別の文書にして、local attention で実 byte と互いに見ないようにする。
        """
        new_doc = F.pad(input_ids == self.cfg.eos_token_id, (1, 0), value=False)[:, :-1]
        if is_byte is not None:
            new_doc = new_doc | ~is_byte
        return new_doc.to(torch.long).cumsum(dim=1)

    def embed(self, input_ids: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
        """byte 埋め込み + hash n-gram 埋め込み。BLT 論文どおり n-gram の種類数 + 1 で割る."""
        x = self.byte_emb(input_ids)
        if self.hash_emb is None:
            return x
        return (x + self.hash_emb(input_ids, doc)) / (len(self.hash_emb.sizes) + 1)

    def _byte_attn_mask(self, input_ids: torch.Tensor, doc: torch.Tensor):
        """local の byte 層の mask: causal ∧ 同一文書 (∧ 直近 local_attn_window byte).

        compile 下の CUDA では flex_attention の BlockMask (窓外・文書外の block を丸ごと
        飛ばす fused kernel)、それ以外 (eager / CPU / 評価) は同じ規則の密 bool mask。
        文書境界も窓も無い入力では None を返し、Attention 側の素の causal (flash) を使う。
        """
        b, t = input_ids.shape
        w = self.cfg.local_attn_window
        if input_ids.is_cuda and torch.compiler.is_compiling():
            from torch.nn.attention.flex_attention import create_block_mask

            def mask_mod(b_idx, h_idx, q_idx, kv_idx):
                keep = (kv_idx <= q_idx) & (doc[b_idx, q_idx] == doc[b_idx, kv_idx])
                if w is not None:
                    keep = keep & (q_idx - kv_idx < w)
                return keep

            return create_block_mask(mask_mod, B=b, H=None, Q_LEN=t, KV_LEN=t,
                                     device=input_ids.device)
        if w is None and not bool((doc[:, -1] != doc[:, 0]).any()):
            return None
        q = torch.arange(t, device=input_ids.device).view(t, 1)
        kv = torch.arange(t, device=input_ids.device).view(1, t)
        allow = (kv <= q).unsqueeze(0) & (doc.unsqueeze(2) == doc.unsqueeze(1))
        if w is not None:
            allow = allow & (q - kv < w).unsqueeze(0)
        return allow.unsqueeze(1)                                # (B, 1, T, T)

    def _global_doc_mask(self, patch_doc: torch.Tensor) -> torch.Tensor:
        """global (patch 階層) 用の block-diagonal + causal マスク (B,1,K,K).

        global は 1 patch 右シフト後の causal。出力位置 j (= patch j の文脈) が
        key 位置 j' に attend できるのは:
          - j'=0 (先頭の学習可能 BOS。文書非依存で常に許可。全マスク行を防ぐ)
          - それ以外は key s[j'] = patch j'-1 が patch j と同一文書のときだけ。
        """
        b, k = patch_doc.shape
        device = patch_doc.device
        causal = torch.tril(torch.ones(k, k, dtype=torch.bool, device=device))
        key_doc = F.pad(patch_doc[:, :-1], (1, 0), value=-1)
        same_doc = patch_doc.unsqueeze(2) == key_doc.unsqueeze(1)  # (B,K,K)
        bos_col = torch.zeros(k, dtype=torch.bool, device=device)
        bos_col[0] = True
        allow = causal.unsqueeze(0) & (same_doc | bos_col.view(1, 1, k))
        return allow.unsqueeze(1)  # (B,1,K,K)

    def _global_mask(self, patch_doc: torch.Tensor):
        """cfg.global_attn_impl に応じて global 用マスクを返す (sdpa=密, flex=BlockMask)."""
        impl = self.cfg.global_attn_impl
        if impl == "flex":
            return self._global_flex_block_mask(patch_doc)
        if impl == "sdpa":
            return self._global_doc_mask(patch_doc)
        raise RuntimeError(f"unsupported global_attn_impl at runtime: {impl!r}")

    def _global_flex_block_mask(self, patch_doc: torch.Tensor):
        """_global_doc_mask と同じ許可規則を flex_attention の BlockMask で表す."""
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

    def compute_patch_starts(self, input_ids: torch.Tensor) -> torch.Tensor:
        """入力の各行をその場で区切る (行頭は ByteLM の文脈が無い。推論・評価用)."""
        cfg = self.cfg
        return compute_patch_starts(
            input_ids, cfg.patching_mode, self.min_patch_len, self.max_patch_len,
            self.entropy_model, cfg.entropy_threshold, eos_token_id=cfg.eos_token_id,
        )

    def _encoder_cross_attn(
        self, layer_idx: int, h: torch.Tensor, patch_id: torch.Tensor, is_byte: torch.Tensor,
        k: int, queries: torch.Tensor | None,
    ) -> torch.Tensor:
        """queries が None (encoder の最初の層) なら patch 内 max-pool の射影で初期化する (BLT)."""
        b, t, dl = h.shape
        if queries is None:
            src = h.masked_fill(~is_byte.unsqueeze(-1), float("-inf"))
            pooled = h.new_full((b, k, dl), float("-inf")).scatter_reduce(
                1, patch_id.unsqueeze(-1).expand(-1, -1, dl), src, reduce="amax", include_self=True,
            )
            pooled = pooled.masked_fill(torch.isneginf(pooled), 0.0)
            queries = self.patch_query_proj(pooled).unflatten(-1, (self.cfg.cross_attn_k, dl))
        return queries + self.encoder_cross_attn[layer_idx].forward_segments(queries, h, patch_id, is_byte)

    def _run_global(
        self, patches: torch.Tensor, attn_mask: "torch.Tensor | None", patch_doc: torch.Tensor,
    ) -> torch.Tensor:
        """1 patch 右シフト + causal global。decoder の cross-attention 用に (B, K, k, dl) で返す.

        新文書先頭の右シフト入力は BOS に置き換える (前文書の patch が residual に残らないように)。
        """
        b = patches.size(0)
        bos = self.global_bos.to(patches.dtype).expand(b, 1, -1)
        g = torch.cat((bos, patches[:, :-1]), dim=1)
        new_doc = F.pad(patch_doc[:, 1:] != patch_doc[:, :-1], (1, 0), value=True)
        g = torch.where(new_doc.unsqueeze(-1), bos, g)
        for layer in self.global_layers:
            g = self._maybe_ckpt(layer, g, attn_mask, seg=patch_doc)
        out = self.global_to_local(self.global_norm(g))
        return out.unflatten(-1, (self.cfg.cross_attn_k, self.cfg.local_hidden_size))

    def _maybe_ckpt(
        self, layer: nn.Module, x: torch.Tensor,
        attn_mask: "torch.Tensor | None" = None,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(layer, x, attn_mask, None, 0, seg, use_reentrant=False)
        return layer(x, attn_mask, seg=seg)

    # ------------------------------------------------------------- forward
    def forward(
        self, input_ids: torch.Tensor, patch_starts: torch.Tensor | None = None,
    ) -> ArborOutput:
        """input_ids (B, T) の各位置の next-byte logits.

        patch_starts (B, T) bool は学習用に src/data/patch_packer.py が作る区切り: 各行の patch は
        cfg.seq_patches 個以下で、global は常に seq_patches 位置 (形状固定)。枠の余りは PAD
        (cfg.pad_token_id) で、どの patch にも入らない。省くと各行をその場で区切り、global の
        長さは batch 内の最大 patch 数になる (host 同期あり。推論・評価用)。
        """
        cfg = self.cfg
        b, t = input_ids.shape
        if t > cfg.max_bytes:
            raise ValueError(f"入力長 {t} が max_bytes={cfg.max_bytes} を超えている")
        if patch_starts is None:
            patch_starts = self.compute_patch_starts(input_ids)
            k = int(patch_starts.sum(1).max())
        else:
            k = cfg.seq_patches
        is_byte = input_ids != cfg.pad_token_id
        # PAD を 1 つの patch に集めると scatter_add が同じ番地に集中して遅いので散らす
        ar = torch.arange(t, device=input_ids.device)
        patch_id = torch.where(is_byte, (patch_starts.long().cumsum(1) - 1).clamp_min(0), ar % k)
        doc = self._byte_doc_ids(input_ids, is_byte)
        mask = self._byte_attn_mask(input_ids, doc)

        h = self.embed(input_ids, doc)
        queries = None
        for i, layer in enumerate(self.encoder_layers):
            h = self._maybe_ckpt(layer, h, mask)
            queries = self._encoder_cross_attn(i, h, patch_id, is_byte, k, queries)
        patches = queries.reshape(b, k, -1)                    # (B, K, k*dl)
        if self.patch_proj is not None:
            patches = self.patch_proj(patches)

        # pad patch は番兵の doc 番号のままで、実文書と混ざらない
        sentinel = 1 << 30
        patch_doc = torch.full((b, k), sentinel, dtype=torch.long, device=input_ids.device)
        patch_doc.scatter_reduce_(1, patch_id, torch.where(is_byte, doc, sentinel),
                                  reduce="amin", include_self=True)
        g = self._run_global(patches, self._global_mask(patch_doc), patch_doc)  # (B, K, k, dl)

        d = h
        for i, layer in enumerate(self.decoder_layers):
            if i < len(self.decoder_cross_attn):
                d = d + self.decoder_cross_attn[i].forward_per_byte(d, g, patch_id)
            d = self._maybe_ckpt(layer, d, mask)
        return ArborOutput(logits=self.head(self.head_norm(d)))

    # ------------------------------------------------------------ utility
    def num_parameters(self) -> dict[str, int]:
        def count(*mods: nn.Module | None) -> int:
            return sum(par.numel() for m in mods if m is not None for par in m.parameters())

        return {
            "total": count(self),
            "global": count(self.global_layers),
            "local_encoder": count(self.encoder_layers, self.encoder_cross_attn, self.patch_query_proj),
            "local_decoder": count(self.decoder_layers, self.decoder_cross_attn, self.global_to_local),
            "hash_ngram": count(self.hash_emb),
            "embedding_head": count(self.byte_emb, self.head, self.patch_proj),
            "entropy_model": count(self.entropy_model),
        }


class ArborByteGenerator:
    """KV cache 付きの逐次バイト生成器 (batch=1).

    フルフォワード (1 バイトごとに全系列再計算) と同じ logits を増分計算で返す:
      - encoder / decoder の byte 層: 全 byte の KV cache を持ち、新しい byte の分だけ計算
      - patch が確定したら、その byte の encoder 表現から patch query を作って global に 1 つ追記
      - byte の decoder cross-attention は、その byte が属する patch の global 出力を使う
      - entropy モードは境界判定用 ByteLM にも KV cache を持つ

    使い方:
        gen = ArborByteGenerator(model)
        logits = gen.prefill(prompt_ids)   # 最終位置の next-byte logits (vocab,)
        logits = gen.push(next_id)         # 1 バイト進める

    context が max_bytes に達したら後半半分を残して内部で自動的に作り直す。
    プロンプト中の EOS (文書の切り替え) はフルフォワードと違い分離しない。
    """

    def __init__(self, model: ArborModel):
        if not isinstance(model, ArborModel):
            raise TypeError("ArborByteGenerator は ArborModel 専用")
        if getattr(model, "has_ssd", False):
            raise NotImplementedError(
                "ArborByteGenerator は global_layer_pattern の SSD 層 (逐次 state 生成) 未対応 (実験用 A/B のみ)"
            )
        self.m = model.eval()
        self.cfg = model.cfg
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self._hash_span = max(model.hash_emb.sizes) if model.hash_emb is not None else 1
        self.reset()

    def reset(self) -> None:
        self.byte_ids: list[int] = []
        self.cur_patch: list[int] = []
        self.n_global = 0
        self.g_caches = [_LayerKVCache() for _ in self.m.global_layers]
        self.g_cur: torch.Tensor | None = None  # 現 patch の global 出力 (1, 1, k, dl)
        if self.cfg.patching_mode in ENTROPY_MODES:
            self.lm_caches = [_LayerKVCache() for _ in self.m.entropy_model.layers]
            self.prev_entropy = 0.0
            self.prev_rest = 0.0  # char_rest head の予測 (直前に読んだ byte の位置)
            self.prev_char_h: float | None = None  # 1 つ前の文字の H (entropy_char)
        self.enc_caches = [_LayerKVCache() for _ in self.m.encoder_layers]
        self.dec_caches = [_LayerKVCache() for _ in self.m.decoder_layers]
        self.cur_enc: list[list[torch.Tensor]] = [[] for _ in self.m.encoder_layers]  # 現 patch の各層出力
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
        return self._append_byte(byte_id)

    def _append_byte(self, byte_id: int) -> torch.Tensor:
        """1 byte 進めて、その位置の next-byte logits (vocab,) を返す."""
        mode = self.cfg.patching_mode
        if mode == "entropy_char":
            # 境界判定にこの byte を読んだ後のエントロピー (多バイト文字の 2 byte 目の予測) も使う
            e_prev = self.prev_entropy
            self._advance_entropy_lm(byte_id, pos=len(self.byte_ids))
            byte = byte_id - BYTE_OFFSET
            rise = False
            if _is_utf8_char_start_byte(byte):
                h = e_prev + (self.prev_entropy if 0xC2 <= byte <= 0xF4 else 0.0)
                if self.cfg.entropy_char_rest and 0xE0 <= byte <= 0xF4:
                    h += self.prev_rest
                rise = self.prev_char_h is not None and h - self.prev_char_h > self.cfg.entropy_threshold
                # 先頭の文字は直前の予測エントロピーが無く H が過小なので比較の基準にしない (patch_candidates)
                self.prev_char_h = h if self.byte_ids else None
            new_patch = self._starts_new_patch(byte_id, rise=rise)
        else:
            new_patch = self._starts_new_patch(byte_id)
        if new_patch:
            self._commit_patch()
        self.cur_patch.append(byte_id)
        self.byte_ids.append(byte_id)
        if mode == "entropy":
            self._advance_entropy_lm(byte_id, pos=len(self.byte_ids) - 1)

        pos = len(self.byte_ids) - 1
        window = self.cfg.local_attn_window
        tail = torch.tensor([self.byte_ids[-self._hash_span:]], dtype=torch.long, device=self.device)
        x = self.m.embed(tail, torch.zeros_like(tail))[:, -1:].to(self.dtype)
        for i, (layer, cache) in enumerate(zip(self.m.encoder_layers, self.enc_caches)):
            if window is not None:
                cache.trim(max(0, window - 1))
            x = layer(x, kv_cache=cache, pos_offset=pos)
            self.cur_enc[i].append(x)
        d = x
        patch_id = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for i, (layer, cache) in enumerate(zip(self.m.decoder_layers, self.dec_caches)):
            if i < len(self.m.decoder_cross_attn):
                d = d + self.m.decoder_cross_attn[i].forward_per_byte(d, self.g_cur, patch_id)
            if window is not None:
                cache.trim(max(0, window - 1))
            d = layer(d, kv_cache=cache, pos_offset=pos)
        return self.m.head(self.m.head_norm(d))[0, -1]

    # ----------------------------------------------------------- internal
    def _starts_new_patch(
        self, next_byte_id: int | None = None, rise: bool = False,
    ) -> bool:
        """次に push されるバイトが新しい patch を始めるか (_patch_starts_reference と同じ規則)."""
        run = len(self.cur_patch)
        if run == 0:
            return False
        cfg, m = self.cfg, self.m
        if m.max_patch_len is not None and run >= m.max_patch_len:
            return True
        if self.byte_ids[-1] == cfg.eos_token_id:
            return True  # 文書先頭は min_len に関係なく区切る
        if run < m.min_patch_len or cfg.patching_mode == "static":
            return False
        if cfg.patching_mode == "utf8":
            return next_byte_id is not None and _is_utf8_char_start_byte(next_byte_id - BYTE_OFFSET)
        if cfg.patching_mode == "space":
            return (self.byte_ids[-1] - BYTE_OFFSET) in _SPACE_BYTES
        if cfg.patching_mode == "entropy_char":
            return rise or (
                m.soft_patch_len > 0 and run >= m.soft_patch_len
                and _is_utf8_char_start_byte(next_byte_id - BYTE_OFFSET)
            )
        return self.prev_entropy > cfg.entropy_threshold

    def _push_global(self, g_in: torch.Tensor) -> None:
        g = g_in.to(self.dtype)
        for layer, cache in zip(self.m.global_layers, self.g_caches):
            g = layer(g, kv_cache=cache, pos_offset=self.n_global)
        self.n_global += 1
        out = self.m.global_to_local(self.m.global_norm(g))
        self.g_cur = out.unflatten(-1, (self.cfg.cross_attn_k, self.cfg.local_hidden_size))

    def _commit_patch(self) -> None:
        queries = None
        for i, states in enumerate(self.cur_enc):
            h = torch.cat(states, dim=1)                                   # (1, 現 patch 長, dl)
            if queries is None:
                queries = self.m.patch_query_proj(h.amax(dim=1)).view(1, self.cfg.cross_attn_k, -1)
            queries = queries + self.m.encoder_cross_attn[i](queries, h)
        patch = queries.reshape(1, 1, -1)
        if self.m.patch_proj is not None:
            patch = self.m.patch_proj(patch)
        self._push_global(patch)
        self.cur_patch = []
        self.cur_enc = [[] for _ in self.m.encoder_layers]

    def _advance_entropy_lm(self, byte_id: int, pos: int) -> None:
        lm = self.m.entropy_model
        x = lm.embed(torch.tensor([[byte_id]], dtype=torch.long, device=self.device))
        x = x.to(self.dtype)
        for layer, cache in zip(lm.layers, self.lm_caches):
            if lm.attention_window is not None:
                cache.trim(max(0, lm.attention_window - 1))
            x = layer(x, kv_cache=cache, pos_offset=pos)
        xn = lm.norm(x)
        logp = F.log_softmax(lm.head(xn).float()[0, -1], dim=-1)
        self.prev_entropy = float(-(logp.exp() * logp).sum())
        rest = lm.char_rest_from_hidden(xn)
        if rest is not None:
            self.prev_rest = float(rest[0, -1])

    def _rebuild(self, keep: int) -> None:
        tail = self.byte_ids[-keep:]
        self.reset()
        for byte_id in tail:
            self._append_byte(byte_id)


def build_arbor(model_cfg: dict[str, Any]) -> ArborModel:
    """config dict (configs/*.yaml の model 節) から構築する.

    patching_mode=entropy で entropy_model_ckpt が指定され、かつ存在する場合は
    そこから凍結 ByteLM の重みを読む (arbor 自体の checkpoint から resume する
    場合は、その後の strict ロードで上書きされるので二重指定でも安全)。
    """
    cfg = ArborConfig.from_dict(model_cfg)
    model = ArborModel(cfg)

    if cfg.patching_mode in ENTROPY_MODES and cfg.entropy_model_ckpt:
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
            if cfg.entropy_char_rest:
                head_file = ckpt / CHAR_REST_HEAD_FILE
                if not head_file.exists():
                    raise FileNotFoundError(
                        f"entropy_char_rest=true だが {head_file} が無い (この ByteLM step 用の head を "
                        "scripts/train_char_rest_head.py で学習すること)"
                    )
                state.update({f"char_rest.{k}": v for k, v in safe_load(str(head_file), device="cpu").items()})
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
        f"local_dec={counts['local_decoder'] / 1e6:.1f}M hash_ngram={counts['hash_ngram'] / 1e6:.1f}M "
        f"emb/head={counts['embedding_head'] / 1e6:.1f}M "
        f"xattn_k={cfg.cross_attn_k} local_window={cfg.local_attn_window or 'doc'} "
        f"entropy_lm={counts['entropy_model'] / 1e6:.1f}M) "
        f"patching={cfg.patching_mode} bitnet={'ON' if cfg.bitnet else 'OFF'} "
        f"bitlinear_layers={n_bit} "
        + (
            f"global_layers={''.join('S' if b.mixer is not None else 'A' for b in model.global_layers)} "
            if getattr(model, "has_ssd", False) else ""
        )
        + 
        "weights=W1.58(absmean ternary) "
        f"activations={_activation_desc(cfg.activation_precision)} "
        "subln=ON backward=STE(detach)"
    )
    return model
