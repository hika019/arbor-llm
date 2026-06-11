"""BitLinear — BitNet b1.58 (W1.58 / A8) 公式レシピ準拠.

"The Era of 1-bit LLMs: Training Tips, Code and FAQ" (Microsoft) の実装に従う:

- 重み: per-tensor absmean scale で ternary {-1, 0, +1} に丸める (1.58 bit)。
- 活性: per-token absmax scale で int8 [-128, 127] に丸める。
- STE は detach トリックで実装する:
      x_q = x + (Q(x) - x).detach()
  これにより forward は量子化値を使い、backward は量子化を恒等写像とみなした
  勾配が「量子化後の重み・活性」で計算される (生の重み/活性で勾配を取るのは
  公式レシピと異なる)。
- bias は持たない (BitNet b1.58 2B4T 仕様)。
- 量子化前の正規化 (SubLN) は層構造側 (arbor.py) が担う。BitNet 2B4T と同じく、
  すべての BitLinear の入力は直前に RMSNorm を通る:
      q/k/v <- input_layernorm,  o <- attn_sub_norm,
      gate/up <- post_attention_layernorm,  down <- ffn_sub_norm

カスタム autograd や Triton カーネルは使わない。純 PyTorch なので torch.compile が
全体を融合でき、CPU でもそのまま動く。学習は BF16 シャドウ重みが master。
推論専用の packed ternary カーネルは将来の最適化として分離する。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """per-token absmax int8 量子化 (値は dequantize して返す)."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return (x * scale).round().clamp(-128, 127) / scale


def weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """per-tensor absmean ternary 量子化 (値は dequantize して返す)."""
    scale = w.abs().mean().clamp_min(eps)
    return (w / scale).round().clamp(-1, 1) * scale


class BitLinear(nn.Module):
    """nn.Linear (bias 無し) の drop-in 置換. W1.58 / A8 + STE."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if bias:
            raise ValueError("BitLinear is bias-free (BitNet b1.58 spec).")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # 出力側 (wo / down) は builder 側で 1/sqrt(2*n_layers) に再スケールする
        nn.init.trunc_normal_(self.weight, std=0.02, a=-0.06, b=0.06)
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # detach トリック STE: forward は量子化値、backward は恒等
        x_q = x + (activation_quant(x) - x).detach()
        w_q = w + (weight_quant(w) - w).detach()
        return F.linear(x_q, w_q)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"
