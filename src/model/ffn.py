"""ReLU² FFN (BitNet b1.58 2B4T 互換).

BLT 標準は SwiGLU (w2(silu(w1(x)) * w3(x))). BitNet b1.58 2B4T の技術報告では
FFN を **squared ReLU** に変更 (`w2(relu(w1(x)) ** 2)`). 量子化下での sparsity
改善のため. ここでは BLT の `FeedForward` 互換 API を保ち, global_transformer
配下の FFN を後段で差し替えるためのモジュールを提供する.

参考: BitNet b1.58 2B4T Technical Report (arXiv:2504.12285)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU2FeedForward(nn.Module):
    """w2( relu(w1(x)) ** 2 )

    SwiGLU と違って 2 つの Linear (w1, w2) のみ. bias なし.
    BLT FeedForward と同等のインタフェース (dim, hidden_dim, multiple_of,
    ffn_dim_multiplier) を取り, BLT 側の構築ロジックと整合させる.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        mp_size: int = 1,
    ) -> None:
        super().__init__()
        # SwiGLU 互換の hidden 計算式は w1+w3 で 2/3 縮小していたが,
        # ReLU² は w1 のみなので「同じ実効パラメータ量」を狙うなら 2/3 を外し,
        # ffn_dim_multiplier を直接掛ける. BitNet 2B4T は intermediate を
        # 明示指定する流派なので multiplier=1.0 + multiple_of アライメントで
        # 揃えれば良い.
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.w1(x))
        return self.w2(h * h)

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        in_std = init_std or (self.dim ** -0.5) / factor
        out_std = init_std or (self.hidden_dim ** -0.5) / factor
        for w, std in ((self.w1, in_std), (self.w2, out_std)):
            nn.init.trunc_normal_(w.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


def swap_swiglu_to_relu2(module: nn.Module) -> int:
    """module 配下の BLT FeedForward (SwiGLU) を ReLU2FeedForward に再帰置換.

    BLT 側の `FeedForward` 子モジュールを名前で発見し, 同等の (dim, hidden_dim)
    で ReLU2 版を組み立てて差し替える. 戻り値は置換した層数.
    """
    # 遅延 import で third_party 依存をモジュール読込時に走らせない.
    from bytelatent.base_transformer import FeedForward as BLTFeedForward  # type: ignore

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, BLTFeedForward):
            new = ReLU2FeedForward(
                dim=child.dim,
                hidden_dim=child.hidden_dim,
                multiple_of=1,  # 既に揃った hidden_dim をそのまま使う
                ffn_dim_multiplier=None,
            )
            # 既存重みは捨てる. 学習開始前なので問題ない.
            setattr(module, name, new)
            count += 1
        else:
            count += swap_swiglu_to_relu2(child)
    return count
