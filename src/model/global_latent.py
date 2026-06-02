"""Global Latent Transformer (BitNet b1.58 化).

BLT の Global 部のみ BitLinear に置き換える。Local Encoder/Decoder は
別途 FP のまま組み立てる (src/model/local_encoder.py, local_decoder.py)。

このファイルは BLT 公式 (third_party/blt/bytelatent) の Global 実装を
再利用しつつ、nn.Linear を BitLinear に差し替える薄いラッパを置く想定。
ここでは差し替えユーティリティだけ提供する。
"""
from __future__ import annotations

import torch.nn as nn

from src.model.bitlinear import BitLinear


def swap_linear_to_bitlinear(module: nn.Module, skip_names: tuple[str, ...] = ()) -> int:
    """module 配下の nn.Linear を再帰的に BitLinear へ置換.

    Returns: 置換した層数。
    skip_names: 名前に含まれていたら置換しない (例: ("embed", "lm_head"))。
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not any(s in name for s in skip_names):
            new = BitLinear(child.in_features, child.out_features, bias=False)
            # 既存重みを BF16 シャドウへ移送 (bias は捨てる)
            with_no_grad_copy = child.weight.detach().to(dtype=new.weight.dtype)
            new.weight.data.copy_(with_no_grad_copy)
            setattr(module, name, new)
            count += 1
        else:
            count += swap_linear_to_bitlinear(child, skip_names)
    return count
