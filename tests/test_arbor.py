"""Arbor v2 モデルの形状・因果性・勾配のテスト (CPU)."""
from __future__ import annotations

import pytest
import torch

from src.model.arbor import ArborConfig, ArborModel, build_arbor

TINY = dict(
    vocab_size=260, patch_size=4, max_bytes=64,
    hidden_size=64, num_heads=4, num_kv_heads=2, intermediate_size=128,
    num_hidden_layers=2,
    local_hidden_size=32, local_num_heads=2, local_num_kv_heads=2,
    local_intermediate_size=64,
    num_local_encoder_layers=1, num_local_decoder_layers=1,
    rope_theta=10000.0,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_arbor(TINY).eval()


def test_forward_shape(model):
    x = torch.randint(4, 260, (2, 32))
    out = model(x)
    assert out.logits.shape == (2, 32, 260)


def test_forward_handles_partial_patch(model):
    # T が patch_size の倍数でなくても内部 pad で処理し、T 分の logits を返す
    x = torch.randint(4, 260, (1, 10))
    out = model(x)
    assert out.logits.shape == (1, 10, 260)


@pytest.mark.parametrize("pos", [4, 7, 13])  # patch 境界 (4) と patch 内部
def test_causality(model, pos):
    """位置 pos のバイトを変えても、位置 < pos の logits は変わらないこと."""
    torch.manual_seed(1)
    a = torch.randint(4, 260, (1, 32))
    b = a.clone()
    b[0, pos] = (a[0, pos] - 4 + 1) % 256 + 4  # 必ず違うバイトに
    with torch.inference_mode():
        la = model(a).logits
        lb = model(b).logits
    assert torch.allclose(la[:, :pos], lb[:, :pos], atol=1e-5), (
        f"position {pos} の変更が過去 (<{pos}) の logits に漏れている"
    )
    # 当該位置以降には影響していること (degenerate でないことの確認)
    assert not torch.allclose(la[:, pos:], lb[:, pos:], atol=1e-5)


def test_partial_patch_padding_does_not_leak(model):
    """端数 patch の内部 pad が、それ以前の位置の logits に影響しないこと."""
    torch.manual_seed(2)
    x = torch.randint(4, 260, (1, 32))
    with torch.inference_mode():
        full = model(x).logits
        trunc = model(x[:, :10]).logits  # 内部で 12 まで pad される
    assert torch.allclose(full[:, :9], trunc[:, :9], atol=1e-5)


def test_gradients_reach_all_parameters():
    torch.manual_seed(0)
    m = ArborModel(ArborConfig.from_dict(TINY))
    x = torch.randint(4, 260, (2, 16))
    out = m(x)
    loss = torch.nn.functional.cross_entropy(
        out.logits.flatten(0, 1), torch.randint(4, 260, (32,))
    )
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"勾配が届いていないパラメータ: {missing[:5]}"
    bad = [n for n, p in m.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, f"非有限の勾配: {bad[:5]}"


def test_global_bos_is_not_zero_initialized():
    """ゼロ初期化の BOS は全層で厳密ゼロ行のまま伝播し、RMSNorm backward の
    1/sqrt(eps) 増幅が複利になって勾配が overflow する (実際に起きた事故)."""
    m = ArborModel(ArborConfig.from_dict(TINY))
    assert m.global_bos.abs().max() > 0


def test_bitnet_flag_swaps_linears():
    from src.model.bitlinear import BitLinear

    bit = ArborModel(ArborConfig.from_dict(TINY))
    fp = ArborModel(ArborConfig.from_dict({**TINY, "bitnet": False}))
    assert sum(1 for m in bit.modules() if isinstance(m, BitLinear)) > 0
    assert sum(1 for m in fp.modules() if isinstance(m, BitLinear)) == 0


def test_param_count_reporting(model):
    counts = model.num_parameters()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["global"] > 0 and counts["local_decoder"] > 0
