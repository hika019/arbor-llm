"""BitLinear (BitNet b1.58 公式レシピ) のテスト (CPU)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model.bitlinear import (
    BitLinear,
    BitLinearGroup,
    activation_quant,
    configure_bitlinear_training_cache,
    weight_quant,
)


def test_weight_quant_is_ternary():
    w = torch.randn(64, 32)
    w_q = weight_quant(w)
    scale = w.abs().mean()
    levels = torch.unique((w_q / scale).round())
    assert set(levels.tolist()) <= {-1.0, 0.0, 1.0}


def test_activation_quant_per_token_grid():
    x = torch.randn(8, 32)
    x_q = activation_quant(x)
    # per-token absmax: 各行が int8 グリッドに乗る
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True)
    grid = (x_q * scale).round()
    assert torch.allclose(x_q * scale, grid, atol=1e-4)
    assert grid.abs().max() <= 128
    # 量子化誤差は 1 ステップ未満
    assert (x_q - x).abs().max() <= (1.0 / scale).max()


def test_forward_matches_quantized_linear():
    torch.manual_seed(0)
    lin = BitLinear(32, 16)
    x = torch.randn(4, 32)
    y = lin(x)
    expected = F.linear(activation_quant(x), weight_quant(lin.weight))
    assert torch.allclose(y, expected, atol=1e-5)


def test_ste_gradients_flow_through_quantization():
    torch.manual_seed(0)
    lin = BitLinear(32, 16)
    x = torch.randn(4, 32, requires_grad=True)
    y = lin(x)
    y.sum().backward()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # STE: 勾配は「量子化後の値」で計算される (公式レシピ準拠)
    w_q = weight_quant(lin.weight.detach())
    expected_grad_x = torch.ones(4, 16) @ w_q
    assert torch.allclose(x.grad, expected_grad_x, atol=1e-5)
    x_q = activation_quant(x.detach())
    expected_grad_w = torch.ones(4, 16).t() @ x_q
    assert torch.allclose(lin.weight.grad, expected_grad_w, atol=1e-5)


def test_training_weight_cache_matches_uncached_forward_and_grad():
    torch.manual_seed(0)
    uncached = BitLinear(32, 16)
    cached = BitLinear(32, 16)
    cached.load_state_dict(uncached.state_dict())
    cached.enable_training_weight_cache(True)

    x1 = torch.randn(4, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = uncached(x1)
    y2 = cached(x2)
    assert torch.allclose(y2, y1, atol=1e-5)

    y1.square().mean().backward()
    y2.square().mean().backward()
    assert torch.allclose(x2.grad, x1.grad, atol=1e-5)
    assert torch.allclose(cached.weight.grad, uncached.weight.grad, atol=1e-5)


def test_bitlinear_group_matches_individual_projections():
    torch.manual_seed(0)
    a = BitLinear(32, 16)
    b = BitLinear(32, 24)
    group = BitLinearGroup((a, b), kind="test")
    group.enable_training_weight_cache(True)

    x_group = torch.randn(3, 32, requires_grad=True)
    x_ind = x_group.detach().clone().requires_grad_(True)
    out_group = group(x_group)
    out_ind = torch.cat((a(x_ind), b(x_ind)), dim=-1)
    assert torch.allclose(out_group, out_ind, atol=1e-5)

    out_group.square().sum().backward()
    grad_x_group = x_group.grad.detach().clone()
    grad_a_group = a.weight.grad.detach().clone()
    grad_b_group = b.weight.grad.detach().clone()

    a.weight.grad = None
    b.weight.grad = None
    out_ind.square().sum().backward()
    assert torch.allclose(grad_x_group, x_ind.grad, atol=1e-5)
    assert torch.allclose(grad_a_group, a.weight.grad, atol=1e-5)
    assert torch.allclose(grad_b_group, b.weight.grad, atol=1e-5)


def test_configure_training_cache_installs_projection_groups():
    from src.model.arbor import ArborConfig, ArborModel

    cfg = ArborConfig.from_dict(
        dict(
            vocab_size=260, patch_size=4, max_bytes=16,
            hidden_size=32, num_heads=4, num_kv_heads=2, intermediate_size=64,
            num_hidden_layers=1,
            local_hidden_size=16, local_num_heads=2, local_num_kv_heads=2,
            local_intermediate_size=32,
            num_local_encoder_layers=1, num_local_decoder_layers=1,
        )
    )
    model = ArborModel(cfg)
    info = configure_bitlinear_training_cache(
        model, enabled="fused", grad_accum_steps=2, max_cache_gib=0.01, min_numel=0
    )
    assert info["enabled"]
    assert info["qkv_groups"] > 0
    assert info["gate_up_groups"] > 0


def test_bias_is_rejected():
    with pytest.raises(ValueError, match="bias"):
        BitLinear(8, 8, bias=True)


def test_pack_unpack_roundtrip():
    from src.model.bitlinear import pack_ternary_weight, unpack_ternary_weight

    torch.manual_seed(0)
    w = torch.randint(-1, 2, (7, 13), dtype=torch.int8)  # 4 で割れない K
    packed = pack_ternary_weight(w)
    assert packed.dtype == torch.uint8 and packed.shape == (7, 4)
    assert torch.equal(unpack_ternary_weight(packed, 13), w)


def test_frozen_inference_matches_eval_forward():
    """推論凍結後の forward が通常 eval forward と (数値誤差内で) 一致すること."""
    torch.manual_seed(0)
    lin = BitLinear(64, 32).eval()
    x = torch.randn(5, 64)
    ref = lin(x)
    lin.freeze_for_inference()
    assert lin.frozen
    out = lin(x)
    assert torch.allclose(out, ref, atol=1e-4), float((out - ref).abs().max())
    # train モードに戻すと学習パスに切り替わる (凍結値は使われない)
    lin.train()
    assert torch.allclose(lin(x), ref, atol=1e-4)
    lin.unfreeze()
    assert not lin.frozen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_packed_kernel_matches_reference_cuda():
    torch.manual_seed(0)
    lin = BitLinear(128, 96).to(device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(9, 128, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).float()
    lin.freeze_for_inference()
    out = lin(x).float()
    assert lin._w_packed is not None or lin._w_dq is not None
    assert torch.allclose(out, ref, atol=3e-2, rtol=1e-2), float((out - ref).abs().max())
