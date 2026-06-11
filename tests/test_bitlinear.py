"""BitLinear (BitNet b1.58 公式レシピ) のテスト (CPU)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model.bitlinear import BitLinear, activation_quant, weight_quant


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


def test_bias_is_rejected():
    with pytest.raises(ValueError, match="bias"):
        BitLinear(8, 8, bias=True)
