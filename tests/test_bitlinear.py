from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model.bitlinear import (
    BitLinear,
    _packed_bitlinear_forward,
    clear_bitlinear_weight_cache,
    pack_ternary_weight,
    quantize_activation_int8,
    quantize_weight_ternary,
    set_bitlinear_weight_cache,
    unpack_ternary_weight,
)


def test_pack_ternary_weight_round_trip_with_padding():
    w_q = torch.tensor(
        [
            [-1, 0, 1, 1, 0],
            [1, -1, 0, -1, 1],
        ],
        dtype=torch.int8,
    )

    packed = pack_ternary_weight(w_q)
    unpacked = unpack_ternary_weight(packed, k=w_q.size(1))

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 2)
    assert torch.equal(unpacked, w_q)


def test_bitlinear_forward_backward_cpu_fallback():
    layer = BitLinear(7, 5)
    x = torch.randn(3, 4, 7, dtype=torch.bfloat16, requires_grad=True)

    y = layer(x)
    y.float().sum().backward()

    assert y.shape == (3, 4, 5)
    assert x.grad is not None
    assert layer.weight.grad is not None


def test_bitlinear_weight_cache_helpers_toggle_modules():
    model = torch.nn.Sequential(BitLinear(7, 5), torch.nn.ReLU(), BitLinear(5, 3))

    assert set_bitlinear_weight_cache(model, True) == 2
    assert all(m.cache_packed_weight for m in model.modules() if isinstance(m, BitLinear))

    x = torch.randn(2, 7, dtype=torch.bfloat16, requires_grad=True)
    y = model(x)
    y.float().sum().backward()

    assert clear_bitlinear_weight_cache(model) == 2
    assert set_bitlinear_weight_cache(model, False) == 2
    assert not any(m.cache_packed_weight for m in model.modules() if isinstance(m, BitLinear))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
def test_packed_triton_forward_matches_reference_cuda():
    x = torch.randn(9, 13, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(11, 13, device="cuda", dtype=torch.bfloat16)
    x_q, sx = quantize_activation_int8(x)
    w_q, sw = quantize_weight_ternary(w)

    actual = _packed_bitlinear_forward(x_q, sx, w_q, sw, x.dtype)
    expected = (F.linear(x_q.float(), w_q.float()) * sx.float() * sw.float()).to(x.dtype)

    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
