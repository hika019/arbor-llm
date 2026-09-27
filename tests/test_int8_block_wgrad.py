"""packed ternary の int8_block dW (token block scale の INT8 GEMM) の CUDA テスト."""
from __future__ import annotations

import pytest
import torch

from src.model.bitlinear import (
    _int8_mblock_wgrad,
    _quantize_a8_rows,
    _quantize_a8_rows_scaled,
    _quantize_dy_mblock_dual,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _grad_like(m: int, n: int, spread: float) -> torch.Tensor:
    """token 方向と出力 channel 方向に 10^±spread の大きさの差がある dY."""
    row = 10 ** (torch.rand(m, 1, device="cuda") * 2 * spread - spread)
    col = 10 ** (torch.rand(1, n, device="cuda") * 2 * spread - spread)
    return (torch.randn(m, n, device="cuda") * row * col * 1e-4).bfloat16()


@pytest.mark.parametrize(
    "shape",
    [(16384, 2304, 768), (1024, 3072, 2048), (1000, 200, 96), (130, 64, 32)],
)
def test_int8_block_wgrad_matches_fp32_per_output_row(shape):
    torch.manual_seed(0)
    m, n, k = shape
    x_int8, inv_sx = _quantize_a8_rows(torch.randn(m, k, device="cuda").bfloat16())
    dy = _grad_like(m, n, spread=2.0)
    ref = dy.float().t() @ (x_int8.float() * inv_sx.float()[:, None])
    got = _int8_mblock_wgrad(dy, x_int8, inv_sx).float()
    # 出力 channel ごとの相対誤差。scale 1 個の量子化だと小さい channel の行が丸ごと 0 になる
    row_rel = (got - ref).norm(dim=1) / ref.norm(dim=1)
    assert row_rel.max().item() < 0.03


def test_int8_block_wgrad_accumulates_into_out():
    torch.manual_seed(0)
    m, n, k = 2048, 256, 128
    x_int8, inv_sx = _quantize_a8_rows(torch.randn(m, k, device="cuda").bfloat16())
    dy = _grad_like(m, n, spread=1.0)
    once = _int8_mblock_wgrad(dy, x_int8, inv_sx).float()
    acc = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    _int8_mblock_wgrad(dy, x_int8, inv_sx, out=acc)
    _int8_mblock_wgrad(dy, x_int8, inv_sx, out=acc)
    torch.testing.assert_close(acc.float(), 2 * once, rtol=2e-2, atol=1e-6)


@pytest.mark.parametrize("shape", [(16384, 2304), (1024, 11264), (1000, 200), (5, 3)])
def test_dual_rows_match_a8_rows_scaled_bitwise(shape):
    torch.manual_seed(0)
    m, n = shape
    dy = _grad_like(m, n, spread=2.0)
    dy.view(-1)[::13] = 0.0
    col_scale = torch.rand(n, device="cuda") + 0.5
    inv_sx = torch.rand(m, device="cuda")
    q_ref, inv_ref = _quantize_a8_rows_scaled(dy, col_scale)
    q, inv, qt, block_scale = _quantize_dy_mblock_dual(dy, col_scale, inv_sx)
    assert torch.equal(q, q_ref) and torch.equal(inv, inv_ref)
    _, _, qt_only, block_scale_only = _quantize_dy_mblock_dual(dy, None, inv_sx)
    assert torch.equal(qt, qt_only) and torch.equal(block_scale, block_scale_only)
    assert qt.shape == (n, m) and block_scale.shape == ((m + 127) // 128, n)
