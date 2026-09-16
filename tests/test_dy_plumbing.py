"""packed ternary backward の fused dY plumbing (`_quantize_dy_dual`) の CUDA テスト."""
from __future__ import annotations


import pytest
import torch

from src.model import bitlinear
from src.model.bitlinear import (
    _cast_fp8_tensorwise_transposed,
    _quantize_a8_rows_scaled,
    _quantize_dy_dual,
    fp8_gemm_supported,
    set_bitlinear_fused_dy_plumbing,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.mark.parametrize(
    "shape",
    [(1024, 3072), (1024, 2048), (1024, 11264), (16384, 768), (16384, 2304),
     (33, 17), (5, 3), (2048, 1000), (4097, 640)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantize_dy_dual_matches_separate_kernels_bitwise(shape, dtype):
    torch.manual_seed(0)
    m, n = shape
    # 勾配らしい広いダイナミックレンジ + round-half-to-even の境界値を混ぜる
    dy = (torch.randn(m, n, device="cuda") * torch.logspace(-6, 0, n, device="cuda")).to(dtype)
    dy.view(-1)[::13] = 0.0
    col_scale = torch.rand(n, device="cuda", dtype=torch.float32) + 0.5
    q_ref, inv_ref = _quantize_a8_rows_scaled(dy, col_scale)
    t_ref, sg_ref = _cast_fp8_tensorwise_transposed(dy)
    q, inv, t, sg = _quantize_dy_dual(dy, col_scale)
    assert torch.equal(q, q_ref)
    assert torch.equal(inv, inv_ref)
    assert torch.equal(t.view(torch.uint8), t_ref.view(torch.uint8))
    assert torch.equal(sg, sg_ref)
    assert t.shape == (n, m) and t.is_contiguous()


def test_quantize_dy_dual_handles_all_zero_rows():
    dy = torch.zeros(64, 256, device="cuda", dtype=torch.bfloat16)
    dy[3, 5] = 1.0
    col_scale = torch.ones(256, device="cuda")
    q_ref, inv_ref = _quantize_a8_rows_scaled(dy, col_scale)
    t_ref, sg_ref = _cast_fp8_tensorwise_transposed(dy)
    q, inv, t, sg = _quantize_dy_dual(dy, col_scale)
    assert torch.equal(q, q_ref) and torch.equal(inv, inv_ref)
    assert torch.equal(t.view(torch.uint8), t_ref.view(torch.uint8))
    assert torch.equal(sg, sg_ref)


@pytest.mark.skipif(not fp8_gemm_supported(), reason="sm89+ CUDA required")
def test_ste_backward_is_bitwise_identical_with_fused_dy_plumbing():
    """fused/separate で dX と dW が完全一致する (GEMM 入力が bit 一致するため)."""
    from src.model.bitlinear import (
        BitLinear,
        BitLinearGroup,
        set_bitlinear_fp8_mode,
        set_bitlinear_ternary_backend,
        set_bitlinear_ternary_wgrad_backend,
    )

    torch.manual_seed(0)
    set_bitlinear_ternary_backend("kmajor_single_dot")
    set_bitlinear_ternary_wgrad_backend("fp8")
    try:
        a = BitLinear(256, 128).cuda().bfloat16()
        b = BitLinear(256, 64).cuda().bfloat16()
        group = BitLinearGroup((a, b), kind="test")
        single = BitLinear(256, 192).cuda().bfloat16()
        for mod in (a, b, group, single):
            mod._fp8_mode = "ternary"
        set_bitlinear_fp8_mode(group, "ternary")
        set_bitlinear_fp8_mode(single, "ternary")
        for mod in (a, b, group, single):
            mod.enable_training_weight_cache(True)
        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

        def run(fused: bool):
            set_bitlinear_fused_dy_plumbing(fused)
            xi = x.clone().requires_grad_(True)
            y = group(xi) + single(xi)
            (y.float() * torch.logspace(-4, 0, y.size(-1), device="cuda")).sum().backward()
            grads = (xi.grad.clone(), a.weight.grad.clone(), b.weight.grad.clone(), single.weight.grad.clone())
            for mod in (a, b, single):
                mod.weight.grad = None
            return grads

        fused = run(True)
        separate = run(False)
        for f, s in zip(fused, separate):
            assert torch.equal(f, s)
        assert all(torch.isfinite(g).all() for g in fused)
    finally:
        set_bitlinear_fused_dy_plumbing(True)
        set_bitlinear_ternary_backend("dot_current")
        set_bitlinear_ternary_wgrad_backend("int8")
    assert bitlinear._fused_dy_plumbing is True


@pytest.mark.parametrize("magnitude", [1e-6, 1e-8, 1e-10])
def test_small_gradients_are_not_quantized_to_zero(magnitude):
    """dY の per-token INT8 scale に活性用の絶対下限 (1e-5) を使うと、mean 損失 / grad_accum で
    小さくなった勾配が行ごと 0 に潰れ dX が消える (2026-09-16 の本走停滞の原因)。
    下限は 0 割り回避の極小値だけにし、勾配の絶対スケールに依存しないこと。"""
    torch.manual_seed(0)
    dy = (torch.randn(64, 256, device="cuda") * magnitude).to(torch.bfloat16)
    col_scale = torch.full((256,), 0.02, device="cuda")  # 三値重みの absmean scale 程度
    for q, inv in (_quantize_a8_rows_scaled(dy, col_scale), _quantize_dy_dual(dy, col_scale)[:2]):
        assert int((q != 0).sum()) > q.numel() // 2, "小さい勾配が 0 に量子化された"
        # 行の最大要素は ±127 に写る (scale が下限に張り付いていない)
        assert int(q.abs().amax(dim=1).min()) == 127
        # dequantize の誤差は量子化 1 段 (inv) の半分以内
        err = (q.float() * inv.unsqueeze(1) - dy.float() * col_scale).abs()
        assert bool((err <= 0.5 * inv.unsqueeze(1) * 1.001).all())


def test_all_zero_dy_row_is_finite():
    dy = torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16)
    col_scale = torch.ones(64, device="cuda")
    q, inv, t, sg = _quantize_dy_dual(dy, col_scale)
    assert torch.isfinite(inv).all() and int(q.abs().sum()) == 0
