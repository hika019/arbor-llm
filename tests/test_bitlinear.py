"""BitLinear (BitNet b1.58 公式レシピ) のテスト (CPU)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.model.bitlinear import (
    BitLinear,
    BitLinearGroup,
    activation_quant,
    check_activation_precision,
    quantize_activation,
    configure_bitlinear_training_cache,
    fp8_gemm_supported,
    set_bitlinear_ternary_backend,
    set_bitlinear_ternary_wgrad_backend,
    set_bitlinear_fp8_mode,
    set_bitlinear_int8_backend,
    weight_quant,
    _cast_a8_dequant_fp8_transposed,
    _cast_fp8_tensorwise_transposed,
    _packed_linear_tile,
    _wgrad_tile,
)


def test_weight_quant_is_ternary():
    w = torch.randn(64, 32)
    w_q = weight_quant(w)
    scale = w.abs().mean()
    levels = torch.unique((w_q / scale).round())
    assert set(levels.tolist()) <= {-1.0, 0.0, 1.0}


def test_bitlinear_default_activation_precision_is_int8():
    layer = BitLinear(16, 8)
    assert layer.activation_precision == "int8"


def test_bitlinear_bf8_activation_weight_stays_ternary():
    torch.manual_seed(0)
    layer = BitLinear(16, 8, activation_precision="bf8")
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    # 重みは activation 精度に関係なく W1.58 ternary のまま
    scale = layer.weight.abs().mean()
    levels = torch.unique((weight_quant(layer.weight) / scale).round())
    assert set(levels.tolist()) <= {-1.0, 0.0, 1.0}


def test_bf8_activation_rounds_to_float8_grid():
    x = torch.randn(4, 32)
    q = quantize_activation(x, "bf8")
    assert q.dtype == x.dtype
    # bf8 fake-quant は float8_e5m2 グリッドに一致する
    torch.testing.assert_close(q, x.to(torch.float8_e5m2).to(x.dtype))


def test_bf16_activation_is_identity():
    x = torch.randn(4, 32)
    torch.testing.assert_close(quantize_activation(x, "bf16"), x)


def test_unknown_activation_precision_is_error():
    with pytest.raises(ValueError, match="activation_precision"):
        check_activation_precision("int4")
    with pytest.raises(ValueError, match="activation_precision"):
        BitLinear(8, 8, activation_precision="fp8")


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
    assert cached._train_w_int8.dtype == torch.int8
    assert cached._train_w_scale.dtype == torch.float32
    assert cached._train_w_fp8 is None
    assert cached.training_cache_bytes < cached.weight.numel() * cached.weight.element_size()


def test_ternary_training_cache_uses_two_packed_layouts():
    layer = BitLinear(33, 17)
    layer._fp8_mode = "ternary"
    layer.enable_training_weight_cache(True)

    assert layer._train_w_int8 is None
    assert layer._train_w_fp8 is None
    assert layer._train_w_fp8_t is None
    assert layer._train_w_packed.dtype == torch.uint8
    assert layer._train_w_packed.shape == (17, 9)
    assert layer._train_w_packed_t.shape == (33, 5)
    assert layer.training_cache_bytes == layer.cache_cost_bytes
    assert layer.training_cache_bytes < layer.weight.numel()


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


def test_fp8_auto_is_rejected_instead_of_falling_back():
    model = torch.nn.Sequential(BitLinear(32, 16), BitLinear(16, 16))
    with pytest.raises(ValueError, match="bitlinear fp8 mode"):
        set_bitlinear_fp8_mode(model, "auto")


def test_unknown_fp8_mode_is_error():
    with pytest.raises(ValueError, match="bitlinear fp8 mode"):
        set_bitlinear_fp8_mode(BitLinear(16, 16), "fp4")


def test_unknown_int8_backend_is_error():
    with pytest.raises(ValueError, match="int8 backend"):
        set_bitlinear_int8_backend("cutlass")
    assert set_bitlinear_int8_backend("auto") == "auto"


def test_ternary_backend_aliases_and_unknown_backend():
    assert set_bitlinear_ternary_backend("tensor-core") == "dot"
    assert set_bitlinear_ternary_backend("tl_dot") == "dot"
    assert set_bitlinear_ternary_backend("current") == "dot_current"
    assert set_bitlinear_ternary_backend("dot") == "dot"
    with pytest.raises(ValueError, match="ternary backend"):
        set_bitlinear_ternary_backend("multiply_free")


def test_ternary_wgrad_backend_aliases_and_unknown_backend():
    assert set_bitlinear_ternary_wgrad_backend("hybrid") == "auto"
    assert set_bitlinear_ternary_wgrad_backend("shape-auto") == "auto"
    assert set_bitlinear_ternary_wgrad_backend("int8") == "int8"
    with pytest.raises(ValueError, match="ternary wgrad backend"):
        set_bitlinear_ternary_wgrad_backend("bf16")


def test_ternary_wgrad_auto_selects_backend_by_shape(monkeypatch):
    import src.model.bitlinear as bitlinear

    monkeypatch.setattr(bitlinear, "fp8_gemm_supported", lambda: True)
    monkeypatch.setattr(
        bitlinear,
        "_fp8_wgrad",
        lambda grad, x, dtype: torch.full((grad.size(1), x.size(1)), 8.0),
    )
    monkeypatch.setattr(
        bitlinear,
        "_lowbit_wgrad",
        lambda grad, x, dtype: torch.full((grad.size(1), x.size(1)), 1.0),
    )
    set_bitlinear_ternary_wgrad_backend("auto")
    try:
        fp8_result = bitlinear._ternary_wgrad(
            torch.empty(16, 32), torch.empty(16, 16), torch.float32
        )
        int8_result = bitlinear._ternary_wgrad(
            torch.empty(16, 16), torch.empty(16, 32), torch.float32
        )
    finally:
        set_bitlinear_ternary_wgrad_backend("int8")
    assert torch.all(fp8_result == 8)
    assert torch.all(int8_result == 1)


def test_a8_dequant_fp8_transposed_matches_materialized_xq_cpu():
    cases = [
        (
            torch.tensor(
                [
                    [0, 0, 0, 0],
                    [1, -2, 3, -4],
                    [127, -126, 0, 64],
                    [-128, 0, 5, -7],
                ],
                dtype=torch.int8,
            ),
            torch.tensor([1e-5 / 127.0, 1e-5 / 127.0, 0.25, 0.125]),
        ),
        (
            torch.tensor([[0, 0, 0, 0], [1, -2, 0, 3]], dtype=torch.int8),
            torch.full((2,), 1e-5 / 127.0),
        ),
    ]
    for x_int8, inv_sx in cases:
        x_q = x_int8.to(torch.float32) * inv_sx.unsqueeze(1)
        actual, actual_scale = _cast_a8_dequant_fp8_transposed(x_int8, inv_sx)
        expected, expected_scale = _cast_fp8_tensorwise_transposed(x_q)

        torch.testing.assert_close(actual_scale, expected_scale)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)


def test_lowbit_tile_presets_cover_small_and_arbor_shapes():
    assert _packed_linear_tile(
        1024, 2048, 2048, grouped_decode=True
    ) == (32, 64, 128, 4)
    assert _packed_linear_tile(
        1024, 2048, 2048, grouped_decode=False
    ) == (
        32, 64, 32, 4
    )
    assert _packed_linear_tile(
        16384, 4096, 768, grouped_decode=False
    ) == (128, 64, 64, 4)
    assert _packed_linear_tile(
        32768, 768, 4096, grouped_decode=False
    ) == (128, 128, 64, 4)
    assert _packed_linear_tile(
        16384, 768, 4096, grouped_decode=False
    ) == (128, 64, 64, 4)
    assert _packed_linear_tile(
        131072, 4096, 768, grouped_decode=False
    ) == (128, 128, 64, 4)
    assert _packed_linear_tile(
        1024, 11264, 2048, grouped_decode=False
    ) == (32, 64, 32, 4)
    assert _wgrad_tile(7, 17, 33) == (32, 32, 32, 4)
    assert _wgrad_tile(1024, 2048, 2048) == (64, 64, 64, 8)


@pytest.mark.skipif(not fp8_gemm_supported(), reason="sm89+ CUDA required")
def test_fp8_bwd_preserves_forward_and_produces_finite_gradients_cuda():
    torch.manual_seed(0)
    ref = BitLinear(32, 32, activation_precision="bf16").to(
        device="cuda", dtype=torch.bfloat16
    )
    fp8 = BitLinear(32, 32, activation_precision="bf16").to(
        device="cuda", dtype=torch.bfloat16
    )
    fp8.load_state_dict(ref.state_dict())
    info = set_bitlinear_fp8_mode(fp8, "bwd")
    assert info["mode"] == "bwd"

    x_ref = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_fp8 = x_ref.detach().clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_fp8 = fp8(x_fp8)
    assert torch.equal(y_fp8, y_ref)

    y_ref.square().mean().backward()
    y_fp8.square().mean().backward()
    assert torch.isfinite(x_fp8.grad).all()
    assert torch.isfinite(fp8.weight.grad).all()
    assert torch.nn.functional.cosine_similarity(
        x_fp8.grad.float().flatten(), x_ref.grad.float().flatten(), dim=0
    ) > 0.98
    assert torch.nn.functional.cosine_similarity(
        fp8.weight.grad.float().flatten(), ref.weight.grad.float().flatten(), dim=0
    ) > 0.98


@pytest.mark.skipif(not fp8_gemm_supported(), reason="sm89+ CUDA required")
def test_fp8_unaligned_shape_is_error_instead_of_bf16_fallback_cuda():
    layer = BitLinear(32, 32, activation_precision="bf16").to(
        device="cuda", dtype=torch.bfloat16
    )
    set_bitlinear_fp8_mode(layer, "bwd")
    x = torch.randn(15, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="暗黙フォールバックは禁止"):
        layer(x)


@pytest.mark.skipif(not fp8_gemm_supported(), reason="sm89+ CUDA required")
def test_native_int8_forward_and_fp8_backward_cuda():
    torch.manual_seed(0)
    ref = BitLinear(32, 32).to(device="cuda", dtype=torch.bfloat16)
    native = BitLinear(32, 32).to(device="cuda", dtype=torch.bfloat16)
    native.load_state_dict(ref.state_dict())
    set_bitlinear_fp8_mode(native, "int8")
    native.enable_training_weight_cache(True)

    assert native._train_w_int8.dtype == torch.int8
    assert native._train_w_fp8.dtype == torch.float8_e4m3fn
    assert native._train_w_fp8_t.is_contiguous()
    x_ref = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_native = x_ref.detach().clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_native = native(x_native)
    torch.testing.assert_close(y_native.float(), y_ref.float(), atol=3e-3, rtol=3e-3)

    y_ref.square().mean().backward()
    y_native.square().mean().backward()
    assert torch.isfinite(x_native.grad).all()
    assert torch.isfinite(native.weight.grad).all()
    assert torch.nn.functional.cosine_similarity(
        x_native.grad.float().flatten(), x_ref.grad.float().flatten(), dim=0
    ) > 0.98
    assert torch.nn.functional.cosine_similarity(
        native.weight.grad.float().flatten(), ref.weight.grad.float().flatten(), dim=0
    ) > 0.98


@pytest.mark.parametrize("ternary_backend", ["dot", "dot_current"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_ternary_forward_dgrad_and_wgrad_cuda(ternary_backend):
    torch.manual_seed(0)
    set_bitlinear_ternary_backend(ternary_backend)
    # packed kernelはFP8/cuBLASの16要素alignment制約を持たない。
    ref = BitLinear(33, 17).to(device="cuda", dtype=torch.bfloat16)
    packed = BitLinear(33, 17).to(device="cuda", dtype=torch.bfloat16)
    packed.load_state_dict(ref.state_dict())
    info = set_bitlinear_fp8_mode(packed, "ternary")
    packed.enable_training_weight_cache(True)

    assert info["mode"] == "ternary"
    assert packed._train_w_int8 is None
    assert packed._train_w_packed.dtype == torch.uint8
    assert packed._train_w_packed_t.dtype == torch.uint8
    x_ref = torch.randn(7, 33, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_packed = x_ref.detach().clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_packed = packed(x_packed)
    torch.testing.assert_close(
        y_packed.float(), y_ref.float(), atol=4e-3, rtol=4e-3
    )

    y_ref.square().mean().backward()
    y_packed.square().mean().backward()
    assert torch.isfinite(x_packed.grad).all()
    assert torch.isfinite(packed.weight.grad).all()
    assert torch.nn.functional.cosine_similarity(
        x_packed.grad.float().flatten(), x_ref.grad.float().flatten(), dim=0
    ) > 0.98
    assert torch.nn.functional.cosine_similarity(
        packed.weight.grad.float().flatten(), ref.weight.grad.float().flatten(), dim=0
    ) > 0.98
    set_bitlinear_ternary_backend("dot")


@pytest.mark.skipif(not fp8_gemm_supported(), reason="sm89+ CUDA required")
def test_packed_ternary_auto_fp8_wgrad_cuda():
    torch.manual_seed(3)
    set_bitlinear_ternary_backend("dot_current")
    set_bitlinear_ternary_wgrad_backend("auto")
    try:
        ref = BitLinear(32, 64).to(device="cuda", dtype=torch.bfloat16)
        packed = BitLinear(32, 64).to(device="cuda", dtype=torch.bfloat16)
        packed.load_state_dict(ref.state_dict())
        set_bitlinear_fp8_mode(packed, "ternary")
        packed.enable_training_weight_cache(True)

        x_ref = torch.randn(
            32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        x_packed = x_ref.detach().clone().requires_grad_(True)
        y_ref = ref(x_ref)
        y_packed = packed(x_packed)
        torch.testing.assert_close(
            y_packed.float(), y_ref.float(), atol=4e-3, rtol=4e-3
        )

        y_ref.square().mean().backward()
        y_packed.square().mean().backward()
        assert torch.isfinite(x_packed.grad).all()
        assert torch.isfinite(packed.weight.grad).all()
        assert torch.nn.functional.cosine_similarity(
            x_packed.grad.float().flatten(), x_ref.grad.float().flatten(), dim=0
        ) > 0.98
        assert torch.nn.functional.cosine_similarity(
            packed.weight.grad.float().flatten(),
            ref.weight.grad.float().flatten(),
            dim=0,
        ) > 0.98
    finally:
        set_bitlinear_ternary_wgrad_backend("int8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_ternary_group_supports_distinct_weight_scales_cuda():
    torch.manual_seed(1)
    a = BitLinear(32, 16).to(device="cuda", dtype=torch.bfloat16)
    b = BitLinear(32, 32).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        b.weight.mul_(3.0)
    group = BitLinearGroup((a, b), kind="test").to(
        device="cuda", dtype=torch.bfloat16
    )
    set_bitlinear_fp8_mode(group, "ternary")
    group.enable_training_weight_cache(True)

    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat((a(x), b(x)), dim=-1)
    actual = group(x)
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=5e-3, rtol=5e-3
    )
    assert group._train_w_scale[:16].mean() < group._train_w_scale[16:].mean()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_ternary_bitlinear_torch_compile_cuda():
    layer = BitLinear(32, 64).to(device="cuda", dtype=torch.bfloat16).train()
    set_bitlinear_fp8_mode(layer, "ternary")
    layer.enable_training_weight_cache(True)
    compiled = torch.compile(layer)
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = compiled(x).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arbor_packed_ternary_caches_all_layers_and_trains_cuda():
    from src.model.arbor import ArborConfig, ArborModel

    cfg = ArborConfig.from_dict(
        dict(
            vocab_size=260, patch_size=4, patch_pooling="mean", max_bytes=64,
            hidden_size=32, num_heads=4, num_kv_heads=2, intermediate_size=64,
            num_hidden_layers=1,
            local_hidden_size=16, local_num_heads=2, local_num_kv_heads=2,
            local_intermediate_size=32,
            num_local_encoder_layers=1, num_local_decoder_layers=1,
        )
    )
    model = ArborModel(cfg).to(device="cuda", dtype=torch.bfloat16).train()
    from src.model.bitlinear import install_arbor_projection_fusions

    install_arbor_projection_fusions(model)
    set_bitlinear_fp8_mode(model, "ternary")
    info = configure_bitlinear_training_cache(
        model,
        enabled="full",
        grad_accum_steps=2,
        max_cache_gib=0.1,
        min_numel=0,
    )
    assert info["cached_layers"] == info["eligible_layers"]
    assert info["cache_format"] == "packed2_dual_layout"
    x = torch.randint(4, 260, (1, 64), device="cuda")
    loss = model(x).logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not fp8_gemm_supported(), reason="sm89+ CUDA required")
def test_arbor_native_int8_caches_all_layers_and_trains_cuda():
    from src.model.arbor import ArborConfig, ArborModel

    cfg = ArborConfig.from_dict(
        dict(
            vocab_size=260, patch_size=4, patch_pooling="mean", max_bytes=64,
            hidden_size=32, num_heads=4, num_kv_heads=2, intermediate_size=64,
            num_hidden_layers=1,
            local_hidden_size=16, local_num_heads=2, local_num_kv_heads=2,
            local_intermediate_size=32,
            num_local_encoder_layers=1, num_local_decoder_layers=1,
        )
    )
    model = ArborModel(cfg).to(device="cuda", dtype=torch.bfloat16).train()
    from src.model.bitlinear import install_arbor_projection_fusions

    install_arbor_projection_fusions(model)
    set_bitlinear_fp8_mode(model, "int8")
    info = configure_bitlinear_training_cache(
        model,
        enabled="full",
        grad_accum_steps=2,
        max_cache_gib=0.1,
        min_numel=0,
    )
    assert info["cached_layers"] == info["eligible_layers"]
    assert info["cache_format"] == "int8+fp8_dual_layout"
    x = torch.randint(4, 260, (1, 64), device="cuda")
    loss = model(x).logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not fp8_gemm_supported(), reason="sm89+ CUDA required")
def test_native_int8_bitlinear_torch_compile_cuda():
    layer = BitLinear(32, 64).to(device="cuda", dtype=torch.bfloat16).train()
    set_bitlinear_fp8_mode(layer, "int8")
    layer.enable_training_weight_cache(True)
    compiled = torch.compile(layer)
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = compiled(x).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a8_quantize_rows_uses_round_half_to_even_cuda():
    from src.model.bitlinear import _quantize_a8_rows

    # amax=127 の行なので inv_scale=1.0。scaled は入力そのままで tie を作る。
    x = torch.tensor(
        [[0.5, 1.5, 2.5, 3.5, 127.0, -0.5, -1.5, -2.5]],
        device="cuda", dtype=torch.float32,
    )
    q, inv_scale = _quantize_a8_rows(x)
    # torch.round と同じ ties-to-even: 0.5->0, 1.5->2, 2.5->2, 3.5->4, -0.5->0, -1.5->-2, -2.5->-2
    assert q[0].tolist() == [0, 2, 2, 4, 127, 0, -2, -2]
    torch.testing.assert_close(inv_scale, torch.ones(1, device="cuda"))
    # 参照 (x*scale).round() と bit 一致することも確認
    ref = (x * (127.0 / x.abs().amax(dim=-1, keepdim=True))).round().clamp(-128, 127)
    torch.testing.assert_close(q.float(), ref, atol=0, rtol=0)


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
    assert lin._w_packed is None
    out = lin(x)
    assert torch.allclose(out, ref, atol=1e-4), float((out - ref).abs().max())
    # train モードに戻すと学習パスに切り替わる (凍結値は使われない)
    lin.train()
    assert torch.allclose(lin(x), ref, atol=1e-4)
    lin.unfreeze()
    assert not lin.frozen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_inference_matches_reference_cuda():
    torch.manual_seed(0)
    lin = BitLinear(128, 96).to(device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(9, 128, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).float()
    lin.freeze_for_inference()
    out = lin(x).float()
    assert lin._w_packed is None
    assert lin._w_dq is not None
    assert torch.allclose(out, ref, atol=3e-2, rtol=1e-2), float((out - ref).abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_packed_inference_matches_reference_cuda(monkeypatch):
    monkeypatch.setenv("ARBOR_PACKED_BITLINEAR_INFERENCE", "1")
    torch.manual_seed(0)
    lin = BitLinear(128, 96).to(device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).float()
    lin.freeze_for_inference()
    assert lin._w_packed is not None
    out = lin(x).float()
    assert torch.allclose(out, ref, atol=3e-2, rtol=1e-2), float(
        (out - ref).abs().max()
    )
