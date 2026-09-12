"""fused ternary pack kernel (src/model/ternary_pack.py) の CUDA テスト."""
from __future__ import annotations

import itertools

import pytest
import torch

from src.model.bitlinear import (
    pack_ternary_weight,
    pack_ternary_weight_kmajor,
    ternary_quantize_int8,
    ternary_scale,
)
from src.model.ternary_pack import fused_pack_supported, pack_ternary_cache_

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _reference(w: torch.Tensor, backend: str):
    w_int8, scale = ternary_quantize_int8(w)
    pack = (
        pack_ternary_weight_kmajor
        if backend.startswith("kmajor")
        else pack_ternary_weight
    )
    return (
        pack(w_int8),
        pack(w_int8.t().contiguous()),
        scale.expand(w.shape[0]).contiguous(),
    )


def _stress_weight(n: int, k: int, dtype: torch.dtype) -> torch.Tensor:
    """量子化境界 (±0.5·scale, ±1.5·scale) 付近の値を多く含む weight."""
    w = (torch.randn(n, k, device="cuda") * 0.02).to(dtype)
    scale = ternary_scale(w)
    flat = w.view(-1)
    flat[::7] = (scale * 0.5).to(dtype)
    flat[1::11] = (-scale * 0.5).to(dtype)
    flat[2::13] = (scale * 1.5).to(dtype)
    flat[3::17] = (-scale * 1.5).to(dtype)
    return w


@pytest.mark.parametrize(
    "shape,dtype,backend",
    list(
        itertools.product(
            [(33, 17), (64, 64), (17, 130), (5, 3), (2304, 768), (2048, 5632)],
            [torch.bfloat16, torch.float16, torch.float32],
            ["kmajor_single_dot", "dot_current"],
        )
    ),
)
def test_fused_pack_matches_torch_path_bitwise(shape, dtype, backend):
    torch.manual_seed(0)
    w = _stress_weight(*shape, dtype)
    assert fused_pack_supported(w)
    ref_packed, ref_packed_t, ref_scale = _reference(w, backend)
    packed = torch.empty_like(ref_packed)
    packed_t = torch.empty_like(ref_packed_t)
    row_scale = torch.empty_like(ref_scale)
    pack_ternary_cache_(
        w, ternary_scale(w), packed, packed_t, row_scale, backend=backend
    )
    assert torch.equal(packed, ref_packed)
    assert torch.equal(packed_t, ref_packed_t)
    assert torch.equal(row_scale, ref_scale)


@pytest.mark.parametrize("backend", ["kmajor_single_dot", "dot_current"])
def test_fused_pack_writes_group_member_slices(backend):
    """member ごとの n_offset 書き込みが連結後 pack と一致する."""
    torch.manual_seed(1)
    members = [
        _stress_weight(768, 2048, torch.bfloat16),
        _stress_weight(256, 2048, torch.bfloat16),
        _stress_weight(256, 2048, torch.bfloat16),
    ]
    cat = torch.cat([ternary_quantize_int8(w)[0] for w in members])
    pack = (
        pack_ternary_weight_kmajor
        if backend.startswith("kmajor")
        else pack_ternary_weight
    )
    ref_packed = pack(cat)
    ref_packed_t = pack(cat.t().contiguous())
    ref_scale = torch.cat(
        [ternary_quantize_int8(w)[1].expand(w.shape[0]) for w in members]
    )
    packed = torch.empty_like(ref_packed)
    packed_t = torch.empty_like(ref_packed_t)
    row_scale = torch.empty_like(ref_scale)
    offset = 0
    for w in members:
        pack_ternary_cache_(
            w, ternary_scale(w), packed, packed_t, row_scale,
            backend=backend, n_offset=offset,
        )
        offset += w.shape[0]
    assert torch.equal(packed, ref_packed)
    assert torch.equal(packed_t, ref_packed_t)
    assert torch.equal(row_scale, ref_scale)


def test_fused_pack_rejects_layout_mismatch():
    w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty(16, 64, dtype=torch.uint8, device="cuda")
    packed_t = torch.empty(16, 64, dtype=torch.uint8, device="cuda")
    row_scale = torch.empty(64, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        pack_ternary_cache_(
            w, ternary_scale(w), packed, packed_t, row_scale, backend="dot_current"
        )
    with pytest.raises(ValueError):
        pack_ternary_cache_(
            w, ternary_scale(w), packed, packed_t, row_scale,
            backend="kmajor_single_dot", n_offset=2,
        )
    with pytest.raises(ValueError):
        pack_ternary_cache_(
            w, ternary_scale(w), packed, packed_t, row_scale[:32],
            backend="kmajor_single_dot",
        )
