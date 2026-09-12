"""optim.param_rounding (nearest | stochastic) の挙動テスト。

bf16 parameter を fp32 master 無しで更新する時、nearest は半 ulp 未満の更新と
weight decay を潰す。stochastic は不偏な確率的丸めで期待値を保つ。
"""
from __future__ import annotations

import copy

import pytest
import torch

from src.train.optim import AdamW8bit, AdamWBF8, AdamWFP32, build_optimizer
from src.train.rounding import (
    resolve_param_rounding,
    round_to_param_dtype,
    step_seed,
)


def _run_tiny_updates(opt_cls, rounding, device, steps=400, **kwargs):
    # |w|=1.0 の bf16 では ulp=2^-7。lr=1e-4 の Adam 更新 (≈lr/step) は半 ulp の
    # 約 1/40 なので nearest では一度も反映されない。
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.full((65536,), 1.0, device=device, dtype=torch.bfloat16))
    opt = opt_cls([p], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
                  param_rounding=rounding, **kwargs)
    for _ in range(steps):
        p.grad = torch.ones_like(p)  # 一定勾配 → Adam 更新 ≈ -lr / step
        opt.step()
    return p.detach().float()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["eager", "auto"])
def test_nearest_loses_tiny_updates_and_stochastic_keeps_them(device, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "cpu" and backend == "auto":
        pytest.skip("auto は CPU では eager と同じ")
    steps = 400
    expected_drift = -1e-4 * steps  # fp32 なら約 -0.04

    nearest = _run_tiny_updates(AdamWFP32, "nearest", device, steps, backend=backend)
    assert torch.all(nearest == 1.0), "nearest は半 ulp 未満の更新を捨てるはず"

    stochastic = _run_tiny_updates(AdamWFP32, "stochastic", device, steps, backend=backend)
    mean_drift = (stochastic - 1.0).mean().item()
    # 65536 要素の平均。1 要素の分散は ulp^2/4 程度なので平均は十分にタイト。
    assert abs(mean_drift - expected_drift) < 0.15 * abs(expected_drift), (mean_drift, expected_drift)
    # 個々の値は bf16 grid 上 (1.0 か 1 - k*ulp) にあるだけで、二値化されていないこと
    assert stochastic.unique().numel() >= 3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_weight_decay_only_acts_with_stochastic_rounding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    results = {}
    for rounding in ("nearest", "stochastic"):
        torch.manual_seed(1)
        p = torch.nn.Parameter(torch.full((32768,), 0.5, device=device, dtype=torch.bfloat16))
        opt = AdamWFP32([p], lr=2e-4, weight_decay=0.1, param_rounding=rounding, backend="eager")
        for _ in range(500):
            p.grad = torch.zeros_like(p)  # 更新 0、decay だけ (相対 2e-5/step)
            opt.step()
        results[rounding] = p.detach().float().mean().item()
    assert results["nearest"] == 0.5
    expected = 0.5 * (1 - 2e-5) ** 500
    assert abs(results["stochastic"] - expected) < 0.2 * (0.5 - expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_and_eager_stochastic_agree_in_expectation():
    torch.manual_seed(3)
    base = torch.randn(200000, device="cuda", dtype=torch.bfloat16) * 0.02
    grads = [torch.randn_like(base) * 0.01 for _ in range(50)]
    out = {}
    for backend in ("eager", "triton"):
        p = torch.nn.Parameter(base.clone())
        opt = AdamWFP32([p], lr=3e-5, weight_decay=0.1, backend=backend,
                        param_rounding="stochastic")
        for g in grads:
            p.grad = g.clone()
            opt.step()
        out[backend] = p.detach().float()
    # 乱数列は違うので要素単位では一致しないが、平均更新量は一致する
    d_e = (out["eager"] - base.float()).mean().item()
    d_t = (out["triton"] - base.float()).mean().item()
    assert abs(d_e - d_t) < 1e-6, (d_e, d_t)
    assert (out["eager"] != base.float()).float().mean() > 0.3
    assert (out["triton"] != base.float()).float().mean() > 0.3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_stochastic_is_deterministic_given_state_and_resumable():
    torch.manual_seed(5)
    p = torch.nn.Parameter(torch.randn(70000, device="cuda", dtype=torch.bfloat16))
    opt = AdamWFP32([p], lr=1e-3, backend="triton", param_rounding="stochastic")
    grads = [torch.randn_like(p) for _ in range(6)]
    for g in grads[:3]:
        p.grad = g.clone()
        opt.step()
    assert "sr_seed" in opt.state[p]
    snapshot_p = p.detach().clone()
    snapshot_state = copy.deepcopy(opt.state_dict())
    for g in grads[3:]:
        p.grad = g.clone()
        opt.step()
    q = torch.nn.Parameter(snapshot_p.clone())
    restored = AdamWFP32([q], lr=1e-3, backend="triton", param_rounding="stochastic")
    restored.load_state_dict(snapshot_state)
    assert restored.state[q]["sr_seed"] == opt.state[p]["sr_seed"]
    for g in grads[3:]:
        q.grad = g.clone()
        restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_stochastic_is_noop_for_fp32_and_rejects_fp16():
    p = torch.nn.Parameter(torch.randn(1000))
    q = torch.nn.Parameter(p.detach().clone())
    a = AdamWFP32([p], lr=1e-3, backend="eager", param_rounding="nearest")
    b = AdamWFP32([q], lr=1e-3, backend="eager", param_rounding="stochastic")
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    a.step()
    b.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)

    h = torch.nn.Parameter(torch.randn(10, dtype=torch.float16))
    h.grad = torch.randn_like(h)
    with pytest.raises(ValueError, match="bf16/fp32"):
        AdamWFP32([h], lr=1e-3, backend="eager", param_rounding="stochastic").step()


@pytest.mark.parametrize("opt_cls", [AdamW8bit, AdamWBF8])
def test_low_bit_state_optimizers_support_stochastic(opt_cls):
    steps = 300
    nearest = _run_tiny_updates(opt_cls, "nearest", "cpu", steps)
    assert torch.all(nearest == 1.0)
    stochastic = _run_tiny_updates(opt_cls, "stochastic", "cpu", steps)
    drift = (stochastic - 1.0).mean().item()
    assert abs(drift - (-1e-4 * steps)) < 0.15 * 1e-4 * steps


def test_round_to_param_dtype_is_unbiased_and_on_grid():
    torch.manual_seed(7)
    x = torch.rand(1_000_000) * 0.01 + 0.5
    r = round_to_param_dtype(x, torch.bfloat16, "stochastic")
    assert r.dtype == torch.bfloat16
    err = (r.float() - x)
    assert abs(err.mean().item()) < 1e-6
    # 隣接 bf16 grid のどちらかに落ちること
    lo = x.to(torch.bfloat16).float()
    ulp = 2.0 ** -8  # [0.5, 1) の bf16 間隔
    assert torch.all((r.float() - x).abs() < ulp + 1e-9)
    assert torch.equal(round_to_param_dtype(x, torch.bfloat16, "nearest"), x.to(torch.bfloat16))
    assert torch.equal(round_to_param_dtype(x, torch.float32, "stochastic"), x)


def test_resolve_and_build_optimizer_rounding():
    assert resolve_param_rounding(None) == "stochastic"
    assert resolve_param_rounding("Nearest") == "nearest"
    with pytest.raises(ValueError, match="param_rounding"):
        resolve_param_rounding("truncate")
    p = torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
    opt = build_optimizer([p], {"lr": 1e-3})
    assert opt.param_rounding == "stochastic"
    opt = build_optimizer([p], {"lr": 1e-3, "param_rounding": "nearest"})
    assert opt.param_rounding == "nearest"
    with pytest.raises(ValueError, match="lion"):
        build_optimizer([p], {"optimizer": "lion", "lr": 1e-3, "param_rounding": "stochastic"})
    assert step_seed(123, 4) != step_seed(123, 5)
    assert 0 <= step_seed(2**31 - 1, 10**6) < 2**31
