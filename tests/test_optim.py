from __future__ import annotations

import copy

import pytest
import torch

from src.train.optim import (
    AdamW8bit,
    AdamWBF8,
    AdamWFP32,
    Lion,
    build_optimizer,
    build_scheduler,
    resolve_state_precision,
    _quantize_dynamic_state,
    _dequantize_dynamic_state,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [1, 4097, 131072])
def test_fused_adamw_matches_eager_with_scheduler_and_skipped_grad(dtype, size):
    torch.manual_seed(18)
    p = torch.nn.Parameter(torch.randn(size, device="cuda", dtype=dtype))
    q = torch.nn.Parameter(p.detach().clone())
    eager = AdamWFP32([p], lr=0.001, betas=(0.9, 0.95), backend="eager")
    fused = AdamWFP32([q], lr=0.001, betas=(0.9, 0.95), backend="triton")
    for step in range(30):
        for opt in (eager, fused):
            opt.param_groups[0]["lr"] = 0.001 * (step + 1) / 30
            opt.param_groups[0]["weight_decay"] = 0.1 if step < 15 else 0.0
        p.grad = None if step == 11 else torch.randn_like(p) * 0.01
        q.grad = None if p.grad is None else p.grad.clone()
        previous_version = q._version
        eager.step()
        fused.step()
        assert (q._version > previous_version) == (q.grad is not None)
        # FP32 division implementations may differ by a few ULPs; preserve
        # the exact BF16/FP16 parameter-rounding contract on this workload.
        torch.testing.assert_close(q, p, rtol=1e-6 if dtype == torch.float32 else 0,
                                   atol=1e-8 if dtype == torch.float32 else 0)
        for key in ("exp_avg", "exp_avg_sq"):
            assert fused.state[q][key].dtype == torch.float32
            torch.testing.assert_close(fused.state[q][key], eager.state[p][key], rtol=1e-6, atol=1e-10)
        assert fused.state[q]["step"] == eager.state[p]["step"]


def test_fused_adamw_backend_validation_and_cpu_auto():
    p = torch.nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError, match="backend"):
        AdamWFP32([p], lr=0.01, backend="invalid")
    opt = AdamWFP32([p], lr=0.01, backend="auto")
    p.grad = torch.ones_like(p)
    opt.step()
    assert torch.isfinite(p).all()
    with pytest.raises(RuntimeError, match="contiguous CUDA"):
        AdamWFP32([p], lr=0.01, backend="triton").step()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adamw_fp32_resume_preserves_moment_values_and_backend(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    p = torch.nn.Parameter(torch.randn(4097, device=device, dtype=torch.bfloat16))
    eager = AdamWFP32([p], lr=0.001, backend="eager")
    for _ in range(3):
        p.grad = torch.randn_like(p)
        eager.step()
    q = torch.nn.Parameter(p.detach().clone())
    restored = AdamWFP32([q], lr=0.001, backend="auto")
    restored.load_state_dict(copy.deepcopy(eager.state_dict()))
    assert restored.backend == "auto"
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(restored.state[q][key], eager.state[p][key], rtol=0, atol=0)
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    eager.step()
    restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_adamw_auto_handles_strided_parameters():
    p = torch.nn.Parameter(torch.randn(7, 5, device="cuda").T)
    q = torch.nn.Parameter(p.detach().clone())
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    AdamWFP32([p], lr=0.01, backend="auto").step()
    AdamWFP32([q], lr=0.01, backend="eager").step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_lion_optimizer_step_updates_parameter():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)

    p.grad = torch.tensor([0.5, -0.5])
    opt.step()

    torch.testing.assert_close(p.detach(), torch.tensor([0.9, -0.9]))
    assert "exp_avg" in opt.state[p]


def test_build_optimizer_accepts_lion():
    model = torch.nn.Linear(2, 1)
    opt = build_optimizer(
        model.parameters(),
        {
            "optimizer": "lion",
            "lr": 1e-3,
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
        },
    )

    assert isinstance(opt, Lion)


def test_adamw_8bit_keeps_moments_as_uint8_indices():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    opt = AdamW8bit([p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)

    p.grad = torch.tensor([0.25, -0.5, 1.0])
    opt.step()

    state = opt.state[p]
    assert state["exp_avg"].dtype == torch.uint8
    assert state["exp_avg_sq"].dtype == torch.uint8
    assert state["exp_avg_scale"].dtype == torch.float32
    assert state["exp_avg_sq_scale"].dtype == torch.float32
    assert state["step"] == 1
    assert not torch.equal(p.detach(), torch.tensor([1.0, -2.0, 3.0]))


def test_build_optimizer_accepts_int8_state_precision():
    model = torch.nn.Linear(2, 1)
    opt = build_optimizer(
        model.parameters(),
        {
            "optimizer": "adamw",
            "state_precision": "int8",
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
    )

    assert isinstance(opt, AdamW8bit)


def test_adamw_fp32_state_precision_applies_to_every_parameter():
    parameters = [
        torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.ones(4097, dtype=torch.bfloat16)),
    ]
    opt = build_optimizer(
        parameters,
        {
            "optimizer": "adamw",
            "state_precision": "fp32",
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
    )
    for p in parameters:
        p.grad = torch.ones_like(p)
    opt.step()

    for p in parameters:
        assert opt.state[p]["exp_avg"].dtype == torch.float32
        assert opt.state[p]["exp_avg_sq"].dtype == torch.float32


def test_adamw_int8_state_precision_applies_to_every_parameter():
    parameters = [
        torch.nn.Parameter(torch.ones(1)),
        torch.nn.Parameter(torch.ones(4097)),
    ]
    opt = build_optimizer(
        parameters,
        {
            "optimizer": "adamw",
            "state_precision": "int8",
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
    )
    for p in parameters:
        p.grad = torch.ones_like(p)
    opt.step()

    for p in parameters:
        assert opt.state[p]["exp_avg"].dtype == torch.uint8
        assert opt.state[p]["exp_avg_sq"].dtype == torch.uint8


def test_adamw_bf8_keeps_moments_as_float8():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    opt = AdamWBF8([p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)

    p.grad = torch.tensor([0.25, -0.5, 1.0])
    opt.step()

    state = opt.state[p]
    assert state["exp_avg"].dtype == torch.float8_e5m2
    assert state["exp_avg_sq"].dtype == torch.float8_e5m2
    assert state["step"] == 1
    assert not torch.equal(p.detach(), torch.tensor([1.0, -2.0, 3.0]))


def test_build_optimizer_accepts_bf8_state_precision():
    parameters = [
        torch.nn.Parameter(torch.ones(1)),
        torch.nn.Parameter(torch.ones(4097)),
    ]
    opt = build_optimizer(
        parameters,
        {
            "optimizer": "adamw",
            "state_precision": "bf8",
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
    )
    assert isinstance(opt, AdamWBF8)
    for p in parameters:
        p.grad = torch.ones_like(p)
    opt.step()
    for p in parameters:
        assert opt.state[p]["exp_avg"].dtype == torch.float8_e5m2
        assert opt.state[p]["exp_avg_sq"].dtype == torch.float8_e5m2


def test_adamw_bf8_checkpoint_restore_preserves_state_dtypes():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = AdamWBF8([p], lr=1e-2)
    p.grad = torch.ones_like(p)
    opt.step()

    p2 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt2 = AdamWBF8([p2], lr=1e-2)
    opt2.load_state_dict(opt.state_dict())
    state = opt2.state[p2]
    assert state["exp_avg"].dtype == torch.float8_e5m2
    assert state["exp_avg_sq"].dtype == torch.float8_e5m2
    p2.grad = torch.ones_like(p2)
    opt2.step()


def test_adamw_8bit_checkpoint_restore_preserves_state_dtypes():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = AdamW8bit([p], lr=1e-2)
    p.grad = torch.ones_like(p)
    opt.step()

    p2 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt2 = AdamW8bit([p2], lr=1e-2)
    opt2.load_state_dict(opt.state_dict())
    state = opt2.state[p2]

    assert state["exp_avg"].dtype == torch.uint8
    assert state["exp_avg_sq"].dtype == torch.uint8
    assert state["exp_avg_scale"].dtype == torch.float32
    assert state["exp_avg_sq_scale"].dtype == torch.float32
    p2.grad = torch.ones_like(p2)
    opt2.step()


def test_adamw_8bit_rejects_legacy_linear_int8_checkpoint():
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW8bit([p], lr=1e-2)
    p.grad = torch.ones_like(p)
    opt.step()
    legacy = copy.deepcopy(opt.state_dict())
    for state in legacy["state"].values():
        state["exp_avg"] = state["exp_avg"].to(torch.int8)
        state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.int8)

    p2 = torch.nn.Parameter(torch.ones(4))
    opt2 = AdamW8bit([p2], lr=1e-2)
    with pytest.raises(ValueError, match="旧linear-int8"):
        opt2.load_state_dict(legacy)


def test_legacy_optimizer_name_is_not_implicitly_converted():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(
            model.parameters(),
            {
                "optimizer": "bnb_adamw_8bit",
                "lr": 1e-3,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        )


def test_precision_must_be_selected_by_state_precision():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(
            model.parameters(),
            {
                "optimizer": "adamw_8bit",
                "lr": 1e-3,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        )


def test_resolve_state_precision_rejects_unknown_value():
    assert resolve_state_precision("fp32") == "fp32"


def test_dynamic_quant_preserves_small_second_moment_values():
    """裾の重い二次モーメント: 大きな値と極小値が混在しても小値が0に潰れないこと."""
    torch.manual_seed(0)
    x = torch.empty(4096)
    x[:] = 1e-6
    x[0] = 1.0  # block absmax を支配する外れ値
    x[10] = 3.2e-5
    q, scale = _quantize_dynamic_state(x, signed=False)
    recon = _dequantize_dynamic_state(q, scale, signed=False)
    assert q.dtype == torch.uint8
    # 旧線形実装では 1e-6/(1.0/127)=0 に underflow していた。dynamic では相対誤差で残る。
    assert (recon[1:] > 0).all()
    rel = (recon[1:] - x[1:]).abs() / x[1:]
    assert rel.max() < 0.2, float(rel.max())


def test_dynamic_quant_signed_roundtrip_relative_error():
    torch.manual_seed(0)
    x = torch.randn(4096) * torch.logspace(-6, 0, 4096)
    q, scale = _quantize_dynamic_state(x, signed=True)
    recon = _dequantize_dynamic_state(q, scale, signed=True)
    mask = x.abs() > x.abs().max() * 1e-6
    rel = (recon[mask] - x[mask]).abs() / x[mask].abs()
    assert rel.max() < 0.2, float(rel.max())


def test_adamw8bit_dynamic_tracks_fp32_on_heavy_tailed_grads():
    """裾の重い勾配で AdamW8bit(dynamic) が AdamWFP32 に追随し発散しないこと."""
    from src.train.optim import AdamWFP32

    torch.manual_seed(0)
    dim = 4096
    target = torch.randn(dim)
    scales = torch.logspace(-3, 1, dim)  # 座標ごとに勾配スケールが桁違い

    def make():
        return torch.nn.Parameter(torch.zeros(dim))

    p8, pf = make(), make()
    o8 = AdamW8bit([p8], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    of = AdamWFP32([pf], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    for _ in range(200):
        g = (torch.randn(dim) + (p8.detach() - target)) * scales
        p8.grad = g.clone()
        pf.grad = g.clone()
        o8.step()
        of.step()
    assert torch.isfinite(p8).all()
    # dynamic 8bit が fp32 と大きく乖離せず、発散していないこと
    denom = pf.detach().norm().clamp_min(1e-6)
    rel = (p8.detach() - pf.detach()).norm() / denom
    assert rel < 0.1, float(rel)
    assert resolve_state_precision("int8") == "int8"
    assert resolve_state_precision("bf8") == "bf8"
    with pytest.raises(ValueError, match="state_precision"):
        resolve_state_precision("int4")


def _lr_at(sched_cfg: dict, steps: int) -> float:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, sched_cfg)
    for _ in range(steps):
        opt.step()
        sched.step()
    return opt.param_groups[0]["lr"]


def test_scheduler_min_lr_ratio():
    cfg = {"total_steps": 100, "warmup_steps": 10, "min_lr_ratio": 0.1}
    assert _lr_at(cfg, 10) == pytest.approx(1.0)        # warmup 終了 = ピーク
    assert _lr_at(cfg, 55) == pytest.approx(0.55)       # 中間 = (1+0.1)/2
    assert _lr_at(cfg, 100) == pytest.approx(0.1)       # 最終 = 下限
    assert _lr_at(cfg, 150) == pytest.approx(0.1)       # 超過しても下限維持


def test_scheduler_min_lr_ratio_default_zero():
    cfg = {"total_steps": 100, "warmup_steps": 10}
    assert _lr_at(cfg, 100) == pytest.approx(0.0)       # 既定は従来どおり 0 まで減衰


def test_scheduler_decay_end_ratio():
    cfg = {"total_steps": 100, "warmup_steps": 10,
           "min_lr_ratio": 0.1, "decay_end_ratio": 0.8}
    assert _lr_at(cfg, 10) == pytest.approx(1.0)        # warmup 終了 = ピーク
    assert _lr_at(cfg, 45) == pytest.approx(0.55)       # 減衰区間の中間 = (1+0.1)/2
    assert _lr_at(cfg, 80) == pytest.approx(0.1)        # 80% 時点で下限到達
    assert _lr_at(cfg, 90) == pytest.approx(0.1)        # 残り 20% は下限で一定
    assert _lr_at(cfg, 100) == pytest.approx(0.1)


def test_scheduler_decay_end_ratio_validation():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError):
        build_scheduler(opt, {"total_steps": 100, "decay_end_ratio": 0.0})
    with pytest.raises(ValueError):
        build_scheduler(opt, {"total_steps": 100, "decay_end_ratio": 1.5})


def test_scheduler_min_lr_ratio_validation():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError):
        build_scheduler(opt, {"total_steps": 100, "min_lr_ratio": 1.0})


def _wsd_cfg(**over):
    cfg = {
        "optimizer": "adamw",
        "state_precision": "fp32",
        "lr": 1.0,  # base lr=1 なので get_last_lr がそのまま lr_lambda 比率になる
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "weight_decay_decay_phase": 0.0,
        "scheduler": "wsd",
        "warmup_steps": 10,
        "total_steps": 100,
        "decay_start_ratio": 0.5,
        "decay_end_ratio": 0.8,
        "min_lr_ratio": 0.02,
    }
    cfg.update(over)
    return cfg


def _step_to(scheduler, target_last_epoch):
    while scheduler.last_epoch < target_last_epoch:
        scheduler.step()


def test_wsd_weight_decay_switches_exactly_at_decay_start():
    cfg = _wsd_cfg()
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer([p], cfg)
    sched = build_scheduler(opt, cfg)
    _step_to(sched, 49)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert sched.get_last_lr()[0] == pytest.approx(1.0, abs=1e-6)
    _step_to(sched, 50)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.0)
    # decay_end (80) で下限、以降一定
    _step_to(sched, 80)
    assert sched.get_last_lr()[0] == pytest.approx(0.02, abs=1e-6)
    _step_to(sched, 95)
    assert sched.get_last_lr()[0] == pytest.approx(0.02, abs=1e-6)


def test_wsd_scheduler_state_dict_roundtrip_restores_wd():
    cfg = _wsd_cfg()
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer([p], cfg)
    sched = build_scheduler(opt, cfg)
    _step_to(sched, 60)  # decay 区間 (WD=0)
    state = sched.state_dict()

    p2 = torch.nn.Parameter(torch.zeros(4))
    opt2 = build_optimizer([p2], cfg)
    sched2 = build_scheduler(opt2, cfg)
    sched2.load_state_dict(state)
    assert sched2.last_epoch == 60
    # 復元後に両者を 1 step 進め、LR/WD が一致すること (WD は last_epoch から再計算)
    sched.step()
    sched2.step()
    assert opt2.param_groups[0]["weight_decay"] == pytest.approx(0.0)
    assert sched2.get_last_lr()[0] == pytest.approx(sched.get_last_lr()[0], abs=1e-6)


def test_unknown_scheduler_is_error():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = build_optimizer([p], {"optimizer": "adamw", "state_precision": "fp32",
                                "lr": 1e-3, "weight_decay": 0.1})
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_scheduler(opt, _wsd_cfg(scheduler="two_stage"))


# ---- Muon ---------------------------------------------------------------------
def _tiny_byte_lm(bitnet: bool = False):
    from src.model.arbor import ByteLM

    torch.manual_seed(0)
    return ByteLM({
        "vocab_size": 260, "max_bytes": 64, "hidden_size": 32, "num_heads": 4,
        "num_kv_heads": 2, "intermediate_size": 48, "num_hidden_layers": 2,
        "bitnet": bitnet,
    })


def test_newton_schulz_orthogonalizes_singular_values():
    from src.train.optim import _newton_schulz_orthogonalize

    torch.manual_seed(1)
    for shape in ((16, 48), (48, 16), (32, 32)):
        g = torch.randn(shape)
        o = _newton_schulz_orthogonalize(g, steps=5).float()
        assert o.shape == g.shape
        sv = torch.linalg.svdvals(o)
        # KellerJordan 係数は 5 反復で特異値を概ね [0.7, 1.2] に寄せる (厳密な 1 ではない)
        assert sv.min() > 0.5 and sv.max() < 1.3, sv


def test_muon_param_split_targets_block_2d_weights_only():
    from src.train.optim import muon_param_split

    model = _tiny_byte_lm()
    muon, adamw = muon_param_split(model)
    names = dict((id(p), n) for n, p in model.named_parameters())
    muon_names = sorted(names[id(p)] for p in muon)
    adamw_names = sorted(names[id(p)] for p in adamw)
    assert all(".attn.w" in n or ".ffn." in n for n in muon_names), muon_names
    assert all(p.ndim == 2 for p in muon)
    assert "embed.weight" in adamw_names and "head.weight" in adamw_names
    assert all("norm" in n or n in ("embed.weight", "head.weight") for n in adamw_names), adamw_names
    assert len(muon) + len(adamw) == len(list(model.parameters()))


def test_muon_step_updates_both_groups_and_matches_rms_scale():
    from src.train.optim import Muon, muon_param_split

    model = _tiny_byte_lm()
    muon, adamw = muon_param_split(model)
    opt = Muon(muon, adamw, lr=1e-2, weight_decay=0.0, param_rounding="nearest")
    assert len(opt.param_groups) == 2 and opt.param_groups[0]["use_muon"]
    before = {id(p): p.detach().clone() for p in model.parameters()}
    x = torch.randint(0, 260, (2, 16))
    model(x).logits.float().logsumexp(-1).mean().backward()
    opt.step()
    for p in muon:
        delta = (p.detach() - before[id(p)]).float()
        assert delta.abs().sum() > 0
        # 更新 RMS ≈ lr · 0.2 · √max(A,B) · rms(直交行列) = lr · 0.2 · √(max/min) 程度
        rows, cols = p.shape
        expected = 1e-2 * 0.2 * (max(rows, cols) / min(rows, cols)) ** 0.5
        rms = delta.pow(2).mean().sqrt().item()
        assert 0.4 * expected < rms < 1.6 * expected, (p.shape, rms, expected)
        assert opt.state[p]["momentum_buffer"].dtype == torch.float32
    for p in adamw:
        assert (p.detach() - before[id(p)]).abs().sum() > 0
        assert opt.state[p]["exp_avg"].dtype == torch.float32


def test_muon_skips_params_without_grad_and_rejects_non_2d():
    from src.train.optim import Muon

    w = torch.nn.Parameter(torch.randn(4, 8))
    b = torch.nn.Parameter(torch.zeros(4))
    opt = Muon([w], [b], lr=1e-2)
    w.grad = None
    b.grad = torch.ones_like(b)
    opt.step()
    assert w not in opt.state or len(opt.state[w]) == 0
    assert "exp_avg" in opt.state[b]
    with pytest.raises(ValueError, match="2D"):
        Muon([b], [w], lr=1e-2)
    with pytest.raises(ValueError, match="Block"):
        Muon([], [w], lr=1e-2)


def test_muon_state_dict_roundtrip_keeps_fp32_state_on_bf16_params():
    from src.train.optim import Muon

    w = torch.nn.Parameter(torch.randn(4, 8).to(torch.bfloat16))
    b = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    opt = Muon([w], [b], lr=1e-2, param_rounding="stochastic")
    w.grad = torch.randn_like(w)
    b.grad = torch.randn_like(b)
    opt.step()
    sd = copy.deepcopy(opt.state_dict())
    w2 = torch.nn.Parameter(w.detach().clone())
    b2 = torch.nn.Parameter(b.detach().clone())
    opt2 = Muon([w2], [b2], lr=1e-2, param_rounding="stochastic")
    opt2.load_state_dict(sd)
    assert opt2.state[w2]["momentum_buffer"].dtype == torch.float32
    assert opt2.state[b2]["exp_avg_sq"].dtype == torch.float32
    torch.testing.assert_close(opt2.state[w2]["momentum_buffer"], opt.state[w]["momentum_buffer"])


def test_build_optimizer_muon_requires_model_and_fp32_state():
    from src.train.optim import Muon

    model = _tiny_byte_lm()
    cfg = {"optimizer": "muon", "lr": 1e-3, "weight_decay": 0.1, "muon_adamw_lr": 5e-4}
    with pytest.raises(ValueError, match="model="):
        build_optimizer(model.parameters(), cfg)
    with pytest.raises(ValueError, match="fp32"):
        build_optimizer(model.parameters(), {**cfg, "state_precision": "int8"}, model=model)
    opt = build_optimizer(model.parameters(), cfg, model=model)
    assert isinstance(opt, Muon)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[1]["lr"] == pytest.approx(5e-4)
    # scheduler は 2 group に同じ倍率を掛ける (adamw_lr の比を保つ)
    sched = build_scheduler(opt, {"total_steps": 100, "warmup_steps": 10, "min_lr_ratio": 0.1})
    for _ in range(5):
        opt.step()
        sched.step()
    lrs = sched.get_last_lr()
    assert lrs[0] == pytest.approx(1e-3 * 0.5) and lrs[1] == pytest.approx(5e-4 * 0.5)


def _wsd_run(sched_cfg: dict, steps: int, progress=None):
    """(lr, weight_decay) の step ごとの列を返す."""
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0, weight_decay=0.5)
    sched = build_scheduler(opt, sched_cfg, progress=progress)
    out = []
    for _ in range(steps):
        opt.step()
        sched.step()
        out.append((opt.param_groups[0]["lr"], opt.param_groups[0]["weight_decay"]))
    return out


def test_scheduler_wsd_shape_and_weight_decay():
    cfg = {"scheduler": "wsd", "total_steps": 100, "warmup_steps": 10, "min_lr_ratio": 0.1,
           "decay_start_ratio": 0.8, "decay_end_ratio": 1.0, "decay_shape": "inv_sqrt",
           "weight_decay": 0.1, "weight_decay_decay_phase": 0.0}
    hist = _wsd_run(cfg, 100)
    at = lambda step: hist[step - 1]  # noqa: E731  (hist[i] = step i+1 の lr)
    assert at(10) == pytest.approx((1.0, 0.1))           # warmup 終了 = ピーク、WD は stable 値
    assert at(50) == pytest.approx((1.0, 0.1))           # stable: 一定
    assert at(79) == pytest.approx((1.0, 0.1))           # decay 直前もピーク
    assert at(80) == pytest.approx((1.0, 0.0))           # decay 開始 (p=0): lr はピーク、WD 切替
    assert at(81)[0] < 1.0
    # 1-sqrt: 区間中間 (p=0.5) は 0.1 + 0.9*(1-sqrt(0.5))
    assert at(90)[0] == pytest.approx(0.1 + 0.9 * (1 - 0.5 ** 0.5))
    assert at(100)[0] == pytest.approx(0.1)              # 終端 = 下限
    lrs = [lr for lr, _ in hist[79:]]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))     # 単調減少


@pytest.mark.parametrize("shape", ["linear", "cosine"])
def test_scheduler_wsd_decay_shapes(shape):
    cfg = {"scheduler": "wsd", "total_steps": 100, "warmup_steps": 10, "min_lr_ratio": 0.1,
           "decay_start_ratio": 0.8, "decay_shape": shape, "weight_decay": 0.1}
    hist = _wsd_run(cfg, 100)
    assert hist[90 - 1][0] == pytest.approx(0.55)        # 中間 (p=0.5) はどちらも (1+0.1)/2
    assert hist[100 - 1][0] == pytest.approx(0.1)


def test_scheduler_wsd_extension_keeps_stable_history():
    """stable 区間で total_steps を増やしても、過去の lr / WD 履歴が変わらないこと
    (延長可能性の要件)。cosine ではこれが成り立たない。"""
    base = {"scheduler": "wsd", "warmup_steps": 10, "min_lr_ratio": 0.1,
            "decay_start_ratio": 0.8, "weight_decay": 0.1}
    # total 100 では step 80 から decay。それより前 (step 79 まで) は total 200 と完全一致
    short = _wsd_run(dict(base, total_steps=100), 79)
    longer = _wsd_run(dict(base, total_steps=200), 79)
    assert short == longer
    # 対照: cosine は同じ step でも total で lr が違う
    cs = {"scheduler": "cosine_warmup", "warmup_steps": 10, "min_lr_ratio": 0.1}
    assert _wsd_run(dict(cs, total_steps=100), 40)[-1][0] != pytest.approx(
        _wsd_run(dict(cs, total_steps=200), 40)[-1][0]
    )


def test_scheduler_wsd_validation():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError, match="decay_start_ratio"):
        build_scheduler(opt, {"scheduler": "wsd", "total_steps": 100, "decay_start_ratio": 0.9,
                              "decay_end_ratio": 0.8})
    with pytest.raises(ValueError, match="decay_shape"):
        build_scheduler(opt, {"scheduler": "wsd", "total_steps": 100, "decay_shape": "exp"})
