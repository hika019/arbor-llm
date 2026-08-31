from __future__ import annotations

import copy

import pytest
import torch

from src.train.optim import (
    AdamW8bit,
    AdamWBF8,
    Lion,
    build_optimizer,
    build_scheduler,
    resolve_state_precision,
)
from src.train.optim import (
    _quantize_dynamic_state,
    _dequantize_dynamic_state,
)


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


def _two_stage_cfg(**over):
    cfg = {
        "optimizer": "adamw",
        "state_precision": "fp32",
        "lr": 1.0,  # base lr=1 なので get_last_lr がそのまま lr_lambda 比率になる
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "weight_decay_stage2": 0.0,
        "scheduler": "two_stage",
        "warmup_steps": 10,
        "total_steps": 100,
        "stage2_start_ratio": 0.5,
        "stage2_peak_lr_ratio": 0.5,
        "decay_end_ratio": 0.8,
        "min_lr_ratio": 0.02,
    }
    cfg.update(over)
    return cfg


def _step_to(scheduler, target_last_epoch):
    while scheduler.last_epoch < target_last_epoch:
        scheduler.step()


def test_two_stage_scheduler_lr_curve_and_weight_decay():
    cfg = _two_stage_cfg()
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer([p], cfg)
    sched = build_scheduler(opt, cfg)

    # warmup 終了 (step=10): ピーク (=1.0) かつ WD は stage1 (0.1)
    _step_to(sched, 10)
    assert sched.get_last_lr()[0] == pytest.approx(1.0, abs=1e-6)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)

    # stage2 開始 (step=50): 連続で stage2_peak (=0.5)、WD が 0 へ切替
    _step_to(sched, 50)
    assert sched.get_last_lr()[0] == pytest.approx(0.5, abs=1e-6)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.0)

    # cooldown 終了 (step=80): min_lr_ratio (=0.02) に到達
    _step_to(sched, 80)
    assert sched.get_last_lr()[0] == pytest.approx(0.02, abs=1e-6)
    # 以降は下限で一定
    _step_to(sched, 95)
    assert sched.get_last_lr()[0] == pytest.approx(0.02, abs=1e-6)


def test_two_stage_weight_decay_switches_exactly_at_stage2():
    cfg = _two_stage_cfg()
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer([p], cfg)
    sched = build_scheduler(opt, cfg)
    _step_to(sched, 49)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    _step_to(sched, 50)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.0)


def test_two_stage_scheduler_state_dict_roundtrip_restores_wd():
    cfg = _two_stage_cfg()
    p = torch.nn.Parameter(torch.zeros(4))
    opt = build_optimizer([p], cfg)
    sched = build_scheduler(opt, cfg)
    _step_to(sched, 60)  # stage2 (WD=0)
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


def test_two_stage_scheduler_validation():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = build_optimizer([p], {"optimizer": "adamw", "state_precision": "fp32",
                                "lr": 1e-3, "weight_decay": 0.1})
    # decay_end_ratio <= stage2_start_ratio は不正
    with pytest.raises(ValueError):
        build_scheduler(opt, _two_stage_cfg(stage2_start_ratio=0.8, decay_end_ratio=0.8))
    # stage2_peak_lr_ratio が (min_lr_ratio, 1] の外は不正
    with pytest.raises(ValueError):
        build_scheduler(opt, _two_stage_cfg(stage2_peak_lr_ratio=1.5))
    with pytest.raises(ValueError):
        build_scheduler(opt, _two_stage_cfg(stage2_peak_lr_ratio=0.0, min_lr_ratio=0.02))
