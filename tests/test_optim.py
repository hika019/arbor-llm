from __future__ import annotations

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


def test_adamw_8bit_keeps_moments_as_int8():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    opt = AdamW8bit([p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)

    p.grad = torch.tensor([0.25, -0.5, 1.0])
    opt.step()

    state = opt.state[p]
    assert state["exp_avg"].dtype == torch.int8
    assert state["exp_avg_sq"].dtype == torch.int8
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
        assert opt.state[p]["exp_avg"].dtype == torch.int8
        assert opt.state[p]["exp_avg_sq"].dtype == torch.int8


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

    assert state["exp_avg"].dtype == torch.int8
    assert state["exp_avg_sq"].dtype == torch.int8
    assert state["exp_avg_scale"].dtype == torch.float32
    assert state["exp_avg_sq_scale"].dtype == torch.float32
    p2.grad = torch.ones_like(p2)
    opt2.step()


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
