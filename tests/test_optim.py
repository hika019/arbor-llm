from __future__ import annotations

import pytest
import torch

from src.train.optim import Lion, build_optimizer, build_scheduler


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


def test_scheduler_min_lr_ratio_validation():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError):
        build_scheduler(opt, {"total_steps": 100, "min_lr_ratio": 1.0})
