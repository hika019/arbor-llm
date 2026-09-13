from __future__ import annotations

import math

import pytest
import torch

from src.train.grad_accum import GradAccumSchedule
from src.train.optim import build_scheduler
from src.train.train import adapt_config_for_device


def test_constant_schedule_when_config_has_no_schedule():
    sched = GradAccumSchedule.from_speed_cfg({"grad_accum_steps": 32})
    assert sched.is_constant
    assert sched.final_accum == 32
    assert sched.accum_at(0) == 32
    assert sched.accum_at(10**9) == 32
    assert sched.lr_scale_at(0) == 1.0


def test_schedule_is_step_function_and_lr_scales_by_sqrt():
    sched = GradAccumSchedule.from_speed_cfg({
        "grad_accum_steps": 32,
        "grad_accum_schedule": [[0, 8], [100, 16], [300, 32]],
    })
    assert not sched.is_constant
    assert [sched.accum_at(s) for s in (0, 99, 100, 299, 300, 10_000)] == [8, 8, 16, 16, 32, 32]
    assert sched.max_accum == 32
    assert sched.lr_scale_at(0) == pytest.approx(math.sqrt(8 / 32))
    assert sched.lr_scale_at(100) == pytest.approx(math.sqrt(16 / 32))
    assert sched.lr_scale_at(300) == 1.0


def test_lr_scaling_none_keeps_lr():
    sched = GradAccumSchedule.from_speed_cfg({
        "grad_accum_steps": 4,
        "grad_accum_schedule": [[0, 1], [10, 4]],
        "grad_accum_lr_scaling": "none",
    })
    assert sched.lr_scale_at(0) == 1.0


@pytest.mark.parametrize(
    "speed_cfg, message",
    [
        ({"grad_accum_steps": 32, "grad_accum_schedule": [[0, 8], [100, 16]]}, "一致"),
        ({"grad_accum_steps": 16, "grad_accum_schedule": [[5, 8], [100, 16]]}, "最初の step"),
        ({"grad_accum_steps": 16, "grad_accum_schedule": [[0, 8], [100, 4], [50, 16]]}, "昇順"),
        ({"grad_accum_steps": 16, "grad_accum_schedule": [[0, 0], [100, 16]]}, "1 以上"),
        ({"grad_accum_steps": 16, "grad_accum_schedule": [[0, 8], [100, 16]], "grad_accum_lr_scaling": "linear"}, "grad_accum_lr_scaling"),
        ({"grad_accum_steps": 16, "grad_accum_schedule": [8, 16]}, "形"),
    ],
)
def test_invalid_schedules_are_rejected(speed_cfg, message):
    with pytest.raises(ValueError, match=message):
        GradAccumSchedule.from_speed_cfg(speed_cfg)


@pytest.mark.parametrize("name", ["cosine_warmup", "two_stage"])
def test_build_scheduler_applies_lr_scale_to_every_scheduler(name):
    sched = GradAccumSchedule.from_speed_cfg({
        "grad_accum_steps": 4,
        "grad_accum_schedule": [[0, 1], [20, 4]],
    })
    optim_cfg = {
        "scheduler": name,
        "lr": 1e-3,
        "warmup_steps": 5,
        "total_steps": 100,
        "min_lr_ratio": 0.1,
        "decay_end_ratio": 0.8,
        "stage2_start_ratio": 0.5,
        "stage2_peak_lr_ratio": 0.5,
        "weight_decay": 0.1,
        "weight_decay_stage2": 0.0,
    }

    def run(lr_scale):
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([p], lr=1e-3)
        s = build_scheduler(opt, optim_cfg, lr_scale=lr_scale)
        lrs = []
        for _ in range(40):
            lrs.append(s.get_last_lr()[0])
            opt.step()
            s.step()
        return lrs

    base = run(None)
    scaled = run(sched.lr_scale_at)
    for step, (b, sc) in enumerate(zip(base, scaled)):
        assert sc == pytest.approx(b * sched.lr_scale_at(step))
    # 定常区間では無補正
    assert scaled[30] == pytest.approx(base[30])
    # warmup 前半は √(1/4) = 0.5 倍
    assert scaled[3] == pytest.approx(base[3] * 0.5)


def test_mps_adaptation_scales_schedule_with_micro_batch():
    cfg = {
        "model": {"gradient_checkpointing": False},
        "optim": {"optimizer": "adamw", "state_precision": "fp32"},
        "speed": {
            "micro_batch_size": 2,
            "grad_accum_steps": 32,
            "grad_accum_schedule": [[0, 8], [1000, 32]],
        },
    }
    resolved = adapt_config_for_device(cfg, torch.device("mps"))
    assert resolved["speed"]["grad_accum_steps"] == 64
    assert resolved["speed"]["grad_accum_schedule"] == [[0, 16], [1000, 64]]
    # 振替後も schedule と定常値の整合が保たれる
    sched = GradAccumSchedule.from_speed_cfg(resolved["speed"])
    assert sched.final_accum == 64
    assert sched.accum_at(0) == 16
