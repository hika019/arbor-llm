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
        "grad_accum_steps": [[0, 8], [100, 16], [300, 32]],
    })
    assert not sched.is_constant
    assert [sched.accum_at(s) for s in (0, 99, 100, 299, 300, 10_000)] == [8, 8, 16, 16, 32, 32]
    assert sched.max_accum == 32
    assert sched.lr_scale_at(0) == pytest.approx(math.sqrt(8 / 32))
    assert sched.lr_scale_at(100) == pytest.approx(math.sqrt(16 / 32))
    assert sched.lr_scale_at(300) == 1.0


def test_lr_scaling_none_keeps_lr():
    sched = GradAccumSchedule.from_speed_cfg({
        "grad_accum_steps": [[0, 1], [10, 4]],
        "grad_accum_lr_scaling": "none",
    })
    assert sched.lr_scale_at(0) == 1.0


@pytest.mark.parametrize(
    "speed_cfg, message",
    [
        ({"grad_accum_steps": 32, "grad_accum_schedule": [[0, 8], [100, 32]]}, "廃止"),
        ({"grad_accum_steps": [[5, 8], [100, 16]]}, "step 0"),
        ({"grad_accum_steps": [[0, 8], [100, 4], [50, 16]]}, "昇順"),
        ({"grad_accum_steps": [[0, 0], [100, 16]]}, "1 以上"),
        ({"grad_accum_steps": [[0, 8], [100, 16]], "grad_accum_lr_scaling": "linear"}, "grad_accum_lr_scaling"),
        ({"grad_accum_steps": [8, 16]}, "形"),
    ],
)
def test_invalid_schedules_are_rejected(speed_cfg, message):
    with pytest.raises(ValueError, match=message):
        GradAccumSchedule.from_speed_cfg(speed_cfg)


@pytest.mark.parametrize("name", ["cosine_warmup", "wsd"])
def test_build_scheduler_applies_lr_scale_to_every_scheduler(name):
    sched = GradAccumSchedule.from_speed_cfg({
        "grad_accum_steps": [[0, 1], [20, 4]],
    })
    optim_cfg = {
        "scheduler": name,
        "lr": 1e-3,
        "warmup_steps": 5,
        "total_steps": 100,
        "min_lr_ratio": 0.1,
        "decay_end_ratio": 0.8,
        "decay_start_ratio": 0.5,
        "weight_decay": 0.1,
        "weight_decay_decay_phase": 0.0,
    }

    def run(lr_scale):
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([p], lr=1e-3)
        s = build_scheduler(opt, optim_cfg, lr_scale=lr_scale) if lr_scale else build_scheduler(opt, optim_cfg)
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
            "grad_accum_steps": [[0, 8], [1000, 32]],
        },
    }
    resolved = adapt_config_for_device(cfg, torch.device("mps"))
    assert resolved["speed"]["grad_accum_steps"] == [[0, 16], [1000, 64]]
    # 振替後も schedule と定常値の整合が保たれる
    sched = GradAccumSchedule.from_speed_cfg(resolved["speed"])
    assert sched.final_accum == 64
    assert sched.accum_at(0) == 16


def test_cumulative_accum_and_bytes_fraction():
    sched = GradAccumSchedule.from_speed_cfg({
        "grad_accum_steps": [[0, 1], [10, 2], [15, 4]],
    })
    assert sched.cumulative_accum(0) == 0
    assert sched.cumulative_accum(10) == 10
    assert sched.cumulative_accum(15) == 10 + 5 * 2
    assert sched.cumulative_accum(20) == 20 + 5 * 4
    frac = sched.bytes_fraction_fn(20)
    assert frac(0) == 0.0
    assert frac(10) == pytest.approx(10 / 40)
    assert frac(15) == pytest.approx(20 / 40)
    assert frac(20) == 1.0 and frac(25) == 1.0
    # 固定 accum では step 割合と一致
    const = GradAccumSchedule.from_speed_cfg({"grad_accum_steps": 3}).bytes_fraction_fn(50)
    assert const(25) == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["cosine_warmup", "wsd"])
def test_scheduler_progress_by_bytes_matches_constant_run_at_equal_bytes(name):
    """accum 1→4 の run は、同じ bytes を消費した時点で固定 accum 4 の run と同じ lr (補正前) になる。"""
    # warmup は step 単位なので、warmup 中に消費する bytes は両者で違う (cosine の起点が
    # bytes 上でわずかにずれる)。等価性を厳密に見るため warmup 0 で比べる。
    optim_cfg = {
        "scheduler": name, "lr": 1e-3, "warmup_steps": 0, "min_lr_ratio": 0.1,
        "decay_end_ratio": 0.8, "decay_start_ratio": 0.5,
        "weight_decay": 0.1, "weight_decay_decay_phase": 0.0,
    }
    # 固定 accum 4 を 100 step = 400 micro-step。schedule 側は accum 1 を 100 step + accum 4 を 75 step = 400 micro-step。
    sched = GradAccumSchedule.from_speed_cfg({
        "grad_accum_steps": [[0, 1], [100, 4]],
    })

    def curve(total, progress, steps):
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([p], lr=1e-3)
        s = build_scheduler(opt, {**optim_cfg, "total_steps": total}, progress=progress)
        out = []
        for _ in range(steps):
            out.append(s.get_last_lr()[0])
            opt.step()
            s.step()
        return out

    const = curve(100, None, 100)
    warm = curve(175, sched.bytes_fraction_fn(175), 175)
    # schedule run の step k>=100 は bytes = 100 + 4(k-100) micro-step、固定 run の step (25 + (k-100)) と同じ bytes
    for k in range(100, 175):
        assert warm[k] == pytest.approx(const[25 + (k - 100)], rel=1e-6), (k, name)
    # 序盤 (accum 1) は bytes が少ないので lr の減衰が遅い (wsd は stable 区間でどちらもピーク)
    if name == "wsd":
        assert warm[50] == pytest.approx(const[50]) == pytest.approx(1e-3)
    else:
        assert warm[50] > const[50]
