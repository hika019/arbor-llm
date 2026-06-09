from __future__ import annotations

import torch

from src.train.optim import Lion, build_optimizer


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
