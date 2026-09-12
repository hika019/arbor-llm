"""Regression tests for the raw Triton AdamW kernel launch guard."""
from __future__ import annotations

import pytest
import torch

from src.train import adamw_triton


class _RecordingDeviceGuard:
    def __init__(self, device, log):
        self._device = device
        self._log = log

    def __enter__(self):
        self._log.append(("enter", self._device))
        return self

    def __exit__(self, exc_type, exc, tb):
        self._log.append(("exit", self._device))
        return False


class _RecordingKernel:
    def __init__(self, log):
        self._log = log

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self._log.append(("launch", grid, args, kwargs))
            return None

        return launch


def test_adamw_update_launches_kernel_inside_parameter_device_guard(monkeypatch):
    # CPU-only regression test exercising the launch-guard contract with both
    # the device context manager and the raw kernel mocked out.
    p = torch.zeros(7)
    grad = torch.ones_like(p)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)

    log = []

    def fake_device(device):
        return _RecordingDeviceGuard(device, log)

    fake_kernel = _RecordingKernel(log)

    class _FakeTriton:
        @staticmethod
        def cdiv(x, y):
            return (x + y - 1) // y

    monkeypatch.setattr(adamw_triton, "triton", _FakeTriton)
    monkeypatch.setattr(adamw_triton, "_adamw_kernel", fake_kernel)
    monkeypatch.setattr(torch.cuda, "device", fake_device, raising=False)

    adamw_triton.adamw_update(
        p, grad, m, v, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0, step=1
    )

    assert [event[0] for event in log] == ["enter", "launch", "exit"]
    assert log[0][1] == p.device
    assert log[2][1] == p.device
    grid, args, kwargs = log[1][1], log[1][2], log[1][3]
    assert grid == (1,)
    assert args[:5] == (p, grad, m, v, p.numel())
    assert kwargs["BLOCK"] == 1024


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least two CUDA devices"
)
def test_adamw_update_launches_on_parameter_device_across_gpus():
    if adamw_triton.triton is None:
        pytest.skip("Triton not importable")

    device = torch.device("cuda:1")
    torch.cuda.set_device(torch.device("cuda:0"))
    p = torch.randn(4096, device=device, dtype=torch.float32)
    grad = torch.randn_like(p)
    m = torch.randn_like(p)
    v = torch.randn_like(p).abs().add_(1e-6)
    if not adamw_triton.supported(p, grad):
        pytest.skip("Triton CUDA backend not available")

    p_expected = p.clone()
    grad0 = grad.clone()
    m0 = m.clone()
    v0 = v.clone()
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-8
    wd = 0.0
    step = 1
    decay = 1.0 - lr * wd
    alpha1 = 1.0 - beta1
    alpha2 = 1.0 - beta2
    correction2 = (1.0 - beta2**step) ** 0.5
    step_size = lr / (1.0 - beta1**step)
    with torch.cuda.device(device):
        m_ref = m0 * beta1 + grad0 * alpha1
        v_ref = v0 * beta2 + alpha2 * grad0 * grad0
        denom = v_ref.sqrt().div_(correction2).add_(eps)
        update = m_ref.div(denom).mul_(step_size)
        p_expected = (p_expected * decay).sub_(update)

    assert torch.cuda.current_device() == 0
    adamw_triton.adamw_update(
        p, grad, m, v, lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd, step=step
    )

    # Launched on cuda:1 while the caller's current device was cuda:0; the
    # guard must both update the right storage and restore the caller device.
    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(p, p_expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(m, m_ref, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(v, v_ref, rtol=1e-6, atol=1e-10)
