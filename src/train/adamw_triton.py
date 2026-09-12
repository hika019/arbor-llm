"""Memory-traffic fusion for Arbor's FP32-moment AdamW (no Tensor Core requirement).

rounding=nearest keeps the eager optimizer's parameter-dtype rounding after
decay and before subtraction. Standard fused AdamW implementations need not
share that contract. rounding=stochastic computes the whole update in FP32 and
rounds once to BF16 with Philox noise in the discarded low 16 bits, so updates
smaller than half an ulp survive in expectation (see src/train/rounding.py).
Hyperparameters and the seed are runtime scalars: changing LR/step/seed must
not trigger JIT.
"""
# Triton's constexpr annotations are JIT DSL values, not Python types.
# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit(do_not_specialize=[
        "decay", "beta1", "beta2", "alpha1", "alpha2", "eps", "correction2", "step_size",
        "seed",
    ])
    def _adamw_kernel(
        P, G, M, V, N: tl.constexpr,
        decay, beta1, beta2, alpha1, alpha2, eps, correction2, step_size, seed,
        BLOCK: tl.constexpr, STOCHASTIC: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        p = tl.load(P + i, mask, other=0).to(tl.float32)
        g = tl.load(G + i, mask, other=0).to(tl.float32)
        m = tl.load(M + i, mask, other=0)
        v = tl.load(V + i, mask, other=0)
        # Match eager's separate mul_ followed by add_/addcmul_; allow
        # explicit FMA only inside those latter operations.
        m = tl.fma(g, alpha1, m * beta1)
        v = tl.fma(alpha2 * g, g, v * beta2)
        denom = libdevice.div_rn(libdevice.sqrt_rn(v), correction2) + eps
        update = libdevice.div_rn(m, denom) * step_size
        dtype: tl.constexpr = P.dtype.element_ty
        if STOCHASTIC:
            # BF16 only (checked by the caller). Add 16 random bits below the
            # BF16 mantissa and truncate: round-up probability equals the
            # discarded fraction, so the rounding is unbiased.
            new = p * decay - update
            bits = new.to(tl.int32, bitcast=True)
            noise = (tl.randint(seed, i) & 0xFFFF).to(tl.int32)
            bits = (bits + noise) & -65536
            new = bits.to(tl.float32, bitcast=True)
            tl.store(P + i, new, mask)
        else:
            p = (p * decay).to(dtype).to(tl.float32)
            update = update.to(dtype).to(tl.float32)
            tl.store(P + i, p - update, mask)
        tl.store(M + i, m, mask)
        tl.store(V + i, v, mask)


def supported(p: torch.Tensor, grad: torch.Tensor) -> bool:
    return (
        triton is not None
        and p.device.type == "cuda"
        # HIP exposes device.type='cuda' too, but its libdevice does not
        # provide the CUDA round-to-nearest division/sqrt used above.
        and torch.version.hip is None
        and p.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and p.is_contiguous()
        and grad.is_contiguous()
    )


def adamw_update(p, grad, m, v, *, lr, beta1, beta2, eps, wd, step,
                 rounding="nearest", seed=0):
    if not p.numel():
        return
    stochastic = rounding == "stochastic" and p.dtype != torch.float32
    if stochastic and p.dtype != torch.bfloat16:
        raise ValueError(f"stochastic rounding requires bf16 parameters, got {p.dtype}")
    # Triton resolves the launch device from torch.cuda.current_device()
    # (driver.active.get_current_device() maps to torch.cuda.current_device).
    # Without this guard a parameter living on a non-current device would be
    # launched on the wrong device. torch.cuda.device only swaps the active
    # device and restores it on exit, so the caller's device/stream are
    # preserved and the kernel runs on p.device's current stream.
    with torch.cuda.device(p.device):
        _adamw_kernel[(triton.cdiv(p.numel(), 1024),)](
            p, grad, m, v, p.numel(),
            1.0 - lr * wd, beta1, beta2, 1.0 - beta1, 1.0 - beta2, eps,
            (1.0 - beta2**step) ** 0.5, lr / (1.0 - beta1**step), int(seed),
            BLOCK=1024, STOCHASTIC=stochastic, enable_fp_fusion=False,
        )
    # Raw kernels mutate storage outside the dispatcher. Preserve autograd's
    # stale-saved-tensor detection just as eager in-place optimizer ops do.
    torch.autograd.graph.increment_version((p, m, v))
