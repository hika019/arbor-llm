"""Alternating eager/fused AdamW benchmark on actual model parameter shapes.

python -m scripts.bench_adamw --config configs/arbor.yaml --iters 30
Measures optimizer only, not training throughput. No checkpoint/data access.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import yaml

from src.model.arbor import build_arbor
from src.train.optim import AdamWFP32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/arbor.yaml")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 1:
        parser.error("iters and warmup must be positive")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with torch.device("meta"):
        model = build_arbor(cfg["model"])
    shapes = [tuple(p.shape) for p in model.parameters() if p.requires_grad]
    del model
    torch.manual_seed(42)
    params = [torch.nn.Parameter(torch.randn(s, device="cuda", dtype=torch.bfloat16))
              for s in shapes]
    for p in params:
        p.grad = torch.randn_like(p)
    opt_cfg = cfg["optim"]
    opt = AdamWFP32(params, lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"]),
                    eps=opt_cfg["eps"], weight_decay=opt_cfg["weight_decay"])
    results = []
    for backend in ("eager", "triton", "triton", "eager"):
        opt.backend = backend
        for _ in range(args.warmup):
            opt.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            opt.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000 / args.iters
        result = {"backend": backend, "wall_ms": ms}
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                      "parameters": sum(p.numel() for p in params),
                      "tensors": len(params), "results": results,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()}))


if __name__ == "__main__":
    main()
