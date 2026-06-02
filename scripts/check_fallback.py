"""WSL2 上で NVIDIA driver の System Memory Fallback が OFF か確認.

VRAM を超える tensor 確保を試し:
  - 即 OOM → fallback OFF (期待)
  - 成功 → fallback ON, RAM に溢れている (パフォーマンス激落ち)
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM = {vram:.2f} GB")

    # VRAM を 1.5x 超える size で試す
    test_gb = int(vram * 1.5 + 1)
    print(f"-> {test_gb} GB の bfloat16 tensor 確保＋書き込みを試行")
    try:
        t0 = time.perf_counter()
        n = int(test_gb * 2**30 / 2)
        x = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        x.fill_(1.0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"成功 in {dt*1000:.0f} ms")
        print("=> System Memory Fallback はまだ ON (要 OFF 設定 + 再起)")
        return 2
    except torch.cuda.OutOfMemoryError as e:
        print("OOM (期待挙動): System Memory Fallback OFF 確定")
        print(f"  msg: {str(e)[:200]}")
        return 0
    except RuntimeError as e:
        # WSL2 + expandable_segments では超過確保が cudaErrorNotReady 等の
        # driver error になることがある. RAM へ溢れていない＝fallback OFF と判断.
        print("RuntimeError (RAM へ溢れず失敗): System Memory Fallback OFF と判断")
        print(f"  msg: {str(e)[:200]}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
