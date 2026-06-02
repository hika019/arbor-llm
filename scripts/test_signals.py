"""StopFlag の挙動を自己 SIGINT で検証する.

期待: SIGINT 1 回で stop.requested=True、ループは次境界で抜ける.
"""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.signals import StopFlag


def main() -> int:
    stop = StopFlag()
    # 0.3 秒後に自分自身へ SIGINT を投げる
    if os.fork() == 0:
        time.sleep(0.3)
        os.kill(os.getppid(), signal.SIGINT)
        os._exit(0)

    steps = 0
    while True:
        time.sleep(0.05)
        steps += 1
        if stop.requested:
            print(f"[test] stop requested at step={steps} -> OK")
            return 0
        if steps > 200:
            print("[test] FAILED: stop.requested never set")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
