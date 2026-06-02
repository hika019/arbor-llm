"""学習スループット計測。tokens/sec を直近 N step の移動平均で見る。"""
from __future__ import annotations

import time
from collections import deque


class ThroughputMeter:
    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._times: deque[float] = deque(maxlen=window)
        self._tokens: deque[int] = deque(maxlen=window)
        self._t0 = time.perf_counter()

    def step(self, tokens: int) -> None:
        now = time.perf_counter()
        self._times.append(now - self._t0)
        self._tokens.append(tokens)
        self._t0 = now

    def tokens_per_sec(self) -> float:
        dt = sum(self._times)
        return sum(self._tokens) / dt if dt > 0 else 0.0
