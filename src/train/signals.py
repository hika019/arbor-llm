"""SIGINT / SIGTERM handling for safe-stop training.

Usage:
    stop = StopFlag()
    while training:
        ...
        if stop.requested:
            save_checkpoint(...)
            break

First signal sets `stop.requested`. A second signal within the same process
triggers immediate exit (handler raises KeyboardInterrupt).
"""
from __future__ import annotations

import signal
import threading


class StopFlag:
    def __init__(self) -> None:
        self._requested = threading.Event()
        self._force = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def _handle(self, signum, frame):  # noqa: ARG002
        if self._requested.is_set():
            # second hit: force exit
            self._force = True
            raise KeyboardInterrupt(f"forced exit on signal {signum}")
        print(f"\n[signals] received {signal.Signals(signum).name}, "
              "will save checkpoint and exit at next safe boundary. "
              "Press again to force-exit.", flush=True)
        self._requested.set()
