"""gradient accumulation 段数の step スケジュール (batch size warmup).

序盤は臨界 batch size (critical batch size) が小さく、大きな実効バッチは token
効率を落とすだけなので、accumulation 段数を小さく始めて step で段階的に増やす。
Ai2 "Critical Batch Size Revisited" (arXiv 2505.23971) は OLMo 1B でこの方式により
同 loss を 43% 少ない optimizer step で達成した。batch を上げるときの lr は
√(batch 比) で合わせる (同論文の square-root scaling rule)。

設定 (speed セクション):
    grad_accum_steps: 32                       # 定常値 (schedule の最終値と一致させる)
    grad_accum_schedule: [[0, 8], [10000, 16], [30000, 32]]   # [step, accum]、step 昇順
    grad_accum_lr_scaling: sqrt                # sqrt | none

schedule は step 関数 (区間の開始 step で切り替え、補間しない)。step で決まるので
resume は決定的。compile の単位は micro-step なので accum が変わっても再コンパイル
は起きない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_LR_SCALINGS = ("sqrt", "none")


@dataclass(frozen=True)
class GradAccumSchedule:
    """(start_step, accum) の昇順列。最後の区間の accum が定常値."""

    points: tuple[tuple[int, int], ...]
    lr_scaling: str = "sqrt"

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("grad_accum_schedule は 1 点以上必要")
        if self.points[0][0] != 0:
            raise ValueError(
                f"grad_accum_schedule の最初の step は 0 にすること: {self.points[0]}"
            )
        prev = -1
        for step, accum in self.points:
            if step <= prev:
                raise ValueError(f"grad_accum_schedule の step は昇順にすること: {self.points}")
            if accum < 1:
                raise ValueError(f"grad_accum_schedule の accum は 1 以上: {(step, accum)}")
            prev = step
        if self.lr_scaling not in _LR_SCALINGS:
            raise ValueError(
                f"grad_accum_lr_scaling は {' | '.join(_LR_SCALINGS)} から選ぶ: {self.lr_scaling!r}"
            )

    @property
    def final_accum(self) -> int:
        return self.points[-1][1]

    @property
    def max_accum(self) -> int:
        return max(accum for _, accum in self.points)

    @property
    def is_constant(self) -> bool:
        return len(self.points) == 1

    def accum_at(self, step: int) -> int:
        accum = self.points[0][1]
        for start, value in self.points:
            if step < start:
                break
            accum = value
        return accum

    def lr_scale_at(self, step: int) -> float:
        """定常 batch 向けに調整した lr に掛ける係数 (√(accum/final) または 1)."""
        if self.lr_scaling == "none":
            return 1.0
        return math.sqrt(self.accum_at(step) / self.final_accum)

    def scaled(self, factor: int) -> "GradAccumSchedule":
        """全区間の accum を factor 倍する (MPS の micro_batch→accum 振替用)."""
        return GradAccumSchedule(
            tuple((step, accum * factor) for step, accum in self.points),
            self.lr_scaling,
        )

    def describe(self) -> str:
        if self.is_constant:
            return f"constant accum={self.final_accum}"
        segs = ", ".join(f"step>={s}:{a}" for s, a in self.points)
        return f"{segs} lr_scaling={self.lr_scaling}"

    @classmethod
    def from_speed_cfg(cls, speed_cfg: dict[str, Any]) -> "GradAccumSchedule":
        steady = int(speed_cfg.get("grad_accum_steps", 1))
        raw = speed_cfg.get("grad_accum_schedule")
        lr_scaling = str(speed_cfg.get("grad_accum_lr_scaling", "sqrt")).lower()
        if raw is None:
            return cls(((0, steady),), lr_scaling)
        try:
            points = tuple((int(step), int(accum)) for step, accum in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "grad_accum_schedule は [[step, accum], ...] の形で指定する"
            ) from exc
        schedule = cls(points, lr_scaling)
        if schedule.final_accum != steady:
            # 定常値が 2 箇所に書かれて食い違うと bytes/update の見積りが狂うので拒否する
            raise ValueError(
                "grad_accum_schedule の最終 accum と grad_accum_steps を一致させること "
                f"(schedule={schedule.final_accum}, grad_accum_steps={steady})"
            )
        return schedule
