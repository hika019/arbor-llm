"""gradient accumulation 段数の step スケジュール (batch size warmup).

序盤は臨界 batch size (critical batch size) が小さく、大きな実効バッチは token
効率を落とすだけなので、accumulation 段数を小さく始めて step で段階的に増やす。
Ai2 "Critical Batch Size Revisited" (arXiv 2505.23971) は OLMo 1B でこの方式により
同 loss を 43% 少ない optimizer step で達成した。batch を上げるときの lr は
√(batch 比) で合わせる (同論文の square-root scaling rule)。

設定 (speed セクション): grad_accum_steps は固定値か step 関数のどちらか。
    grad_accum_steps: 32                                   # 固定
    grad_accum_steps: [[0, 8], [10000, 16], [30000, 32]]   # [step, accum]、step 昇順 (最終値が定常値)
    grad_accum_lr_scaling: sqrt                            # sqrt | none

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
            raise ValueError("grad_accum_steps の schedule は 1 点以上必要")
        if self.points[0][0] != 0:
            raise ValueError(
                f"grad_accum_steps の schedule は step 0 から始めること: {self.points[0]}"
            )
        prev = -1
        for step, accum in self.points:
            if step <= prev:
                raise ValueError(f"grad_accum_steps の schedule は step 昇順にすること: {self.points}")
            if accum < 1:
                raise ValueError(f"grad_accum_steps の accum は 1 以上: {(step, accum)}")
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

    def cumulative_accum(self, step: int) -> int:
        """step 0..step-1 の accum の総和 (= micro-step 数 ∝ 消費 bytes)."""
        total = 0
        for i, (start, accum) in enumerate(self.points):
            if step <= start:
                break
            end = self.points[i + 1][0] if i + 1 < len(self.points) else step
            total += accum * (min(step, end) - start)
        return total

    def bytes_fraction_fn(self, total_steps: int):
        """step → 消費 bytes の割合 [0, 1] を返す関数。

        accum が変わる run では step 割合と bytes 割合がずれる (序盤ほど bytes/step が
        小さい)。lr の cosine / decay_start / decay_end の進行を bytes 割合で測ることで、
        固定 accum の run と「同じ bytes で同じ lr」になる (batch size warmup の
        A/B が lr schedule の違いに汚染されないため)。
        """
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1: {total_steps}")
        denom = float(self.cumulative_accum(total_steps))

        def fraction(step: int) -> float:
            return min(1.0, self.cumulative_accum(max(0, step)) / denom)

        return fraction

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
        if "grad_accum_schedule" in speed_cfg:
            raise ValueError(
                "speed.grad_accum_schedule は廃止。grad_accum_steps に [[step, accum], ...] を書く"
            )
        raw = speed_cfg.get("grad_accum_steps", 1)
        lr_scaling = str(speed_cfg.get("grad_accum_lr_scaling", "sqrt")).lower()
        if isinstance(raw, (int, float, str)):
            return cls(((0, int(raw)),), lr_scaling)
        try:
            points = tuple((int(step), int(accum)) for step, accum in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "grad_accum_steps は整数か [[step, accum], ...] の形で指定する"
            ) from exc
        return cls(points, lr_scaling)
