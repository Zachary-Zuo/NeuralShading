"""checkpoint 选择策略。

`median_then_p95`：P1 v1 规则，validation directional L1 state-median 优先、p95 决胜。
`tail_guard`：先剔除 validation p95 > 该 run 至今最小 p95 × 1.25 的 checkpoint，再取 median 最小
（`p1_audit.md` §4.2：M2-S 的 best@4500 p95 0.586 被更晚的 7500（p95 0.340）取代）。
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CHECKPOINT_SELECTIONS = ("median_then_p95", "tail_guard")
TAIL_GUARD_P95_RATIO = 1.25


def directional_summary(record: Mapping[str, Any]) -> tuple[float, float]:
    summary = record["primary"]["directional_l1_by_state"]
    return float(summary["median"]), float(summary["p95"])


def checkpoint_score(
    record: Mapping[str, Any],
    minimum_p95: float,
    strategy: str,
) -> tuple[float, float]:
    """越小越好；`tail_guard` 下被剔除的 checkpoint 得 `(inf, p95)`。"""

    if strategy not in CHECKPOINT_SELECTIONS:
        raise ValueError(f"unsupported checkpoint selection {strategy!r}")
    median, p95 = directional_summary(record)
    if strategy == "tail_guard" and p95 > TAIL_GUARD_P95_RATIO * minimum_p95:
        return (math.inf, p95)
    return (median, p95)


def select_checkpoint(
    history: Sequence[Mapping[str, Any]],
    strategy: str,
) -> Mapping[str, Any] | None:
    """按 runner 的在线规则回放 validation history：每次都用「至今最小 p95」重评 best 与当前记录。"""

    best: Mapping[str, Any] | None = None
    minimum_p95 = math.inf
    for record in history:
        minimum_p95 = min(minimum_p95, directional_summary(record)[1])
        if best is None or checkpoint_score(record, minimum_p95, strategy) < checkpoint_score(
            best, minimum_p95, strategy
        ):
            best = record
    return best
