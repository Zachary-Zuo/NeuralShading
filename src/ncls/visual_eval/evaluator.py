"""两个平台共用的进程内图像评估接口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from torch import nn
    from ncls.learning.method import Method
    from ncls.learning.training.config import TrainingConfig
    from ncls.learning.training.plan import VisualEvalSettings


@dataclass(frozen=True)
class VisualContext:
    step: int
    method: Method
    config: TrainingConfig
    source_snapshot_ids: tuple[str, ...]
    settings: VisualEvalSettings
    output: Path


@dataclass(frozen=True)
class VisualResult:
    images: Mapping[str, Path]
    elapsed_seconds: float


class VisualEvaluator(Protocol):
    def evaluate(self, model: nn.Module, context: VisualContext) -> VisualResult | None: ...


class NoVisualEvaluation:
    def evaluate(self, model: nn.Module, context: VisualContext) -> None:
        return None
