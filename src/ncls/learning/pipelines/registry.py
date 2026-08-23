from __future__ import annotations

from collections.abc import Callable

from .base import LearningPipeline, LearningPipelineDescriptor
from .dense_evaluator import (
    DenseLinearE1Pipeline,
    DenseLog1pE1Pipeline,
    DenseStandardizedLog1pE1Pipeline,
)
from .legacy_ltc_k2 import LegacyLtcK2Pipeline


_FACTORIES: dict[str, Callable[[], LearningPipeline]] = {}


def register_pipeline(factory: Callable[[], LearningPipeline]) -> None:
    pipeline = factory()
    pipeline_id = pipeline.descriptor.pipeline_id
    if pipeline_id in _FACTORIES:
        raise ValueError(f"learning pipeline {pipeline_id!r} is already registered")
    _FACTORIES[pipeline_id] = factory


def create_pipeline(pipeline_id: str) -> LearningPipeline:
    try:
        return _FACTORIES[pipeline_id]()
    except KeyError as error:
        raise ValueError(f"unsupported learning pipeline {pipeline_id!r}") from error


def pipeline_descriptors() -> tuple[LearningPipelineDescriptor, ...]:
    return tuple(_FACTORIES[pipeline_id]().descriptor for pipeline_id in sorted(_FACTORIES))


register_pipeline(LegacyLtcK2Pipeline)
register_pipeline(DenseLinearE1Pipeline)
register_pipeline(DenseLog1pE1Pipeline)
register_pipeline(DenseStandardizedLog1pE1Pipeline)
