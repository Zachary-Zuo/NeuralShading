from __future__ import annotations

from collections.abc import Callable

from .base import LearningPipeline, LearningPipelineDescriptor


_FACTORIES: dict[str, Callable[[], LearningPipeline]] = {}


def register_pipeline(factory: Callable[[], LearningPipeline]) -> None:
    pipeline = factory()
    name = pipeline.descriptor.name
    if name in _FACTORIES:
        raise ValueError(f"learning pipeline {name!r} is already registered")
    _FACTORIES[name] = factory


def create_pipeline(name: str) -> LearningPipeline:
    try:
        return _FACTORIES[name]()
    except KeyError as error:
        raise ValueError(f"unsupported learning pipeline {name!r}") from error


def pipeline_descriptors() -> tuple[LearningPipelineDescriptor, ...]:
    return tuple(_FACTORIES[name]().descriptor for name in sorted(_FACTORIES))
