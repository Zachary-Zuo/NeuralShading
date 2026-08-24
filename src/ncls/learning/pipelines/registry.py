from __future__ import annotations

from collections.abc import Callable

from .base import LearningPipeline, LearningPipelineDescriptor
from .dense_evaluator import (
    AnalyticResidualE1Pipeline,
    AnalyticResidualEnergyShapeE1Pipeline,
    DenseLinearE1Pipeline,
    DenseLog1pE1Pipeline,
    DenseEnergyShapeE1Pipeline,
    DenseStandardizedLog1pE1Pipeline,
)
from .legacy_ltc_k2 import LegacyLtcK2Pipeline
from .plane_factorized import (
    PlaneFactorizedAnalyticResidualE1Pipeline,
    PlaneFactorizedEnergyShapeE1Pipeline,
)
from .shared_evaluator import (
    AnalyticResidualSharedEvaluatorE2Pipeline,
    DenseLatentSharedEvaluatorE2Pipeline,
    PerStateAnalyticResidualSharedEvaluatorE2Pipeline,
    SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline,
    NoiseAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline,
    BoundaryCapacityPerStateAnalyticResidualSharedEvaluatorE2Pipeline,
    FactorizedLatentAnalyticResidualSharedEvaluatorE2Pipeline,
    SparseDictionaryAnalyticResidualSharedEvaluatorE2Pipeline,
    TargetTensorEncoderAnalyticResidualSharedEvaluatorE2Pipeline,
    TargetEncoderRefinementAnalyticResidualSharedEvaluatorE2Pipeline,
    TargetEncoderSeTailRefinementAnalyticResidualSharedEvaluatorE2Pipeline,
)
from .source_compiler import LayerStackSourceCompilerAnalyticResidualE3Pipeline


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
register_pipeline(AnalyticResidualE1Pipeline)
register_pipeline(AnalyticResidualEnergyShapeE1Pipeline)
register_pipeline(DenseEnergyShapeE1Pipeline)
register_pipeline(PlaneFactorizedEnergyShapeE1Pipeline)
register_pipeline(PlaneFactorizedAnalyticResidualE1Pipeline)
register_pipeline(DenseLatentSharedEvaluatorE2Pipeline)
register_pipeline(AnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(PerStateAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(SourceAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(NoiseAwarePerStateAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(BoundaryCapacityPerStateAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(SparseDictionaryAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(FactorizedLatentAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(TargetTensorEncoderAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(TargetEncoderRefinementAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(TargetEncoderSeTailRefinementAnalyticResidualSharedEvaluatorE2Pipeline)
register_pipeline(LayerStackSourceCompilerAnalyticResidualE3Pipeline)
