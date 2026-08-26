from .base import LearningPipeline, LearningPipelineDescriptor
from .registry import create_pipeline, pipeline_descriptors, register_pipeline
from .lobe_residual import register_lobe_residual_pipelines
from .p1_evaluator import register_p1_pipelines
from .unified_neural import register_unified_neural_pipelines
from .nvidia_neural_appearance import register_nvidia_neural_appearance_pipeline

register_p1_pipelines()
register_lobe_residual_pipelines()
register_unified_neural_pipelines()
register_nvidia_neural_appearance_pipeline()

__all__ = [
    "LearningPipeline",
    "LearningPipelineDescriptor",
    "create_pipeline",
    "pipeline_descriptors",
    "register_pipeline",
]
