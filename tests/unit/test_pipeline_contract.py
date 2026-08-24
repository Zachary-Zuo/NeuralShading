from __future__ import annotations

import pytest

from ncls.learning.pipelines import LearningPipelineDescriptor, pipeline_descriptors


def _descriptor() -> LearningPipelineDescriptor:
    return LearningPipelineDescriptor(
        name="film-evaluator-s-v1",
        stage="P1",
        data={
            "reader": "reference-corpus-v1",
            "partition": "parametric-v1",
            "source_adapter": "layer-stack-v1",
        },
        model={
            "representation": "film-v1",
            "architecture": "modulated-mlp-v1",
            "latent": "autodecoder-v1",
        },
        fitting={"path": "gradient", "loss": "response-v1"},
        runtime={"compiler": "none", "exporter": "method-bundle-v1"},
        supported_families=("layer-stack",),
        scope="P1 LayerStack evaluator candidate",
    )


def test_pipeline_identity_is_readable_structured_and_hash_exact() -> None:
    descriptor = _descriptor()
    assert descriptor.name == "film-evaluator-s-v1"
    assert descriptor.model["architecture"] == "modulated-mlp-v1"
    assert len(descriptor.sha256) == 64
    changed = LearningPipelineDescriptor(
        **{**descriptor.__dict__, "model": {**descriptor.model, "latent": "encoder-v1"}}
    )
    assert changed.sha256 != descriptor.sha256
    assert pipeline_descriptors() == ()


def test_pipeline_contract_rejects_unstructured_component_sets() -> None:
    descriptor = _descriptor()
    with pytest.raises(ValueError, match="model fields"):
        LearningPipelineDescriptor(
            **{**descriptor.__dict__, "model": {"architecture": "mlp-v1"}}
        )
