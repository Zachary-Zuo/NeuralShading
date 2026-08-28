import pytest
import torch

from ncls.learning.batches import (
    EvaluatorBatch,
    MethodSamplerBatch,
    TrainingConditioning,
    TrainingRouteRequest,
)


def _conditioning(device: str = "cpu") -> TrainingConditioning:
    return TrainingConditioning(
        "fixture.family@1",
        ("a" * 64, "b" * 64),
        {
            "source_index": torch.zeros(2, dtype=torch.int64, device=device),
            "wo": torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device
            ),
        },
        {"source": "test"},
    )


def test_typed_batches_only_require_semantic_route_tensors() -> None:
    conditioning = _conditioning()
    evaluator = EvaluatorBatch(
        conditioning,
        torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]),
        torch.ones((2, 1, 3)),
    )
    sampler = MethodSamplerBatch(conditioning, torch.full((2, 2), 0.5))

    assert set(evaluator.tensors) == {"source_index", "wo", "wi", "target_f"}
    assert set(sampler.tensors) == {"source_index", "wo", "sample_u"}
    assert "target" not in sampler.tensors
    assert "wi" not in sampler.tensors


def test_evaluator_batch_rejects_non_f_target_shape() -> None:
    conditioning = _conditioning()
    wi = torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]])
    with pytest.raises(ValueError, match="target_f"):
        EvaluatorBatch(conditioning, wi, torch.ones((2, 2, 3)))


def test_method_sampler_batch_rejects_dummy_target_by_construction() -> None:
    conditioning = _conditioning()
    with pytest.raises(TypeError):
        MethodSamplerBatch(
            conditioning,
            torch.ones((2, 2)),
            target_f=torch.zeros(2, 1, 3),  # type: ignore[call-arg]
        )


def test_training_route_request_uses_explicit_kind() -> None:
    request = TrainingRouteRequest(
        "evaluator", "reference-evaluator", 2, 1, 0, 17, {}
    )
    assert request.kind == "reference-evaluator"
    with pytest.raises(ValueError, match="kind"):
        TrainingRouteRequest("legacy", "unknown", 2, 1, 0, 17, {})  # type: ignore[arg-type]
