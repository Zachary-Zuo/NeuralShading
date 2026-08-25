from __future__ import annotations

import pytest
import numpy as np
import torch

import ncls.learning.pipelines.p1_evaluator as p1_evaluator_module
from ncls.learning.pipelines import (
    LearningPipelineDescriptor,
    create_pipeline,
    pipeline_descriptors,
)
from ncls.learning.pipelines.p1_evaluator import _fit_scales


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
    assert {item.name for item in pipeline_descriptors()} == {
        "film-evaluator-s-v1",
        "film-evaluator-m-v1",
        "film-evaluator-l-v1",
        "analytic-residual-s-v1",
        "analytic-residual-m-v1",
        "analytic-residual-l-v1",
        "per-state-teacher-l-v1",
    }


def test_pipeline_contract_rejects_unstructured_component_sets() -> None:
    descriptor = _descriptor()
    with pytest.raises(ValueError, match="model fields"):
        LearningPipelineDescriptor(
            **{**descriptor.__dict__, "model": {"architecture": "mlp-v1"}}
        )


def test_p1_film_pipeline_returns_linear_nonnegative_f() -> None:
    pipeline = create_pipeline("film-evaluator-s-v1")
    pipeline.load_training_state({
        "contract": "ncls.p1-output-scale@3",
        "state_ids": ["state-a", "state-b"],
        "target_scale": [[0.1, 0.2, 0.3], [0.2, 0.1, 0.05]],
        "output_scale": [[0.1, 0.2, 0.3], [0.2, 0.1, 0.05]],
        "initial_output_ratio": 0.01,
    })
    model = pipeline.create_model({
        "state_count": 2,
        "width": 128,
        "latent_dim": 32,
        "prepare_blocks": 2,
        "evaluate_blocks": 3,
        "fourier_bands": 4,
    })
    wo = torch.tensor([[0.0, 0.0, 1.0], [0.4, 0.0, 0.9165151]])
    wi = torch.tensor([
        [[0.0, 0.0, 1.0], [0.2, 0.0, 0.9797959]],
        [[-0.4, 0.0, 0.9165151], [0.0, 0.2, 0.9797959]],
    ])
    batch = {
        "state_index": torch.tensor([0, 1]),
        "wo": wo,
        "wi": wi,
        "mean": torch.full((2, 2, 3), 0.05),
        "solid_angle_weight": torch.full((2, 2), 0.5),
    }
    prediction = pipeline.predict_f(model, batch, None, torch.device("cpu"))
    assert prediction.shape == (2, 2, 3)
    assert torch.all(torch.isfinite(prediction))
    assert torch.all(prediction >= 0.0)
    loss = pipeline.training_loss(prediction, batch)
    assert torch.isfinite(loss) and loss > 0.0
    near_black_batch = {**batch, "mean": torch.zeros_like(batch["mean"])}
    near_black_prediction = torch.full_like(prediction, 0.006)
    near_black_loss = pipeline.training_loss(near_black_prediction, near_black_batch)
    assert torch.isfinite(near_black_loss)
    assert near_black_loss < 20.0
    costs = pipeline.parameter_costs(model)
    assert costs["C_prepare_macs"] > 0
    assert costs["C_eval_macs"] > 0
    assert costs["analytic_core_state_bytes"] == 0


def test_p1_output_scale_is_fitted_in_linear_f_space() -> None:
    class Store:
        state_count = 1

        @staticmethod
        def state_strings(field: str) -> np.ndarray:
            assert field == "state_id"
            return np.asarray(["state-a"], dtype=object)

        @staticmethod
        def iter_batches(
            indices: np.ndarray,
            batch_size: int,
            *,
            fields: tuple[str, ...] | None = None,
        ):
            del indices, batch_size
            assert fields == ("state_index", "wi", "mean")
            yield {
                "state_index": np.asarray([0], dtype=np.int64),
                "wo": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                "wi": np.asarray(
                    [[[np.sqrt(0.75), 0.0, 0.5], [0.0, 0.0, 1.0]]],
                    dtype=np.float32,
                ),
                "mean": np.full((1, 2, 3), 0.1, dtype=np.float32),
            }

    fitted = _fit_scales(Store(), np.asarray([0], dtype=np.int64))
    assert fitted["contract"] == "ncls.p1-output-scale@3"
    np.testing.assert_allclose(fitted["target_scale"], [[0.1, 0.1, 0.1]], rtol=1e-6)
    assert np.all(np.asarray(fitted["output_scale"]) > 0.19)
    assert 0.0 < fitted["initial_output_ratio"] <= 0.25


def test_p1_residual_scale_scan_requests_wo(monkeypatch: pytest.MonkeyPatch) -> None:
    class Store:
        state_count = 1

        @staticmethod
        def state_strings(field: str) -> np.ndarray:
            assert field == "state_id"
            return np.asarray(["state-a"], dtype=object)

        @staticmethod
        def iter_batches(
            indices: np.ndarray,
            batch_size: int,
            *,
            fields: tuple[str, ...] | None = None,
        ):
            del indices, batch_size
            assert fields == ("state_index", "wo", "wi", "mean")
            yield {
                "state_index": np.asarray([0], dtype=np.int64),
                "wo": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                "wi": np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32),
                "mean": np.full((1, 1, 3), 0.1, dtype=np.float32),
            }

    monkeypatch.setattr(
        p1_evaluator_module,
        "direct_top_bsdf",
        lambda direct_top, state_index, wo, wi: torch.zeros_like(wi),
    )
    fitted = _fit_scales(
        Store(),
        np.asarray([0], dtype=np.int64),
        direct_top={"contract": "test"},
    )
    assert fitted["contract"] == "ncls.p1-output-scale@3"


def test_per_state_teacher_preserves_gradients_through_state_routing() -> None:
    pipeline = create_pipeline("per-state-teacher-l-v1")
    pipeline.load_training_state({
        "contract": "ncls.p1-output-scale@3",
        "state_ids": ["state-a", "state-b"],
        "target_scale": [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
        "output_scale": [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        "initial_output_ratio": 0.01,
    })
    model = pipeline.create_model({
        "state_count": 2,
        "width": 16,
        "block_count": 1,
        "fourier_bands": 1,
    })
    prediction = model(
        torch.tensor([0, 1]),
        torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 0.9797959]]),
        torch.tensor([
            [[0.0, 0.0, 1.0], [0.4, 0.0, 0.9165151]],
            [[0.0, 0.2, 0.9797959], [-0.2, 0.0, 0.9797959]],
        ]),
    )
    prediction.sum().backward()
    assert all(
        any(parameter.grad is not None for parameter in network.parameters())
        for network in model.networks
    )
