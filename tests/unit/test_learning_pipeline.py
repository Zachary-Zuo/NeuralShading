from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ncls.bundle import MethodBundle, export_legacy_ltc_k2_checkpoint
from ncls.core.material import DiffuseInterface, HomogeneousMedium, LayerStackIR, RoughDielectricInterface
from ncls.data import CollectionConfig, collect_reference_dataset
from ncls.data.providers import LayerStackProvider, LayerStackProviderConfig
from ncls.learning.data import LayerStackReferenceStore, ReferenceQueryStore
from ncls.learning.direct_fit import DirectFitConfig, run_direct_fit
from ncls.learning.evaluation import evaluate_checkpoint, evaluate_evaluator_gate
from ncls.learning.features import CONTINUOUS_FEATURE_COUNT, FEATURE_CONTRACT_ID, encode_layer_stack
from ncls.learning.losses import reference_se_group_tail_loss
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training import TrainingConfig, train
from ncls.learning.training.checkpoint import load_checkpoint


class _ConstantEvaluator:
    def __init__(self, light_count: int) -> None:
        self.light_count = light_count

    def evaluate_query_groups(
        self,
        materials,
        view_directions,
        *,
        sample_count_per_replica: int,
        query_group_seeds: np.ndarray,
        light_directions: np.ndarray | None = None,
        sample_offset: int = 0,
    ):
        shape = (len(materials), self.light_count, 3)
        mean_a = np.full(shape, 0.2, dtype=np.float32)
        mean_b = np.full(shape, 0.22, dtype=np.float32)
        variance_a = np.full(shape, 0.01, dtype=np.float32)
        variance_b = np.full(shape, 0.01, dtype=np.float32)
        return mean_a, variance_a + mean_a * mean_a, mean_b, variance_b + mean_b * mean_b


def _dataset(path: Path) -> None:
    collection = CollectionConfig(
        view_count=1,
        light_count=4,
        seed=29,
        split_direction_scramble=False,
    )
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=3,
            local_state_count=1,
            samples_per_replica=4,
            query_group_batch=3,
            max_depth=4,
        ),
        evaluator=_ConstantEvaluator(4),
    )
    collect_reference_dataset(
        path,
        [provider],
        collection,
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )


def _e1_dataset(path: Path) -> None:
    collection = CollectionConfig(
        view_count=2,
        validation_view_count=1,
        test_view_count=1,
        adversarial_view_count=1,
        light_count=8,
        seed=41,
        split_direction_scramble=True,
    )
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=3,
            local_state_count=1,
            samples_per_replica=4,
            query_group_batch=3,
            max_depth=4,
        ),
        evaluator=_ConstantEvaluator(8),
    )
    collect_reference_dataset(
        path,
        [provider],
        collection,
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )


def test_learning_base_store_is_source_independent_and_layer_stack_decode_is_explicit(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.h5"
    _dataset(dataset_path)
    with ReferenceQueryStore(dataset_path) as common:
        batch = common.batch(np.asarray([0, 1, 2]))
        assert "interface_kinds" not in batch
        assert batch["mean"].shape == (3, 4, 3)
        assert batch["lights"].shape == (3, 4, 3)
    with LayerStackReferenceStore(dataset_path) as store:
        assert store.features.continuous.shape == (3, 8, CONTINUOUS_FEATURE_COUNT)
        assert {name: len(indices) for name, indices in store.source_split_indices.items()} == {
            "train": 1,
            "validation": 1,
            "test": 1,
        }
        assert {
            name: len(store.partition_indices("ncls.source-state-split@1", name))
            for name in ("train", "validation", "test")
        } == {"train": 1, "validation": 1, "test": 1}
        batch = store.batch(np.asarray([0, 1, 2]))
        assert batch["standard_error"].shape == (3, 4, 3)
        np.testing.assert_allclose(batch["standard_error"], np.sqrt(0.0101 / 8.0), rtol=2e-5)

    _, encoded, _ = encode_layer_stack(
        LayerStackIR(
            (RoughDielectricInterface(0.1, 0.2, 1.5, 0.0), DiffuseInterface((0.5, 0.5, 0.5))),
            (HomogeneousMedium(thickness=0.2),),
        )
    )
    assert encoded[0, 14:16].tolist() == [0.0, 1.0]


def test_training_writes_tensorboard_best_last_and_keeps_test_held_out(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.h5"
    run_path = tmp_path / "run"
    _dataset(dataset_path)
    config = TrainingConfig(
        model_parameters={"width": 8},
        steps=2,
        batch_size=2,
        learning_rate=1e-3,
        learning_rate_schedule="cosine",
        final_learning_rate_fraction=0.1,
        validation_interval=1,
        checkpoint_interval=1,
        max_validation_query_groups=1,
        seed=31,
        device="cpu",
    )
    manifest = train(dataset_path, run_path, config)
    assert manifest["status"] == "complete"
    assert manifest["pipeline_id"] == "legacy-ltc-k2-p1-deployment-regression@1"
    assert manifest["pipeline_contract"]["target_transform_id"] == "ncls.identity-linear-response@1"
    assert manifest["partition_policy_id"] == "ncls.source-state-split@1"
    assert manifest["held_out_test_accessed"] is False
    assert manifest["feature_contract"]["feature_contract_id"] == FEATURE_CONTRACT_ID
    for name in ("best.pt", "last.pt"):
        assert (run_path / "checkpoints" / name).is_file()
        assert (run_path / "checkpoints" / f"{name}.sha256").is_file()

    events = EventAccumulator(str(run_path / "tensorboard"))
    events.Reload()
    scalar_tags = set(events.Tags()["scalars"])
    assert "train/loss" in scalar_tags
    learning_rates = [event.value for event in events.Scalars("train/learning_rate")]
    assert learning_rates[0] == pytest.approx(1e-3)
    assert learning_rates[-1] < learning_rates[0]
    assert "validation/relative_l1_median" in scalar_tags
    assert not any(tag.startswith("test/") for tag in scalar_tags)

    result = evaluate_checkpoint(
        dataset_path,
        run_path / "checkpoints" / "best.pt",
        split="test",
        output_path=run_path / "test_metrics.json",
        device_name="cpu",
    )
    assert result["split"] == "test"
    assert result["metrics"]["query_group_count"] == 1
    persisted_manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert persisted_manifest["held_out_test_accessed"] is False

    bundle_path = tmp_path / "bundle"
    bundle = export_legacy_ltc_k2_checkpoint(
        run_path / "checkpoints" / "best.pt",
        bundle_path,
        source_run_manifest=run_path / "run_manifest.json",
        created_at="2026-08-23T00:00:00+00:00",
    )
    assert bundle.manifest.backend_id == "legacy-ltc-k2"
    assert bundle.manifest.runtime_class == "realtime"
    assert bundle.manifest.compiler["runtime_implementation"] == "slang"
    assert bundle.manifest.backend_descriptor["shader_entry_points"]["prepare"] == "LegacyLtcK2P1Backend.prepare"
    layout = json.loads(bundle.file("weight_layout").read_text(encoding="utf-8"))
    assert bundle.file("weights").stat().st_size == 4 * layout["total_floats"]
    assert bundle.file("compiler_shader").is_file()
    assert bundle.file("parity_material").stat().st_size == 752
    assert bundle.file("parity").is_file()
    MethodBundle.open(bundle_path)

    weight_path = bundle.file("weights")
    weight_path.write_bytes(weight_path.read_bytes() + b"corrupt")
    with np.testing.assert_raises_regex(ValueError, "content hash mismatch"):
        MethodBundle.open(bundle_path)


def test_dense_e1_pipeline_fits_transform_from_selected_train_queries_only(tmp_path: Path) -> None:
    dataset_path = tmp_path / "e1-dataset.h5"
    run_path = tmp_path / "e1-run"
    _e1_dataset(dataset_path)
    with ReferenceQueryStore(dataset_path) as store:
        state_id = str(store.dataset.state_strings("state_id")[0])
        expected_train = store.select_indices(
            store.partition_indices("ncls.query-role-within-state@1", "train"),
            {"state_ids": [state_id]},
        )
        response = np.asarray(
            store.dataset.stream["responses/mean"][expected_train], dtype=np.float64
        ).reshape(-1, 3)
        peak = np.max(np.abs(response), axis=0)
        expected_scale = np.maximum(
            np.quantile(np.maximum(response, 0.0), 0.9, axis=0),
            np.maximum(1e-3 * peak, 1e-6),
        )

    config = TrainingConfig(
        pipeline_id="dense-latent-small-mlp-log1p-e1@1",
        research_stage="e1-single-material-capacity",
        model_parameters={
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "direction_encoding_id": "ncls.local-cartesian-directions@1",
            "fourier_band_count": 1,
        },
        dataset_selection={"state_ids": [state_id]},
        steps=2,
        batch_size=2,
        learning_rate=1e-3,
        validation_interval=1,
        checkpoint_interval=1,
        max_validation_query_groups=1,
        seed=43,
        device="cpu",
        selection_metric="solid_angle_normalized_l1.median",
    )
    manifest = train(dataset_path, run_path, config)
    assert manifest["status"] == "complete"
    assert manifest["lifecycle_query_group_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert manifest["dataset_selection"] == {"state_ids": [state_id]}
    assert manifest["fitted_training_state"]["fit_scope"] == "final-train-query-groups-only"
    np.testing.assert_allclose(
        manifest["fitted_training_state"]["target_channel_scale"], expected_scale
    )
    assert manifest["model_costs"]["B_shared_fp32"] == 0
    assert manifest["model_costs"]["B_asset_fp32"] > 0

    checkpoint = load_checkpoint(run_path / "checkpoints" / "best.pt")
    assert checkpoint["fitted_training_state_sha256"] == manifest["fitted_training_state_sha256"]
    result = evaluate_checkpoint(
        dataset_path,
        run_path / "checkpoints" / "best.pt",
        split="test",
        output_path=run_path / "test_metrics.json",
        device_name="cpu",
    )
    metrics = result["metrics"]
    assert metrics["query_group_count"] == 1
    for name in (
        "solid_angle_normalized_l1",
        "linear_relative_l1",
        "log_l1",
        "energy_relative_error",
        "peak_ratio",
        "peak_angle_degrees",
        "top_5_percent_energy_recall",
        "model_error_over_reference_standard_error",
        "finite_rate",
        "nonnegative_rate",
        "reciprocity_relative_l1",
    ):
        assert name in metrics
    assert "ncls.layer-stack@1" in metrics["by_family"]
    adversarial = evaluate_checkpoint(
        dataset_path,
        run_path / "checkpoints" / "best.pt",
        split="adversarial_probe",
        output_path=run_path / "adversarial_metrics.json",
        device_name="cpu",
    )
    assert adversarial["evaluation_role"] == "adversarial_probe"
    assert adversarial["metrics"]["query_group_count"] == 1
    gate = evaluate_evaluator_gate(
        run_path / "run_manifest.json",
        run_path / "test_metrics.json",
        run_path / "adversarial_metrics.json",
        Path(__file__).parents[2] / "configs" / "research" / "e1-evaluator-gates-v1.json",
        run_path / "gate_result.json",
    )
    assert gate["passed"] is False
    assert gate["gate_id"] == "ncls.e1-single-material-evaluator-acceptance@1"
    assert (run_path / "gate_result.json").is_file()


def test_standardized_log1p_state_matches_train_only_response_statistics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "standardized-dataset.h5"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline("dense-latent-small-mlp-standardized-log1p-e1@1")
    with pipeline.open_store(str(dataset_path)) as store:
        state_id = str(store.dataset.state_strings("state_id")[0])
        indices = store.select_indices(
            pipeline.lifecycle_indices(store, "train"), {"state_ids": [state_id]}
        )
        response = np.maximum(
            np.asarray(store.dataset.stream["responses/mean"][indices], dtype=np.float64), 0.0
        ).reshape(-1, 3)
        scale = np.maximum(np.quantile(response, 0.5, axis=0), 1e-8)
        transformed = np.log1p(response / scale)
        state = pipeline.fit_training_state(store, indices)
        assert state["format_version"] == 2
        np.testing.assert_allclose(state["target_channel_scale"], scale)
        np.testing.assert_allclose(state["target_channel_mean"], np.mean(transformed, axis=0))
        np.testing.assert_allclose(
            state["target_channel_standard_deviation"],
            np.maximum(np.std(transformed, axis=0), 1e-6),
        )
        pipeline.load_training_state(state)


def test_analytic_residual_pipeline_uses_layer_stack_core_and_signed_train_state(tmp_path: Path) -> None:
    dataset_path = tmp_path / "analytic-residual-dataset.h5"
    _e1_dataset(dataset_path)
    pipeline_ids = (
        "analytic-core-neural-residual-standardized-e1@1",
        "analytic-core-neural-residual-energy-shape-e1@1",
    )
    for pipeline_id in pipeline_ids:
        pipeline = create_pipeline(pipeline_id)
        with pipeline.open_store(str(dataset_path)) as store:
            state_id = str(store.dataset.state_strings("state_id")[0])
            indices = store.select_indices(
                pipeline.lifecycle_indices(store, "train"), {"state_ids": [state_id]}
            )
            state = pipeline.fit_training_state(store, indices)
            assert state["target_transform_id"] == "ncls.train-only-standardized-asinh-analytic-residual@1"
            pipeline.load_training_state(state)
            model = pipeline.create_model({
                "latent_dimension": 4,
                "width": 8,
                "prepare_layer_count": 1,
                "evaluate_layer_count": 1,
                "direction_encoding_id": "ncls.half-difference-directions@1",
                "fourier_band_count": 1,
                "output_bias": 0.0,
            })
            batch = {
                name: torch.as_tensor(value)
                for name, value in store.batch(indices[:1]).items()
            }
            prediction = pipeline.predict(model, batch, store, torch.device("cpu"))
            loss = pipeline.training_loss(prediction, batch)
            assert prediction.shape == batch["mean"].shape
            assert torch.all(torch.isfinite(prediction))
            assert torch.isfinite(loss)
            loss.backward()


def test_multiscale_half_slope_encoding_has_fixed_small_evaluate_input() -> None:
    pipeline = create_pipeline("dense-latent-small-mlp-standardized-log1p-e1@1")
    model = pipeline.create_model({
        "latent_dimension": 4,
        "width": 8,
        "prepare_layer_count": 1,
        "evaluate_layer_count": 1,
        "direction_encoding_id": "ncls.multiscale-half-slope-directions@1",
        "fourier_band_count": 1,
    })
    wo = torch.tensor([[0.0, 0.0, 1.0], [0.4, 0.0, 0.9165151]])
    wi = torch.tensor([
        [[0.0, 0.0, 1.0], [0.001, 0.0, 0.9999995]],
        [[-0.4, 0.0, 0.9165151], [-0.399, 0.001, 0.916949]],
    ])
    raw = model(wo, wi)
    assert raw.shape == (2, 2, 3)
    assert torch.all(torch.isfinite(raw))
    assert model.evaluate_network[0].in_features == model.prepared_dimension + 30


def test_energy_shape_loss_penalizes_missing_integrated_response() -> None:
    pipeline = create_pipeline("dense-latent-small-mlp-energy-shape-e1@1")
    pipeline._training_state = {
        "target_channel_scale": [0.1, 0.1, 0.1],
        "target_channel_mean": [0.0, 0.0, 0.0],
        "target_channel_standard_deviation": [1.0, 1.0, 1.0],
    }
    target = torch.tensor([[[1.0, 0.5, 0.25], [0.1, 0.05, 0.025]]])
    batch = {
        "mean": target,
        "standard_error": torch.zeros_like(target),
        "solid_angle_weight": torch.ones((1, 2)),
    }
    matching = pipeline.training_loss(target.clone(), batch)
    missing = pipeline.training_loss(torch.full_like(target, 1e-6), batch)
    assert matching < missing


def test_shared_e2_pipeline_uses_material_slots_and_reports_state_partitions(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "shared-e2-dataset.h5"
    run_path = tmp_path / "shared-e2-run"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline("dense-latent-shared-small-mlp-energy-shape-e2@1")
    assert pipeline.descriptor.latent_inference_id == (
        "ncls.optimized-target-visible-dense-material-latent-table@1"
    )
    assert pipeline.descriptor.compiler_id == "ncls.none-target-visible-capacity-study@1"
    with pipeline.open_store(str(dataset_path)) as store:
        indices = pipeline.lifecycle_indices(store, "train")
        state = pipeline.fit_training_state(store, indices)
        assert state["fit_scope"] == "final-train-query-groups-only"
        assert state["latent_scope"] == "target-visible-selected-states"
        assert len(state["state_ids"]) == 3
        assert state["train_query_group_count_by_state"] == [2, 2, 2]
        pipeline.load_training_state(state)
        model = pipeline.create_model({
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.local-cartesian-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
        })
        batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(indices[[0, 2, 4]]).items()
        }
        prediction = pipeline.predict(model, batch, store, torch.device("cpu"))
        loss = pipeline.training_loss(prediction, batch)
        assert prediction.shape == batch["mean"].shape
        assert torch.isfinite(loss)
        loss.backward()
        costs = pipeline.parameter_costs(model)
        assert costs["material_count"] == 3
        assert costs["B_asset_fp32"] == 16
        assert costs["B_asset_fp32_total"] == 48
        assert costs["B_shared_fp32"] > 0

    config = TrainingConfig(
        pipeline_id="dense-latent-shared-small-mlp-energy-shape-e2@1",
        research_stage="e2-shared-representation-capacity",
        model_parameters={
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.local-cartesian-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
        },
        steps=2,
        batch_size=3,
        learning_rate=1e-3,
        validation_interval=1,
        checkpoint_interval=1,
        max_validation_query_groups=3,
        seed=47,
        device="cpu",
        selection_metric="solid_angle_normalized_l1.median",
    )
    manifest = train(dataset_path, run_path, config)
    assert manifest["lifecycle_query_group_counts"] == {
        "train": 6,
        "validation": 3,
        "test": 3,
    }
    assert manifest["lifecycle_source_state_counts"] == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert manifest["model_costs"]["B_asset_fp32"] == 16
    assert manifest["model_costs"]["B_shared_fp32"] > 0
    result = evaluate_checkpoint(
        dataset_path,
        run_path / "checkpoints" / "best.pt",
        split="test",
        device_name="cpu",
    )
    assert len(result["metrics"]["by_state"]) == 3
    assert set(result["metrics"]["by_source_split"]) == {"train", "validation", "test"}


def test_shared_analytic_residual_uses_explicit_layer_stack_adapter(tmp_path: Path) -> None:
    dataset_path = tmp_path / "shared-analytic-e2-dataset.h5"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline("analytic-core-shared-neural-residual-energy-shape-e2@1")
    assert pipeline.descriptor.source_adapter_id == "ncls.layer-stack-direct-top-adapter@1"
    assert pipeline.descriptor.candidate_id == "ncls.analytic-core-neural-residual@1"
    with pipeline.open_store(str(dataset_path)) as store:
        indices = pipeline.lifecycle_indices(store, "train")
        state = pipeline.fit_training_state(store, indices)
        assert state["target_transform_id"] == (
            "ncls.train-only-standardized-asinh-analytic-residual@1"
        )
        pipeline.load_training_state(state)
        model = pipeline.create_model({
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.half-difference-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
        })
        batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(indices[[0, 2, 4]]).items()
        }
        prediction = pipeline.predict(model, batch, store, torch.device("cpu"))
        loss = pipeline.training_loss(prediction, batch)
        assert prediction.shape == batch["mean"].shape
        assert torch.all(torch.isfinite(prediction))
        assert torch.isfinite(loss)
        loss.backward()
        metrics = pipeline.additional_metric_distributions(
            model.eval(), batch, store, torch.device("cpu")
        )
        assert set(metrics) == {
            "reciprocity_relative_l1",
            "analytic_core_solid_angle_normalized_l1",
        }


def test_per_state_shared_analytic_residual_accounts_transform_per_asset(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "per-state-shared-analytic-e2-dataset.h5"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline("analytic-core-shared-neural-residual-energy-shape-e2@2")
    assert pipeline.descriptor.target_transform_id == (
        "ncls.train-only-per-state-standardized-asinh-analytic-residual@1"
    )
    with pipeline.open_store(str(dataset_path)) as store:
        indices = pipeline.lifecycle_indices(store, "train")
        state = pipeline.fit_training_state(store, indices)
        state_count = len(state["state_ids"])
        assert state["format_version"] == 4
        assert state["target_channel_scale"] is None
        assert np.asarray(state["target_channel_scale_by_state"]).shape == (state_count, 3)
        assert np.all(np.asarray(state["target_channel_scale_by_state"]) > 0.0)
        pipeline.load_training_state(state)
        model = pipeline.create_model({
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.half-difference-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
        })
        batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(indices[[0, 2, 4]]).items()
        }
        prediction = pipeline.predict(model, batch, store, torch.device("cpu"))
        loss = pipeline.training_loss(prediction, batch)
        assert torch.isfinite(loss)
        loss.backward()
        costs = pipeline.parameter_costs(model)
        assert costs["B_asset_target_transform_fp32"] == 36
        assert costs["B_asset_fp32"] == 52
        assert costs["B_asset_fp32_total"] == state_count * 52


def test_source_aware_shared_residual_reports_deviation_from_source_asymmetry(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "source-aware-shared-analytic-e2-dataset.h5"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline("analytic-core-shared-neural-residual-energy-shape-e2@3")
    with pipeline.open_store(str(dataset_path)) as store:
        indices = pipeline.lifecycle_indices(store, "train")
        pipeline.load_training_state(pipeline.fit_training_state(store, indices))
        model = pipeline.create_model({
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.half-difference-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
        }).eval()
        batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(indices[[0, 2, 4]]).items()
        }
        metrics = pipeline.additional_metric_distributions(
            model, batch, store, torch.device("cpu")
        )
        assert set(metrics) == {
            "reciprocity_relative_l1",
            "source_reciprocity_deviation_relative_l1",
            "analytic_core_solid_angle_normalized_l1",
        }
        assert metrics["source_reciprocity_deviation_relative_l1"].shape == (3,)
        assert np.all(np.isfinite(metrics["source_reciprocity_deviation_relative_l1"]))


def test_noise_aware_shared_residual_uses_peak_support_and_finite_loss(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "noise-aware-shared-analytic-e2-dataset.h5"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline("analytic-core-shared-neural-residual-energy-shape-e2@4")
    target = torch.tensor([[[1.0, 0.0, 0.0], [0.96, 0.0, 0.0]]])
    prediction = torch.tensor([[[0.9, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    lights = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    support_angle = pipeline._peak_support_angle(prediction, target, lights)
    torch.testing.assert_close(support_angle, torch.zeros_like(support_angle))
    with pipeline.open_store(str(dataset_path)) as store:
        indices = pipeline.lifecycle_indices(store, "train")
        pipeline.load_training_state(pipeline.fit_training_state(store, indices))
        model = pipeline.create_model({
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.half-difference-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
        })
        batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(indices[[0, 2, 4]]).items()
        }
        predicted = pipeline.predict(model, batch, store, torch.device("cpu"))
        loss = pipeline.training_loss(predicted, batch)
        assert torch.isfinite(loss)
        loss.backward()
        metrics = pipeline.additional_metric_distributions(
            model.eval(), batch, store, torch.device("cpu")
        )
        assert "peak_support_angle_degrees" in metrics
        assert metrics["peak_support_angle_degrees"].shape == (3,)
    boundary = create_pipeline("analytic-core-shared-neural-residual-energy-shape-e2@5")
    assert boundary.descriptor.loss_id == (
        "ncls.per-state-standardized-asinh-residual-energy-shape-reciprocity@1"
    )
    assert boundary.descriptor.metric_suite_id == pipeline.descriptor.metric_suite_id


@pytest.mark.parametrize(
    (
        "pipeline_id", "extra_parameters", "candidate_id", "compiler_id",
        "asset_bytes", "total_bytes",
    ),
    (
        (
            "sparse-latent-dictionary-analytic-residual-e2@1",
            {"dictionary_size": 8, "top_k": 2},
            "ncls.sparse-latent-dictionary-top-k-mixture@1",
            "ncls.none-target-visible-capacity-study@1",
            48,
            144,
        ),
        (
            "factorized-latent-analytic-residual-e2@1",
            {"factor_rank": 2},
            "ncls.plane-tensor-factorization@1",
            "ncls.none-target-visible-capacity-study@1",
            44,
            132,
        ),
        (
            "target-tensor-encoder-analytic-residual-e2@1",
            {"encoder_width": 8, "encoder_layer_count": 1},
            "ncls.target-tensor-encoder-shared-decoder@1",
            "ncls.none-target-visible-response-compression@1",
            52,
            156,
        ),
    ),
)
def test_structured_e2_latents_use_common_source_aware_lifecycle(
    tmp_path: Path,
    pipeline_id: str,
    extra_parameters: dict[str, int],
    candidate_id: str,
    compiler_id: str,
    asset_bytes: int,
    total_bytes: int,
) -> None:
    dataset_path = tmp_path / f"{pipeline_id}.h5"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline(pipeline_id)
    assert pipeline.descriptor.candidate_id == candidate_id
    assert pipeline.descriptor.compiler_id == compiler_id
    assert pipeline.descriptor.source_adapter_id == "ncls.layer-stack-direct-top-adapter@1"
    with pipeline.open_store(str(dataset_path)) as store:
        indices = pipeline.lifecycle_indices(store, "train")
        pipeline.load_training_state(pipeline.fit_training_state(store, indices))
        model = pipeline.create_model({
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.half-difference-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
            **extra_parameters,
        })
        batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(indices[[0, 2, 4]]).items()
        }
        prediction = pipeline.predict(model, batch, store, torch.device("cpu"))
        loss = pipeline.training_loss(prediction, batch)
        assert prediction.shape == batch["mean"].shape
        assert torch.isfinite(loss)
        loss.backward()
        costs = pipeline.parameter_costs(model)
        assert costs["material_count"] == 3
        assert costs["B_asset_target_transform_fp32"] == 36
        assert costs["B_asset_fp32"] == asset_bytes
        assert costs["B_asset_fp32_total"] == total_bytes
        assert costs["C_prepare_macs"] > 0
        assert costs["C_eval_macs"] > 0
        metrics = pipeline.additional_metric_distributions(
            model.eval(), batch, store, torch.device("cpu")
        )
        assert "source_reciprocity_deviation_relative_l1" in metrics
        assert "peak_support_angle_degrees" in metrics

    run_path = tmp_path / f"{pipeline_id}-run"
    config = TrainingConfig(
        pipeline_id=pipeline_id,
        research_stage="e2-shared-representation-capacity",
        model_parameters={
            "latent_dimension": 4,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.half-difference-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
            **extra_parameters,
        },
        steps=1,
        batch_size=3,
        learning_rate=1e-3,
        validation_interval=1,
        checkpoint_interval=1,
        max_validation_query_groups=3,
        seed=53,
        device="cpu",
        selection_metric="solid_angle_normalized_l1.median",
    )
    manifest = train(dataset_path, run_path, config)
    assert manifest["model_costs"]["B_asset_fp32"] == asset_bytes
    result = evaluate_checkpoint(
        dataset_path,
        run_path / "checkpoints" / "best.pt",
        split="test",
        device_name="cpu",
    )
    assert len(result["metrics"]["by_state"]) == 3


def test_structured_e2_latent_parameters_are_strictly_validated(tmp_path: Path) -> None:
    dataset_path = tmp_path / "structured-latent-validation.h5"
    _e1_dataset(dataset_path)
    cases = (
        (
            "sparse-latent-dictionary-analytic-residual-e2@1",
            {"dictionary_size": 4, "top_k": 5},
        ),
        (
            "factorized-latent-analytic-residual-e2@1",
            {"factor_rank": 0},
        ),
        (
            "target-tensor-encoder-analytic-residual-e2@1",
            {"encoder_width": 0, "encoder_layer_count": 1},
        ),
    )
    for pipeline_id, extra_parameters in cases:
        pipeline = create_pipeline(pipeline_id)
        with pipeline.open_store(str(dataset_path)) as store:
            indices = pipeline.lifecycle_indices(store, "train")
            pipeline.load_training_state(pipeline.fit_training_state(store, indices))
            with pytest.raises(ValueError):
                pipeline.create_model({
                    "latent_dimension": 4,
                    "width": 8,
                    "prepare_layer_count": 1,
                    "evaluate_layer_count": 1,
                    "activation": "gelu",
                    "direction_encoding_id": "ncls.half-difference-directions@1",
                    "fourier_band_count": 1,
                    "output_bias": 0.0,
                    **extra_parameters,
                })


def test_target_tensor_encoder_fitted_state_records_only_train_response_points(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "target-tensor-encoder-input.h5"
    _e1_dataset(dataset_path)
    pipeline = create_pipeline("target-tensor-encoder-analytic-residual-e2@1")
    with pipeline.open_store(str(dataset_path)) as store:
        train_indices = pipeline.lifecycle_indices(store, "train")
        state = pipeline.fit_training_state(store, train_indices)
        assert state["format_version"] == 5
        assert state["target_encoder_input_query_role"] == "train"
        assert state["target_encoder_input_shape"] == [3, 16, 10]
        assert len(state["target_encoder_input_sha256"]) == 64
        assert state["train_query_group_count"] == len(train_indices)
        pipeline.load_training_state(state)
        assert pipeline._target_encoder_input is not None
        assert pipeline._target_encoder_input.shape == (3, 16, 10)
        model = pipeline.create_model({
            "latent_dimension": 4,
            "encoder_width": 8,
            "encoder_layer_count": 1,
            "width": 8,
            "prepare_layer_count": 1,
            "evaluate_layer_count": 1,
            "activation": "gelu",
            "direction_encoding_id": "ncls.half-difference-directions@1",
            "fourier_band_count": 1,
            "output_bias": 0.0,
        })
        assert "target_encoder_input" not in model.state_dict()
        costs = pipeline.parameter_costs(model)
        assert costs["B_compiler_input_fp32"] == 16 * 10 * 4
        assert costs["B_compiler_input_fp32_total"] == 3 * 16 * 10 * 4


def test_target_encoder_bounded_refinement_loads_frozen_source_checkpoint(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "target-refinement.h5"
    source_run = tmp_path / "target-encoder-source"
    _e1_dataset(dataset_path)
    common_model_parameters = {
        "latent_dimension": 4,
        "encoder_width": 8,
        "encoder_layer_count": 1,
        "width": 8,
        "prepare_layer_count": 1,
        "evaluate_layer_count": 1,
        "activation": "gelu",
        "direction_encoding_id": "ncls.half-difference-directions@1",
        "fourier_band_count": 1,
        "output_bias": 0.0,
    }
    source_config = TrainingConfig(
        pipeline_id="target-tensor-encoder-analytic-residual-e2@1",
        research_stage="e2-shared-representation-capacity",
        model_parameters=common_model_parameters,
        steps=1,
        batch_size=3,
        learning_rate=1e-3,
        validation_interval=1,
        checkpoint_interval=1,
        max_validation_query_groups=3,
        seed=59,
        device="cpu",
        selection_metric="solid_angle_normalized_l1.median",
    )
    train(dataset_path, source_run, source_config)
    source_checkpoint = source_run / "checkpoints" / "best.pt"
    refinement_pipeline_ids = (
        "target-encoder-initialization-bounded-refinement-e2@1",
        "target-encoder-se-tail-bounded-refinement-e2@1",
    )
    for pipeline_index, pipeline_id in enumerate(refinement_pipeline_ids):
        refinement_run = tmp_path / f"target-encoder-refinement-{pipeline_index}"
        refinement_config = TrainingConfig(
            schema_version=6,
            pipeline_id=pipeline_id,
            research_stage="e2-shared-representation-capacity",
            model_parameters={**common_model_parameters, "refinement_bound": 0.25},
            steps=2,
            batch_size=3,
            learning_rate=1e-2,
            validation_interval=1,
            checkpoint_interval=1,
            max_validation_query_groups=3,
            seed=61,
            device="cpu",
            selection_metric="solid_angle_normalized_l1.median",
            initialization_checkpoint=str(source_checkpoint),
        )
        manifest = train(dataset_path, refinement_run, refinement_config)
        assert manifest["initialization"]["source_pipeline_id"] == (
            "target-tensor-encoder-analytic-residual-e2@1"
        )
        assert manifest["initialization"]["refinement_parameter_count"] == 12
        assert manifest["trainable_parameter_count"] == 12
        assert manifest["model_costs"]["B_asset_fp32"] == 52
        checkpoint = load_checkpoint(refinement_run / "checkpoints" / "best.pt")
        assert checkpoint["initialization"]["sha256"] == (
            manifest["initialization"]["sha256"]
        )
        result = evaluate_checkpoint(
            dataset_path,
            refinement_run / "checkpoints" / "best.pt",
            split="test",
            device_name="cpu",
        )
        assert len(result["metrics"]["by_state"]) == 3


def test_reference_se_group_tail_loss_matches_group_metric_tail() -> None:
    prediction = torch.tensor([
        [[0.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0]],
        [[2.0, 2.0, 2.0]],
        [[3.0, 3.0, 3.0]],
    ], requires_grad=True)
    target = torch.zeros_like(prediction)
    standard_error = torch.ones_like(prediction)
    loss = reference_se_group_tail_loss(
        prediction, target, standard_error, tail_fraction=0.25
    )
    assert loss.item() == pytest.approx(torch.log1p(torch.tensor(3.0)).item())
    loss.backward()
    assert torch.count_nonzero(prediction.grad) == 3

    with pytest.raises(ValueError, match="tail_fraction"):
        reference_se_group_tail_loss(
            prediction, target, standard_error, tail_fraction=0.0
        )


def test_e3_source_compiler_uses_only_native_source_and_source_train_statistics(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "source-compiler.h5"
    run_path = tmp_path / "source-compiler-run"
    _e1_dataset(dataset_path)
    pipeline_id = "layer-stack-source-state-compiler-analytic-residual-e3@1"
    pipeline = create_pipeline(pipeline_id)
    model_parameters = {
        "latent_dimension": 4,
        "compiler_width": 8,
        "compiler_type_width": 2,
        "compiler_layer_count": 1,
        "width": 8,
        "prepare_layer_count": 1,
        "evaluate_layer_count": 1,
        "activation": "gelu",
        "direction_encoding_id": "ncls.half-difference-directions@1",
        "fourier_band_count": 1,
        "output_bias": 0.0,
    }
    with pipeline.open_store(str(dataset_path)) as store:
        train_indices = pipeline.lifecycle_indices(store, "train")
        validation_indices = pipeline.lifecycle_indices(store, "validation")
        test_indices = pipeline.lifecycle_indices(store, "test")
        adversarial_indices = pipeline.evaluation_indices(store, "adversarial_probe")
        assert [len(value) for value in (
            train_indices, validation_indices, test_indices, adversarial_indices
        )] == [2, 1, 1, 1]
        state = pipeline.fit_training_state(store, train_indices)
        assert state["fit_scope"] == (
            "source-train-states-and-train-query-groups-only"
        )
        assert state["source_train_state_count"] == 1
        assert len(state["target_transform_supervision_by_state"]) == 1
        source_train_state_ids = set(state["state_ids"])
        all_state_ids = set(map(str, store.dataset.state_strings("state_id")))
        assert source_train_state_ids < all_state_ids
        pipeline.load_training_state(state)
        model = pipeline.create_model(model_parameters)
        train_batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(train_indices).items()
        }
        prediction = pipeline.predict(
            model, train_batch, store, torch.device("cpu")
        )
        loss = pipeline.training_loss(prediction, train_batch)
        assert prediction.shape == train_batch["mean"].shape
        assert torch.isfinite(loss)
        loss.backward()
        model.eval()
        test_batch = {
            name: torch.as_tensor(value)
            for name, value in store.batch(test_indices).items()
        }
        with torch.no_grad():
            test_prediction = pipeline.predict(
                model, test_batch, store, torch.device("cpu")
            )
        assert test_prediction.shape == test_batch["mean"].shape
        costs = pipeline.parameter_costs(model)
        assert costs["B_asset_fp32"] == (4 + 9) * 4
        assert costs["B_compiler_input_native_bytes"] == 836

    config = TrainingConfig(
        pipeline_id=pipeline_id,
        research_stage="e3-source-compiler-generalization",
        model_parameters=model_parameters,
        steps=1,
        batch_size=1,
        learning_rate=1e-3,
        validation_interval=1,
        checkpoint_interval=1,
        max_validation_query_groups=1,
        seed=67,
        device="cpu",
        selection_metric="solid_angle_normalized_l1.median",
    )
    manifest = train(dataset_path, run_path, config)
    assert manifest["partition_policy_id"] == "ncls.source-state-and-query-role@1"
    assert manifest["lifecycle_source_state_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert manifest["held_out_test_accessed"] is False
    result = evaluate_checkpoint(
        dataset_path,
        run_path / "checkpoints" / "best.pt",
        split="test",
        device_name="cpu",
    )
    assert result["metrics"]["query_group_count"] == 1
    assert set(result["metrics"]["by_source_split"]) == {"test"}


def test_plane_factorized_pipeline_uses_the_common_e1_lifecycle(tmp_path: Path) -> None:
    dataset_path = tmp_path / "plane-factorized-dataset.h5"
    _e1_dataset(dataset_path)
    pipeline_ids = (
        "plane-factorized-small-mlp-energy-shape-e1@1",
        "plane-factorized-analytic-residual-energy-shape-e1@1",
    )
    for pipeline_id in pipeline_ids:
        pipeline = create_pipeline(pipeline_id)
        assert pipeline.descriptor.candidate_id == "ncls.plane-tensor-factorization@1"
        with pipeline.open_store(str(dataset_path)) as store:
            state_id = str(store.dataset.state_strings("state_id")[0])
            indices = store.select_indices(
                pipeline.lifecycle_indices(store, "train"), {"state_ids": [state_id]}
            )
            pipeline.load_training_state(pipeline.fit_training_state(store, indices))
            model = pipeline.create_model({
                "plane_resolution": 8,
                "plane_feature_dimension": 2,
                "material_latent_dimension": 4,
                "width": 8,
                "evaluate_layer_count": 1,
                "activation": "gelu",
                "output_bias": 0.0,
            })
            batch = {
                name: torch.as_tensor(value)
                for name, value in store.batch(indices[:1]).items()
            }
            prediction = pipeline.predict(model, batch, store, torch.device("cpu"))
            loss = pipeline.training_loss(prediction, batch)
            assert prediction.shape == batch["mean"].shape
            assert torch.isfinite(loss)
            loss.backward()
            costs = pipeline.parameter_costs(model)
            assert costs["plane_texel_fetches_prepare"] == 4
            assert costs["plane_texel_fetches_eval"] == 20
            assert costs["B_shared_fp32"] == 0


def test_direct_fit_is_a_separate_representation_ceiling_run(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.h5"
    output = tmp_path / "direct-fit"
    _dataset(dataset_path)
    result = run_direct_fit(
        dataset_path,
        output,
        split="validation",
        config=DirectFitConfig(
            family="ltc",
            lobe_count=2,
            fit_batch=1,
            steps=2,
            restarts=1,
            learning_rate=0.01,
            seed=37,
            device="cpu",
        ),
        max_query_groups=1,
    )
    assert result["format_name"] == "ncls.representation-ceiling"
    assert result["representation_id"] == "legacy-ltc-k2@1"
    assert result["query_group_count"] == 1
    assert (output / "parameters.npz").is_file()
    assert any((output / "tensorboard").iterdir())
