from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
