"""锁定03 exact-core candidate、数据身份、sampler协议与relative selection。"""

from __future__ import annotations

from pathlib import Path
import json
import struct

import numpy as np
import torch

from ncls.core.scattering import BackendCapability, BackendDescriptor
from ncls.learning.data import UnifiedScatteringTrainingStore
from ncls.learning.evaluation.offline_cook import UnifiedOfflineCookConfig
from ncls.learning.evaluation.sampler_correctness import (
    independent_unified_sampler_pdf,
    load_unified_sampler_protocol,
    unified_sampler_protocol_sha256,
)
from ncls.learning.evaluation.unified_selection import (
    build_unified_selection_from_artifacts,
    build_unified_selection_manifest,
    load_unified_selection_protocol,
    paired_state_difference,
)
from ncls.learning.evaluation.unified_parity import (
    _batch_across_shards,
    _selected_validation_groups,
)
from ncls.learning.models import UnifiedNeuralModel
from ncls.learning.pipelines import create_pipeline
from ncls.learning.slang import UNIFIED_LAYOUT, render_unified_layout_slang, unified_layout_sha256
from ncls.learning.pipelines.sampler_objective import sampler_cross_entropy
from ncls.learning.unified_artifacts import (
    _runtime_adapter,
    pack_unified_record,
    pack_unified_shared_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_unified_parity_selects_validation_views_across_shards() -> None:
    references = np.asarray([
        (view % 2, state * 8 + view // 2)
        for state in range(2)
        for view in range(16)
    ], dtype=np.int64)

    class FakeStore:
        state_count = 2

        @staticmethod
        def partition_indices(policy_id: str, role: str) -> np.ndarray:
            assert policy_id == "parametric-v1" and role == "validation"
            return references

        @staticmethod
        def batch(
            selected: np.ndarray, *, fields: tuple[str, ...]
        ) -> dict[str, np.ndarray]:
            assert len(np.unique(selected[:, 0])) == 1
            states = selected[:, 1] // 8
            views = (selected[:, 1] % 8) * 2 + selected[:, 0]
            values = {
                "state_index": states.astype(np.int64),
                "wo": np.column_stack((
                    np.zeros(len(selected)),
                    np.zeros(len(selected)),
                    1.0 - views / 16.0,
                )).astype(np.float32),
                "wi": np.zeros((len(selected), 4, 3), dtype=np.float32),
            }
            return {name: values[name] for name in fields}

    class FakePipeline:
        class descriptor:
            partition_policy_id = "parametric-v1"

    selected = _selected_validation_groups(FakeStore(), FakePipeline())
    np.testing.assert_array_equal(selected, np.asarray([[1, 7], [1, 15]]))
    batch = _batch_across_shards(
        FakeStore(), selected, fields=("state_index", "wo", "wi")
    )
    np.testing.assert_array_equal(batch["state_index"], np.asarray([0, 1]))
    np.testing.assert_allclose(batch["wo"][:, 2], np.asarray([1.0 / 16.0] * 2))


def test_unified_layout_matches_frozen_budget() -> None:
    assert UNIFIED_LAYOUT["compiled_material"]["total_bytes"] == 128
    assert UNIFIED_LAYOUT["state"] == {"payload_half_count": 27, "stride_bytes": 64}
    assert UNIFIED_LAYOUT["realtime"]["prepare_macs"] == 23 * 64 + 64 * 64 + 64 * 27
    assert UNIFIED_LAYOUT["realtime"]["evaluate_macs"] == 17 * 32 + 32 * 32 + 32 * 3
    assert len(unified_layout_sha256()) == 64
    generated = PROJECT_ROOT / "shaders/ncls/backends/unified_neural/unified_neural_layout.slang"
    assert generated.read_text(encoding="utf-8") == render_unified_layout_slang()


def test_unified_pipeline_runtime_classes_and_costs() -> None:
    residual = create_pipeline("core-frame-neural-v1")
    offline_control = create_pipeline("nvidia-frame-two-lobe-layer-stack-budget-adapted-v1")
    assert residual.descriptor.deployment_candidate
    assert not offline_control.descriptor.deployment_candidate
    assert residual.parameter_costs(None)["C_eval_macs"] == 1664
    assert offline_control.parameter_costs(None)["runtime_class"] == "diagnostic"
    assert offline_control.parameter_costs(None)["C_eval_macs"] > 2000


def test_unified_data_identities_are_not_directory_guesses() -> None:
    assert UnifiedScatteringTrainingStore.ENTRY_ID == "47ef20138007703f2d1b644bcb4ca4b084001da4ec975f1b712587d3e7e35a89"
    assert UnifiedScatteringTrainingStore.BASE_CORPUS_ID == "0513d0c837b109f74cbf6fd4f811e05c6bc68c02226bd6d443f3225ef5dd64b7"
    assert UnifiedScatteringTrainingStore.SUPPLEMENT_CORPUS_ID == "f6931474890ab7642f244b84df2736e2a5fc1f9e169b5f7a620494184d99e4f3"


def test_unified_curriculum_blocks_early_stopping_until_frozen_base_segment() -> None:
    store = object.__new__(UnifiedScatteringTrainingStore)
    contract = store.training_lifecycle_contract(25_000)
    assert contract == {
        "contract": "ncls.unified-scattering-curriculum@1",
        "curriculum_steps": 20_000,
        "base_target_start_step": 17_501,
        "post_curriculum_base_start_step": 20_001,
        "early_stopping_floor_step": 20_000,
        "total_steps": 25_000,
    }
    assert store.training_lifecycle_contract(2)["early_stopping_floor_step"] == 2


def test_unified_training_step_routes_exact_base_boundaries() -> None:
    class FakeBase:
        batch_calls = 0

        @staticmethod
        def sample_batch_indices(
            train_indices: np.ndarray, batch_size: int, rng: object
        ) -> np.ndarray:
            del batch_size, rng
            return train_indices

        @classmethod
        def batch(
            cls, selected: np.ndarray, *, fields: tuple[str, ...]
        ) -> dict[str, np.ndarray]:
            del selected, fields
            cls.batch_calls += 1
            return {
                "state_index": np.asarray([0], dtype=np.int64),
                "wo": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                "wi": np.zeros((1, 64, 3), dtype=np.float32),
                "mean": np.zeros((1, 64, 3), dtype=np.float32),
                "solid_angle_weight": np.ones((1, 64), dtype=np.float32),
            }

    class FakeCurriculum:
        @staticmethod
        def batch(
            state_ids: list[str],
            view_indices: np.ndarray,
            light_indices: np.ndarray,
            *,
            training_progress: float,
        ) -> dict[str, np.ndarray]:
            del state_ids, view_indices, light_indices
            source = "base-v5" if training_progress >= 0.875 else "mollified-reference"
            radius = 0.0 if source == "base-v5" else 1.4644660941
            return {
                "wo": np.tile([[0.0, 0.0, 1.0]], (64, 1)).astype(np.float32),
                "wi": np.zeros((64, 3), dtype=np.float32),
                "response": np.zeros((64, 3), dtype=np.float32),
                "target_source": np.full(64, source, dtype=object),
                "mollification_progress": np.full(64, training_progress, dtype=np.float32),
                "mollification_radius_degrees": np.full(64, radius, dtype=np.float32),
            }

    class FakeRng:
        @staticmethod
        def choice(
            values: np.ndarray, *, size: int, replace: bool
        ) -> np.ndarray:
            del values, replace
            return np.full(size, "state", dtype=object)

        @staticmethod
        def integers(
            low: int, high: int, *, size: int, dtype: type[np.int64]
        ) -> np.ndarray:
            del low, high
            return np.zeros(size, dtype=dtype)

    store = object.__new__(UnifiedScatteringTrainingStore)
    store.base = FakeBase()
    store.curriculum = FakeCurriculum()
    store._state_ids = ("state",)
    store._state_index = {"state": 0}
    store._base_training_references = None
    store._base_training_cache = {}
    train_indices = np.asarray([[0, 0]], dtype=np.int64)
    expected = {
        17_500: ("mollified-reference", 1.4644660941),
        17_501: ("base-v5", 0.0),
        20_000: ("base-v5", 0.0),
        20_001: ("base-v5", None),
        20_002: ("base-v5", None),
    }
    for step, (source, radius) in expected.items():
        batch = store.training_batch(
            train_indices, 1, FakeRng(), step=step, total_steps=25_000
        )
        assert batch["target_source"].tolist() == [source]
        if radius is None:
            assert "mollification_radius_degrees" not in batch
        else:
            assert np.isclose(batch["mollification_radius_degrees"][0], radius)
    assert FakeBase.batch_calls == 1


def test_positive_residual_source_has_no_prediction_clamp_or_lobe_fallback() -> None:
    source = (PROJECT_ROOT / "shaders/ncls/backends/unified_neural/unified_neural_core.slang").read_text(encoding="utf-8")
    decode = source[source.index("float3 nclsUnifiedDecodeResidual") : source.index("NclsUnifiedLtcProposal nclsDecodeUnifiedLtc")]
    assert "nclsUnifiedSoftplus(raw.x)" in decode
    assert "return core + residual" in decode
    assert "clamp(core" not in decode
    assert "correction" not in decode


def _single_state_model() -> UnifiedNeuralModel:
    return UnifiedNeuralModel(
        state_count=1,
        response_scale=[[1.0, 1.0, 1.0]],
        top_rows=[{
            "interface_kind": 3,
            "alpha": [0.2, 0.2],
            "relative_ior": 1.0,
            "eta": [0.0, 0.0, 0.0],
            "k": [0.0, 0.0, 0.0],
            "color": [0.5, 0.5, 0.5],
            "tangent_rotation": 0.0,
        }],
        evaluator="core-frame-neural-v1",
        runtime_class="realtime",
    )


def test_sampler_stage_detaches_everything_except_selected_head() -> None:
    model = _single_state_model()
    assert not model.nvidia_sampler_w.requires_grad and not model.ltc_sampler_w.requires_grad
    model.set_sampler_training("ltc-k2")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"ltc_sampler_w", "ltc_sampler_b"}


def test_sampler_cross_entropy_has_explicit_zero_energy_cosine_target() -> None:
    wi = torch.tensor([[[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]])
    area = torch.tensor([[1.0, 1.0]])
    evaluator = torch.zeros((1, 2, 3))
    proposal = torch.full((1, 2), 0.25, requires_grad=True)
    loss, relative = sampler_cross_entropy(evaluator, wi, area, proposal)
    assert torch.isfinite(loss) and torch.isfinite(relative)
    loss.backward()
    assert proposal.grad is not None and torch.isfinite(proposal.grad).all()


def test_sampler_correctness_protocol_is_frozen_before_formal_results() -> None:
    protocol = load_unified_sampler_protocol()
    assert protocol["coverage"]["source_role"] == "validation"
    assert protocol["coverage"]["ordinals"] == [0, 5, 10, 15]
    assert protocol["sample_pdf"]["samples_per_state_view"] == 262_144
    assert protocol["mc_unbiasedness"]["replicas"] == 64
    assert len(unified_sampler_protocol_sha256()) == 64


def test_offline_cook_protocol_fits_only_latent_without_test_role() -> None:
    config = UnifiedOfflineCookConfig.load(
        PROJECT_ROOT / "configs/evaluation/unified-offline-cook-v1.json"
    )
    assert config.steps == 2_000
    assert config.latent_initialization == "zero-v1"
    assert config.evaluation_roles == ("validation", "dense_slice")
    assert "test" not in config.evaluation_roles
    assert len(config.sha256) == 64


def test_independent_sampler_oracle_is_not_the_slang_sample_pdf_path() -> None:
    prepared = np.zeros(27, dtype=np.float64)
    wo = np.asarray([0.25, -0.1, 0.963], dtype=np.float64)
    wo /= np.linalg.norm(wo)
    wi = np.asarray([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]], dtype=np.float64)
    ltc = independent_unified_sampler_pdf(prepared, wo, wi, "ltc-k2")
    np.testing.assert_allclose(ltc, wi[:, 2] / np.pi, rtol=2e-12, atol=0.0)
    nvidia = independent_unified_sampler_pdf(
        prepared, wo, wi, "nvidia-diffuse-ggx9"
    )
    np.testing.assert_allclose(
        nvidia,
        [0.30249914481264867, 0.18856931996247672],
        rtol=2e-7,
        atol=1e-9,
    )


def test_paired_selection_evidence_is_relative_not_an_absolute_quality_gate() -> None:
    baseline = {f"state-{index:02d}": 10.0 for index in range(30)}
    candidate = {state_id: 8.0 for state_id in baseline}
    comparison = paired_state_difference(baseline, candidate, iterations=1000)
    assert comparison["credible_improvement"]
    assert not comparison["credible_regression"]


def _synthetic_selection_cell(
    evaluator: str,
    sampler: str,
    *,
    variance: float = 1.0,
    implementation: bool = True,
    evaluator_convergence: bool = True,
    sampler_convergence: bool = True,
    correctness: bool = True,
) -> dict[str, object]:
    states = {f"state-{index:02d}": 1.0 for index in range(30)}
    metrics = {
        "directional_l1_by_state": dict(states),
        "signed_energy_absolute_error_by_state": dict(states),
        "cosine_relative_variance_by_state": {
            state_id: variance for state_id in states
        },
        "single_query_time_microseconds_by_state": dict(states),
    }
    artifact = {"uri": "artifact.bin", "sha256": "a" * 64}
    return {
        "evaluator": evaluator,
        "sampler": sampler,
        "evaluator_checkpoint": dict(artifact),
        "sampler_checkpoint": dict(artifact),
        "compiled_set": dict(artifact),
        "implementation_correctness": {"passed": implementation},
        "evaluator_convergence": {"passed": evaluator_convergence},
        "sampler_convergence": {"passed": sampler_convergence},
        "sampler_correctness": {**artifact, "passed": correctness},
        "checkpoint_parity": {**artifact, "passed": True},
        "slang_implementation_sha256": "b" * 64,
        "layout_sha256": "c" * 64,
        "metrics": metrics,
        "cost": {"B_asset": 128, "B_shared": 1024},
    }


def test_unified_selection_manifest_applies_pareto_and_illegal_baseline_rules() -> None:
    cells = {
        "A": _synthetic_selection_cell(
            "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1", "nvidia-diffuse-ggx9"
        ),
        "B": _synthetic_selection_cell(
            "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1", "ltc-k2", variance=0.8
        ),
        "C": _synthetic_selection_cell(
            "core-frame-neural-v1", "nvidia-diffuse-ggx9"
        ),
        "D": _synthetic_selection_cell("core-frame-neural-v1", "ltc-k2"),
    }
    manifest = build_unified_selection_manifest(
        cells,
        data_id="1" * 64,
        source_git_commit="4" * 40,
    )
    assert manifest["selected_cell"] == "B"
    assert len(manifest["selection_id"]) == 64

    cells["A"]["evaluator_convergence"] = {"passed": False}
    cells["B"]["evaluator_convergence"] = {"passed": False}
    cells["C"]["evaluator_convergence"] = {"passed": True}
    fallback = build_unified_selection_manifest(
        cells,
        data_id="1" * 64,
        source_git_commit="4" * 40,
    )
    assert fallback["selected_cell"] == "C"


def test_unified_selection_assembles_only_identity_matched_formal_artifacts(
    tmp_path: Path,
) -> None:
    from ncls.learning.evaluation.unified_selection import _sha256_json

    states = {
        f"state-{index:02d}": {
            "structure_family_id": (
                f"layers-01-kind-{index:02d}"
                if index < 15
                else f"layers-04-kind-{index:02d}"
            ),
            "directional_l1": 0.012 + index * 0.001,
            "signed_energy": {"signed_relative_bias": 0.01},
        }
        for index in range(30)
    }
    data_id, slang_id, layout_id = "1" * 64, "2" * 64, "3" * 64
    implementation_identity = {
        "identity_sha256": "9" * 64,
        "slang_implementation_sha256": slang_id,
        "layout_sha256": layout_id,
    }
    evaluator_sha = {"direct": "4" * 64, "core": "5" * 64}
    pipelines = {
        "direct": "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1",
        "core": "core-frame-neural-v1",
    }

    def write_hashed(name: str, value: dict[str, object], field: str = "report_sha256") -> Path:
        value[field] = _sha256_json(value)
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    audit = write_hashed("audit.json", {
        "format_name": "p1-audit-report",
        "format_version": 1,
        "scope": {"kind": "unified-scattering-03-formal-evaluation"},
        "roles": ["test"],
        "data": {"data_id": data_id},
        "checkpoints": {
            label: {
                "checkpoint": {
                    "uri": f"{label}.pt",
                    "sha256": evaluator_sha[label],
                    "pipeline": pipelines[label],
                    "implementation_identity": implementation_identity,
                },
                "roles": {
                    "test": {
                        "states": states,
                        "directional_l1_by_state": {"median": 0.04, "p95": 0.09},
                    }
                },
            }
            for label in ("direct", "core")
        },
    })
    input_cells: dict[str, object] = {}
    for cell_id, label, sampler, variance in (
        ("A", "direct", "nvidia-diffuse-ggx9", 1.0),
        ("B", "direct", "ltc-k2", 0.8),
        ("C", "core", "nvidia-diffuse-ggx9", 1.0),
        ("D", "core", "ltc-k2", 1.0),
    ):
        sampler_sha = ("6" if cell_id in "AC" else "7") * 64
        implementation = write_hashed(f"{cell_id}-implementation.json", {
            "format_name": "unified-method-correctness-report",
            "format_version": 1,
            "status": "complete",
            "passed": True,
            "data_id": data_id,
            "pipeline": pipelines[label],
            "slang_implementation_sha256": slang_id,
            "layout_sha256": layout_id,
        })
        evaluator_convergence = write_hashed(
            f"{cell_id}-evaluator-convergence.json",
            {
                "format_name": "unified-training-convergence-report",
                "format_version": 1,
                "status": "complete",
                "stage": "evaluator",
                "passed": True,
                "data_id": data_id,
                "pipeline": pipelines[label],
                "checkpoint": {
                    "uri": f"{label}.pt",
                    "sha256": evaluator_sha[label],
                },
                "slang_implementation_sha256": slang_id,
                "layout_sha256": layout_id,
            },
        )
        sampler_convergence = write_hashed(
            f"{cell_id}-sampler-convergence.json",
            {
                "format_name": "unified-training-convergence-report",
                "format_version": 1,
                "status": "complete",
                "stage": "sampler",
                "passed": True,
                "data_id": data_id,
                "pipeline": pipelines[label],
                "sampler": sampler,
                "evaluator_checkpoint": {
                    "uri": f"{label}.pt",
                    "sha256": evaluator_sha[label],
                },
                "checkpoint": {
                    "uri": f"{cell_id}.pt",
                    "sha256": sampler_sha,
                },
                "slang_implementation_sha256": slang_id,
                "layout_sha256": layout_id,
            },
        )
        correctness = write_hashed(f"{cell_id}-correctness.json", {
            "format_name": "unified-sampler-correctness-report",
            "format_version": 1,
            "status": "complete",
            "passed": True,
            "data_id": data_id,
            "pipeline": pipelines[label],
            "sampler": sampler,
            "evaluator_checkpoint": {
                "uri": f"{label}.pt",
                "sha256": evaluator_sha[label],
                "implementation_identity": implementation_identity,
            },
            "sampler_checkpoint": {
                "uri": f"{cell_id}.pt",
                "sha256": sampler_sha,
                "implementation_identity": implementation_identity,
            },
            "slang_implementation_sha256": slang_id,
            "layout_sha256": layout_id,
            "cases": [
                {
                    "state_id": state_id,
                    "mc_unbiasedness": {"cosine_relative_variance": variance},
                }
                for state_id in states
                for _ in range(4)
            ],
        })
        compiled = write_hashed(f"{cell_id}-compiled.json", {
            "format_name": "unified-compiled-material-set",
            "format_version": 1,
            "pipeline": pipelines[label],
            "sampler": sampler,
            "data_id": data_id,
            "evaluator_checkpoint_sha256": evaluator_sha[label],
            "sampler_checkpoint_sha256": sampler_sha,
            "evaluator_implementation_identity": implementation_identity,
            "sampler_implementation_identity": implementation_identity,
            "slang_implementation_sha256": slang_id,
            "layout_sha256": layout_id,
            "cost": {"B_asset": 128, "B_shared": 1024},
        }, field="compiled_set_id")
        compiled_id = json.loads(compiled.read_text(encoding="utf-8"))["compiled_set_id"]
        parity = write_hashed(f"{cell_id}-parity.json", {
            "format_name": "unified-checkpoint-parity-report",
            "format_version": 1,
            "passed": True,
            "pipeline": pipelines[label],
            "sampler": sampler,
            "data_id": data_id,
            "compiled_set_id": compiled_id,
            "evaluator_checkpoint_sha256": evaluator_sha[label],
            "sampler_checkpoint_sha256": sampler_sha,
            "slang_implementation_sha256": slang_id,
            "layout_sha256": layout_id,
        })
        benchmark = write_hashed(f"{cell_id}-benchmark.json", {
            "schema": {"name": "p1-query-benchmark", "version": 1},
            "data_id": data_id,
            "pipeline": pipelines[label],
            "checkpoint_sha256": evaluator_sha[label],
            "single_query_time_microseconds_by_state": {
                state_id: 1.0 for state_id in states
            },
        })
        input_cells[cell_id] = {
            "audit": str(audit),
            "checkpoint_label": label,
            "implementation_correctness": str(implementation),
            "evaluator_convergence": str(evaluator_convergence),
            "sampler_convergence": str(sampler_convergence),
            "sampler_correctness": str(correctness),
            "benchmark": str(benchmark),
            "compiled": str(compiled),
            "parity": str(parity),
        }
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({
        "format_name": "unified-selection-inputs",
        "format_version": 1,
        "cells": input_cells,
    }), encoding="utf-8")
    manifest = build_unified_selection_from_artifacts(
        inputs,
        tmp_path / "selection.json",
        source_git_commit="8" * 40,
    )
    assert manifest["selected_cell"] == "B"
    assert manifest["cells"]["B"]["checkpoint_parity"]["passed"]


def test_compiled_material_record_uses_json_abi_offsets() -> None:
    row = {
        "interface_kind": 2,
        "alpha": [1.0, 1.0],
        "relative_ior": 1.0,
        "eta": [0.0, 0.0, 0.0],
        "k": [0.0, 0.0, 0.0],
        "color": [0.2, 0.3, 0.4],
        "tangent_rotation": 0.0,
    }
    record = pack_unified_record(
        row,
        torch.arange(16).numpy(),
        torch.tensor([1.0, 2.0, 3.0]).numpy(),
        3,
    )
    fields = UNIFIED_LAYOUT["compiled_material"]["fields"]
    assert len(record) == 128
    assert struct.unpack_from("<I", record, fields["layout_version"]["offset"])[0] == 1
    assert struct.unpack_from("<I", record, fields["flags"]["offset"])[0] == 3


def test_shared_parameter_offsets_are_generated_from_the_model() -> None:
    model = _single_state_model()
    payload, layout = pack_unified_shared_parameters(
        model, "nvidia-diffuse-ggx9"
    )
    assert "nvidia_sampler_w" in layout and "ltc_sampler_w" not in layout
    ordered = sorted(layout.items(), key=lambda item: item[1]["offset_elements"])
    expected_offset = 0
    for _name, record in ordered:
        assert record["offset_elements"] == expected_offset
        expected_offset += record["element_count"]
    assert len(payload) == expected_offset * 2
    costs = create_pipeline("core-frame-neural-v1").parameter_costs(model)
    adapter = _runtime_adapter(
        layout,
        record_stride=128,
        state_stride=64,
        cost=costs,
    )
    descriptor = BackendDescriptor.from_dict(adapter["backend_descriptor"])
    assert descriptor.capabilities & BackendCapability.SAMPLE
    assert descriptor.capabilities & BackendCapability.PDF
    assert adapter["shader_defines"]["NCLS_UNIFIED_PREPARE_W0_OFFSET"] == str(
        layout["prepare_w0"]["offset_elements"]
    )
    assert adapter["compiled_material_stride"] == 128
    assert adapter["packed_state_stride"] == 64
