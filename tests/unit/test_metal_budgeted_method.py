from __future__ import annotations

import torch

from ncls.learning.conformance import validate_objective_outputs
from ncls.learning.batches import (
    EvaluatorBatch,
    MethodSamplerBatch,
    TrainingConditioning,
)
from ncls.learning.methods.metal.method import METHOD
from ncls.learning.methods.metal.model import METAL_BUDGETED_REQUIRED_CONTEXT


def _values(batch: int = 2) -> dict[str, torch.Tensor]:
    slots, patch = 4, 8
    presence = torch.zeros((batch, 32), dtype=torch.int64)
    presence[:, :12] = 1
    uv = torch.tensor([[0.2, 0.7], [0.73, 0.11]])[:batch]
    patches = torch.rand(batch, slots, 2, 4, patch, patch)
    return {
        "source_index": torch.arange(batch, dtype=torch.int64),
        "wo": torch.nn.functional.normalize(
            torch.tensor([[0.25, -0.1, 1.0], [-0.15, 0.2, 1.0]])[:batch], dim=1
        ),
        "uv": uv,
        "uv_dx": torch.tensor([[1.0 / 4096.0, 0.0]]).expand(batch, -1).clone(),
        "uv_dy": torch.tensor([[0.0, 1.0 / 4096.0]]).expand(batch, -1).clone(),
        "paired_uv": uv + torch.tensor([[1.0 / 4096.0, 0.0]]),
        "paired_uv_dx": torch.tensor([[1.0 / 4096.0, 0.0]]).expand(batch, -1).clone(),
        "paired_uv_dy": torch.tensor([[0.0, 1.0 / 4096.0]]).expand(batch, -1).clone(),
        "mip_level": torch.tensor([0.35, 1.65])[:batch],
        "metal_mip_fraction": torch.tensor([0.35, 0.65])[:batch],
        "metal_texture_patches": patches,
        "metal_paired_texture_patches": patches + 0.03,
        "metal_texture_slot_mask": torch.ones(batch, slots, dtype=torch.bool),
        "metal_texture_role_class": torch.tensor([[0, 1, 2, 3]]).expand(batch, -1).clone(),
        "metal_graph_index": torch.arange(batch, dtype=torch.int64),
        "metal_schema_index": torch.arange(batch, dtype=torch.int64),
        "metal_recipe_index": torch.arange(batch, dtype=torch.int64),
        "metal_identity_index": torch.arange(batch, dtype=torch.int64),
        "metal_finish_index": torch.arange(batch, dtype=torch.int64),
        "metal_asset_index": torch.arange(batch, dtype=torch.int64),
        "metal_typed_semantic_id": torch.arange(32, dtype=torch.int64)[None].expand(batch, -1).clone(),
        "metal_typed_type_id": torch.remainder(
            torch.arange(32, dtype=torch.int64), 8
        )[None].expand(batch, -1).clone(),
        "metal_typed_responsibility_id": torch.remainder(
            torch.arange(32, dtype=torch.int64), 6
        )[None].expand(batch, -1).clone(),
        "metal_typed_discrete": torch.remainder(
            torch.arange(32, dtype=torch.int64), 7
        )[None].expand(batch, -1).clone(),
        "metal_typed_continuous": torch.linspace(-1.0, 1.0, batch * 32 * 4).reshape(batch, 32, 4),
        "metal_typed_presence": presence,
        "metal_canonical_optical": torch.linspace(0.1, 0.9, batch * 16).reshape(batch, 16),
        "metal_access_state": torch.tensor(
            [
                [1.2, 0.8, 0.1, -0.2, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0],
                [0.9, 1.1, -0.1, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0],
            ]
        )[:batch],
        "metal_frame_state": torch.zeros(batch, 8),
        "metal_distribution_id": torch.zeros(batch, dtype=torch.int64),
    }


def _batches() -> dict[str, EvaluatorBatch | MethodSamplerBatch]:
    values = _values()
    snapshot_ids = ("a" * 64, "b" * 64)
    evaluator_conditioning = TrainingConditioning(
        "mdl.program@1", snapshot_ids, values, {"fixture": True}
    )
    wi = torch.nn.functional.normalize(
        torch.tensor([[[0.1, 0.3, 1.0]], [[-0.2, 0.4, 1.0]]]), dim=-1
    )
    target = torch.tensor([[[0.7, 0.5, 0.3]], [[0.3, 0.55, 0.8]]])
    evaluator = EvaluatorBatch(
        evaluator_conditioning, wi, target, target + 0.02
    )
    sampler_values = {
        name: value
        for name, value in values.items()
        if not name.startswith("paired_")
        and name != "metal_paired_texture_patches"
    }
    sampler_conditioning = TrainingConditioning(
        "mdl.program@1", snapshot_ids, sampler_values, {"fixture": True}
    )
    sampler = MethodSamplerBatch(
        sampler_conditioning, torch.tensor([[0.31, 0.77], [0.81, 0.19]])
    )
    return {"evaluator": evaluator, "sampler": sampler}


def _phase(name: str = "joint-response-fit") -> dict[str, object]:
    return {
        "name": name,
        "phase_step": 3,
        "recipes": {
            "appearance_calibration": {
                "schema": "train-only-reference-rgb-percentiles@1",
                "route": "evaluator",
                "sample_count": 16384,
                "seed": 2026090401,
                "scale_percentile": 0.5,
                "peak_percentile": 0.95,
                "scale_clamp": [2.0**-12, 2.0**8],
            },
            "proposal_weight": {
                "schema": "linear-nonzero-ramp@1",
                "start": 0.05,
                "end": 0.5,
                "ramp_steps": 10,
            },
        },
    }


def _initialize_calibration(model: torch.nn.Module) -> None:
    target = torch.linspace(0.01, 1.0, 16384)[:, None].expand(-1, 3).clone()
    result = METHOD.initialize_training_state(
        model,
        {"appearance-calibration": {"target_f": target}},
        {
            "schema": "ncls.train-only-initialization@1",
            "training_config_sha256": "a" * 64,
            "reference_program_identity": "b" * 64,
            "reference_execution_plan_identity": "c" * 64,
            "native_asset_collection_identity": "d" * 64,
            "query_stream_identity": "e" * 64,
            "requests": [],
        },
    )
    assert result["appearance_calibration_sample_count"] == 16384


def test_budgeted_method_descriptor_and_parameter_registry_are_exact() -> None:
    descriptor = METHOD.descriptor
    model = METHOD.create_trainable(METAL_BUDGETED_REQUIRED_CONTEXT)
    groups = METHOD.parameter_registry(model)
    assert descriptor.method_key == "metal-budgeted-neural-material"
    assert descriptor.runtime_abi == "ncls.metal-budgeted-method@3"
    assert descriptor.cost_claims["C_eval_macs"] == 11_392
    assert descriptor.cost_claims["B_prepared"] == 160
    assert descriptor.cost_claims["P_trainable"] == 30_825
    assert descriptor.cost_claims["P_runtime_prepare_evaluate"] == 14_313
    assert descriptor.cost_claims["B_runtime_fp16_weights"] == 28_626


def test_budgeted_joint_objective_reports_standard_losses_and_gradients() -> None:
    torch.manual_seed(20260904)
    model = METHOD.create_trainable(METAL_BUDGETED_REQUIRED_CONTEXT)
    _initialize_calibration(model)
    loss, metrics = METHOD.training_objective(
        model, _batches(), _phase()
    )
    validate_objective_outputs(
        METHOD.descriptor,
        "joint-response-fit",
        metrics,
    )
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert metrics["loss/optimization_total"].ndim == 0
    assert metrics["loss/appearance"].ndim == 0
    assert metrics["loss/proposal"].ndim == 0
    assert metrics["loss/proposal_weight"] > 0.0
    for key in (
        "appearance/log_rgb",
        "appearance/linear_rgb",
        "appearance/chroma",
        "appearance/peak_rgb",
        "appearance/spatial_gradient",
        "appearance/core",
        "appearance/semantic_runtime",
    ):
        assert key in metrics
        assert bool(torch.isfinite(metrics[key]))
    for group, parameters in METHOD.parameter_registry(model).items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients, group
        assert all(bool(torch.isfinite(value).all()) for value in gradients), group
        assert any(bool(torch.count_nonzero(value)) for value in gradients), group


def test_budgeted_proposal_objective_detaches_nonproposal_parameters() -> None:
    torch.manual_seed(20260904)
    model = METHOD.create_trainable(METAL_BUDGETED_REQUIRED_CONTEXT)
    loss, metrics = METHOD._proposal_objective(
        model, _batches(), qat=False
    )
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert bool(torch.isfinite(metrics["proposal/density_nll"]))
    for group, parameters in METHOD.parameter_registry(model).items():
        gradients = [parameter.grad for parameter in parameters]
        if group == "proposal_sampler":
            assert all(value is not None for value in gradients)
            assert any(bool(torch.count_nonzero(value)) for value in gradients)
        else:
            assert all(value is None for value in gradients)


def test_budgeted_qat_quantizes_weights_and_direct_auxiliary_stays_training_only() -> None:
    context = {
        **METAL_BUDGETED_REQUIRED_CONTEXT,
        "profile_id": "metal_budgeted_direct_control_v3",
    }
    model = METHOD.create_trainable(context)
    _initialize_calibration(model)
    loss, metrics = METHOD.training_objective(
        model, _batches(), _phase("deployment-qat-refine")
    )
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert bool(torch.isfinite(metrics["qat/runtime_weight_mae"]))
    assert float(metrics["appearance/direct_core_auxiliary"]) > 0.0


def test_budgeted_calibration_is_train_only_state_and_checkpoint_visible() -> None:
    model = METHOD.create_trainable(METAL_BUDGETED_REQUIRED_CONTEXT)
    requests = METHOD.initialization_requests(
        {
            "phases": [_phase()],
        }
    )
    assert requests[0].sample_count == 16384
    assert not bool(model.appearance_calibrated.item())
    _initialize_calibration(model)
    assert bool(model.appearance_calibrated.item())
    assert len(model.appearance_calibration_identity_hex) == 64
    state = METHOD.export_training_state(model)
    restored = METHOD.create_trainable(METAL_BUDGETED_REQUIRED_CONTEXT)
    METHOD.restore_training_state(restored, state)
    assert restored.appearance_calibration_identity_hex == (
        model.appearance_calibration_identity_hex
    )


def test_all_budgeted_profiles_match_the_public_checkpoint_tensor_schema() -> None:
    for profile_id in (
        "metal_budgeted_hybrid_v3",
        "metal_budgeted_direct_control_v3",
        "metal_budgeted_hybrid_role_detail_v4",
        "metal_budgeted_hybrid_center_detail_v5",
        "metal_budgeted_hybrid_dual_local_v6",
    ):
        model = METHOD.create_trainable(
            {**METAL_BUDGETED_REQUIRED_CONTEXT, "profile_id": profile_id}
        )
        state = METHOD.export_training_state(model)
        assert set(state) == {
            field.name for field in METHOD.descriptor.tensor_state_schema
        }
