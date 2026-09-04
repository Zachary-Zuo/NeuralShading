from __future__ import annotations

import pytest
import torch

from ncls.learning.batches import (
    AssetTileBatch,
    EvaluatorBatch,
    MethodSamplerBatch,
    TrainingConditioning,
)
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalModel,
)
from ncls.learning.source_adaptation import DenseNativeAssetCollection, NativeAssetRole


pytestmark = pytest.mark.slangpy


def _conditioning(device: torch.device) -> dict[str, torch.Tensor]:
    batch, slots, patch = 1, 4, 8
    presence = torch.zeros((batch, 32), dtype=torch.int64, device=device)
    presence[:, :8] = 1
    return {
        "source_index": torch.zeros(batch, dtype=torch.int64, device=device),
        "wo": torch.nn.functional.normalize(
            torch.tensor([[0.25, -0.1, 1.0]], device=device), dim=1
        ),
        "uv": torch.tensor([[0.2, 0.7]], device=device),
        "uv_dx": torch.tensor([[1.0 / 1024.0, 0.0]], device=device),
        "uv_dy": torch.tensor([[0.0, 1.0 / 1024.0]], device=device),
        "mip_level": torch.tensor([0.35], device=device),
        "metal_mip_fraction": torch.tensor([0.35], device=device),
        "metal_texture_patches": torch.rand(
            (batch, slots, 2, 4, patch, patch), device=device
        ),
        "metal_texture_slot_mask": torch.ones(
            (batch, slots), dtype=torch.bool, device=device
        ),
        "metal_texture_role_class": torch.tensor(
            [[0, 1, 2, 3]], dtype=torch.int64, device=device
        ),
        "metal_graph_index": torch.tensor([3], dtype=torch.int64, device=device),
        "metal_schema_index": torch.tensor([2], dtype=torch.int64, device=device),
        "metal_recipe_index": torch.tensor([1], dtype=torch.int64, device=device),
        "metal_identity_index": torch.tensor([5], dtype=torch.int64, device=device),
        "metal_finish_index": torch.tensor([4], dtype=torch.int64, device=device),
        "metal_asset_index": torch.tensor([2], dtype=torch.int64, device=device),
        "metal_typed_semantic_id": torch.arange(32, dtype=torch.int64, device=device)[None, :],
        "metal_typed_type_id": torch.remainder(
            torch.arange(32, dtype=torch.int64, device=device), 8
        )[None, :],
        "metal_typed_responsibility_id": torch.remainder(
            torch.arange(32, dtype=torch.int64, device=device), 6
        )[None, :],
        "metal_typed_discrete": torch.remainder(
            torch.arange(32, dtype=torch.int64, device=device), 7
        )[None, :],
        "metal_typed_continuous": torch.linspace(-1.0, 1.0, 128, device=device).reshape(1, 32, 4),
        "metal_typed_presence": presence,
        "metal_canonical_optical": torch.linspace(0.1, 0.9, 16, device=device)[None, :],
        "metal_access_state": torch.tensor(
            [[1.2, 0.8, 0.1, -0.2, 0.9238795, 0.3826834, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0]],
            device=device,
        ),
        "metal_frame_state": torch.tensor(
            [[1.0, 0.25, 0.0, 1.0, 0, 0, 0, 0]], device=device
        ),
    }


def _asset_batch(device: torch.device) -> AssetTileBatch:
    values = torch.rand((8, 8, 4), dtype=torch.float32)
    roles = (
        NativeAssetRole("color", "base-color", 0, 1, "linear", "box-mip"),
        NativeAssetRole("normal", "normal-tangent", 1, 1, "linear", "normal-renormalize"),
        NativeAssetRole("rough", "roughness", 2, 1, "linear", "box-mip"),
        NativeAssetRole("packed", "packed-correlated", 3, 1, "linear", "box-mip"),
    )
    collection = DenseNativeAssetCollection(
        ((values,),),
        ("fixture-asset",),
        "fixture-schema",
        "fixture-domain",
        "surface-uv",
        "wrap",
        roles,
    )
    request = next(collection.iter_tile_requests(0, "fixture-domain", 64, 0))
    tile = collection.acquire_tile(request, device)
    return AssetTileBatch(collection.descriptors, (tile,), {"fixture": True})


def test_full_metal_evaluator_has_finite_nonnegative_f_and_all_group_gradients() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(20260830)
    model = MetalModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to(device)
    values = _conditioning(device)
    asset_batch = _asset_batch(device)
    try:
        codec_loss, codec_metrics = model.codec_objective(asset_batch)
        spatial = model.spatial_state(values)
        pure, teacher = model.compile_program_states(values)
        prepared = model.prepare_from_components(pure, spatial, values)
        teacher_prepared = model.prepare_from_components(teacher, spatial, values)
        wi = torch.nn.functional.normalize(
            torch.tensor([[[0.1, 0.3, 1.0]]], device=device), dim=-1
        )
        evaluated = model.evaluate_prepared(prepared, values["wo"], wi)
        teacher_evaluated = model.evaluate_prepared(teacher_prepared, values["wo"], wi)
        loss = (
            codec_loss
            + evaluated.f.mean()
            + evaluated.core_f.mean()
            + evaluated.residual_lobes.mean()
            + evaluated.multiplicative.mean()
            + evaluated.free_tail.mean()
            + teacher_evaluated.f.mean()
            + prepared.proposal_state.mean() * 1e-3
        )
        loss.backward()
    finally:
        asset_batch.release()
    assert bool(torch.isfinite(evaluated.f).all())
    assert bool((evaluated.f >= 0.0).all())
    assert bool(evaluated.valid.all())
    assert evaluated.f.shape == (1, 1, 3)
    groups = METHOD_DEFINITION.parameter_registry(model)
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients, name
        assert all(bool(torch.isfinite(value).all()) for value in gradients), name
        assert any(bool(torch.any(value != 0)) for value in gradients), name
    assert all(bool(torch.isfinite(value)) for value in codec_metrics.values())


def test_typed_edit_and_bundle_replacement_are_separate_model_inputs() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    model = MetalModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to(device).eval()
    values = _conditioning(device)
    with torch.no_grad():
        base_program, _ = model.compile_program_states(values)
        edited = dict(values)
        edited["metal_typed_continuous"] = values["metal_typed_continuous"].clone()
        edited["metal_typed_continuous"][:, 0, 0] += 0.5
        edited_program, _ = model.compile_program_states(edited)
        replacement = dict(values)
        replacement["metal_texture_patches"] = 1.0 - values["metal_texture_patches"]
        base_spatial = model.spatial_state(values)
        replacement_spatial = model.spatial_state(replacement)
        replacement_program, _ = model.compile_program_states(replacement)
        discrete_base = dict(values)
        discrete_base["metal_typed_presence"] = torch.zeros_like(
            values["metal_typed_presence"]
        )
        discrete_base["metal_typed_presence"][:, 0] = 1
        discrete_base["metal_typed_type_id"] = values[
            "metal_typed_type_id"
        ].clone()
        discrete_base["metal_typed_type_id"][:, 0] = 6
        discrete_base["metal_typed_discrete"] = torch.zeros_like(
            values["metal_typed_discrete"]
        )
        discrete_base["metal_typed_continuous"] = torch.zeros_like(
            values["metal_typed_continuous"]
        )
        discrete_program, _ = model.compile_program_states(discrete_base)
        discrete_edit = dict(discrete_base)
        discrete_edit["metal_typed_discrete"] = discrete_base[
            "metal_typed_discrete"
        ].clone()
        discrete_edit["metal_typed_discrete"][:, 0] = 3
        discrete_edited_program, _ = model.compile_program_states(discrete_edit)
        absent_payload_edit = dict(discrete_base)
        absent_payload_edit["metal_typed_semantic_id"] = discrete_base[
            "metal_typed_semantic_id"
        ].clone()
        absent_payload_edit["metal_typed_semantic_id"][:, 1] = 191
        absent_payload_edit["metal_typed_discrete"] = discrete_base[
            "metal_typed_discrete"
        ].clone()
        absent_payload_edit["metal_typed_discrete"][:, 1] = 63
        absent_payload_edit["metal_typed_continuous"] = discrete_base[
            "metal_typed_continuous"
        ].clone()
        absent_payload_edit["metal_typed_continuous"][:, 1, :] = 100.0
        absent_payload_program, _ = model.compile_program_states(absent_payload_edit)
        empty = dict(discrete_base)
        empty["metal_typed_presence"] = torch.zeros_like(
            discrete_base["metal_typed_presence"]
        )
        empty_program, _ = model.compile_program_states(empty)
    assert not torch.equal(base_program.compiler_latent, edited_program.compiler_latent)
    assert not torch.equal(base_spatial.structured, replacement_spatial.structured)
    torch.testing.assert_close(
        base_program.compiler_latent, replacement_program.compiler_latent
    )
    assert not torch.equal(
        discrete_program.compiler_latent,
        discrete_edited_program.compiler_latent,
    )
    torch.testing.assert_close(
        discrete_program.compiler_latent,
        absent_payload_program.compiler_latent,
    )
    assert bool(torch.isfinite(empty_program.compiler_latent).all())


def test_full_model_bfloat16_forward_keeps_sensitive_outputs_finite() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    model = MetalModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to(device)
    values = _conditioning(device)
    wi = torch.nn.functional.normalize(
        torch.tensor([[[0.1, 0.3, 1.0]]], device=device), dim=-1
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        spatial = model.spatial_state(values)
        program, _ = model.compile_program_states(values)
        prepared = model.prepare_from_components(program, spatial, values)
        evaluated = model.evaluate_prepared(prepared, values["wo"], wi)
        loss = evaluated.f.mean()
    loss.backward()
    assert bool(torch.isfinite(evaluated.f).all())
    assert bool((evaluated.f >= 0.0).all())


def test_full_metal_sample_pdf_and_throughput_weight_share_one_prepared_state() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(20260831)
    model = MetalModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to(device)
    values = _conditioning(device)
    spatial = model.spatial_state(values)
    program, _ = model.compile_program_states(values)
    prepared = model.prepare_from_components(program, spatial, values)
    sampled = model.sample_prepared(
        prepared,
        values["wo"],
        torch.tensor([[0.731, 0.217]], device=device),
    )
    independent = model.pdf_prepared(
        prepared, values["wo"], sampled.wi
    )
    expected_weight = (
        sampled.f
        * sampled.wi[..., 2:3]
        / independent.forward[..., None]
    )
    assert bool(sampled.valid.all())
    assert bool((sampled.wi[..., 2] > 0.0).all())
    assert bool(torch.isfinite(sampled.weight).all())
    torch.testing.assert_close(sampled.forward_pdf, independent.forward)
    torch.testing.assert_close(sampled.reverse_pdf, independent.reverse)
    torch.testing.assert_close(sampled.weight, expected_weight)


def test_proposal_group_has_finite_gradients_and_descends_on_fixed_density_target() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(20260831)
    model = MetalModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to(device)
    proposal_parameters = tuple(
        METHOD_DEFINITION.parameter_registry(model)["proposal_sampler"]
    )
    optimizer = torch.optim.Adam(proposal_parameters, lr=2e-3)
    values = _conditioning(device)
    wo = values["wo"]
    target_wi = torch.stack((-wo[:, 0], -wo[:, 1], wo[:, 2]), dim=1)
    target_wi = torch.nn.functional.normalize(target_wi, dim=1)[:, None, :]

    def density_loss() -> tuple[torch.Tensor, torch.Tensor]:
        spatial = model.spatial_state(values)
        program = model.typed_compiler(values)
        prepared = model.prepare_from_components(program, spatial, values)
        density = model.pdf_prepared(prepared, wo, target_wi)
        assert bool(density.valid.all())
        return -torch.log(torch.clamp(density.forward, min=1e-12)).mean(), density.forward.mean()

    initial_loss, initial_density = density_loss()
    observed_finite_nonzero = False
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = density_loss()
        loss.backward()
        gradients = [parameter.grad for parameter in proposal_parameters]
        assert all(gradient is not None for gradient in gradients)
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        observed_finite_nonzero |= any(
            bool(torch.count_nonzero(gradient)) for gradient in gradients
        )
        optimizer.step()
    final_loss, final_density = density_loss()
    assert observed_finite_nonzero
    assert float(final_loss.detach()) < float(initial_loss.detach())
    assert float(final_density.detach()) > float(initial_density.detach())


def test_joint_proposal_objective_detaches_every_nonproposal_parameter() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(20260903)
    model = MetalModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to(device)
    values = _conditioning(device)
    conditioning = TrainingConditioning(
        "mdl.program@1", ("a" * 64,), values, {"fixture": True}
    )
    wi = torch.nn.functional.normalize(
        torch.tensor([[[0.1, 0.3, 1.0]]], device=device), dim=-1
    )
    evaluator = EvaluatorBatch(
        conditioning,
        wi,
        torch.full_like(wi, 0.25),
    )
    sampler = MethodSamplerBatch(
        conditioning,
        torch.tensor([[0.731, 0.217]], device=device),
    )
    loss, _ = METHOD_DEFINITION._proposal_objective(
        model, {"evaluator": evaluator, "sampler": sampler}
    )
    loss.backward()

    registry = METHOD_DEFINITION.parameter_registry(model)
    for group, parameters in registry.items():
        gradients = [parameter.grad for parameter in parameters]
        if group == "proposal_sampler":
            assert all(value is not None for value in gradients)
            assert any(bool(torch.count_nonzero(value)) for value in gradients)
        else:
            assert all(value is None for value in gradients), group


def test_end_to_end_step_zero_updates_evaluator_codec_teacher_and_proposal() -> None:
    pytest.importorskip("slangpy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(20260903)
    model = MetalModel.from_context(
        METAL_FUSED_REQUIRED_CONTEXT
    ).to(device)
    values = _conditioning(device)
    conditioning = TrainingConditioning(
        "mdl.program@1", ("a" * 64,), values, {"fixture": True}
    )
    wi = torch.nn.functional.normalize(
        torch.tensor([[[0.1, 0.3, 1.0]]], device=device), dim=-1
    )
    batches = {
        "asset": _asset_batch(device),
        "evaluator": EvaluatorBatch(conditioning, wi, torch.full_like(wi, 0.25)),
        "sampler": MethodSamplerBatch(
            conditioning, torch.tensor([[0.731, 0.217]], device=device)
        ),
    }
    try:
        loss, metrics = METHOD_DEFINITION.training_objective(
            model,
            batches,
            {
                "name": "joint-coarse-to-fine",
                "phase_step": 0,
                "recipes": {
                    "proposal_weight": {
                        "schema": "linear-nonzero-ramp@1",
                        "start": 0.05,
                        "end": 1.0,
                        "ramp_steps": 5000,
                    }
                },
            },
        )
        loss.backward()
    finally:
        batches["asset"].release()
    assert metrics["proposal_objective_weight"] == pytest.approx(0.05)
    assert float(metrics["response_robust_loss"]) > 0.0
    assert float(metrics["proposal_density_fit_loss"]) > 0.0
    for group, parameters in METHOD_DEFINITION.parameter_registry(model).items():
        gradients = [value.grad for value in parameters if value.grad is not None]
        assert gradients, group
        assert all(bool(torch.isfinite(value).all()) for value in gradients), group
        assert any(bool(torch.count_nonzero(value)) for value in gradients), group
