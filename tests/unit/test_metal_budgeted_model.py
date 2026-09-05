from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from ncls.learning.methods.metal.model import (
    MetalBudgetedModel,
    pack_metal_budgeted_prepared_state,
    pack_metal_budgeted_program_state,
)
from ncls.learning.methods.metal.profile import (
    METAL_BUDGETED_CENTER_DETAIL_PROFILE,
    METAL_BUDGETED_DIRECT_PROFILE,
    METAL_BUDGETED_DUAL_LOCAL_PROFILE,
    METAL_BUDGETED_HYBRID_PROFILE,
    METAL_BUDGETED_ROLE_DETAIL_PROFILE,
)
from ncls.learning.methods.metal.compiler import (
    MetalBudgetedOptimizedProgramStateControl,
)
from ncls.learning.methods.metal.sampler import (
    metal_budgeted_proposal_pdf,
    metal_budgeted_sample_proposal,
)


def _conditioning(batch: int = 2) -> dict[str, torch.Tensor]:
    slots, patch = 4, 8
    presence = torch.zeros((batch, 32), dtype=torch.int64)
    presence[:, :12] = 1
    return {
        "source_index": torch.arange(batch, dtype=torch.int64),
        "wo": torch.nn.functional.normalize(
            torch.tensor([[0.25, -0.1, 1.0], [-0.15, 0.2, 1.0]])[:batch], dim=1
        ),
        "uv": torch.tensor([[0.2, 0.7], [0.73, 0.11]])[:batch],
        "uv_dx": torch.tensor([[1.0 / 1024.0, 0.0]]).expand(batch, -1).clone(),
        "uv_dy": torch.tensor([[0.0, 1.0 / 1024.0]]).expand(batch, -1).clone(),
        "mip_level": torch.tensor([0.35, 1.65])[:batch],
        "metal_mip_fraction": torch.tensor([0.35, 0.65])[:batch],
        "metal_texture_patches": torch.rand(batch, slots, 2, 4, patch, patch),
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
        "metal_frame_state": torch.tensor(
            [[1.0, 0.25, 0.0, 1.0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]]
        )[:batch],
        "metal_distribution_id": torch.zeros(batch, dtype=torch.int64),
    }


def _directions(batch: int) -> torch.Tensor:
    values = torch.tensor(
        [[[0.1, 0.3, 1.0], [-0.35, 0.05, 1.0], [0.7, -0.2, 0.2]]]
    ).expand(batch, -1, -1)
    return torch.nn.functional.normalize(values, dim=-1)


def test_budgeted_compiler_masks_absent_payload_and_bypasses_deterministic_state() -> None:
    torch.manual_seed(20260904)
    model = MetalBudgetedModel().eval()
    values = _conditioning(2)
    edited = dict(values)
    edited["metal_typed_continuous"] = values["metal_typed_continuous"].clone()
    edited["metal_typed_continuous"][:, 20:] = 100.0
    edited["metal_access_state"] = values["metal_access_state"].clone()
    edited["metal_access_state"][:, 0] += 2.0
    with torch.no_grad():
        base = model.compile_program_state(values)
        changed = model.compile_program_state(edited)
    torch.testing.assert_close(base.compiler_condition, changed.compiler_condition)
    torch.testing.assert_close(changed.access_state, edited["metal_access_state"])
    torch.testing.assert_close(changed.frame_state, edited["metal_frame_state"])
    assert torch.equal(changed.resource_variant, edited["metal_asset_index"])


def test_budgeted_prepare_packs_exact_160_bytes_and_uses_both_asset_planes() -> None:
    torch.manual_seed(20260904)
    model = MetalBudgetedModel().eval()
    values = _conditioning(2)
    with torch.no_grad():
        program = model.compile_program_state(values)
        asset = model.sample_asset(values, program)
        prepared = model.prepare_from_components(program, asset, values["wo"])
        program_fp16, program_u32 = pack_metal_budgeted_program_state(program)
        prepared_fp16, prepared_u32 = pack_metal_budgeted_prepared_state(prepared)
        replaced = dict(values)
        replaced["metal_budgeted_detail"] = torch.zeros(2, 4)
        replaced["metal_budgeted_context"] = torch.ones(2, 4)
        replaced_asset = model.sample_asset(replaced, program)
        replaced_prepared = model.prepare_from_components(
            program, replaced_asset, values["wo"]
        )
    assert program_fp16.numel() * program_fp16.element_size() + program_u32.numel() * program_u32.element_size() == 2 * 160
    assert prepared_fp16.numel() * prepared_fp16.element_size() + prepared_u32.numel() * prepared_u32.element_size() == 2 * 160
    assert prepared.semantic_state.shape == (2, 24)
    assert prepared.analytic_lobes.shape == (2, 2, 8)
    assert prepared.proposal_state.shape == (2, 3, 4)
    assert not torch.equal(prepared.semantic_state, replaced_prepared.semantic_state)
    assert not torch.equal(prepared.frames, replaced_prepared.frames)
    gram = prepared.frames @ prepared.frames.transpose(-1, -2)
    torch.testing.assert_close(
        gram,
        torch.eye(3).expand_as(gram),
        rtol=1e-5,
        atol=1e-5,
    )


def test_budgeted_hybrid_and_direct_share_shape_and_produce_nonnegative_rgb() -> None:
    torch.manual_seed(20260904)
    hybrid = MetalBudgetedModel(METAL_BUDGETED_HYBRID_PROFILE).eval()
    direct = MetalBudgetedModel(METAL_BUDGETED_DIRECT_PROFILE).eval()
    direct.load_state_dict(hybrid.state_dict())
    values = _conditioning(2)
    wi = _directions(2)
    with torch.no_grad():
        hybrid_eval = hybrid.evaluate_prepared(
            hybrid.prepare(values), values["wo"], wi
        )
        direct_eval = direct.evaluate_prepared(
            direct.prepare(values), values["wo"], wi
        )
    assert hybrid_eval.f.shape == direct_eval.f.shape == (2, 3, 3)
    assert bool(torch.isfinite(hybrid_eval.f).all())
    assert bool(torch.isfinite(direct_eval.f).all())
    assert bool((hybrid_eval.f >= 0.0).all())
    assert bool((direct_eval.f >= 0.0).all())
    assert bool((hybrid_eval.rgb_gate >= 0.0).all())
    assert bool((hybrid_eval.rgb_gate <= 1.0).all())
    assert torch.count_nonzero(direct_eval.rgb_gate) == 0
    torch.testing.assert_close(direct_eval.f, direct_eval.positive_f)
    assert bool((direct_eval.direct_core_auxiliary > 0.0).all())
    assert [tuple(parameter.shape) for parameter in hybrid.evaluator.parameters()] == [
        tuple(parameter.shape) for parameter in direct.evaluator.parameters()
    ]


def test_budgeted_evaluator_consumes_the_full_prepared_semantic_state() -> None:
    torch.manual_seed(20260905)
    model = MetalBudgetedModel().eval()
    values = _conditioning(2)
    wi = _directions(2)
    with torch.no_grad():
        prepared = model.prepare(values)
        baseline = model.evaluate_prepared(prepared, values["wo"], wi).f
        changed = prepared.semantic_state.clone()
        changed[:, 8:] += 0.5
        replaced = replace(prepared, semantic_state=changed)
        modified = model.evaluate_prepared(replaced, values["wo"], wi).f
    assert not torch.equal(baseline, modified)


def test_budgeted_detail_plane_has_a_direct_frame_semantic_path() -> None:
    torch.manual_seed(20260905)
    model = MetalBudgetedModel().eval()
    values = _conditioning(2)
    with torch.no_grad():
        program = model.compile_program_state(values)
        asset = model.sample_asset(values, program, qat=False)
        for parameter in model.prepared_model.semantic_decoder.parameters():
            parameter.zero_()
        prepared = model.prepare_from_components(program, asset, values["wo"])
    torch.testing.assert_close(prepared.semantic_state[:, :4], asset.detail)
    assert torch.count_nonzero(prepared.semantic_state[:, 4:]) == 0


def test_budgeted_role_detail_keeps_role_changes_in_their_channel() -> None:
    torch.manual_seed(20260905)
    model = MetalBudgetedModel(METAL_BUDGETED_ROLE_DETAIL_PROFILE).eval()
    values = _conditioning(1)
    changed = dict(values)
    changed["metal_texture_patches"] = values["metal_texture_patches"].clone()
    changed["metal_texture_patches"][:, 0, :, :, 3:5, 3:5] += 2.0
    with torch.no_grad():
        program = model.compile_program_state(values)
        baseline = model.sample_asset(values, program, qat=False).detail
        modified = model.sample_asset(changed, program, qat=False).detail

    assert not torch.equal(baseline[:, :1], modified[:, :1])
    torch.testing.assert_close(baseline[:, 1:], modified[:, 1:])


def test_budgeted_center_detail_uses_the_requested_patch_texel() -> None:
    torch.manual_seed(20260905)
    baseline_model = MetalBudgetedModel(METAL_BUDGETED_HYBRID_PROFILE).eval()
    center_model = MetalBudgetedModel(METAL_BUDGETED_CENTER_DETAIL_PROFILE).eval()
    center_model.load_state_dict(baseline_model.state_dict())
    values = _conditioning(1)
    values["metal_mip_fraction"].zero_()
    values["metal_texture_patches"].zero_()
    values["metal_texture_patches"][:, :, 0, :, 4, 4] = 1.0
    with torch.no_grad():
        program = baseline_model.compile_program_state(values)
        baseline = baseline_model.sample_asset(values, program, qat=False).detail
        centered = center_model.sample_asset(values, program, qat=False).detail

    assert not torch.equal(baseline, centered)


def test_budgeted_dual_local_detail_observes_signed_x_derivative() -> None:
    model = MetalBudgetedModel(METAL_BUDGETED_DUAL_LOCAL_PROFILE).eval()
    values = _conditioning(1)
    values["metal_texture_slot_mask"][:, 1:] = False
    values["metal_texture_patches"].zero_()
    with torch.no_grad():
        for parameter in model.asset.detail_encoder.parameters():
            parameter.zero_()
        model.asset.detail_encoder[0].weight[0, 4] = 1.0
        model.asset.detail_encoder[2].weight[0, 0] = 1.0
        values["metal_texture_patches"][:, 0, 0, 0, 4, 5] = 1.0
        positive, _, _ = model.asset._encode_source_patches(
            values, torch.zeros(1, dtype=torch.int64)
        )
        values["metal_texture_patches"][:, 0, 0, 0, 4, 5] = 0.0
        values["metal_texture_patches"][:, 0, 0, 0, 4, 3] = 1.0
        negative, _, _ = model.asset._encode_source_patches(
            values, torch.zeros(1, dtype=torch.int64)
        )

    assert positive[0, 0] > 0.0
    assert negative[0, 0] < 0.0
    torch.testing.assert_close(positive[:, 1:], torch.zeros_like(positive[:, 1:]))
    torch.testing.assert_close(negative[:, 1:], torch.zeros_like(negative[:, 1:]))


def test_optimized_program_state_control_cannot_change_deterministic_contract() -> None:
    model = MetalBudgetedModel().eval()
    program = model.compile_program_state(_conditioning(2))
    control = MetalBudgetedOptimizedProgramStateControl(program)
    with torch.no_grad():
        control.compiler_condition.add_(0.25)
        control.primary_lobe.add_(0.1)
    optimized = control()
    assert not torch.equal(program.compiler_condition, optimized.compiler_condition)
    torch.testing.assert_close(program.access_state, optimized.access_state)
    torch.testing.assert_close(program.frame_state, optimized.frame_state)
    assert torch.equal(program.resource_variant, optimized.resource_variant)
    assert torch.equal(program.resource_and_flags, optimized.resource_and_flags)


def test_beckmann_exception_is_deterministic_and_shared_with_proposal() -> None:
    model = MetalBudgetedModel().eval()
    values = _conditioning(2)
    values["metal_distribution_id"] = torch.tensor([1, 0], dtype=torch.int64)
    with torch.no_grad():
        program = model.compile_program_state(values)
        prepared = model.prepare_from_components(
            program, model.sample_asset(values, program), values["wo"]
        )
    assert torch.equal(program.resource_and_flags[:, 6], torch.tensor([1, 0]))
    assert torch.equal(
        prepared.proposal_state[:, :, 3],
        torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
    )
    invalid = _conditioning(1)
    invalid["metal_distribution_id"] = torch.tensor([2], dtype=torch.int64)
    with pytest.raises(ValueError, match="GGX=0 or Beckmann=1"):
        model.compile_program_state(invalid)


def test_budgeted_half_degeneracy_and_beckmann_path_fail_closed_finitely() -> None:
    torch.manual_seed(20260904)
    model = MetalBudgetedModel().eval()
    values = _conditioning(1)
    prepared = model.prepare(values)
    wi = -values["wo"][:, None, :]
    evaluated = model.evaluate_prepared(prepared, values["wo"], wi)
    assert not bool(evaluated.valid.any())
    assert bool(torch.isfinite(evaluated.f).all())
    assert torch.count_nonzero(evaluated.f) == 0
    valid_wi = _directions(1)
    valid = model.evaluate_prepared(prepared, values["wo"], valid_wi)
    assert bool(torch.isfinite(valid.analytic_f).all())
    assert bool((valid.analytic_f >= 0.0).all())


def test_budgeted_three_component_sampler_is_normalized_and_sample_pdf_matches() -> None:
    batch = 3
    state = torch.tensor(
        [
            [0.58, 0.22, 0.4, 0.0],
            [0.40, 0.35, 0.55, 1.0],
            [0.02, 1.0, 1.0, 2.0],
        ]
    )[None].expand(batch, -1, -1).clone()
    angles = torch.tensor([[0.0, 0.4]]).expand(batch, -1)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    zeros, ones = torch.zeros_like(cosine), torch.ones_like(cosine)
    frames = torch.stack(
        (
            torch.stack((cosine, sine, zeros), dim=-1),
            torch.stack((-sine, cosine, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )
    wo = torch.nn.functional.normalize(
        torch.tensor([[0.2, -0.1, 1.0]]).expand(batch, -1), dim=1
    )
    z_count, phi_count = 64, 128
    z = (torch.arange(z_count, dtype=torch.float32) + 0.5) / z_count
    phi = 2.0 * math.pi * (
        torch.arange(phi_count, dtype=torch.float32) + 0.5
    ) / phi_count
    zz, pp = torch.meshgrid(z, phi, indexing="ij")
    radius = torch.sqrt(torch.clamp(1.0 - zz.square(), min=0.0))
    wi = torch.stack(
        (radius * torch.cos(pp), radius * torch.sin(pp), zz), dim=-1
    ).reshape(1, -1, 3).expand(batch, -1, -1)
    density = metal_budgeted_proposal_pdf(
        state, frames, torch.ones(batch, dtype=torch.bool), wo, wi
    )
    integral = 2.0 * math.pi * density.forward.mean(dim=1)
    torch.testing.assert_close(integral, torch.ones_like(integral), rtol=6e-3, atol=6e-3)

    sample_u = torch.tensor([[0.13, 0.77], [0.61, 0.29], [0.995, 0.41]])
    sampled = metal_budgeted_sample_proposal(
        state, frames, torch.ones(batch, dtype=torch.bool), wo, sample_u
    )
    independent = metal_budgeted_proposal_pdf(
        state, frames, torch.ones(batch, dtype=torch.bool), wo, sampled.wi
    )
    assert bool(sampled.valid.all())
    torch.testing.assert_close(sampled.forward_pdf, independent.forward)
    torch.testing.assert_close(sampled.reverse_pdf, independent.reverse)


def test_budgeted_sampler_uses_fallback_tangent_for_collinear_reflection_axis() -> None:
    root_half = math.sqrt(0.5)
    tangent = torch.tensor([root_half, 0.0, -root_half])
    bitangent = torch.tensor([0.0, 1.0, 0.0])
    normal = torch.tensor([root_half, 0.0, root_half])
    frame = torch.stack((tangent, bitangent, normal))
    frames = frame[None, None].expand(1, 2, -1, -1).clone()
    wo = torch.tensor([[-root_half, 0.0, root_half]])
    state = torch.tensor(
        [[[0.98, 0.2, 0.3, 0.0], [0.0, 0.4, 0.4, 0.0], [0.02, 1.0, 1.0, 2.0]]]
    )
    sampled = metal_budgeted_sample_proposal(
        state,
        frames,
        torch.ones(1, dtype=torch.bool),
        wo,
        torch.tensor([[0.1, 0.25]]),
    )
    assert bool(sampled.valid.all())
    independent = metal_budgeted_proposal_pdf(
        state, frames, torch.ones(1, dtype=torch.bool), wo, sampled.wi
    )
    torch.testing.assert_close(sampled.forward_pdf, independent.forward)


def test_budgeted_model_all_parameters_receive_finite_nonzero_gradient() -> None:
    torch.manual_seed(20260904)
    model = MetalBudgetedModel()
    values = _conditioning(2)
    prepared = model.prepare(values)
    evaluated = model.evaluate_prepared(prepared, values["wo"], _directions(2))
    loss = (
        evaluated.f.mean()
        + 0.01 * prepared.semantic_state.square().mean()
        + 0.01 * prepared.proposal_state.square().mean()
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool(torch.count_nonzero(parameter.grad)), name
