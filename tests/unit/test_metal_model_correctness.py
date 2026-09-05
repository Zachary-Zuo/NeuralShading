from dataclasses import fields, replace

import pytest
import torch

from ncls.learning.methods.metal.compiler import MetalBudgetedOptimizedProgramStateControl, MetalBudgetedProgramState
from ncls.learning.methods.metal.evaluator import _local_frames, _orthonormal_frame
from ncls.learning.methods.metal.asset import MetalBudgetedAssetSample
from ncls.learning.methods.metal.model import MetalBudgetedModel, pack_metal_budgeted_prepared_state
from ncls.learning.methods.metal.profile import METAL_SPATIAL_PROFILE


def _program():
    lobe = torch.tensor([[0., 0.3, 1., 0.015, 1., 3., 4 * torch.pi, 0.7]])
    return MetalBudgetedProgramState(
        torch.linspace(-1., 1.04, 8)[None], lobe, lobe.clone(),
        torch.linspace(-1., 1., 8)[None], torch.tensor([[0., 0.2, 0.8]]),
        torch.tensor([734]), torch.arange(8)[None], torch.arange(16.)[None], torch.arange(8.)[None], {},
    )


def test_control_zero_delta_is_exact_initial_including_angles_and_boundary_values():
    initial = _program()
    control = MetalBudgetedOptimizedProgramStateControl(initial)
    actual = control()
    for field in fields(initial):
        if field.name != "trace":
            assert torch.equal(getattr(actual, field.name), getattr(initial, field.name)), field.name
    loss = actual.compiler_condition.sum() + actual.primary_lobe.sum() + actual.spatial_scale_bias.sum() + actual.proposal_prior[:, 1].sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for name, p in control.named_parameters() if name != "secondary_lobe")
    immutable = {name: value.clone() for name, value in control.named_buffers()}
    with torch.no_grad():
        for p in control.parameters():
            p.add_(10)
    changed = control()
    for name, value in control.named_buffers():
        assert torch.equal(value, immutable[name])
    assert changed.primary_lobe[0, 6] > 4 * torch.pi
    assert torch.all(changed.primary_lobe[:, 3:5] >= 0.015)
    torch.testing.assert_close(changed.proposal_prior.sum(dim=1), torch.ones(1))


def test_control_rejects_illegal_initial_instead_of_silently_projecting_it():
    initial = _program()
    illegal = initial.primary_lobe.clone()
    illegal[0, 3] = 0.001
    with pytest.raises(ValueError, match="outside its legal domain"):
        MetalBudgetedOptimizedProgramStateControl(replace(initial, primary_lobe=illegal))


def test_frame_is_orthonormal_continuous_and_preserves_native_axes():
    z = torch.tensor([1., 0.9990001, 0.9989999, 0.7, 0.001], dtype=torch.float64)
    normal = torch.stack((torch.sqrt(1. - z.square()), torch.zeros_like(z), z), dim=1)
    frame = _orthonormal_frame(normal)
    torch.testing.assert_close(frame @ frame.transpose(-1, -2), torch.eye(3, dtype=z.dtype)[None].expand(5, -1, -1), rtol=0, atol=1e-14)
    torch.testing.assert_close(torch.cross(frame[:, 0], frame[:, 1], dim=-1), normal, rtol=0, atol=1e-14)
    torch.testing.assert_close(frame[0], torch.eye(3, dtype=z.dtype), rtol=0, atol=0)
    assert torch.linalg.vector_norm(frame[1] - frame[2]) < 2 * torch.linalg.vector_norm(normal[1] - normal[2])
    rotated = _local_frames(torch.zeros(1, 2, 2), torch.tensor([[0., torch.pi / 2]]))
    torch.testing.assert_close(rotated[0, 1, 0], torch.tensor([0., 1., 0.]), rtol=0, atol=1e-7)


def test_reverse_pdf_equals_independent_reverse_prepare_with_different_semantic_frame():
    torch.manual_seed(394)
    model = MetalBudgetedModel(METAL_SPATIAL_PROFILE)
    program = _program()
    # 合法正 prior 保证两个 specular proposal 均参与本 witness。
    program = replace(program, proposal_prior=torch.tensor([[0.4, 0.3, 0.3]]), resource_and_flags=torch.zeros(1, 8, dtype=torch.int64))
    latent = torch.randn(1, 9, 8) * 0.3
    feature = torch.cat((latent, torch.zeros(1, 9, 5), torch.ones(1, 9, 1)), dim=-1)
    asset = MetalBudgetedAssetSample(latent[:, 0, :4], latent[:, 0, 4:], torch.zeros(1, dtype=torch.int64),
                                     torch.ones(1, dtype=torch.bool), {}, feature, latent, torch.zeros(1, 8))
    wo = torch.nn.functional.normalize(torch.tensor([[0.3, -0.2, 1.]]), dim=-1)
    wi = torch.nn.functional.normalize(torch.tensor([[-0.6, 0.1, 0.7]]), dim=-1)
    forward = model.prepare_from_components(program, asset, wo)
    reverse = model.prepare_from_components(program, asset, wi)
    assert not torch.equal(forward.semantic_state, reverse.semantic_state)
    assert torch.equal(forward.proposal_frames, reverse.proposal_frames)
    assert torch.equal(forward.proposal_state, reverse.proposal_state)
    first_pdf = model.pdf_prepared(forward, wo, wi[:, None])
    second_pdf = model.pdf_prepared(reverse, wi, wo[:, None])
    assert first_pdf.valid.all() and second_pdf.valid.all()
    torch.testing.assert_close(first_pdf.reverse, second_pdf.forward, rtol=0, atol=0)
    half, flags = pack_metal_budgeted_prepared_state(forward)
    assert half.nelement() * half.element_size() + flags.nelement() * flags.element_size() == 176
    assert METAL_SPATIAL_PROFILE.runtime_prepare_dense_macs == 7664


def test_finite_zero_is_valid_but_nonfinite_internal_response_stays_invalid():
    model = MetalBudgetedModel(METAL_SPATIAL_PROFILE)
    latent = torch.zeros(1, 9, 8)
    feature = torch.cat((latent, torch.zeros(1, 9, 5), torch.ones(1, 9, 1)), dim=-1)
    asset = MetalBudgetedAssetSample(latent[:, 0, :4], latent[:, 0, 4:], torch.zeros(1, dtype=torch.int64),
                                     torch.ones(1, dtype=torch.bool), {}, feature, latent, torch.zeros(1, 8))
    direction = torch.tensor([[0., 0., 1.]])
    prepared = model.prepare_from_components(_program(), asset, direction)
    lobes = prepared.analytic_lobes.clone()
    lobes[..., 5] = 0.
    prepared = replace(prepared, analytic_lobes=lobes)
    with torch.no_grad():
        for parameter in model.evaluator.parameters():
            parameter.zero_()
        model.evaluator.layers[-1].bias[:3] = -1000.
    zero = model.evaluate_prepared(prepared, direction, direction[:, None])
    assert zero.valid.all() and torch.equal(zero.f, torch.zeros_like(zero.f))
    corrupted = prepared.semantic_state.clone()
    corrupted[:, 0] = float("nan")
    bad = replace(prepared, semantic_state=corrupted)
    invalid = model.evaluate_prepared(bad, direction, direction[:, None])
    sample = model.sample_prepared(bad, direction, torch.tensor([[0.3, 0.7]]))
    assert not invalid.valid.any() and not sample.valid.any()
    assert torch.equal(invalid.f, torch.zeros_like(invalid.f))
    assert torch.equal(sample.weight, torch.zeros_like(sample.weight))
