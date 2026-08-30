from __future__ import annotations

import math

import torch

from ncls.learning.models.metal_fused_profile import load_metal_fused_layout
from ncls.learning.models.metal_sampler import (
    METAL_PROPOSAL_COMPONENTS,
    METAL_PROPOSAL_COMPONENT_COUNT,
    METAL_PROPOSAL_DISTRIBUTION_IDS,
    METAL_PROPOSAL_FRAME_INDICES,
    METAL_PROPOSAL_SPECULAR_FLAGS,
    metal_proposal_pdf,
    metal_sample_proposal,
)


def _frames(batch: int) -> torch.Tensor:
    angles = torch.tensor((0.0, 0.35, -0.45, 0.7))
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    tangent = torch.stack((cosine, torch.zeros_like(cosine), -sine), dim=1)
    bitangent = torch.tensor((0.0, 1.0, 0.0)).expand(4, -1)
    normal = torch.stack((sine, torch.zeros_like(sine), cosine), dim=1)
    return torch.stack((tangent, bitangent, normal), dim=1)[None, ...].expand(
        batch, -1, -1, -1
    ).clone()


def _states() -> torch.Tensor:
    states = torch.zeros((METAL_PROPOSAL_COMPONENT_COUNT, 11, 8))
    states[..., 1:3] = torch.tensor((0.35, 0.55))
    states[..., 3] = 0.2
    states[..., 4] = 1.0
    states[..., 5] = torch.tensor(METAL_PROPOSAL_FRAME_INDICES)
    states[..., 6] = torch.tensor(METAL_PROPOSAL_DISTRIBUTION_IDS)
    states[..., 7] = 1.0
    for index in range(METAL_PROPOSAL_COMPONENT_COUNT):
        if index == METAL_PROPOSAL_COMPONENT_COUNT - 1:
            states[index, index, 0] = 1.0
        else:
            states[index, index, 0] = 0.98
            states[index, -1, 0] = 0.02
    return states


def test_metal_proposal_layout_and_python_component_order_are_identical() -> None:
    proposal = load_metal_fused_layout()["proposal_reservation"]
    assert tuple(proposal["components"]) == METAL_PROPOSAL_COMPONENTS
    assert tuple(proposal["component_frame_indices"]) == METAL_PROPOSAL_FRAME_INDICES
    assert tuple(proposal["component_distribution_ids"]) == METAL_PROPOSAL_DISTRIBUTION_IDS
    assert tuple(proposal["component_specular_flags"]) == METAL_PROPOSAL_SPECULAR_FLAGS
    assert proposal["state_fields"] == [
        "normalized_weight",
        "alpha_x",
        "alpha_y",
        "rotation_radians",
        "active",
        "frame_index",
        "distribution_id",
        "energy_clue",
    ]


def test_every_folded_component_mixture_is_normalized_on_shading_hemisphere() -> None:
    states = _states()
    batch = states.shape[0]
    z_count, phi_count = 128, 256
    z = (torch.arange(z_count, dtype=torch.float32) + 0.5) / z_count
    phi = 2.0 * math.pi * (
        torch.arange(phi_count, dtype=torch.float32) + 0.5
    ) / phi_count
    zz, pp = torch.meshgrid(z, phi, indexing="ij")
    radius = torch.sqrt(torch.clamp(1.0 - zz.square(), min=0.0))
    directions = torch.stack(
        (radius * torch.cos(pp), radius * torch.sin(pp), zz), dim=-1
    ).reshape(1, -1, 3).expand(batch, -1, -1)
    wo = torch.nn.functional.normalize(
        torch.tensor((0.3, -0.2, 1.0))[None, :].expand(batch, -1), dim=1
    )
    density = metal_proposal_pdf(
        states,
        _frames(batch),
        torch.ones(batch, dtype=torch.bool),
        wo,
        directions,
    )
    integral = 2.0 * math.pi * density.forward.mean(dim=1)
    torch.testing.assert_close(
        integral,
        torch.ones_like(integral),
        rtol=3e-3,
        atol=3e-3,
    )
    assert bool(torch.isfinite(density.forward).all())
    assert bool((density.forward > 0.0).all())


def test_sample_reuses_exact_independent_pdf_and_invalid_state_fails_closed() -> None:
    states = _states()
    batch = states.shape[0]
    frames = _frames(batch)
    wo = torch.nn.functional.normalize(
        torch.tensor((0.2, 0.1, 1.0))[None, :].expand(batch, -1), dim=1
    )
    sample_u = torch.stack(
        (
            (torch.arange(batch, dtype=torch.float32) + 0.5) / batch,
            torch.frac((torch.arange(batch, dtype=torch.float32) + 0.5) * 0.61803398875),
        ),
        dim=1,
    )
    sampled = metal_sample_proposal(
        states, frames, torch.ones(batch, dtype=torch.bool), wo, sample_u
    )
    independent = metal_proposal_pdf(
        states,
        frames,
        torch.ones(batch, dtype=torch.bool),
        wo,
        sampled.wi,
    )
    assert bool(sampled.valid.all())
    assert bool((sampled.wi[..., 2] > 0.0).all())
    torch.testing.assert_close(sampled.forward_pdf, independent.forward)
    torch.testing.assert_close(sampled.reverse_pdf, independent.reverse)

    broken = states.clone()
    broken[..., 0] = 0.0
    invalid = metal_sample_proposal(
        broken, frames, torch.ones(batch, dtype=torch.bool), wo, sample_u
    )
    assert not bool(invalid.valid.any())
    assert bool(torch.isfinite(invalid.wi).all())
    assert bool(torch.equal(invalid.forward_pdf, torch.zeros_like(invalid.forward_pdf)))

    invalid_random = sample_u.clone()
    invalid_random[0, 0] = torch.nan
    invalid_random[1, 1] = 1.0
    invalid = metal_sample_proposal(
        states, frames, torch.ones(batch, dtype=torch.bool), wo, invalid_random
    )
    assert not bool(invalid.valid[:2].any())
    assert bool(torch.isfinite(invalid.wi).all())
    assert bool(torch.equal(invalid.forward_pdf[:2], torch.zeros_like(invalid.forward_pdf[:2])))


def test_invalid_grazing_inactive_zero_energy_and_degenerate_frames_fail_closed() -> None:
    state = _states()[:1]
    frames = _frames(1)
    valid = torch.ones(1, dtype=torch.bool)
    wo = torch.nn.functional.normalize(torch.tensor([[0.2, -0.1, 1.0]]), dim=1)
    wi = torch.tensor([[[0.0, 0.0, 1.0]]])

    cases: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    inactive = state.clone()
    inactive[0, 0, 4] = 0.0
    cases.append((inactive, frames, valid, wo))
    zero_energy = state.clone()
    zero_energy[..., 7] = 0.0
    cases.append((zero_energy, frames, valid, wo))
    nonfinite = state.clone()
    nonfinite[0, 0, 3] = torch.nan
    cases.append((nonfinite, frames, valid, wo))
    degenerate = frames.clone()
    degenerate[0, 0, 2] = 0.0
    cases.append((state, degenerate, valid, wo))
    grazing_wo = wo.clone()
    grazing_wo[:, 2] = 0.0
    cases.append((state, frames, valid, grazing_wo))
    cases.append((state, frames, torch.zeros_like(valid), wo))

    for proposal_state, proposal_frames, prepared_valid, outgoing in cases:
        density = metal_proposal_pdf(
            proposal_state,
            proposal_frames,
            prepared_valid,
            outgoing,
            wi,
        )
        assert not bool(density.valid.any())
        assert bool(torch.equal(density.forward, torch.zeros_like(density.forward)))
        assert bool(torch.equal(density.reverse, torch.zeros_like(density.reverse)))
        assert bool(torch.isfinite(density.component_pdfs).all())

    grazing_wi = wi.clone()
    grazing_wi[..., 2] = 0.0
    density = metal_proposal_pdf(state, frames, valid, wo, grazing_wi)
    assert not bool(density.valid.any())
    assert bool(torch.equal(density.forward, torch.zeros_like(density.forward)))


def test_specular_axis_tangent_alignment_uses_deterministic_valid_basis() -> None:
    state = _states()[2:3]
    state[..., 3] = 0.0
    frames = _frames(1)
    # For frame 1, reflecting -tangent around the frame normal produces the
    # tangent itself.  Its projected authored tangent is exactly degenerate,
    # but the distribution remains well-defined through the fallback basis.
    wo = -frames[:, 1, 0, :]
    wi = torch.tensor([[[0.0, 0.0, 1.0]]])
    density = metal_proposal_pdf(
        state, frames, torch.ones(1, dtype=torch.bool), wo, wi
    )
    assert bool(density.valid.all())
    assert bool(torch.isfinite(density.forward).all())
    assert bool((density.forward > 0.0).all())
