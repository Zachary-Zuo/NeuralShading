from __future__ import annotations

from dataclasses import dataclass
import math

import torch


METAL_BUDGETED_PROPOSAL_COMPONENTS = (
    "primary-specular",
    "secondary-specular",
    "full-hemisphere-fallback",
)
METAL_BUDGETED_PROPOSAL_COMPONENT_COUNT = 3
METAL_BUDGETED_PROPOSAL_STATE_WIDTH = 4
METAL_BUDGETED_DISTRIBUTION_GGX = 0
METAL_BUDGETED_DISTRIBUTION_BECKMANN = 1
METAL_BUDGETED_DISTRIBUTION_UNIFORM = 2
METAL_BUDGETED_DISTRIBUTION_IDS = (0, 1, 2)
METAL_BUDGETED_FRAME_INDICES = (0, 1, 0)

_MIN_ALPHA = 0.015
_MIN_COSINE = 1e-6
_UNIT_RANDOM_EPSILON = 2.0**-24


@dataclass(frozen=True)
class MetalBudgetedProposalPdf:
    forward: torch.Tensor
    reverse: torch.Tensor
    valid: torch.Tensor
    component_pdfs: torch.Tensor


@dataclass(frozen=True)
class MetalBudgetedProposalSample:
    wi: torch.Tensor
    forward_pdf: torch.Tensor
    reverse_pdf: torch.Tensor
    valid: torch.Tensor
    component: torch.Tensor
    component_pdfs: torch.Tensor


def _safe_normalize(
    value: torch.Tensor, fallback: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    length_squared = torch.sum(value * value, dim=-1, keepdim=True)
    valid = torch.isfinite(value).all(dim=-1) & (length_squared[..., 0] > 1e-12)
    normalized = value * torch.rsqrt(torch.clamp(length_squared, min=1e-12))
    return torch.where(valid[..., None], normalized, fallback.expand_as(value)), valid


def _fallback_tangent(axis: torch.Tensor) -> torch.Tensor:
    z_axis = torch.zeros_like(axis)
    z_axis[..., 2] = 1.0
    x_axis = torch.zeros_like(axis)
    x_axis[..., 0] = 1.0
    helper = torch.where((torch.abs(axis[..., 2:3]) < 0.9), z_axis, x_axis)
    tangent, _ = _safe_normalize(torch.cross(helper, axis, dim=-1), x_axis)
    return tangent


def _decode_state(
    proposal_state: torch.Tensor, prepared_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if proposal_state.ndim != 3 or proposal_state.shape[1:] != (
        METAL_BUDGETED_PROPOSAL_COMPONENT_COUNT,
        METAL_BUDGETED_PROPOSAL_STATE_WIDTH,
    ):
        raise ValueError("Metal budgeted proposal state must have shape [batch,3,4]")
    if prepared_valid.shape != (proposal_state.shape[0],):
        raise ValueError("Metal budgeted prepared validity must have shape [batch]")
    finite = torch.isfinite(proposal_state).all(dim=(1, 2))
    raw_weights = proposal_state[..., 0]
    alpha = proposal_state[..., 1:3]
    raw_distribution = proposal_state[..., 3]
    distribution = torch.round(raw_distribution).to(torch.int64)
    known_distribution = (
        (distribution == METAL_BUDGETED_DISTRIBUTION_GGX)
        | (distribution == METAL_BUDGETED_DISTRIBUTION_BECKMANN)
        | (distribution == METAL_BUDGETED_DISTRIBUTION_UNIFORM)
    )
    valid = (
        prepared_valid
        & finite
        & torch.all(raw_weights >= 0.0, dim=1)
        & (raw_weights[:, -1] > 0.0)
        & torch.all((alpha >= _MIN_ALPHA) & (alpha <= 1.0), dim=(1, 2))
        & torch.all(known_distribution, dim=1)
        & torch.all(torch.abs(raw_distribution - distribution) <= 0.25, dim=1)
    )
    weights = torch.where(finite[:, None], torch.clamp(raw_weights, min=0.0), 0.0)
    total = weights.sum(dim=1, keepdim=True)
    valid = valid & torch.isfinite(total[:, 0]) & (total[:, 0] > 0.0)
    return weights / torch.clamp(total, min=1e-12), distribution, valid


def _component_basis(
    frames: torch.Tensor, wo: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = wo.shape[0]
    if frames.shape != (batch, 2, 3, 3):
        raise ValueError("Metal budgeted proposal frames must have shape [batch,2,3,3]")
    if wo.shape != (batch, 3):
        raise ValueError("Metal budgeted outgoing direction must have shape [batch,3]")
    frame_indices = torch.tensor(
        METAL_BUDGETED_FRAME_INDICES, dtype=torch.int64, device=wo.device
    )
    selected = frames.index_select(1, frame_indices)
    tangent_seed = selected[..., 0, :]
    normal_seed = selected[..., 2, :]
    base_normal = torch.zeros_like(normal_seed)
    base_normal[..., 2] = 1.0
    normal, normal_valid = _safe_normalize(normal_seed, base_normal)
    normal = torch.where(normal[..., 2:3] >= 0.0, normal, -normal)
    reflected = (
        2.0 * torch.sum(wo[:, None, :] * normal, dim=-1, keepdim=True) * normal
        - wo[:, None, :]
    )
    specular = torch.tensor(
        (True, True, False), dtype=torch.bool, device=wo.device
    )
    axis_seed = torch.where(specular[None, :, None], reflected, base_normal)
    axis, axis_valid = _safe_normalize(axis_seed, base_normal)
    tangent_seed = tangent_seed - torch.sum(
        tangent_seed * axis, dim=-1, keepdim=True
    ) * axis
    tangent, _ = _safe_normalize(
        tangent_seed, _fallback_tangent(axis)
    )
    bitangent = torch.cross(axis, tangent, dim=-1)
    tangent_valid = torch.isfinite(tangent).all(dim=2) & (
        torch.sum(tangent.square(), dim=2) > 1e-12
    )
    valid = (
        normal_valid.all(dim=1)
        & axis_valid.all(dim=1)
        & tangent_valid.all(dim=1)
        & torch.isfinite(frames).all(dim=(1, 2, 3))
    )
    return tangent, bitangent, axis, valid


def _local_component_pdf(
    local: torch.Tensor, alpha: torch.Tensor, distribution: torch.Tensor
) -> torch.Tensor:
    z = local[..., 2]
    positive = z > 0.0
    safe_z = torch.clamp(z, min=1e-8)
    ax = torch.clamp(alpha[:, None, :, 0], min=_MIN_ALPHA, max=1.0)
    ay = torch.clamp(alpha[:, None, :, 1], min=_MIN_ALPHA, max=1.0)
    slope_radius = (
        (local[..., 0] / (ax * safe_z)).square()
        + (local[..., 1] / (ay * safe_z)).square()
    )
    denominator = (
        (local[..., 0] / ax).square()
        + (local[..., 1] / ay).square()
        + safe_z.square()
    )
    ggx = safe_z / (
        math.pi * ax * ay * torch.clamp(denominator.square(), min=1e-20)
    )
    beckmann_log = (
        -slope_radius
        - math.log(math.pi)
        - torch.log(ax)
        - torch.log(ay)
        - 3.0 * torch.log(safe_z)
    )
    beckmann = torch.exp(torch.clamp(beckmann_log, max=80.0))
    uniform = torch.full_like(safe_z, 0.5 / math.pi)
    distribution = distribution[:, None, :]
    value = torch.where(
        distribution == METAL_BUDGETED_DISTRIBUTION_GGX,
        ggx,
        torch.where(
            distribution == METAL_BUDGETED_DISTRIBUTION_BECKMANN,
            beckmann,
            uniform,
        ),
    )
    return torch.where(positive & torch.isfinite(value), value, 0.0)


def metal_budgeted_component_pdfs(
    proposal_state: torch.Tensor,
    frames: torch.Tensor,
    prepared_valid: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if wi.ndim != 3 or wi.shape[0] != proposal_state.shape[0] or wi.shape[2] != 3:
        raise ValueError("Metal budgeted incident directions must have shape [batch,count,3]")
    weights, distribution, state_valid = _decode_state(proposal_state, prepared_valid)
    tangent, bitangent, axis, basis_valid = _component_basis(frames, wo)
    local = torch.stack(
        (
            torch.einsum("bdk,bck->bdc", wi, tangent),
            torch.einsum("bdk,bck->bdc", wi, bitangent),
            torch.einsum("bdk,bck->bdc", wi, axis),
        ),
        dim=-1,
    )
    mirrored = wi.clone()
    mirrored[..., 2] = -mirrored[..., 2]
    mirrored_local = torch.stack(
        (
            torch.einsum("bdk,bck->bdc", mirrored, tangent),
            torch.einsum("bdk,bck->bdc", mirrored, bitangent),
            torch.einsum("bdk,bck->bdc", mirrored, axis),
        ),
        dim=-1,
    )
    alpha = proposal_state[..., 1:3]
    components = _local_component_pdf(local, alpha, distribution) + _local_component_pdf(
        mirrored_local, alpha, distribution
    )
    direction_valid = (
        torch.isfinite(wo).all(dim=1) & (wo[:, 2] > _MIN_COSINE)
    )[:, None] & torch.isfinite(wi).all(dim=2) & (wi[..., 2] > _MIN_COSINE)
    valid = state_valid & basis_valid
    components = torch.where(
        valid[:, None, None] & direction_valid[:, :, None], components, 0.0
    )
    density = torch.sum(weights[:, None, :] * components, dim=2)
    query_valid = valid[:, None] & direction_valid & torch.isfinite(density) & (density > 0.0)
    return torch.where(query_valid, density, 0.0), components, query_valid


def metal_budgeted_proposal_pdf(
    proposal_state: torch.Tensor,
    frames: torch.Tensor,
    prepared_valid: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
) -> MetalBudgetedProposalPdf:
    forward, components, forward_valid = metal_budgeted_component_pdfs(
        proposal_state, frames, prepared_valid, wo, wi
    )
    batch, count = wi.shape[:2]
    repeated_state = proposal_state[:, None].expand(-1, count, -1, -1).reshape(
        batch * count,
        METAL_BUDGETED_PROPOSAL_COMPONENT_COUNT,
        METAL_BUDGETED_PROPOSAL_STATE_WIDTH,
    )
    repeated_frames = frames[:, None].expand(-1, count, -1, -1, -1).reshape(
        batch * count, 2, 3, 3
    )
    repeated_valid = prepared_valid[:, None].expand(-1, count).reshape(-1)
    reverse_wo = wi.reshape(batch * count, 3)
    reverse_wi = wo[:, None, :].expand(-1, count, -1).reshape(
        batch * count, 1, 3
    )
    reverse, _, reverse_valid = metal_budgeted_component_pdfs(
        repeated_state, repeated_frames, repeated_valid, reverse_wo, reverse_wi
    )
    return MetalBudgetedProposalPdf(
        forward=forward,
        reverse=reverse.reshape(batch, count),
        valid=forward_valid & reverse_valid.reshape(batch, count),
        component_pdfs=components,
    )


def _sample_local(
    distribution: torch.Tensor, alpha: torch.Tensor, sample_u: torch.Tensor
) -> torch.Tensor:
    u0 = torch.clamp(
        sample_u[:, 0], _UNIT_RANDOM_EPSILON, 1.0 - _UNIT_RANDOM_EPSILON
    )
    phi = 2.0 * math.pi * sample_u[:, 1]
    cosine, sine = torch.cos(phi), torch.sin(phi)
    ggx_radius = torch.sqrt(u0 / torch.clamp(1.0 - u0, min=_UNIT_RANDOM_EPSILON))
    beckmann_radius = torch.sqrt(
        -torch.log(torch.clamp(1.0 - u0, min=_UNIT_RANDOM_EPSILON))
    )
    radius = torch.where(
        distribution == METAL_BUDGETED_DISTRIBUTION_BECKMANN,
        beckmann_radius,
        ggx_radius,
    )
    specular = torch.stack(
        (
            alpha[:, 0] * radius * cosine,
            alpha[:, 1] * radius * sine,
            torch.ones_like(radius),
        ),
        dim=1,
    )
    specular = specular * torch.rsqrt(
        torch.clamp(torch.sum(specular.square(), dim=1, keepdim=True), min=1e-12)
    )
    uniform = torch.stack(
        (
            torch.sqrt(torch.clamp(1.0 - u0.square(), min=0.0)) * cosine,
            torch.sqrt(torch.clamp(1.0 - u0.square(), min=0.0)) * sine,
            u0,
        ),
        dim=1,
    )
    return torch.where(
        (distribution == METAL_BUDGETED_DISTRIBUTION_UNIFORM)[:, None],
        uniform,
        specular,
    )


def metal_budgeted_sample_proposal(
    proposal_state: torch.Tensor,
    frames: torch.Tensor,
    prepared_valid: torch.Tensor,
    wo: torch.Tensor,
    sample_u: torch.Tensor,
) -> MetalBudgetedProposalSample:
    batch = proposal_state.shape[0]
    if sample_u.shape != (batch, 2):
        raise ValueError("Metal budgeted proposal requires one float2 random tuple")
    random_valid = torch.isfinite(sample_u).all(dim=1) & torch.all(
        (sample_u >= 0.0) & (sample_u < 1.0), dim=1
    )
    safe_u = torch.where(random_valid[:, None], sample_u, 0.0)
    weights, distribution, state_valid = _decode_state(proposal_state, prepared_valid)
    cdf = torch.cumsum(weights.detach(), dim=1)
    component = torch.sum(safe_u[:, 0:1] >= cdf, dim=1).to(torch.int64)
    component = torch.clamp(component, max=METAL_BUDGETED_PROPOSAL_COMPONENT_COUNT - 1)
    previous = torch.where(
        component > 0,
        torch.gather(cdf, 1, torch.clamp(component - 1, min=0)[:, None])[:, 0],
        torch.zeros_like(safe_u[:, 0]),
    )
    selected_weight = torch.gather(weights.detach(), 1, component[:, None])[:, 0]
    remapped = torch.stack(
        (
            torch.clamp(
                (safe_u[:, 0] - previous) / torch.clamp(selected_weight, min=1e-12),
                0.0,
                1.0 - _UNIT_RANDOM_EPSILON,
            ),
            safe_u[:, 1],
        ),
        dim=1,
    )
    tangent, bitangent, axis, basis_valid = _component_basis(
        frames.detach(), wo.detach()
    )
    gather3 = component[:, None, None].expand(-1, 1, 3)
    selected_tangent = torch.gather(tangent, 1, gather3)[:, 0]
    selected_bitangent = torch.gather(bitangent, 1, gather3)[:, 0]
    selected_axis = torch.gather(axis, 1, gather3)[:, 0]
    selected_alpha = torch.gather(
        proposal_state.detach()[..., 1:3],
        1,
        component[:, None, None].expand(-1, 1, 2),
    )[:, 0]
    selected_distribution = torch.gather(distribution, 1, component[:, None])[:, 0]
    local = _sample_local(selected_distribution, selected_alpha, remapped)
    wi = (
        local[:, 0:1] * selected_tangent
        + local[:, 1:2] * selected_bitangent
        + local[:, 2:3] * selected_axis
    )
    wi[:, 2] = torch.abs(wi[:, 2])
    fallback = torch.zeros_like(wi)
    fallback[:, 2] = 1.0
    wi, wi_valid = _safe_normalize(wi, fallback)
    valid = state_valid & basis_valid & random_valid & wi_valid & (wo[:, 2] > _MIN_COSINE)
    density = metal_budgeted_proposal_pdf(
        proposal_state, frames, prepared_valid & valid, wo, wi[:, None, :]
    )
    valid = valid & density.valid[:, 0]
    return MetalBudgetedProposalSample(
        wi=wi[:, None, :],
        forward_pdf=torch.where(valid[:, None], density.forward, 0.0),
        reverse_pdf=torch.where(valid[:, None], density.reverse, 0.0),
        valid=valid[:, None],
        component=torch.where(valid, component, -1)[:, None],
        component_pdfs=density.component_pdfs,
    )


__all__ = [
    "METAL_BUDGETED_DISTRIBUTION_IDS",
    "METAL_BUDGETED_FRAME_INDICES",
    "METAL_BUDGETED_PROPOSAL_COMPONENTS",
    "METAL_BUDGETED_PROPOSAL_COMPONENT_COUNT",
    "METAL_BUDGETED_PROPOSAL_STATE_WIDTH",
    "MetalBudgetedProposalPdf",
    "MetalBudgetedProposalSample",
    "metal_budgeted_component_pdfs",
    "metal_budgeted_proposal_pdf",
    "metal_budgeted_sample_proposal",
]
