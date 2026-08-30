from __future__ import annotations

from dataclasses import dataclass
import math

import torch


METAL_PROPOSAL_COMPONENTS = (
    "core-conductor-ggx",
    "core-conductor-beckmann",
    "core-coat-specular",
    "core-diffuse-contamination",
    "core-broad-scatter",
    "core-secondary-specular",
    "positive-residual-0",
    "positive-residual-1",
    "positive-residual-2",
    "positive-residual-3",
    "full-hemisphere-fallback",
)
METAL_PROPOSAL_COMPONENT_COUNT = len(METAL_PROPOSAL_COMPONENTS)
METAL_PROPOSAL_STATE_WIDTH = 8

METAL_DISTRIBUTION_GGX = 0
METAL_DISTRIBUTION_BECKMANN = 1
METAL_DISTRIBUTION_COSINE = 2
METAL_DISTRIBUTION_UNIFORM = 3

# Each component reuses one of the four prepared frames.  Specular components
# reflect around the selected normal; diffuse/broad/fallback components use it
# directly as their directional axis.  The final fold is always relative to the
# renderer shading hemisphere, so every component remains normalized there.
METAL_PROPOSAL_FRAME_INDICES = (0, 0, 1, 0, 2, 3, 0, 1, 2, 3, 0)
METAL_PROPOSAL_DISTRIBUTION_IDS = (
    METAL_DISTRIBUTION_GGX,
    METAL_DISTRIBUTION_BECKMANN,
    METAL_DISTRIBUTION_GGX,
    METAL_DISTRIBUTION_COSINE,
    METAL_DISTRIBUTION_COSINE,
    METAL_DISTRIBUTION_GGX,
    METAL_DISTRIBUTION_GGX,
    METAL_DISTRIBUTION_GGX,
    METAL_DISTRIBUTION_GGX,
    METAL_DISTRIBUTION_GGX,
    METAL_DISTRIBUTION_UNIFORM,
)
METAL_PROPOSAL_SPECULAR_FLAGS = (
    True,
    True,
    True,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    False,
)
_MIN_COSINE = 1e-6
_MIN_ALPHA = 0.01
_UNIT_RANDOM_EPSILON = 2.0**-24


@dataclass(frozen=True)
class MetalProposalPdf:
    forward: torch.Tensor
    reverse: torch.Tensor
    valid: torch.Tensor
    component_pdfs: torch.Tensor


@dataclass(frozen=True)
class MetalProposalSample:
    wi: torch.Tensor
    forward_pdf: torch.Tensor
    reverse_pdf: torch.Tensor
    valid: torch.Tensor
    component: torch.Tensor
    component_pdfs: torch.Tensor


def _safe_normalize(value: torch.Tensor, fallback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length_squared = torch.sum(value * value, dim=-1, keepdim=True)
    valid = torch.isfinite(value).all(dim=-1) & (length_squared[..., 0] > 1e-12)
    normalized = value * torch.rsqrt(torch.clamp(length_squared, min=1e-12))
    return torch.where(valid[..., None], normalized, fallback), valid


def _component_constants(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(METAL_PROPOSAL_FRAME_INDICES, dtype=torch.int64, device=device),
        torch.tensor(METAL_PROPOSAL_DISTRIBUTION_IDS, dtype=torch.int64, device=device),
        torch.tensor(METAL_PROPOSAL_SPECULAR_FLAGS, dtype=torch.bool, device=device),
    )


def _decode_weights(
    proposal_state: torch.Tensor,
    prepared_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if proposal_state.ndim != 3 or proposal_state.shape[1:] != (
        METAL_PROPOSAL_COMPONENT_COUNT,
        METAL_PROPOSAL_STATE_WIDTH,
    ):
        raise ValueError("Metal proposal state must have shape [batch,11,8]")
    if prepared_valid.shape != (proposal_state.shape[0],):
        raise ValueError("Metal prepared validity must have shape [batch]")
    finite = torch.isfinite(proposal_state).all(dim=(1, 2))
    raw_weights = proposal_state[..., 0]
    energy_clue = proposal_state[..., 7]
    active = proposal_state[..., 4] > 0.5
    frame_indices, distributions, _ = _component_constants(proposal_state.device)
    alpha = proposal_state[..., 1:3]
    valid = (
        prepared_valid
        & finite
        & torch.all(raw_weights >= 0.0, dim=1)
        & torch.all(energy_clue >= 0.0, dim=1)
        & (torch.sum(energy_clue, dim=1) > 0.0)
        & torch.all((raw_weights <= 0.0) | active, dim=1)
        & (raw_weights[:, -1] > 0.0)
        & active[:, -1]
        & torch.all((alpha >= _MIN_ALPHA) & (alpha <= 1.0), dim=(1, 2))
        & torch.all(
            torch.abs(proposal_state[..., 5] - frame_indices[None, :]) <= 0.25,
            dim=1,
        )
        & torch.all(
            torch.abs(proposal_state[..., 6] - distributions[None, :]) <= 0.25,
            dim=1,
        )
    )
    weights = torch.where(finite[:, None], torch.clamp(raw_weights, min=0.0), 0.0)
    total = torch.sum(weights, dim=1, keepdim=True)
    valid = valid & (total[:, 0] > 0.0) & torch.isfinite(total[:, 0])
    weights = weights / torch.clamp(total, min=1e-12)
    return weights, valid


def _fallback_tangent(axis: torch.Tensor) -> torch.Tensor:
    z_axis = torch.zeros_like(axis)
    z_axis[..., 2] = 1.0
    x_axis = torch.zeros_like(axis)
    x_axis[..., 0] = 1.0
    helper = torch.where((torch.abs(axis[..., 2:3]) < 0.9), z_axis, x_axis)
    value = torch.cross(helper, axis, dim=-1)
    return value * torch.rsqrt(torch.clamp(torch.sum(value * value, dim=-1, keepdim=True), min=1e-12))


def _component_basis(
    proposal_state: torch.Tensor,
    frames: torch.Tensor,
    wo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = proposal_state.shape[0]
    if frames.shape != (batch, 4, 3, 3):
        raise ValueError("Metal proposal frames must have shape [batch,4,3,3]")
    if wo.shape != (batch, 3):
        raise ValueError("Metal proposal outgoing direction must have shape [batch,3]")
    frame_indices, distributions, specular = _component_constants(wo.device)
    selected = frames.index_select(1, frame_indices)
    frame_tangent = selected[..., 0, :]
    frame_normal = selected[..., 2, :]
    base_normal = torch.zeros_like(frame_normal)
    base_normal[..., 2] = 1.0
    frame_normal, normal_valid = _safe_normalize(frame_normal, base_normal)
    frame_normal = torch.where(
        frame_normal[..., 2:3] >= 0.0, frame_normal, -frame_normal
    )
    tangent_seed = frame_tangent - torch.sum(
        frame_tangent * frame_normal, dim=-1, keepdim=True
    ) * frame_normal
    frame_tangent, tangent_valid = _safe_normalize(
        tangent_seed, _fallback_tangent(frame_normal)
    )
    frame_bitangent = torch.cross(frame_normal, frame_tangent, dim=-1)
    rotation = proposal_state[..., 3]
    cosine = torch.cos(rotation)[..., None]
    sine = torch.sin(rotation)[..., None]
    rotated_tangent = cosine * frame_tangent + sine * frame_bitangent
    reflected = (
        2.0
        * torch.sum(wo[:, None, :] * frame_normal, dim=-1, keepdim=True)
        * frame_normal
        - wo[:, None, :]
    )
    axis_seed = torch.where(specular[None, :, None], reflected, frame_normal)
    axis, axis_valid = _safe_normalize(axis_seed, base_normal)
    tangent_seed = rotated_tangent - torch.sum(
        rotated_tangent * axis, dim=-1, keepdim=True
    ) * axis
    tangent, _ = _safe_normalize(
        tangent_seed, _fallback_tangent(axis)
    )
    bitangent = torch.cross(axis, tangent, dim=-1)
    basis_valid = (
        normal_valid.all(dim=1)
        & tangent_valid.all(dim=1)
        & axis_valid.all(dim=1)
        & torch.isfinite(frames).all(dim=(1, 2, 3))
    )
    return tangent, bitangent, axis, basis_valid


def _local_component_pdf(
    local: torch.Tensor,
    alpha: torch.Tensor,
    distributions: torch.Tensor,
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
    cosine = safe_z / math.pi
    uniform = torch.full_like(safe_z, 0.5 / math.pi)
    distribution = distributions[None, None, :]
    value = torch.where(
        distribution == METAL_DISTRIBUTION_GGX,
        ggx,
        torch.where(
            distribution == METAL_DISTRIBUTION_BECKMANN,
            beckmann,
            torch.where(
                distribution == METAL_DISTRIBUTION_COSINE, cosine, uniform
            ),
        ),
    )
    return torch.where(positive & torch.isfinite(value), value, 0.0)


def metal_proposal_component_pdfs(
    proposal_state: torch.Tensor,
    frames: torch.Tensor,
    prepared_valid: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if wi.ndim != 3 or wi.shape[0] != proposal_state.shape[0] or wi.shape[2] != 3:
        raise ValueError("Metal proposal incident directions must have shape [batch,count,3]")
    weights, state_valid = _decode_weights(proposal_state, prepared_valid)
    tangent, bitangent, axis, basis_valid = _component_basis(
        proposal_state, frames, wo
    )
    _, distributions, _ = _component_constants(wi.device)
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
    component_pdf = _local_component_pdf(local, alpha, distributions)
    component_pdf = component_pdf + _local_component_pdf(
        mirrored_local, alpha, distributions
    )
    direction_valid = (
        torch.isfinite(wo).all(dim=1)
        & (wo[:, 2] > _MIN_COSINE)
    )[:, None] & torch.isfinite(wi).all(dim=2) & (wi[..., 2] > _MIN_COSINE)
    valid = state_valid & basis_valid
    component_pdf = torch.where(
        (valid[:, None, None] & direction_valid[:, :, None]),
        component_pdf,
        0.0,
    )
    pdf = torch.sum(weights[:, None, :] * component_pdf, dim=2)
    finite_pdf = torch.isfinite(pdf) & (pdf > 0.0)
    valid_queries = valid[:, None] & direction_valid & finite_pdf
    pdf = torch.where(valid_queries, pdf, 0.0)
    return pdf, component_pdf, valid_queries


def metal_proposal_pdf(
    proposal_state: torch.Tensor,
    frames: torch.Tensor,
    prepared_valid: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
) -> MetalProposalPdf:
    forward, components, forward_valid = metal_proposal_component_pdfs(
        proposal_state, frames, prepared_valid, wo, wi
    )
    batch, direction_count = wi.shape[:2]
    repeated_state = proposal_state[:, None, :, :].expand(
        -1, direction_count, -1, -1
    ).reshape(batch * direction_count, METAL_PROPOSAL_COMPONENT_COUNT, -1)
    repeated_frames = frames[:, None, :, :, :].expand(
        -1, direction_count, -1, -1, -1
    ).reshape(batch * direction_count, 4, 3, 3)
    repeated_valid = prepared_valid[:, None].expand(-1, direction_count).reshape(-1)
    reverse_wo = wi.reshape(batch * direction_count, 3)
    reverse_wi = wo[:, None, :].expand(-1, direction_count, -1).reshape(
        batch * direction_count, 1, 3
    )
    reverse, _, reverse_valid = metal_proposal_component_pdfs(
        repeated_state,
        repeated_frames,
        repeated_valid,
        reverse_wo,
        reverse_wi,
    )
    reverse = reverse.reshape(batch, direction_count)
    reverse_valid = reverse_valid.reshape(batch, direction_count)
    return MetalProposalPdf(
        forward,
        reverse,
        forward_valid & reverse_valid,
        components,
    )


def _sample_local(
    distribution: torch.Tensor,
    alpha: torch.Tensor,
    sample_u: torch.Tensor,
) -> torch.Tensor:
    u0 = torch.clamp(
        sample_u[:, 0], min=_UNIT_RANDOM_EPSILON, max=1.0 - _UNIT_RANDOM_EPSILON
    )
    phi = 2.0 * math.pi * sample_u[:, 1]
    cosine = torch.cos(phi)
    sine = torch.sin(phi)
    ggx_radius = torch.sqrt(u0 / torch.clamp(1.0 - u0, min=_UNIT_RANDOM_EPSILON))
    beckmann_radius = torch.sqrt(-torch.log(torch.clamp(1.0 - u0, min=_UNIT_RANDOM_EPSILON)))
    slope_radius = torch.where(
        distribution == METAL_DISTRIBUTION_BECKMANN,
        beckmann_radius,
        ggx_radius,
    )
    slope = torch.stack(
        (
            alpha[:, 0] * slope_radius * cosine,
            alpha[:, 1] * slope_radius * sine,
            torch.ones_like(u0),
        ),
        dim=1,
    )
    slope = slope * torch.rsqrt(
        torch.clamp(torch.sum(slope * slope, dim=1, keepdim=True), min=1e-12)
    )
    disk_radius = torch.sqrt(u0)
    cosine_sample = torch.stack(
        (
            disk_radius * cosine,
            disk_radius * sine,
            torch.sqrt(torch.clamp(1.0 - u0, min=0.0)),
        ),
        dim=1,
    )
    uniform_sample = torch.stack(
        (
            torch.sqrt(torch.clamp(1.0 - u0.square(), min=0.0)) * cosine,
            torch.sqrt(torch.clamp(1.0 - u0.square(), min=0.0)) * sine,
            u0,
        ),
        dim=1,
    )
    return torch.where(
        (distribution == METAL_DISTRIBUTION_COSINE)[:, None],
        cosine_sample,
        torch.where(
            (distribution == METAL_DISTRIBUTION_UNIFORM)[:, None],
            uniform_sample,
            slope,
        ),
    )


def metal_sample_proposal(
    proposal_state: torch.Tensor,
    frames: torch.Tensor,
    prepared_valid: torch.Tensor,
    wo: torch.Tensor,
    sample_u: torch.Tensor,
) -> MetalProposalSample:
    batch = proposal_state.shape[0]
    if sample_u.shape != (batch, 2):
        raise ValueError("Metal proposal requires one float2 random tuple per state")
    random_valid = torch.isfinite(sample_u).all(dim=1) & torch.all(
        (sample_u >= 0.0) & (sample_u < 1.0), dim=1
    )
    safe_sample_u = torch.where(random_valid[:, None], sample_u, 0.0)
    weights, state_valid = _decode_weights(proposal_state, prepared_valid)
    detached_weights = weights.detach()
    cdf = torch.cumsum(detached_weights, dim=1)
    component = torch.sum(safe_sample_u[:, 0:1] >= cdf, dim=1).to(torch.int64)
    component = torch.clamp(component, max=METAL_PROPOSAL_COMPONENT_COUNT - 1)
    previous = torch.where(
        component > 0,
        torch.gather(cdf, 1, torch.clamp(component - 1, min=0)[:, None])[:, 0],
        torch.zeros_like(sample_u[:, 0]),
    )
    selected_weight = torch.gather(
        detached_weights, 1, component[:, None]
    )[:, 0]
    remapped = torch.stack(
        (
            torch.clamp(
                (safe_sample_u[:, 0] - previous)
                / torch.clamp(selected_weight, min=1e-12),
                min=0.0,
                max=1.0 - _UNIT_RANDOM_EPSILON,
            ),
            safe_sample_u[:, 1],
        ),
        dim=1,
    )
    tangent, bitangent, axis, basis_valid = _component_basis(
        proposal_state.detach(), frames.detach(), wo.detach()
    )
    gather_index = component[:, None, None].expand(-1, 1, 3)
    selected_tangent = torch.gather(tangent, 1, gather_index)[:, 0, :]
    selected_bitangent = torch.gather(bitangent, 1, gather_index)[:, 0, :]
    selected_axis = torch.gather(axis, 1, gather_index)[:, 0, :]
    selected_alpha = torch.gather(
        proposal_state.detach()[..., 1:3],
        1,
        component[:, None, None].expand(-1, 1, 2),
    )[:, 0, :]
    _, distributions, _ = _component_constants(wo.device)
    selected_distribution = distributions.index_select(0, component)
    local = _sample_local(selected_distribution, selected_alpha, remapped)
    wi = (
        local[:, 0:1] * selected_tangent
        + local[:, 1:2] * selected_bitangent
        + local[:, 2:3] * selected_axis
    )
    wi = wi.clone()
    wi[:, 2] = torch.abs(wi[:, 2])
    wi, wi_valid = _safe_normalize(
        wi,
        torch.tensor((0.0, 0.0, 1.0), dtype=wi.dtype, device=wi.device).expand_as(wi),
    )
    wi = wi[:, None, :].detach()
    density = metal_proposal_pdf(
        proposal_state, frames, prepared_valid, wo, wi
    )
    valid = (
        density.valid
        & state_valid[:, None]
        & basis_valid[:, None]
        & wi_valid[:, None]
        & random_valid[:, None]
        & (selected_weight[:, None] > 0.0)
    )
    return MetalProposalSample(
        wi,
        torch.where(valid, density.forward, 0.0),
        torch.where(valid, density.reverse, 0.0),
        valid,
        component,
        density.component_pdfs,
    )


__all__ = [
    "METAL_DISTRIBUTION_BECKMANN",
    "METAL_DISTRIBUTION_COSINE",
    "METAL_DISTRIBUTION_GGX",
    "METAL_DISTRIBUTION_UNIFORM",
    "METAL_PROPOSAL_COMPONENTS",
    "METAL_PROPOSAL_COMPONENT_COUNT",
    "METAL_PROPOSAL_DISTRIBUTION_IDS",
    "METAL_PROPOSAL_FRAME_INDICES",
    "METAL_PROPOSAL_SPECULAR_FLAGS",
    "METAL_PROPOSAL_STATE_WIDTH",
    "MetalProposalPdf",
    "MetalProposalSample",
    "metal_proposal_component_pdfs",
    "metal_proposal_pdf",
    "metal_sample_proposal",
]
