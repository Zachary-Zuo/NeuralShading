from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.models.metal_budgeted_asset import MetalBudgetedAssetSample
from ncls.learning.models.metal_budgeted_compiler import MetalBudgetedProgramState
from ncls.learning.models.metal_budgeted_profile import MetalBudgetedProfile


def _safe_normalize(
    value: torch.Tensor, fallback: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    length_squared = torch.sum(value * value, dim=-1, keepdim=True)
    valid = torch.isfinite(value).all(dim=-1) & (length_squared[..., 0] > 1e-12)
    normalized = value * torch.rsqrt(torch.clamp(length_squared, min=1e-12))
    return torch.where(valid[..., None], normalized, fallback.expand_as(value)), valid


def _orthonormal_frame(normal: torch.Tensor) -> torch.Tensor:
    y_axis = torch.zeros_like(normal)
    y_axis[..., 1] = 1.0
    x_axis = torch.zeros_like(normal)
    x_axis[..., 0] = 1.0
    helper = torch.where((normal[..., 2:3].abs() < 0.999), y_axis, x_axis)
    tangent, _ = _safe_normalize(torch.cross(helper, normal, dim=-1), x_axis)
    bitangent = torch.cross(normal, tangent, dim=-1)
    return torch.stack((tangent, bitangent, normal), dim=-2)


def _local_frames(slopes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    if slopes.ndim != 3 or slopes.shape[1:] != (2, 2):
        raise ValueError("Metal budgeted local frame slopes must have shape [batch,2,2]")
    if angles.shape != slopes.shape[:2]:
        raise ValueError("Metal budgeted local frame angles must have shape [batch,2]")
    normal_seed = torch.cat(
        (slopes, torch.ones_like(slopes[..., :1])), dim=-1
    )
    fallback = torch.zeros_like(normal_seed)
    fallback[..., 2] = 1.0
    normal, _ = _safe_normalize(normal_seed, fallback)
    base = _orthonormal_frame(normal)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    tangent = cosine[..., None] * base[..., 0, :] + sine[..., None] * base[..., 1, :]
    bitangent = -sine[..., None] * base[..., 0, :] + cosine[..., None] * base[..., 1, :]
    return torch.stack((tangent, bitangent, normal), dim=-2)


@dataclass(frozen=True)
class MetalBudgetedPreparedState:
    program: MetalBudgetedProgramState
    semantic_state: torch.Tensor
    view_state: torch.Tensor
    compact_frame_state: torch.Tensor
    frames: torch.Tensor
    analytic_lobes: torch.Tensor
    proposal_state: torch.Tensor
    access_state: torch.Tensor
    identity_and_flags: torch.Tensor
    valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


class MetalBudgetedPrepare(nn.Module):
    """两次 asset read 后执行固定 24→32→32→24 语义解码。"""

    def __init__(self, profile: MetalBudgetedProfile) -> None:
        super().__init__()
        self.profile = profile
        widths = profile.semantic_decoder_layers
        self.semantic_decoder = nn.Sequential(
            nn.Linear(widths[0], widths[1]),
            nn.SiLU(),
            nn.Linear(widths[1], widths[2]),
            nn.SiLU(),
            nn.Linear(widths[2], widths[3]),
        )
        self.proposal_adapter = nn.Linear(widths[-1], profile.proposal_component_count)

    @staticmethod
    def _view_features(wo: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                wo,
                wo.square(),
                wo[:, 2:3],
                torch.ones_like(wo[:, 2:3]),
            ),
            dim=1,
        )

    @staticmethod
    def _modulate_lobes(
        base: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (
                torch.clamp(base[..., :3] + 0.25 * torch.tanh(delta[..., :3]), 0.0, 1.0),
                torch.clamp(
                    base[..., 3:5] * torch.exp(0.75 * torch.tanh(delta[..., 3:5])),
                    0.015,
                    1.0,
                ),
                base[..., 5:6] * F.softplus(delta[..., 5:6] + 1.0),
                base[..., 6:7] + 0.5 * torch.pi * torch.tanh(delta[..., 6:7]),
                torch.clamp(
                    base[..., 7:8] * (2.0 * torch.sigmoid(delta[..., 7:8])),
                    0.0,
                    1.0,
                ),
            ),
            dim=-1,
        )

    def forward(
        self,
        program: MetalBudgetedProgramState,
        asset: MetalBudgetedAssetSample,
        wo: torch.Tensor,
    ) -> MetalBudgetedPreparedState:
        if wo.ndim != 2 or wo.shape[1] != 3:
            raise ValueError("Metal budgeted outgoing direction must have shape [batch,3]")
        fallback = torch.zeros_like(wo)
        fallback[:, 2] = 1.0
        normalized_wo, wo_valid = _safe_normalize(wo, fallback)
        view_state = self._view_features(normalized_wo)
        decoder_input = torch.cat(
            (
                program.compiler_condition,
                asset.detail,
                asset.context,
                view_state,
            ),
            dim=1,
        )
        if decoder_input.shape[1] != self.profile.semantic_decoder_layers[0]:
            raise RuntimeError("Metal budgeted semantic decoder input width drifted")
        decoded_semantic = self.semantic_decoder(decoder_input)
        semantic = torch.cat(
            (
                decoded_semantic[:, :4] + asset.detail,
                decoded_semantic[:, 4:],
            ),
            dim=1,
        )
        base_lobes = program.analytic_lobes
        lobe_delta = semantic[:, 8:].reshape(
            wo.shape[0], self.profile.analytic_lobe_count, 8
        )
        lobes = self._modulate_lobes(base_lobes, lobe_delta)
        frame_slopes = 0.75 * torch.tanh(semantic[:, :4]).reshape(
            wo.shape[0], self.profile.analytic_lobe_count, 2
        )
        frames = _local_frames(frame_slopes, lobes[..., 6])
        compact_frame_state = torch.cat(
            (
                frame_slopes.flatten(start_dim=1),
                torch.cos(lobes[..., 6]),
                torch.sin(lobes[..., 6]),
            ),
            dim=1,
        )
        luminance = wo.new_tensor((0.2126, 0.7152, 0.0722))
        clue = (
            torch.sum(lobes[..., :3] * luminance, dim=-1)
            * lobes[..., 5]
            * lobes[..., 7]
        )
        raw_weight = torch.cat(
            (
                program.proposal_prior[:, :2] * torch.clamp(clue, min=1e-5),
                program.proposal_prior[:, 2:3],
            ),
            dim=1,
        ).float() * torch.exp(
            torch.clamp(self.proposal_adapter(semantic).float(), min=-4.0, max=4.0)
        )
        normalized = raw_weight / torch.clamp(raw_weight.sum(dim=1, keepdim=True), min=1e-12)
        fallback_floor = 0.02
        weights = (1.0 - fallback_floor) * normalized
        weights = torch.cat(
            (weights[:, :2], weights[:, 2:3] + fallback_floor), dim=1
        ).to(wo.dtype)
        alpha = torch.cat(
            (
                lobes[..., 3:5],
                torch.ones((wo.shape[0], 1, 2), dtype=wo.dtype, device=wo.device),
            ),
            dim=1,
        )
        distribution = torch.stack(
            (
                program.resource_and_flags[:, 6].to(wo.dtype),
                torch.zeros(wo.shape[0], dtype=wo.dtype, device=wo.device),
                torch.full((wo.shape[0],), 2.0, dtype=wo.dtype, device=wo.device),
            ),
            dim=1,
        )
        proposal_state = torch.stack(
            (weights, alpha[..., 0], alpha[..., 1], distribution), dim=-1
        )
        access_state = program.access_state[:, :4]
        identity_and_flags = torch.stack(
            (
                program.resource_variant,
                asset.mip_choice,
                program.resource_and_flags[:, 6].to(torch.int64),
                (program.access_state[:, 7] > 0.5).to(torch.int64),
            ),
            dim=1,
        )
        finite = (
            torch.isfinite(semantic).all(dim=1)
            & torch.isfinite(lobes).all(dim=(1, 2))
            & torch.isfinite(proposal_state).all(dim=(1, 2))
        )
        valid = asset.valid & wo_valid & finite & (normalized_wo[:, 2] > 0.0)
        return MetalBudgetedPreparedState(
            program=program,
            semantic_state=semantic,
            view_state=view_state,
            compact_frame_state=compact_frame_state,
            frames=frames,
            analytic_lobes=lobes,
            proposal_state=proposal_state,
            access_state=access_state,
            identity_and_flags=identity_and_flags,
            valid=valid,
            trace={
                **program.trace,
                **asset.trace,
                "semantic_runtime": semantic.square().mean(),
                "prepared_frame_slope": frame_slopes.square().mean(),
                "prepared_lobes": lobes.square().mean(),
                "prepared_proposal": weights.square().mean(),
            },
        )


@dataclass(frozen=True)
class MetalBudgetedDirectionalFeatures:
    values: torch.Tensor
    half_vector: torch.Tensor
    half_valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


class MetalBudgetedDirectionalRepresentation(nn.Module):
    def __init__(self, profile: MetalBudgetedProfile) -> None:
        super().__init__()
        self.profile = profile

    def forward(
        self,
        prepared: MetalBudgetedPreparedState,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> MetalBudgetedDirectionalFeatures:
        if wi.ndim != 3 or wo.shape != (wi.shape[0], 3):
            raise ValueError("Metal budgeted directions must be wo[B,3], wi[B,D,3]")
        directions = wi.shape[1]
        expanded_wo = wo[:, None, :].expand(-1, directions, -1)
        fallback = torch.zeros_like(wi)
        fallback[..., 2] = 1.0
        half_vector, half_valid = _safe_normalize(expanded_wo + wi, fallback)
        half_frame = _orthonormal_frame(half_vector)
        difference = torch.einsum("bdfc,bdc->bdf", half_frame, wi)
        feature_frame = prepared.frames[:, 1]
        wo_feature = torch.einsum("bfc,bc->bf", feature_frame, wo)
        wi_feature = torch.einsum("bfc,bdc->bdf", feature_frame, wi)
        condition = prepared.semantic_state[:, :, None].transpose(1, 2).expand(
            -1, directions, -1
        )
        values = torch.cat(
            (
                expanded_wo,
                wi,
                half_vector,
                difference,
                wo_feature[:, None, :].expand(-1, directions, -1),
                wi_feature,
                torch.sum(expanded_wo * wi, dim=-1, keepdim=True),
                half_valid.to(wi.dtype)[..., None],
                condition,
            ),
            dim=-1,
        )
        if values.shape[-1] != self.profile.directional_width:
            raise RuntimeError("Metal budgeted directional feature width drifted")
        return MetalBudgetedDirectionalFeatures(
            values=values,
            half_vector=half_vector,
            half_valid=half_valid,
            trace={
                "direction_half": half_vector.square().mean(),
                "direction_difference": difference.square().mean(),
                "direction_two_frame": (
                    wo_feature.square().mean() + wi_feature.square().mean()
                ),
            },
        )


def _anisotropic_ggx(
    wo: torch.Tensor,
    wi: torch.Tensor,
    half_vector: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    ax = torch.clamp(alpha[..., 0:1], min=0.015)
    ay = torch.clamp(alpha[..., 1:2], min=0.015)
    hx, hy, hz = half_vector.unbind(dim=-1)
    denominator = (
        (hx[..., None] / ax).square()
        + (hy[..., None] / ay).square()
        + hz[..., None].square()
    )
    distribution = 1.0 / (
        math.pi * ax * ay * torch.clamp(denominator.square(), min=1e-8)
    )
    cos_o = torch.clamp(wo[..., 2:3], min=1e-5)
    cos_i = torch.clamp(wi[..., 2:3], min=1e-5)
    lambda_o = 0.5 * (
        torch.sqrt(
            1.0
            + ((ax * wo[..., 0:1]).square() + (ay * wo[..., 1:2]).square())
            / cos_o.square()
        )
        - 1.0
    )
    lambda_i = 0.5 * (
        torch.sqrt(
            1.0
            + ((ax * wi[..., 0:1]).square() + (ay * wi[..., 1:2]).square())
            / cos_i.square()
        )
        - 1.0
    )
    geometry = 1.0 / (1.0 + lambda_o + lambda_i)
    return distribution * geometry / torch.clamp(4.0 * cos_o * cos_i, min=1e-6)


def _anisotropic_beckmann(
    wo: torch.Tensor,
    wi: torch.Tensor,
    half_vector: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    ax = torch.clamp(alpha[..., 0:1], min=0.015)
    ay = torch.clamp(alpha[..., 1:2], min=0.015)
    hz = torch.clamp(half_vector[..., 2:3], min=1e-4)
    exponent = -(
        (half_vector[..., 0:1] / ax).square()
        + (half_vector[..., 1:2] / ay).square()
    ) / hz.square()
    distribution = torch.exp(torch.clamp(exponent, min=-80.0)) / (
        math.pi * ax * ay * hz.pow(4)
    )
    cos_o = torch.clamp(wo[..., 2:3], min=1e-5)
    cos_i = torch.clamp(wi[..., 2:3], min=1e-5)
    return distribution / torch.clamp(4.0 * cos_o * cos_i, min=1e-6)


@dataclass(frozen=True)
class MetalBudgetedEvaluation:
    f: torch.Tensor
    analytic_f: torch.Tensor
    positive_f: torch.Tensor
    rgb_gate: torch.Tensor
    direct_core_auxiliary: torch.Tensor
    valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


class MetalBudgetedEvaluator(nn.Module):
    def __init__(self, profile: MetalBudgetedProfile) -> None:
        super().__init__()
        self.profile = profile
        widths = profile.evaluator_layers
        self.layers = nn.ModuleList(
            nn.Linear(left, right) for left, right in zip(widths, widths[1:])
        )
        nn.init.constant_(self.layers[-1].bias[:3], -5.0)

    @staticmethod
    def _analytic_lobes(
        prepared: MetalBudgetedPreparedState,
        wo: torch.Tensor,
        wi: torch.Tensor,
        half_vector: torch.Tensor,
    ) -> torch.Tensor:
        frames = prepared.frames
        wo_local = torch.einsum("blfc,bc->blf", frames, wo)[:, None, :, :]
        wi_local = torch.einsum("blfc,bdc->bdlf", frames, wi)
        half_local = torch.einsum("blfc,bdc->bdlf", frames, half_vector)
        state = prepared.analytic_lobes[:, None, :, :]
        alpha = state[..., 3:5]
        ggx = _anisotropic_ggx(wo_local, wi_local, half_local, alpha)
        beckmann = _anisotropic_beckmann(wo_local, wi_local, half_local, alpha)
        primary_beckmann = (
            prepared.program.resource_and_flags[:, 6] == 1
        )[:, None, None, None]
        selector = torch.cat(
            (primary_beckmann, torch.zeros_like(primary_beckmann)), dim=2
        )
        shape = torch.where(selector, beckmann, ggx)
        voh = torch.clamp(
            torch.sum(wo_local * half_local, dim=-1, keepdim=True), 0.0, 1.0
        )
        color = state[..., :3]
        fresnel = color + (1.0 - color) * (1.0 - voh).pow(5.0)
        active = (
            (wo_local[..., 2:3] > 0.0)
            & (wi_local[..., 2:3] > 0.0)
        ).to(wi.dtype)
        return fresnel * shape * state[..., 5:6] * state[..., 7:8] * active

    def forward(
        self,
        prepared: MetalBudgetedPreparedState,
        features: MetalBudgetedDirectionalFeatures,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> MetalBudgetedEvaluation:
        hidden = features.values
        layer_trace = []
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index + 1 < len(self.layers):
                hidden = F.silu(hidden)
            layer_trace.append(hidden.square().mean())
        analytic_lobes = self._analytic_lobes(
            prepared, wo, wi, features.half_vector
        )
        analytic = analytic_lobes.sum(dim=2)
        if self.profile.evaluator_mode == "hybrid":
            positive = F.softplus(hidden[..., :3])
            gate = torch.sigmoid(hidden[..., 3:6])
            direct_auxiliary = torch.zeros_like(positive)
            result = positive + gate * analytic
        else:
            positive = F.softplus(hidden[..., :3])
            gate = torch.zeros_like(positive)
            direct_auxiliary = F.softplus(hidden[..., 3:6])
            result = positive
        hemisphere = (wo[:, 2] > 0.0)[:, None] & (wi[..., 2] > 0.0)
        final_valid = prepared.valid[:, None] & hemisphere & features.half_valid
        finite = torch.isfinite(result).all(dim=-1)
        final_valid = final_valid & finite
        result = torch.where(final_valid[..., None], result, 0.0)
        return MetalBudgetedEvaluation(
            f=result,
            analytic_f=analytic,
            positive_f=positive,
            rgb_gate=gate,
            direct_core_auxiliary=direct_auxiliary,
            valid=final_valid,
            trace={
                **prepared.trace,
                **features.trace,
                "analytic_lobes": analytic_lobes.square().mean(),
                "positive_rgb": positive.square().mean(),
                "rgb_gate": gate.square().mean(),
                "direct_core_auxiliary": direct_auxiliary.square().mean(),
                "evaluator_layers": torch.stack(layer_trace).mean(),
            },
        )


__all__ = [
    "MetalBudgetedDirectionalFeatures",
    "MetalBudgetedDirectionalRepresentation",
    "MetalBudgetedEvaluation",
    "MetalBudgetedEvaluator",
    "MetalBudgetedPrepare",
    "MetalBudgetedPreparedState",
]
