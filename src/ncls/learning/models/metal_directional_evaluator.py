from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.models.metal_fused_profile import MetalFusedProfile
from ncls.learning.models.metal_typed_compiler import MetalMaterialProgramState


def _safe_normalize(value: torch.Tensor, fallback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    valid = length > 1e-7
    normalized = value / torch.clamp(length, min=1e-7)
    return torch.where(valid, normalized, fallback.expand_as(normalized)), valid.squeeze(-1)


def _orthonormal_frame(normal: torch.Tensor) -> torch.Tensor:
    fallback = torch.tensor((0.0, 1.0, 0.0), dtype=normal.dtype, device=normal.device)
    alternate = torch.tensor((1.0, 0.0, 0.0), dtype=normal.dtype, device=normal.device)
    helper = torch.where((normal[..., 2:3].abs() < 0.999), fallback, alternate)
    tangent, _ = _safe_normalize(torch.cross(helper.expand_as(normal), normal, dim=-1), alternate)
    bitangent = torch.cross(normal, tangent, dim=-1)
    return torch.stack((tangent, bitangent, normal), dim=-2)


def _to_frame(direction: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...fc,...c->...f", frame, direction)


class MetalAngularFeatureBank(nn.Module):
    def __init__(self, profile: MetalFusedProfile) -> None:
        super().__init__()
        resolutions = (16, 32, 64, 128)
        if profile.angular_levels != len(resolutions):
            raise ValueError("Metal full angular bank requires four levels")
        self.levels = nn.ParameterList(
            nn.Parameter(torch.empty(1, profile.angular_channels, size, size))
            for size in resolutions
        )
        for level in self.levels:
            nn.init.normal_(level, std=0.02)
        self.difference_x = nn.Parameter(
            torch.empty(profile.angular_difference_rank, 64)
        )
        self.difference_y = nn.Parameter(
            torch.empty(profile.angular_difference_rank, 64)
        )
        nn.init.normal_(self.difference_x, std=0.02)
        nn.init.normal_(self.difference_y, std=0.02)

    @staticmethod
    def _sample_1d(table: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        # Coordinate interpolation is a precision-sensitive random-access
        # primitive.  Its deployed form is FP32 as well, so do not let outer
        # BF16 autocast change either the index calculation or lerp weight.
        with torch.autocast(device_type=table.device.type, enabled=False):
            table_fp32 = table.float()
            position = (
                torch.clamp(0.5 * (coordinate.float() + 1.0), 0.0, 1.0)
                * (table.shape[1] - 1)
            )
            low = torch.floor(position).to(torch.int64)
            high = torch.clamp(low + 1, max=table.shape[1] - 1)
            weight = position - low.to(position.dtype)
            values_low = table_fp32[:, low].movedim(0, -1)
            values_high = table_fp32[:, high].movedim(0, -1)
            return torch.lerp(values_low, values_high, weight[..., None])

    def forward(
        self, warped_half: torch.Tensor, difference: torch.Tensor
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        flat = warped_half.reshape(-1, 2)
        multiscale = []
        with torch.autocast(device_type=flat.device.type, enabled=False):
            grid = flat.float()[:, None, None, :]
            for level in self.levels:
                expanded = level.float().expand(flat.shape[0], -1, -1, -1)
                sampled = F.grid_sample(
                    expanded,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )[:, :, 0, 0]
                multiscale.append(sampled)
        angular = torch.cat(multiscale, dim=-1).reshape(*warped_half.shape[:-1], -1)
        factor = self._sample_1d(self.difference_x, difference[..., 0])
        factor = factor * self._sample_1d(self.difference_y, difference[..., 1])
        return torch.cat((angular, factor), dim=-1), {
            "angular_multiscale": angular.square().mean(),
            "angular_difference": factor.square().mean(),
        }


@dataclass(frozen=True)
class MetalDirectionalFeatures:
    values: torch.Tensor
    half_vector: torch.Tensor
    difference: torch.Tensor
    half_valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


class MetalDirectionalRepresentation(nn.Module):
    def __init__(self, profile: MetalFusedProfile) -> None:
        super().__init__()
        self.profile = profile
        self.angular_bank = MetalAngularFeatureBank(profile)

    def forward(
        self,
        wo: torch.Tensor,
        wi: torch.Tensor,
        learned_frames: torch.Tensor,
        structured: torch.Tensor,
        compiler_latent: torch.Tensor,
        view_token: torch.Tensor,
    ) -> MetalDirectionalFeatures:
        if wi.ndim != 3 or wo.shape != (wi.shape[0], 3):
            raise ValueError("Metal directional inputs must be wo[B,3], wi[B,D,3]")
        batch, directions = wi.shape[:2]
        expanded_wo = wo[:, None, :].expand(-1, directions, -1)
        fallback = torch.zeros_like(wi)
        fallback[..., 2] = 1.0
        half_vector, half_valid = _safe_normalize(expanded_wo + wi, fallback)
        half_frame = _orthonormal_frame(half_vector)
        difference = _to_frame(wi, half_frame)
        slope = half_vector[..., :2] / torch.clamp(half_vector[..., 2:3], min=0.02)
        warped_half = slope / (1.0 + slope.abs())
        angular, angular_trace = self.angular_bank(warped_half, difference)
        frames = learned_frames[:, None, :, :, :].expand(-1, directions, -1, -1, -1)
        wo_frames = torch.einsum("bdfkc,bdc->bdfk", frames, expanded_wo)
        wi_frames = torch.einsum("bdfkc,bdc->bdfk", frames, wi)
        learned = torch.cat((wo_frames, wi_frames), dim=-1).flatten(start_dim=2)
        raw = torch.cat(
            (
                expanded_wo,
                wi,
                torch.sum(expanded_wo * wi, dim=-1, keepdim=True),
                expanded_wo[..., 2:3],
                wi[..., 2:3],
                half_valid.to(wi.dtype)[..., None],
            ),
            dim=-1,
        )
        half_difference = torch.cat((half_vector, difference, warped_half), dim=-1)
        repeated = lambda value: value[:, None, :].expand(-1, directions, -1)
        values = torch.cat(
            (
                raw,
                half_difference,
                learned,
                angular,
                repeated(structured),
                repeated(compiler_latent),
                repeated(view_token),
            ),
            dim=-1,
        )
        return MetalDirectionalFeatures(
            values,
            half_vector,
            difference,
            half_valid,
            {
                "direction_raw": raw.square().mean(),
                "direction_half_difference": half_difference.square().mean(),
                "direction_learned_frames": learned.square().mean(),
                **angular_trace,
            },
        )


class _ConditionedResidualBlock(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.linear0 = nn.Linear(width, width)
        self.linear1 = nn.Linear(width, width)
        self.film_scale = nn.Linear(rank, width)
        self.film_bias = nn.Linear(rank, width)
        self.lora_down = nn.Linear(width, rank, bias=False)
        self.lora_up = nn.Linear(rank, width, bias=False)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = 0.25 * torch.tanh(self.film_scale(condition))[:, None, :]
        bias = 0.25 * torch.tanh(self.film_bias(condition))[:, None, :]
        hidden = self.linear1(F.silu(self.linear0(value)))
        lora = self.lora_up(self.lora_down(value))
        return F.silu(value + hidden * (1.0 + scale) + lora * scale + bias)


@dataclass(frozen=True)
class MetalEvaluation:
    f: torch.Tensor
    core_f: torch.Tensor
    residual_lobes: torch.Tensor
    multiplicative: torch.Tensor
    free_tail: torch.Tensor
    valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


def _anisotropic_ggx(
    wo: torch.Tensor,
    wi: torch.Tensor,
    half_vector: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    ax = torch.clamp(alpha[..., 0:1], min=0.01)
    ay = torch.clamp(alpha[..., 1:2], min=0.01)
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
    ax = torch.clamp(alpha[..., 0:1], min=0.01)
    ay = torch.clamp(alpha[..., 1:2], min=0.01)
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


class MetalHybridEvaluator(nn.Module):
    def __init__(self, profile: MetalFusedProfile, directional_width: int) -> None:
        super().__init__()
        self.profile = profile
        self.input = nn.Linear(directional_width, profile.evaluator_width)
        self.blocks = nn.ModuleList(
            _ConditionedResidualBlock(
                profile.evaluator_width, profile.asset_adapter_rank
            )
            for _ in range(profile.evaluator_blocks)
        )
        self.correction_head = nn.Linear(profile.evaluator_width, 3)
        self.residual_amplitude_head = nn.Linear(
            profile.evaluator_width, profile.residual_lobe_count * 3
        )
        self.tail_head = nn.Linear(profile.evaluator_width, 3)
        nn.init.constant_(self.tail_head.bias, -5.0)
        self.analytic_gain_log = nn.Parameter(
            torch.zeros(profile.core_lobe_count, 3)
        )

    def _analytic_core(
        self,
        wo: torch.Tensor,
        wi: torch.Tensor,
        half_vector: torch.Tensor,
        core_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, directions = wi.shape[:2]
        wo_expanded = wo[:, None, :].expand(-1, directions, -1)
        state = core_state[:, None, :, :]
        wo_lobe = wo_expanded[:, :, None, :]
        wi_lobe = wi[:, :, None, :]
        half_lobe = half_vector[:, :, None, :]
        color = state[..., :3]
        alpha = state[..., 3:5]
        energy = state[..., 6:7]
        active = state[..., 7:8]
        fresnel_power = state[..., 8:9]
        voh = torch.clamp(
            torch.sum(wo_lobe * half_lobe, dim=-1, keepdim=True), 0.0, 1.0
        )
        fresnel = color + (1.0 - color) * (1.0 - voh).pow(5.0 + 3.0 * fresnel_power)
        ggx = _anisotropic_ggx(wo_lobe, wi_lobe, half_lobe, alpha)
        beckmann = _anisotropic_beckmann(wo_lobe, wi_lobe, half_lobe, alpha)
        cosine_i = torch.clamp(wi_lobe[..., 2:3], min=0.0)
        lambert = color / math.pi
        broad = color * (0.2 + 0.8 * cosine_i) / math.pi
        coat_f0 = 0.04 + 0.24 * color
        coat_fresnel = coat_f0 + (1.0 - coat_f0) * (1.0 - voh).pow(5.0)
        candidates = torch.stack(
            (
                fresnel * ggx,
                fresnel * beckmann,
                coat_fresnel * ggx,
                lambert,
                broad,
                fresnel * torch.sqrt(torch.clamp(ggx, min=0.0)),
            ),
            dim=-2,
        )
        selector = torch.eye(
            self.profile.core_lobe_count,
            dtype=wi.dtype,
            device=wi.device,
        )[None, None, :, :, None]
        lobes = torch.sum(candidates[:, :, :, None, :] * selector, dim=2)
        gain = torch.exp(self.analytic_gain_log)[None, None, :, :]
        lobes = lobes * energy * active * gain
        return torch.sum(lobes, dim=2), lobes

    def _positive_residual_lobes(
        self,
        amplitude: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        half_vector: torch.Tensor,
        residual_state: torch.Tensor,
    ) -> torch.Tensor:
        directions = wi.shape[1]
        state = residual_state[:, None, :, :]
        color_scale = state[..., :3]
        alpha = state[..., 3:5]
        active = state[..., 5:6]
        skew = state[..., 6:7]
        wo_lobe = wo[:, None, None, :].expand(-1, directions, self.profile.residual_lobe_count, -1)
        wi_lobe = wi[:, :, None, :].expand(-1, -1, self.profile.residual_lobe_count, -1)
        half_lobe = half_vector[:, :, None, :].expand_as(wi_lobe)
        alpha = torch.clamp(alpha * (1.0 + 0.35 * skew), min=0.01)
        shape = _anisotropic_ggx(wo_lobe, wi_lobe, half_lobe, alpha)
        return amplitude * color_scale * active * shape

    def forward(
        self,
        features: MetalDirectionalFeatures,
        wo: torch.Tensor,
        wi: torch.Tensor,
        program_state: MetalMaterialProgramState,
        core_state: torch.Tensor,
        residual_state: torch.Tensor,
        valid: torch.Tensor,
    ) -> MetalEvaluation:
        hidden = F.silu(self.input(features.values))
        block_trace = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, program_state.block_condition[:, index, :])
            block_trace.append(hidden.square().mean())
        core, core_lobes = self._analytic_core(
            wo, wi, features.half_vector, core_state
        )
        correction = program_state.correction_bound[:, None, :] * torch.tanh(
            self.correction_head(hidden)
        )
        multiplicative = torch.exp(correction)
        amplitude = F.softplus(self.residual_amplitude_head(hidden)).reshape(
            wi.shape[0], wi.shape[1], self.profile.residual_lobe_count, 3
        )
        residual_lobes = self._positive_residual_lobes(
            amplitude,
            wo,
            wi,
            features.half_vector,
            residual_state,
        )
        free_tail = (
            F.softplus(self.tail_head(hidden))
            * program_state.tail_scale[:, None, :]
        )
        result = core * multiplicative + residual_lobes.sum(dim=2) + free_tail
        hemisphere = (wo[:, 2] > 0.0)[:, None] & (wi[..., 2] > 0.0)
        final_valid = valid[:, None] & hemisphere & features.half_valid
        result = torch.where(final_valid[..., None], result, torch.zeros_like(result))
        if result.is_floating_point():
            finite = torch.isfinite(result).all()
            if finite.device.type == "cuda":
                torch._assert_async(finite)
            elif not bool(finite):
                raise RuntimeError("Metal hybrid evaluator produced non-finite f")
        return MetalEvaluation(
            result,
            core,
            residual_lobes,
            multiplicative,
            free_tail,
            final_valid,
            {
                **features.trace,
                "analytic_core": core_lobes.square().mean(),
                "multiplicative_correction": multiplicative.square().mean(),
                "positive_residual_lobes": residual_lobes.square().mean(),
                "free_positive_tail": free_tail.square().mean(),
                "hybrid_blocks": torch.stack(block_trace).mean(),
            },
        )


__all__ = [
    "MetalAngularFeatureBank",
    "MetalDirectionalFeatures",
    "MetalDirectionalRepresentation",
    "MetalEvaluation",
    "MetalHybridEvaluator",
]
