from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _safe_normalize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length_squared = torch.sum(value * value, dim=-1, keepdim=True)
    valid = length_squared > 1e-16
    fallback = torch.zeros_like(value)
    fallback[..., 2] = 1.0
    normalized = torch.where(
        valid,
        value * torch.rsqrt(torch.clamp(length_squared, min=1e-16)),
        fallback,
    )
    return normalized, valid.to(value.dtype)


def prepare_features(wo: torch.Tensor) -> torch.Tensor:
    radial = torch.sqrt(torch.clamp(wo[..., 0] ** 2 + wo[..., 1] ** 2, min=0.0))
    grazing = -torch.log(torch.clamp(wo[..., 2], min=1e-4)) / 10.0
    return torch.cat(
        (
            wo,
            wo[..., 2:3],
            radial[..., None],
            grazing[..., None],
            wo[..., 2:3] ** 2,
        ),
        dim=-1,
    )


def direction_features(
    wo: torch.Tensor,
    wi: torch.Tensor,
    fourier_bands: int,
) -> torch.Tensor:
    if wo.ndim != 2 or wi.ndim != 3 or wo.shape[0] != wi.shape[0]:
        raise ValueError("direction features require wo [G,3] and wi [G,N,3]")
    expanded_wo = wo[:, None, :].expand(-1, wi.shape[1], -1)
    half_vector, half_valid = _safe_normalize(expanded_wo + wi)
    half_z = torch.clamp(half_vector[..., 2:3], min=1e-5)
    slope = half_vector[..., :2] / half_z
    log_slope = torch.sign(slope) * torch.log1p(torch.abs(slope))

    radial = torch.sqrt(torch.clamp(
        half_vector[..., 0:1] ** 2 + half_vector[..., 1:2] ** 2,
        min=0.0,
    ))
    tangent = torch.cat(
        (-half_vector[..., 1:2], half_vector[..., 0:1], torch.zeros_like(radial)),
        dim=-1,
    )
    fallback = torch.zeros_like(tangent)
    fallback[..., 0] = 1.0
    tangent = torch.where(
        radial > 1e-6,
        tangent / torch.clamp(radial, min=1e-6),
        fallback,
    )
    bitangent = torch.linalg.cross(half_vector, tangent, dim=-1)
    difference = torch.stack(
        (
            torch.sum(wi * tangent, dim=-1),
            torch.sum(wi * bitangent, dim=-1),
            torch.sum(wi * half_vector, dim=-1),
        ),
        dim=-1,
    )
    raw = torch.cat(
        (
            expanded_wo,
            wi,
            expanded_wo[..., 2:3],
            wi[..., 2:3],
            torch.sum(expanded_wo * wi, dim=-1, keepdim=True),
            half_vector,
            log_slope,
            difference,
            half_valid,
        ),
        dim=-1,
    )
    if fourier_bands == 0:
        return raw
    frequencies = torch.pow(
        torch.tensor(2.0, dtype=wi.dtype, device=wi.device),
        torch.arange(fourier_bands, dtype=wi.dtype, device=wi.device),
    )
    phase = math.pi * log_slope[..., None] * frequencies
    fourier = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(-2)
    return torch.cat((raw, fourier), dim=-1)


def direction_feature_count(fourier_bands: int) -> int:
    return 18 + 4 * fourier_bands


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_1 = nn.Linear(width, width)
        self.linear_2 = nn.Linear(width, width)

    def forward(
        self,
        value: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.linear_2(F.gelu(self.linear_1(self.norm(value))))
        residual = (1.0 + 0.1 * torch.tanh(gamma)) * residual + 0.1 * beta
        return (value + residual) * (2.0 ** -0.5)


class ConditionedSharedEvaluator(nn.Module):
    """P1 M1：view-conditioned prepare + 逐层 FiLM 的共享 evaluator。"""

    def __init__(
        self,
        *,
        state_count: int,
        output_scale: Sequence[Sequence[float]],
        width: int,
        latent_dim: int,
        prepare_blocks: int,
        evaluate_blocks: int,
        fourier_bands: int,
        initial_output_ratio: float,
        output_mode: str = "direct",
    ) -> None:
        super().__init__()
        if state_count < 1 or width < 16 or latent_dim < 4:
            raise ValueError("invalid conditioned evaluator dimensions")
        if output_mode not in {"direct", "residual"}:
            raise ValueError("conditioned evaluator output mode is unsupported")
        scale = torch.as_tensor(output_scale, dtype=torch.float32)
        if scale.shape != (state_count, 3) or torch.any(scale <= 0.0):
            raise ValueError("output scale must contain positive RGB values per state")
        self.state_count = state_count
        self.width = width
        self.fourier_bands = fourier_bands
        self.output_mode = output_mode
        self.register_buffer("output_scale", scale)
        self.latent = nn.Embedding(state_count, latent_dim)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.02)
        block_count = prepare_blocks + evaluate_blocks
        self.condition = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.Linear(width, width + 2 * width * block_count),
        )
        self.prepare_input = nn.Linear(7 + width, width)
        self.prepare_layers = nn.ModuleList(
            FiLMResidualBlock(width) for _ in range(prepare_blocks)
        )
        self.evaluate_input = nn.Linear(
            direction_feature_count(fourier_bands) + width,
            width,
        )
        self.evaluate_layers = nn.ModuleList(
            FiLMResidualBlock(width) for _ in range(evaluate_blocks)
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        if output_mode == "direct":
            initial_bias = math.log(math.expm1(initial_output_ratio))
            nn.init.constant_(self.head[-1].bias, initial_bias)

    def _film(
        self,
        condition: torch.Tensor,
        block_index: int,
        direction_axis: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.width + block_index * 2 * self.width
        gamma = condition[..., start : start + self.width]
        beta = condition[..., start + self.width : start + 2 * self.width]
        if direction_axis:
            gamma = gamma[:, None, :]
            beta = beta[:, None, :]
        return gamma, beta

    def forward(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        states = state_index.long()
        latent = self.latent(states)
        condition = self.condition(latent)
        context = condition[..., : self.width]
        prepared = self.prepare_input(torch.cat((prepare_features(wo), context), dim=-1))
        block_index = 0
        for layer in self.prepare_layers:
            gamma, beta = self._film(condition, block_index, False)
            prepared = layer(prepared, gamma, beta)
            block_index += 1
        directional = direction_features(wo, wi, self.fourier_bands)
        prepared_queries = prepared[:, None, :].expand(-1, wi.shape[1], -1)
        hidden = self.evaluate_input(torch.cat((directional, prepared_queries), dim=-1))
        for layer in self.evaluate_layers:
            gamma, beta = self._film(condition, block_index, True)
            hidden = layer(hidden, gamma, beta)
            block_index += 1
        raw = self.head(hidden)
        scale = self.output_scale[states][:, None, :]
        if self.output_mode == "residual":
            return raw * scale
        return F.softplus(raw) * scale


class PlainResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_1 = nn.Linear(width, width)
        self.linear_2 = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.linear_2(F.gelu(self.linear_1(self.norm(value))))
        return (value + residual) * (2.0 ** -0.5)


class PerStateNetwork(nn.Module):
    def __init__(
        self,
        feature_count: int,
        width: int,
        block_count: int,
        initial_output_ratio: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(feature_count, width)
        self.blocks = nn.Sequential(*(PlainResidualBlock(width) for _ in range(block_count)))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        initial_bias = math.log(math.expm1(initial_output_ratio))
        nn.init.constant_(self.head[-1].bias, initial_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input(features)))


class PerStateTeacher(nn.Module):
    """P1 T：每个 state 一套独立权重，不经过共享 latent 瓶颈。"""

    def __init__(
        self,
        *,
        state_count: int,
        output_scale: Sequence[Sequence[float]],
        width: int,
        block_count: int,
        fourier_bands: int,
        initial_output_ratio: float,
    ) -> None:
        super().__init__()
        scale = torch.as_tensor(output_scale, dtype=torch.float32)
        if scale.shape != (state_count, 3) or torch.any(scale <= 0.0):
            raise ValueError("teacher output scale must contain positive RGB values per state")
        self.state_count = state_count
        self.fourier_bands = fourier_bands
        self.register_buffer("output_scale", scale)
        feature_count = direction_feature_count(fourier_bands)
        self.networks = nn.ModuleList(
            PerStateNetwork(feature_count, width, block_count, initial_output_ratio)
            for _ in range(state_count)
        )

    def forward(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        features = direction_features(wo, wi, self.fourier_bands)
        result = torch.zeros((*features.shape[:-1], 3), dtype=features.dtype, device=features.device)
        for state in torch.unique(state_index.long()).tolist():
            mask = state_index.long() == state
            raw = self.networks[state](features[mask])
            result[mask] = F.softplus(raw) * self.output_scale[state]
        return result
