from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE_ID = "ncls.small-view-conditioned-mlp@1"
DIRECTION_ENCODING_IDS = (
    "ncls.local-cartesian-directions@1",
    "ncls.fourier-cartesian-directions@1",
    "ncls.half-difference-directions@1",
    "ncls.multiscale-half-slope-directions@1",
)
ACTIVATION_IDS = ("relu", "silu", "gelu")


def _activation(name: str) -> type[nn.Module]:
    return {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[name]


def _fourier(value: torch.Tensor, band_count: int) -> torch.Tensor:
    parts = [value]
    for band in range(band_count):
        frequency = torch.pi * float(2**band)
        parts.extend((torch.sin(frequency * value), torch.cos(frequency * value)))
    return torch.cat(parts, dim=-1)


def _safe_normalize(value: torch.Tensor) -> torch.Tensor:
    fallback = torch.zeros_like(value)
    fallback[..., 2] = 1.0
    length_squared = torch.sum(value * value, dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(torch.clamp(length_squared, min=1e-12))
    return torch.where(length_squared > 1e-12, normalized, fallback)


def _mlp(
    input_dimension: int,
    output_dimension: int,
    width: int,
    hidden_layer_count: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dimension = input_dimension
    for _ in range(hidden_layer_count):
        layers.extend((nn.Linear(dimension, width), _activation(activation)()))
        dimension = width
    layers.append(nn.Linear(dimension, output_dimension))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class NeuralEvaluatorModelConfig:
    latent_dimension: int = 16
    width: int = 64
    prepare_layer_count: int = 1
    evaluate_layer_count: int = 3
    activation: str = "silu"
    direction_encoding_id: str = "ncls.fourier-cartesian-directions@1"
    fourier_band_count: int = 4
    output_bias: float = -3.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NeuralEvaluatorModelConfig":
        allowed = {
            "latent_dimension", "width", "prepare_layer_count", "evaluate_layer_count",
            "activation", "direction_encoding_id", "fourier_band_count",
            "output_bias",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"neural evaluator received unsupported model parameters: {sorted(unknown)}")
        return cls(**{name: value[name] for name in allowed if name in value})

    def __post_init__(self) -> None:
        if min(
            self.latent_dimension,
            self.width,
            self.evaluate_layer_count,
            self.fourier_band_count,
        ) < 1 or self.prepare_layer_count < 0:
            raise ValueError("neural evaluator dimensions must be positive")
        if self.activation not in ACTIVATION_IDS:
            raise ValueError(f"unsupported neural evaluator activation {self.activation!r}")
        if self.direction_encoding_id not in DIRECTION_ENCODING_IDS:
            raise ValueError(f"unsupported direction encoding {self.direction_encoding_id!r}")


class SingleMaterialNeuralEvaluator(nn.Module):
    """一个 material latent、可复用 view prepare 和逐 wi 小 MLP。"""

    architecture_id = ARCHITECTURE_ID

    def __init__(self, config: NeuralEvaluatorModelConfig) -> None:
        super().__init__()
        self.config = config
        self.material_latent = nn.Parameter(torch.zeros(config.latent_dimension))
        if config.direction_encoding_id == "ncls.fourier-cartesian-directions@1":
            view_dimension = 3 * (1 + 2 * config.fourier_band_count)
            light_dimension = view_dimension
        elif config.direction_encoding_id == "ncls.half-difference-directions@1":
            view_dimension = 3
            light_dimension = 7
        elif config.direction_encoding_id == "ncls.multiscale-half-slope-directions@1":
            view_dimension = 3
            light_dimension = 30
        else:
            view_dimension = light_dimension = 3
        prepare_input = config.latent_dimension + view_dimension
        if config.prepare_layer_count:
            self.prepare_network = _mlp(
                prepare_input,
                config.width,
                config.width,
                config.prepare_layer_count,
                config.activation,
            )
            self.prepared_dimension = config.width
        else:
            self.prepare_network = nn.Identity()
            self.prepared_dimension = prepare_input
        self.evaluate_network = _mlp(
            self.prepared_dimension + light_dimension,
            3,
            config.width,
            config.evaluate_layer_count,
            config.activation,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.material_latent, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.evaluate_network[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.bias, self.config.output_bias)

    def _view_features(self, wo: torch.Tensor) -> torch.Tensor:
        if self.config.direction_encoding_id == "ncls.fourier-cartesian-directions@1":
            return _fourier(wo, self.config.fourier_band_count)
        return wo

    def _light_features(self, wo: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        if self.config.direction_encoding_id == "ncls.fourier-cartesian-directions@1":
            return _fourier(wi, self.config.fourier_band_count)
        if self.config.direction_encoding_id in {
            "ncls.half-difference-directions@1",
            "ncls.multiscale-half-slope-directions@1",
        }:
            view = wo[:, None, :].expand_as(wi)
            half = _safe_normalize(view + wi)
            difference = _safe_normalize(wi - view)
            cosine = torch.sum(view * wi, dim=-1, keepdim=True)
            base = torch.cat((half, difference, cosine), dim=-1)
            if self.config.direction_encoding_id == "ncls.half-difference-directions@1":
                return base
            slope = half[..., :2] / torch.clamp(torch.abs(half[..., 2:3]), min=1e-4)
            slope_x, slope_y = slope[..., 0:1], slope[..., 1:2]
            quadratic = torch.cat(
                (slope, slope_x * slope_x, slope_x * slope_y, slope_y * slope_y), dim=-1
            )
            peak_parts = []
            radial_squared = torch.sum(slope * slope, dim=-1, keepdim=True)
            for scale in (0.001, 0.002, 0.004, 0.008, 0.02, 0.08):
                inverse_variance = 0.5 / (scale * scale)
                peak_parts.extend((
                    torch.exp(-slope_x * slope_x * inverse_variance),
                    torch.exp(-slope_y * slope_y * inverse_variance),
                    torch.exp(-radial_squared * inverse_variance),
                ))
            return torch.cat((base, quadratic, *peak_parts), dim=-1)
        return wi

    def prepare(self, wo: torch.Tensor) -> torch.Tensor:
        latent = self.material_latent[None, :].expand(len(wo), -1)
        return self.prepare_network(torch.cat((latent, self._view_features(wo)), dim=-1))

    def forward(self, wo: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        if wo.ndim != 2 or wi.ndim != 3 or len(wo) != len(wi):
            raise ValueError("neural evaluator expects wo [group,3] and wi [group,direction,3]")
        prepared = self.prepare(wo)
        light_features = self._light_features(wo, wi)
        repeated = prepared[:, None, :].expand(-1, wi.shape[1], -1)
        return self.evaluate_network(torch.cat((repeated, light_features), dim=-1))

    def cost_summary(self) -> dict[str, int]:
        latent_count = self.material_latent.numel()
        shared_count = sum(parameter.numel() for name, parameter in self.named_parameters() if name != "material_latent")

        def linear_macs(module: nn.Module) -> int:
            return sum(
                child.in_features * child.out_features
                for child in module.modules()
                if isinstance(child, nn.Linear)
            )

        return {
            "parameter_count": latent_count + shared_count,
            "B_asset_fp32": 4 * latent_count,
            "B_shared_fp32": 4 * shared_count,
            "C_prepare_macs": linear_macs(self.prepare_network),
            "C_eval_macs": linear_macs(self.evaluate_network),
        }


def positive_response(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw)
