from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .neural_evaluator import SingleMaterialEvaluatorModel


ARCHITECTURE_ID = "ncls.plane-factorized-neural-evaluator@1"


def _activation(name: str) -> type[nn.Module]:
    try:
        return {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[name]
    except KeyError as error:
        raise ValueError(f"unsupported plane evaluator activation {name!r}") from error


@dataclass(frozen=True)
class PlaneFactorizedModelConfig:
    plane_resolution: int = 32
    plane_feature_dimension: int = 4
    material_latent_dimension: int = 8
    width: int = 64
    evaluate_layer_count: int = 2
    activation: str = "gelu"
    output_bias: float = -0.75

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlaneFactorizedModelConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"plane evaluator received unsupported model parameters: {sorted(unknown)}")
        return cls(**{name: value[name] for name in allowed if name in value})

    def __post_init__(self) -> None:
        if min(
            self.plane_resolution,
            self.plane_feature_dimension,
            self.material_latent_dimension,
            self.width,
            self.evaluate_layer_count,
        ) < 1:
            raise ValueError("plane evaluator dimensions must be positive")
        if self.plane_resolution < 2:
            raise ValueError("plane evaluator requires at least a 2x2 plane")
        if self.plane_feature_dimension > 4:
            raise ValueError("plane evaluator v1 stores one RGBA-width feature vector per texel")
        _activation(self.activation)


class PlaneFactorizedNeuralEvaluator(SingleMaterialEvaluatorModel):
    """六个成对方向 plane + 小 MLP；每个 query 只做固定次数双线性读取。"""

    architecture_id = ARCHITECTURE_ID
    plane_names = ("wo_xy", "wi_xy", "wo_x_wi_x", "wo_x_wi_y", "wo_y_wi_x", "wo_y_wi_y")

    def __init__(self, config: PlaneFactorizedModelConfig) -> None:
        super().__init__()
        self.config = config
        shape = (
            len(self.plane_names),
            config.plane_feature_dimension,
            config.plane_resolution,
            config.plane_resolution,
        )
        self.planes = nn.Parameter(torch.empty(shape))
        self.material_latent = nn.Parameter(torch.zeros(config.material_latent_dimension))
        prepared_dimension = 3 + config.material_latent_dimension + config.plane_feature_dimension
        evaluate_input_dimension = prepared_dimension + 3 + 5 * config.plane_feature_dimension
        layers: list[nn.Module] = []
        dimension = evaluate_input_dimension
        for _ in range(config.evaluate_layer_count):
            layers.extend((nn.Linear(dimension, config.width), _activation(config.activation)()))
            dimension = config.width
        layers.append(nn.Linear(dimension, 3))
        self.evaluate_network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.planes, mean=0.0, std=0.02)
        nn.init.normal_(self.material_latent, mean=0.0, std=0.02)
        for module in self.evaluate_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.evaluate_network[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.bias, self.config.output_bias)

    @staticmethod
    def _bilinear(plane: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        resolution = plane.shape[-1]
        position = torch.clamp(0.5 * coordinates + 0.5, 0.0, 1.0) * (resolution - 1)
        lower = torch.floor(position).long()
        upper = torch.clamp(lower + 1, max=resolution - 1)
        fraction = position - lower.to(position.dtype)
        flat_shape = coordinates.shape[:-1]
        x0, y0 = lower[..., 0].reshape(-1), lower[..., 1].reshape(-1)
        x1, y1 = upper[..., 0].reshape(-1), upper[..., 1].reshape(-1)

        def gather(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return plane[:, y, x].movedim(0, -1).reshape(*flat_shape, plane.shape[0])

        v00, v10 = gather(x0, y0), gather(x1, y0)
        v01, v11 = gather(x0, y1), gather(x1, y1)
        fx, fy = fraction[..., 0:1], fraction[..., 1:2]
        return torch.lerp(torch.lerp(v00, v10, fx), torch.lerp(v01, v11, fx), fy)

    def prepare(self, wo: torch.Tensor) -> torch.Tensor:
        wo_feature = self._bilinear(self.planes[0], wo[..., :2])
        latent = self.material_latent[None, :].expand(len(wo), -1)
        return torch.cat((wo, latent, wo_feature), dim=-1)

    def forward(self, wo: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        if wo.ndim != 2 or wi.ndim != 3 or len(wo) != len(wi):
            raise ValueError("plane evaluator expects wo [group,3] and wi [group,direction,3]")
        prepared = self.prepare(wo)[:, None, :].expand(-1, wi.shape[1], -1)
        wi_feature = self._bilinear(self.planes[1], wi[..., :2])
        wo_xy = wo[:, None, :2].expand(-1, wi.shape[1], -1)
        cross_coordinates = (
            torch.stack((wo_xy[..., 0], wi[..., 0]), dim=-1),
            torch.stack((wo_xy[..., 0], wi[..., 1]), dim=-1),
            torch.stack((wo_xy[..., 1], wi[..., 0]), dim=-1),
            torch.stack((wo_xy[..., 1], wi[..., 1]), dim=-1),
        )
        cross_features = [
            self._bilinear(self.planes[index], coordinates)
            for index, coordinates in enumerate(cross_coordinates, start=2)
        ]
        return self.evaluate_network(torch.cat((prepared, wi, wi_feature, *cross_features), dim=-1))

    def cost_summary(self) -> dict[str, int]:
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        evaluate_macs = sum(
            module.in_features * module.out_features
            for module in self.evaluate_network.modules()
            if isinstance(module, nn.Linear)
        )
        plane_bytes = 4 * self.planes.numel()
        return {
            "parameter_count": parameter_count,
            "B_asset_fp32": 4 * parameter_count,
            "B_shared_fp32": 0,
            "C_prepare_macs": 0,
            "C_eval_macs": evaluate_macs,
            "plane_bytes_fp32": plane_bytes,
            "plane_texel_fetches_prepare": 4,
            "plane_texel_fetches_eval": 20,
            "plane_lerp_components_prepare": self.config.plane_feature_dimension,
            "plane_lerp_components_eval": 5 * self.config.plane_feature_dimension,
        }
