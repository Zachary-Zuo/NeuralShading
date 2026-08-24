from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE_ID = "ncls.small-view-conditioned-mlp@1"
SHARED_ARCHITECTURE_ID = "ncls.shared-small-view-conditioned-mlp@1"
SPARSE_DICTIONARY_ARCHITECTURE_ID = (
    "ncls.shared-small-view-conditioned-mlp-sparse-latent-dictionary@1"
)
FACTORIZED_LATENT_ARCHITECTURE_ID = (
    "ncls.shared-small-view-conditioned-mlp-factorized-material-latent@1"
)
TARGET_TENSOR_ENCODER_ARCHITECTURE_ID = (
    "ncls.target-tensor-encoder-shared-small-view-conditioned-mlp@1"
)
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


def _linear_macs(module: nn.Module) -> int:
    return sum(
        child.in_features * child.out_features
        for child in module.modules()
        if isinstance(child, nn.Linear)
    )


class SingleMaterialEvaluatorModel(nn.Module):
    """E1 单材质 evaluator 的最小模型接口；具体表示仍由各 pipeline 决定。"""

    architecture_id: str

    def cost_summary(self) -> dict[str, int]:
        raise NotImplementedError


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


class SingleMaterialNeuralEvaluator(SingleMaterialEvaluatorModel):
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

        return {
            "parameter_count": latent_count + shared_count,
            "B_asset_fp32": 4 * latent_count,
            "B_shared_fp32": 4 * shared_count,
            "C_prepare_macs": _linear_macs(self.prepare_network),
            "C_eval_macs": _linear_macs(self.evaluate_network),
        }


class SharedMaterialNeuralEvaluator(SingleMaterialNeuralEvaluator):
    """共享 prepare/evaluate 权重，并为每个 target-visible state 优化一条 latent。"""

    architecture_id = SHARED_ARCHITECTURE_ID

    def __init__(self, config: NeuralEvaluatorModelConfig, material_count: int) -> None:
        if material_count < 1:
            raise ValueError("shared evaluator requires at least one material")
        super().__init__(config)
        del self.material_latent
        self.material_latents = nn.Embedding(material_count, config.latent_dimension)
        nn.init.normal_(self.material_latents.weight, mean=0.0, std=0.02)
        self.material_count = material_count

    def prepare(self, wo: torch.Tensor, material_slots: torch.Tensor) -> torch.Tensor:
        if material_slots.ndim != 1 or len(material_slots) != len(wo):
            raise ValueError("shared evaluator material slots must match wo groups")
        latent = self.material_latents(material_slots.long())
        return self.prepare_network(torch.cat((latent, self._view_features(wo)), dim=-1))

    def forward(
        self,
        wo: torch.Tensor,
        wi: torch.Tensor,
        material_slots: torch.Tensor,
    ) -> torch.Tensor:
        if wo.ndim != 2 or wi.ndim != 3 or len(wo) != len(wi):
            raise ValueError("shared evaluator expects wo [group,3] and wi [group,direction,3]")
        prepared = self.prepare(wo, material_slots)
        light_features = self._light_features(wo, wi)
        repeated = prepared[:, None, :].expand(-1, wi.shape[1], -1)
        return self.evaluate_network(torch.cat((repeated, light_features), dim=-1))

    def cost_summary(self) -> dict[str, int]:
        total_latent_count = self.material_latents.weight.numel()
        shared_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name != "material_latents.weight"
        )

        per_material_count = self.config.latent_dimension
        return {
            "parameter_count": total_latent_count + shared_count,
            "material_count": self.material_count,
            "B_asset_fp32": 4 * per_material_count,
            "B_asset_fp32_total": 4 * total_latent_count,
            "B_shared_fp32": 4 * shared_count,
            "C_prepare_macs": _linear_macs(self.prepare_network),
            "C_eval_macs": _linear_macs(self.evaluate_network),
        }


class SparseDictionaryMaterialNeuralEvaluator(SharedMaterialNeuralEvaluator):
    """每个材质只保存 top-k 字典索引/权重，字典与 decoder 跨材质共享。"""

    architecture_id = SPARSE_DICTIONARY_ARCHITECTURE_ID

    def __init__(
        self,
        config: NeuralEvaluatorModelConfig,
        material_count: int,
        *,
        dictionary_size: int,
        top_k: int,
    ) -> None:
        if dictionary_size < 2 or dictionary_size > 65535:
            raise ValueError("sparse latent dictionary size must lie in [2, 65535]")
        if top_k < 1 or top_k > dictionary_size:
            raise ValueError("sparse latent top_k must lie in [1, dictionary_size]")
        super().__init__(config, material_count)
        del self.material_latents
        self.dictionary_size = dictionary_size
        self.top_k = top_k
        self.latent_dictionary = nn.Parameter(torch.empty(
            dictionary_size, config.latent_dimension
        ))
        self.material_logits = nn.Parameter(torch.empty(material_count, dictionary_size))
        nn.init.normal_(self.latent_dictionary, mean=0.0, std=0.02)
        nn.init.normal_(self.material_logits, mean=0.0, std=0.01)

    def _material_latent_for_slots(self, material_slots: torch.Tensor) -> torch.Tensor:
        logits = self.material_logits[material_slots.long()]
        values, indices = torch.topk(logits, self.top_k, dim=-1, sorted=True)
        weights = torch.softmax(values, dim=-1)
        codewords = self.latent_dictionary[indices]
        return torch.sum(weights[..., None] * codewords, dim=-2)

    def prepare(self, wo: torch.Tensor, material_slots: torch.Tensor) -> torch.Tensor:
        if material_slots.ndim != 1 or len(material_slots) != len(wo):
            raise ValueError("sparse dictionary material slots must match wo groups")
        latent = self._material_latent_for_slots(material_slots)
        return self.prepare_network(torch.cat((latent, self._view_features(wo)), dim=-1))

    def cost_summary(self) -> dict[str, int]:
        logits_count = self.material_logits.numel()
        dictionary_count = self.latent_dictionary.numel()
        decoder_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name not in {"material_logits", "latent_dictionary"}
        )
        # Runtime 资产使用 uint16 codeword ID 与 fp32 mixing weight；完整 logits 只在
        # target-visible 优化期存在，不伪装成部署资产。
        per_material_bytes = self.top_k * (2 + 4)
        return {
            "parameter_count": logits_count + dictionary_count + decoder_count,
            "optimized_training_parameter_count": logits_count,
            "material_count": self.material_count,
            "dictionary_size": self.dictionary_size,
            "top_k": self.top_k,
            "B_asset_sparse_indices_u16": 2 * self.top_k,
            "B_asset_sparse_weights_fp32": 4 * self.top_k,
            "B_asset_fp32": per_material_bytes,
            "B_asset_fp32_total": per_material_bytes * self.material_count,
            "B_shared_fp32": 4 * (dictionary_count + decoder_count),
            "C_prepare_latent_mix_macs": self.top_k * self.config.latent_dimension,
            "C_prepare_macs": (
                _linear_macs(self.prepare_network)
                + self.top_k * self.config.latent_dimension
            ),
            "C_eval_macs": _linear_macs(self.evaluate_network),
        }


class FactorizedMaterialNeuralEvaluator(SharedMaterialNeuralEvaluator):
    """把 state×latent 表分解为材质系数与共享低秩 basis。"""

    architecture_id = FACTORIZED_LATENT_ARCHITECTURE_ID

    def __init__(
        self,
        config: NeuralEvaluatorModelConfig,
        material_count: int,
        *,
        factor_rank: int,
    ) -> None:
        if factor_rank < 1:
            raise ValueError("factorized material latent rank must be positive")
        super().__init__(config, material_count)
        del self.material_latents
        self.factor_rank = factor_rank
        self.material_factors = nn.Parameter(torch.empty(material_count, factor_rank))
        self.latent_basis = nn.Parameter(torch.empty(factor_rank, config.latent_dimension))
        nn.init.normal_(self.material_factors, mean=0.0, std=0.1)
        nn.init.normal_(self.latent_basis, mean=0.0, std=0.1)

    def _material_latent_for_slots(self, material_slots: torch.Tensor) -> torch.Tensor:
        return self.material_factors[material_slots.long()] @ self.latent_basis

    def prepare(self, wo: torch.Tensor, material_slots: torch.Tensor) -> torch.Tensor:
        if material_slots.ndim != 1 or len(material_slots) != len(wo):
            raise ValueError("factorized latent material slots must match wo groups")
        latent = self._material_latent_for_slots(material_slots)
        return self.prepare_network(torch.cat((latent, self._view_features(wo)), dim=-1))

    def cost_summary(self) -> dict[str, int]:
        factor_count = self.material_factors.numel()
        basis_count = self.latent_basis.numel()
        decoder_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name not in {"material_factors", "latent_basis"}
        )
        per_material_count = self.factor_rank
        return {
            "parameter_count": factor_count + basis_count + decoder_count,
            "material_count": self.material_count,
            "factor_rank": self.factor_rank,
            "B_asset_fp32": 4 * per_material_count,
            "B_asset_fp32_total": 4 * factor_count,
            "B_shared_fp32": 4 * (basis_count + decoder_count),
            "C_prepare_latent_factor_macs": (
                self.factor_rank * self.config.latent_dimension
            ),
            "C_prepare_macs": (
                _linear_macs(self.prepare_network)
                + self.factor_rank * self.config.latent_dimension
            ),
            "C_eval_macs": _linear_macs(self.evaluate_network),
        }


class TargetTensorEncoderMaterialNeuralEvaluator(SharedMaterialNeuralEvaluator):
    """压缩期 DeepSets encoder 读取 train-only response tensor，runtime 烘焙其 latent。"""

    architecture_id = TARGET_TENSOR_ENCODER_ARCHITECTURE_ID

    def __init__(
        self,
        config: NeuralEvaluatorModelConfig,
        target_encoder_input: torch.Tensor,
        *,
        encoder_width: int,
        encoder_layer_count: int,
    ) -> None:
        if target_encoder_input.ndim != 3 or min(target_encoder_input.shape) < 1:
            raise ValueError("target tensor encoder input must be [material,point,feature]")
        if encoder_width < 1 or encoder_layer_count < 1:
            raise ValueError("target tensor encoder dimensions must be positive")
        material_count, point_count, feature_count = target_encoder_input.shape
        super().__init__(config, material_count)
        del self.material_latents
        self.encoder_width = encoder_width
        self.encoder_layer_count = encoder_layer_count
        self.encoder_point_count = point_count
        self.encoder_feature_count = feature_count
        self.register_buffer(
            "target_encoder_input",
            target_encoder_input.detach().to(dtype=torch.float32).contiguous(),
            persistent=False,
        )
        self.point_encoder = _mlp(
            feature_count,
            encoder_width,
            encoder_width,
            encoder_layer_count,
            config.activation,
        )
        self.latent_head = _mlp(
            2 * encoder_width,
            config.latent_dimension,
            encoder_width,
            1,
            config.activation,
        )
        self._cached_eval_latents: torch.Tensor | None = None
        for module in (self.point_encoder, self.latent_head):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    nn.init.zeros_(child.bias)

    def train(self, mode: bool = True) -> TargetTensorEncoderMaterialNeuralEvaluator:
        if mode:
            self._cached_eval_latents = None
        return super().train(mode)

    def _encode_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        material_count, point_count, feature_count = inputs.shape
        encoded = self.point_encoder(inputs.reshape(material_count * point_count, feature_count))
        encoded = encoded.reshape(material_count, point_count, self.encoder_width)
        pooled = torch.cat((torch.mean(encoded, dim=1), torch.amax(encoded, dim=1)), dim=-1)
        return self.latent_head(pooled)

    def _material_latent_for_slots(self, material_slots: torch.Tensor) -> torch.Tensor:
        if not self.training and not torch.is_grad_enabled():
            if self._cached_eval_latents is None:
                self._cached_eval_latents = self._encode_inputs(self.target_encoder_input)
            return self._cached_eval_latents[material_slots.long()]
        unique_slots, inverse = torch.unique(
            material_slots.long(), sorted=True, return_inverse=True
        )
        encoded = self._encode_inputs(self.target_encoder_input[unique_slots])
        return encoded[inverse]

    def prepare(self, wo: torch.Tensor, material_slots: torch.Tensor) -> torch.Tensor:
        if material_slots.ndim != 1 or len(material_slots) != len(wo):
            raise ValueError("target tensor encoder material slots must match wo groups")
        latent = self._material_latent_for_slots(material_slots)
        return self.prepare_network(torch.cat((latent, self._view_features(wo)), dim=-1))

    def cost_summary(self) -> dict[str, int]:
        encoder_count = sum(
            parameter.numel()
            for module in (self.point_encoder, self.latent_head)
            for parameter in module.parameters()
        )
        encoder_parameter_ids = {
            id(parameter)
            for module in (self.point_encoder, self.latent_head)
            for parameter in module.parameters()
        }
        decoder_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in encoder_parameter_ids
        )
        per_material_count = self.config.latent_dimension
        compiler_macs = (
            self.encoder_point_count * _linear_macs(self.point_encoder)
            + _linear_macs(self.latent_head)
        )
        return {
            "parameter_count": encoder_count + decoder_count,
            "compiler_parameter_count": encoder_count,
            "material_count": self.material_count,
            "B_asset_fp32": 4 * per_material_count,
            "B_asset_fp32_total": 4 * per_material_count * self.material_count,
            "B_shared_fp32": 4 * decoder_count,
            "B_compiler_shared_fp32": 4 * encoder_count,
            "B_compiler_input_fp32": (
                4 * self.encoder_point_count * self.encoder_feature_count
            ),
            "B_compiler_input_fp32_total": (
                4
                * self.encoder_point_count
                * self.encoder_feature_count
                * self.material_count
            ),
            "C_compile_encoder_macs": compiler_macs,
            "C_prepare_macs": _linear_macs(self.prepare_network),
            "C_eval_macs": _linear_macs(self.evaluate_network),
        }


def positive_response(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw)
