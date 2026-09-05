from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.slang.runtime import SlangModuleSession
from ncls.paths import PROJECT_ROOT
from ncls.learning.source_adaptation import NativeAssetCollection


_CORE_PATH = PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang"


def _parameter(rows: int, columns: int | None = None) -> nn.Parameter:
    shape = (rows,) if columns is None else (rows, columns)
    value = torch.empty(shape, dtype=torch.float32)
    if columns is None:
        nn.init.zeros_(value)
    else:
        nn.init.kaiming_uniform_(value, nonlinearity="relu")
    return nn.Parameter(value)


class NvidiaModel(nn.Module):
    """论文 encoder、hierarchical z8 与两个 decoder 的 FP32 master。"""

    def __init__(
        self,
        *,
        native_feature_count: int,
        latent_width: int,
        latent_height: int,
        latent_mip_count: int,
    ) -> None:
        super().__init__()
        if native_feature_count < 1 or latent_width < 1 or latent_height < 1:
            raise ValueError("NVIDIA encoder input and latent extent must be positive")
        maximum_mips = 1 + max(latent_width, latent_height).bit_length() - 1
        if not 1 <= latent_mip_count <= maximum_mips:
            raise ValueError("NVIDIA latent mip count is outside the texture extent")
        self.native_feature_count = int(native_feature_count)
        self.latent_width = int(latent_width)
        self.latent_height = int(latent_height)
        self.latent_mip_count = int(latent_mip_count)
        self.phase_name = "bootstrap"

        self.encoder_w0 = _parameter(64, self.native_feature_count)
        self.encoder_b0 = _parameter(64)
        self.encoder_w1 = _parameter(64, 64)
        self.encoder_b1 = _parameter(64)
        self.encoder_w2 = _parameter(64, 64)
        self.encoder_b2 = _parameter(64)
        self.encoder_w3 = _parameter(64, 64)
        self.encoder_b3 = _parameter(64)
        self.encoder_out_w = _parameter(8, 64)
        self.encoder_out_b = _parameter(8)

        level_values = []
        level_shapes = []
        level_offsets = [0]
        width, height = self.latent_width, self.latent_height
        for _ in range(self.latent_mip_count):
            value = torch.empty((height * width, 8), dtype=torch.float32)
            nn.init.normal_(value, std=0.02)
            level_values.append(value)
            level_shapes.append((height, width))
            level_offsets.append(level_offsets[-1] + height * width)
            width, height = max(1, width // 2), max(1, height // 2)
        self.latent_texels = nn.Parameter(
            torch.cat(level_values, dim=0), requires_grad=False
        )
        self._mip_shapes = tuple(level_shapes)
        self.register_buffer(
            "_mip_offsets_tensor",
            torch.tensor(level_offsets, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_mip_shapes_tensor",
            torch.tensor(level_shapes, dtype=torch.int64),
            persistent=False,
        )

        self.frame_w = _parameter(12, 8)
        self.evaluate_w0 = _parameter(64, 20)
        self.evaluate_b0 = _parameter(64)
        self.evaluate_w1 = _parameter(64, 64)
        self.evaluate_b1 = _parameter(64)
        self.evaluate_w2 = _parameter(64, 64)
        self.evaluate_b2 = _parameter(64)
        self.evaluate_out_w = _parameter(3, 64)
        self.evaluate_out_b = _parameter(3)

        self.sampler_w0 = _parameter(32, 11)
        self.sampler_b0 = _parameter(32)
        self.sampler_w1 = _parameter(32, 32)
        self.sampler_b1 = _parameter(32)
        self.sampler_w2 = _parameter(32, 32)
        self.sampler_b2 = _parameter(32)
        self.sampler_out_w = _parameter(9, 32)
        self.sampler_out_b = _parameter(9)

        self._session: SlangModuleSession | None = None

    @property
    def session(self) -> SlangModuleSession:
        if self._session is None:
            self._session = SlangModuleSession(_CORE_PATH)
        return self._session

    @property
    def mip_shapes(self) -> tuple[tuple[int, int], ...]:
        return self._mip_shapes

    @property
    def latent_levels(self) -> tuple[torch.Tensor, ...]:
        levels = []
        offsets = self._mip_offsets_tensor.tolist()
        for index, (height, width) in enumerate(self._mip_shapes):
            levels.append(
                self.latent_texels[offsets[index] : offsets[index + 1]]
                .reshape(height, width, 8)
                .permute(2, 0, 1)
            )
        return tuple(levels)

    def set_training_phase(self, phase_name: str) -> None:
        if phase_name not in {"bootstrap", "finetune"}:
            raise ValueError(f"unsupported NVIDIA training phase {phase_name!r}")
        self.phase_name = phase_name
        for name, parameter in self.named_parameters():
            if name.startswith("encoder_"):
                parameter.requires_grad_(phase_name == "bootstrap")
            elif name == "latent_texels":
                parameter.requires_grad_(phase_name == "finetune")
            else:
                parameter.requires_grad_(True)

    def encode(self, native_features: torch.Tensor) -> torch.Tensor:
        if native_features.ndim != 2 or native_features.shape[1] != self.native_feature_count:
            raise ValueError("native feature tensor disagrees with NVIDIA encoder input")
        hidden = F.relu(F.linear(native_features, self.encoder_w0, self.encoder_b0))
        hidden = F.relu(F.linear(hidden, self.encoder_w1, self.encoder_b1))
        hidden = F.relu(F.linear(hidden, self.encoder_w2, self.encoder_b2))
        hidden = F.relu(F.linear(hidden, self.encoder_w3, self.encoder_b3))
        return F.linear(hidden, self.encoder_out_w, self.encoder_out_b)

    def materialize(self, native_assets: NativeAssetCollection) -> None:
        if len(native_assets.descriptors) != 1:
            raise ValueError("NVIDIA reproduction materializes exactly one native asset")
        descriptor = native_assets.descriptors[0]
        if len(descriptor.domains) != 1:
            raise ValueError("NVIDIA reproduction requires one canonical native domain")
        domain = descriptor.domains[0]
        level_shapes = domain.level_shapes
        if len(level_shapes) != self.latent_mip_count:
            raise ValueError("native asset collection does not match latent mip count")
        if domain.channel_count != self.native_feature_count:
            raise ValueError("native asset channels disagree with encoder")
        if level_shapes != self.mip_shapes:
            raise ValueError("native asset extents disagree with latent hierarchy")
        with torch.no_grad():
            offsets = self._mip_offsets_tensor.tolist()
            destinations = []
            for level_index, (height, width) in enumerate(self.mip_shapes):
                destination = self.latent_texels[
                    offsets[level_index] : offsets[level_index + 1]
                ].reshape(height, width, int(self.latent_texels.shape[1]))
                destinations.append(destination)
            for request in native_assets.iter_tile_requests(
                0, domain.domain_id, 262_144, 0
            ):
                tile = native_assets.acquire_tile(request, self.latent_texels.device)
                try:
                    features = tile.core.reshape(-1, domain.channel_count)
                    encoded = self.encode(features)
                    origin_y, origin_x = tile.origin_yx
                    core_height, core_width = tile.core_shape
                    destinations[tile.mip_level][
                        origin_y : origin_y + core_height,
                        origin_x : origin_x + core_width,
                    ].copy_(
                        encoded.reshape(
                            core_height, core_width, int(self.latent_texels.shape[1])
                        )
                    )
                finally:
                    tile.release()
        self.set_training_phase("finetune")

    @staticmethod
    def _bilinear_wrap(level: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        height, width = int(level.shape[1]), int(level.shape[2])
        wrapped = torch.remainder(uv, 1.0)
        x = wrapped[:, 0] * width - 0.5
        y = wrapped[:, 1] * height - 0.5
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        tx = (x - x0).unsqueeze(1)
        ty = (y - y0).unsqueeze(1)
        x0w, x1w = torch.remainder(x0, width), torch.remainder(x0 + 1, width)
        y0w, y1w = torch.remainder(y0, height), torch.remainder(y0 + 1, height)
        v00 = level[:, y0w, x0w].transpose(0, 1)
        v10 = level[:, y0w, x1w].transpose(0, 1)
        v01 = level[:, y1w, x0w].transpose(0, 1)
        v11 = level[:, y1w, x1w].transpose(0, 1)
        return (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + (v01 * (1.0 - tx) + v11 * tx) * ty

    def fetch_latent(self, uv: torch.Tensor, mip_level: torch.Tensor) -> torch.Tensor:
        if uv.ndim != 2 or uv.shape[1] != 2 or mip_level.shape != (uv.shape[0],):
            raise ValueError("latent fetch requires uv [batch,2] and mip_level [batch]")
        selected = torch.clamp(torch.round(mip_level).long(), 0, self.latent_mip_count - 1)
        shapes = self._mip_shapes_tensor[selected]
        height, width = shapes[:, 0], shapes[:, 1]
        wrapped = torch.remainder(uv, 1.0)
        x = wrapped[:, 0] * width - 0.5
        y = wrapped[:, 1] * height - 0.5
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        tx, ty = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
        x0w, x1w = torch.remainder(x0, width), torch.remainder(x0 + 1, width)
        y0w, y1w = torch.remainder(y0, height), torch.remainder(y0 + 1, height)
        offset = self._mip_offsets_tensor[selected]

        def gather(x_index: torch.Tensor, y_index: torch.Tensor) -> torch.Tensor:
            return self.latent_texels[offset + y_index * width + x_index]

        v00 = gather(x0w, y0w)
        v10 = gather(x1w, y0w)
        v01 = gather(x0w, y1w)
        v11 = gather(x1w, y1w)
        return (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + (
            v01 * (1.0 - tx) + v11 * tx
        ) * ty

    def latent_for_batch(self, tensors: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(tensors, torch.Tensor):
            return tensors
        if self.phase_name == "bootstrap":
            return self.encode(tensors["native_features"])
        return self.fetch_latent(tensors["uv"], tensors["mip_level"])

    def _linear_slang(
        self,
        callable_name: str,
        weights: torch.Tensor,
        bias: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        shape = value.shape[:-1]
        flat = value.reshape(-1, value.shape[-1])
        role = "Training" if torch.is_grad_enabled() else "Inference"
        result = getattr(self.session.module, callable_name + role)(
            weights[:, None, :], bias[:, None], flat[None, :, :]
        )
        return result.transpose(0, 1).reshape(*shape, weights.shape[0])

    def _frame_raw_slang(self, latent: torch.Tensor) -> torch.Tensor:
        role = "Training" if torch.is_grad_enabled() else "Inference"
        function = getattr(self.session.module, "nclsNvidiaNeuralFrameDot8" + role)
        return function(self.frame_w[:, None, :], latent[None, :, :]).transpose(0, 1)

    @staticmethod
    def _learned_frames(frame_raw: torch.Tensor) -> torch.Tensor:
        raw = frame_raw.reshape(len(frame_raw), 2, 6)
        normal_offset = torch.tensor(
            (0.0, 0.0, 1.0), dtype=raw.dtype, device=raw.device
        )
        tangent_offset = torch.tensor(
            (1.0, 0.0, 0.0), dtype=raw.dtype, device=raw.device
        )
        normal_value = raw[..., :3] + normal_offset
        tangent_value = raw[..., 3:] + tangent_offset
        normal = normal_value * torch.rsqrt(
            torch.sum(normal_value * normal_value, dim=-1, keepdim=True)
        )
        tangent = tangent_value * torch.rsqrt(
            torch.sum(tangent_value * tangent_value, dim=-1, keepdim=True)
        )
        bitangent = torch.linalg.cross(normal, tangent, dim=-1)
        return torch.stack((tangent, bitangent, normal), dim=2)

    def _evaluation_input(
        self,
        latent: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        frame_raw = F.linear(latent, self.frame_w)
        frames = self._learned_frames(frame_raw)
        fixed = torch.sum(wo[:, None, None, :] * frames, dim=-1)
        queried = torch.einsum("bdc,bfkc->bdfk", wi, frames)
        fixed = fixed[:, None, :, :].expand(-1, wi.shape[1], -1, -1)
        projected = torch.cat((fixed, queried), dim=-1).reshape(
            len(latent), wi.shape[1], 12
        )
        latent_queries = latent[:, None, :].expand(-1, wi.shape[1], -1)
        return torch.cat((projected, latent_queries), dim=-1)

    def evaluate_f(
        self, latent: torch.Tensor, wo: torch.Tensor, wi: torch.Tensor
    ) -> torch.Tensor:
        inputs = self._evaluation_input(latent, wo, wi)
        hidden = F.relu(F.linear(inputs, self.evaluate_w0, self.evaluate_b0))
        hidden = F.relu(F.linear(hidden, self.evaluate_w1, self.evaluate_b1))
        hidden = F.relu(F.linear(hidden, self.evaluate_w2, self.evaluate_b2))
        raw = F.linear(hidden, self.evaluate_out_w, self.evaluate_out_b)
        return torch.exp(raw - 3.0)

    def evaluate_f_slang(
        self, latent: torch.Tensor, wo: torch.Tensor, wi: torch.Tensor
    ) -> torch.Tensor:
        """独立部署数学oracle；训练主路径不通过Slang autograd。"""

        frame_raw = self._frame_raw_slang(latent)
        direction_count = wi.shape[1]
        latent_queries = latent[:, None, :].expand(-1, direction_count, -1).contiguous()
        frame_queries = frame_raw[:, None, :].expand(-1, direction_count, -1).contiguous()
        view_queries = wo[:, None, :].expand(-1, direction_count, -1).contiguous()
        role = "Training" if torch.is_grad_enabled() else "Inference"
        inputs = getattr(self.session.module, "nclsNvidiaNeuralEvaluationInput" + role)(
            latent_queries, frame_queries, view_queries, wi
        )
        hidden = getattr(self.session.module, "nclsNvidiaNeuralRelu" + role)(
            self._linear_slang("nclsNvidiaNeuralEvaluatorDot20", self.evaluate_w0, self.evaluate_b0, inputs)
        )
        hidden = getattr(self.session.module, "nclsNvidiaNeuralRelu" + role)(
            self._linear_slang("nclsNvidiaNeuralEvaluatorDot64A", self.evaluate_w1, self.evaluate_b1, hidden)
        )
        hidden = getattr(self.session.module, "nclsNvidiaNeuralRelu" + role)(
            self._linear_slang("nclsNvidiaNeuralEvaluatorDot64B", self.evaluate_w2, self.evaluate_b2, hidden)
        )
        raw = self._linear_slang(
            "nclsNvidiaNeuralEvaluatorDot64Out", self.evaluate_out_w, self.evaluate_out_b, hidden
        )
        return getattr(self.session.module, "nclsNvidiaNeuralDecodeF" + role)(raw)

    def forward(
        self, latent: torch.Tensor, wo: torch.Tensor, wi: torch.Tensor
    ) -> torch.Tensor:
        return self.evaluate_f(latent, wo, wi)

    def sampler_raw(self, latent: torch.Tensor, wo: torch.Tensor, *, detach_latent: bool) -> torch.Tensor:
        if detach_latent:
            latent = latent.detach()
        inputs = torch.cat((wo, latent), dim=-1)
        hidden = F.relu(F.linear(inputs, self.sampler_w0, self.sampler_b0))
        hidden = F.relu(F.linear(hidden, self.sampler_w1, self.sampler_b1))
        hidden = F.relu(F.linear(hidden, self.sampler_w2, self.sampler_b2))
        return F.linear(hidden, self.sampler_out_w, self.sampler_out_b)

    def sampler_pdf_with_head(
        self,
        latent: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        sampler: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction model only contains its native GGX9 head")
        raw = self.sampler_raw(latent, wo, detach_latent=True)
        return self._sampler_pdf_from_raw(raw, wo, wi), raw

    def _sampler_pdf_from_raw(
        self, raw: torch.Tensor, wo: torch.Tensor, wi: torch.Tensor
    ) -> torch.Tensor:
        alpha = 1e-4 + 0.5 * (
            1.0 + raw[:, :2] * torch.rsqrt(1.0 + raw[:, :2] * raw[:, :2])
        )
        rho = torch.clamp(
            raw[:, 2] * torch.rsqrt(1.0 + raw[:, 2] * raw[:, 2]),
            -0.999999,
            0.999999,
        )
        specular_slope = raw[:, 3:5] * torch.sqrt(1.0 + raw[:, 3:5] * raw[:, 3:5])
        diffuse_slope = raw[:, 5:7] * torch.sqrt(1.0 + raw[:, 5:7] * raw[:, 5:7])
        weights = torch.softmax(raw[:, 7:9], dim=-1)

        view = wo[:, None, :]
        half_value = view + wi
        half_vector = half_value * torch.rsqrt(
            torch.clamp(torch.sum(half_value * half_value, dim=-1, keepdim=True), min=1e-20)
        )
        half_vector = half_vector * torch.where(
            half_vector[..., 2:3] >= 0.0, 1.0, -1.0
        )
        wo_dot_half = torch.sum(view * half_vector, dim=-1)
        slope_x = -half_vector[..., 0] / half_vector[..., 2] - specular_slope[:, None, 0]
        slope_y = -half_vector[..., 1] / half_vector[..., 2] - specular_slope[:, None, 1]
        sqrt_one_minus_rho = torch.sqrt(torch.clamp(1.0 - rho * rho, min=1e-12))
        normalization = 1.0 / (alpha[:, 0] * alpha[:, 1] * sqrt_one_minus_rho)
        slope_x_standard = slope_x / alpha[:, None, 0]
        slope_y_standard = (
            alpha[:, None, 0] * slope_y
            - rho[:, None] * alpha[:, None, 1] * slope_x
        ) * normalization[:, None]
        radius_squared = slope_x_standard * slope_x_standard + slope_y_standard * slope_y_standard
        p22 = (1.0 / (math.pi * (1.0 + radius_squared) ** 2)) * normalization[:, None]
        half_pdf = p22 / (half_vector[..., 2] ** 3)
        specular_pdf = half_pdf / (4.0 * torch.abs(wo_dot_half))

        inverse_length = torch.rsqrt(
            1.0 + torch.sum(diffuse_slope * diffuse_slope, dim=-1)
        )
        diffuse_normal = torch.cat(
            (-diffuse_slope, torch.ones_like(diffuse_slope[:, :1])), dim=-1
        ) * inverse_length[:, None]
        diffuse_pdf = torch.clamp(
            torch.sum(wi * diffuse_normal[:, None, :], dim=-1), min=0.0
        ) / math.pi
        return weights[:, None, 0] * specular_pdf + weights[:, None, 1] * diffuse_pdf

    def sampler_pdf_slang(
        self, raw: torch.Tensor, wo: torch.Tensor, wi: torch.Tensor
    ) -> torch.Tensor:
        direction_count = wi.shape[1]
        raw_queries = raw[:, None, :].expand(-1, direction_count, -1).contiguous()
        view_queries = wo[:, None, :].expand(-1, direction_count, -1).contiguous()
        return self.session.module.nclsNvidiaNeuralSamplerPdf(
            raw_queries, view_queries, wi
        )

    def sampler_sample_with_head(
        self,
        latent: torch.Tensor,
        wo: torch.Tensor,
        sample_u: torch.Tensor,
        sampler: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从当前 learned proposal 取样；方向路径无梯度，score path只经过PDF。"""

        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction model only contains its native GGX9 head")
        if sample_u.shape != (latent.shape[0], 2):
            raise ValueError("NVIDIA sampler requires one float2 sample per latent")
        raw = self.sampler_raw(latent, wo, detach_latent=True)
        with torch.no_grad():
            sampled = self.session.module.nclsNvidiaNeuralSamplerSample(
                raw.detach(), wo, sample_u
            )
        wi = sampled[:, None, :3].contiguous()
        valid = sampled[:, None, 3] > 0.5
        pdf = self._sampler_pdf_from_raw(raw, wo, wi)
        return wi, pdf, raw, valid

    def sampler_pdf(
        self, latent: torch.Tensor, wo: torch.Tensor, wi: torch.Tensor, sampler: str
    ) -> torch.Tensor:
        return self.sampler_pdf_with_head(latent, wo, wi, sampler)[0]

    def sampler_parameter_count(self, sampler: str) -> int:
        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction model only contains its native GGX9 head")
        return 2793
