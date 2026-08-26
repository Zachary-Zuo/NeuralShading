from __future__ import annotations

import torch
from torch import nn

from ncls.learning.slang.runtime import SlangModuleSession
from ncls.paths import PROJECT_ROOT


_CORE_PATH = PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang"


def _parameter(rows: int, columns: int | None = None) -> nn.Parameter:
    shape = (rows,) if columns is None else (rows, columns)
    value = torch.empty(shape, dtype=torch.float32)
    if columns is None:
        nn.init.zeros_(value)
    else:
        nn.init.kaiming_uniform_(value, nonlinearity="relu")
    return nn.Parameter(value)


class NvidiaNeuralAppearanceModel(nn.Module):
    """只持有原方法 FP32 master 参数；方向前向全部调用同一 Slang core。"""

    def __init__(self, *, state_count: int) -> None:
        super().__init__()
        if state_count < 1:
            raise ValueError("NVIDIA baseline requires at least one material state")
        self.latent = nn.Parameter(torch.empty((state_count, 8), dtype=torch.float32))
        nn.init.normal_(self.latent, std=0.02)

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

    def _linear(
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
            weights[:, None, :],
            bias[:, None],
            flat[None, :, :],
        )
        return result.transpose(0, 1).reshape(*shape, weights.shape[0])

    def _frame_raw(self, latent: torch.Tensor) -> torch.Tensor:
        role = "Training" if torch.is_grad_enabled() else "Inference"
        function = getattr(self.session.module, "nclsNvidiaNeuralFrameDot8" + role)
        result = function(
            self.frame_w[:, None, :],
            latent[None, :, :],
        )
        return result.transpose(0, 1)

    def response(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        """返回论文网络原生的 cosine-weighted response。"""

        states = state_index.long()
        latent = self.latent[states]
        frame_raw = self._frame_raw(latent)
        direction_count = wi.shape[1]
        latent_queries = latent[:, None, :].expand(-1, direction_count, -1).contiguous()
        frame_queries = frame_raw[:, None, :].expand(-1, direction_count, -1).contiguous()
        view_queries = wo[:, None, :].expand(-1, direction_count, -1).contiguous()
        role = "Training" if torch.is_grad_enabled() else "Inference"
        input_function = getattr(
            self.session.module, "nclsNvidiaNeuralEvaluationInput" + role
        )
        inputs = input_function(
            latent_queries,
            frame_queries,
            view_queries,
            wi,
        )
        hidden = self._linear(
            "nclsNvidiaNeuralEvaluatorDot20",
            self.evaluate_w0,
            self.evaluate_b0,
            inputs,
        )
        relu = getattr(self.session.module, "nclsNvidiaNeuralRelu" + role)
        hidden = relu(hidden)
        hidden = self._linear(
            "nclsNvidiaNeuralEvaluatorDot64A",
            self.evaluate_w1,
            self.evaluate_b1,
            hidden,
        )
        hidden = relu(hidden)
        hidden = self._linear(
            "nclsNvidiaNeuralEvaluatorDot64B",
            self.evaluate_w2,
            self.evaluate_b2,
            hidden,
        )
        hidden = relu(hidden)
        raw = self._linear(
            "nclsNvidiaNeuralEvaluatorDot64Out",
            self.evaluate_out_w,
            self.evaluate_out_b,
            hidden,
        )
        decode = getattr(
            self.session.module, "nclsNvidiaNeuralDecodeResponse" + role
        )
        return decode(raw)

    def forward(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        response = self.response(state_index, wo, wi)
        role = "Training" if torch.is_grad_enabled() else "Inference"
        adapter = getattr(
            self.session.module, "nclsNvidiaNeuralResponseToBareF" + role
        )
        return adapter(response, wi)

    def sampler_raw(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        *,
        detach_latent: bool,
    ) -> torch.Tensor:
        latent = self.latent[state_index.long()]
        if detach_latent:
            latent = latent.detach()
        role = "Training" if torch.is_grad_enabled() else "Inference"
        input_function = getattr(
            self.session.module, "nclsNvidiaNeuralSamplerInput" + role
        )
        inputs = input_function(latent, wo)
        hidden = self._linear(
            "nclsNvidiaNeuralSamplerDot11",
            self.sampler_w0,
            self.sampler_b0,
            inputs,
        )
        relu = getattr(self.session.module, "nclsNvidiaNeuralRelu" + role)
        hidden = relu(hidden)
        hidden = self._linear(
            "nclsNvidiaNeuralSamplerDot32A",
            self.sampler_w1,
            self.sampler_b1,
            hidden,
        )
        hidden = relu(hidden)
        hidden = self._linear(
            "nclsNvidiaNeuralSamplerDot32B",
            self.sampler_w2,
            self.sampler_b2,
            hidden,
        )
        hidden = relu(hidden)
        return self._linear(
            "nclsNvidiaNeuralSamplerDot32Out",
            self.sampler_out_w,
            self.sampler_out_b,
            hidden,
        )

    def sampler_pdf_with_head(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        sampler: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction model only contains its native GGX9 head")
        raw = self.sampler_raw(state_index, wo, detach_latent=True)
        direction_count = wi.shape[1]
        raw_queries = raw[:, None, :].expand(-1, direction_count, -1).contiguous()
        view_queries = wo[:, None, :].expand(-1, direction_count, -1).contiguous()
        if torch.is_grad_enabled():
            pdf = self.session.module.nclsNvidiaNeuralSamplerPdfTraining(
                raw_queries,
                view_queries,
                wi,
            )
        else:
            pdf = self.session.module.nclsNvidiaNeuralSamplerPdf(
                raw_queries,
                view_queries,
                wi,
            )
        return pdf, raw

    def sampler_pdf(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        sampler: str,
    ) -> torch.Tensor:
        return self.sampler_pdf_with_head(state_index, wo, wi, sampler)[0]

    def set_sampler_training(self, sampler: str) -> None:
        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction model only contains its native GGX9 head")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("sampler_"))

    def reset_sampler_parameters(self, sampler: str) -> None:
        """matched stage 不继承 joint reproduction 的 sampler head。"""

        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction model only contains its native GGX9 head")
        for name, parameter in self.named_parameters():
            if name.startswith("sampler_"):
                if parameter.ndim == 1:
                    nn.init.zeros_(parameter)
                else:
                    nn.init.kaiming_uniform_(parameter, nonlinearity="relu")

    def sampler_parameter_count(self, sampler: str) -> int:
        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("NVIDIA reproduction model only contains its native GGX9 head")
        return 2793
