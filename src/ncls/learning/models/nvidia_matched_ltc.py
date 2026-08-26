from __future__ import annotations

import torch
from torch import nn

from ncls.learning.slang import NvidiaMatchedLtcSlangSession

from .nvidia_neural_appearance import NvidiaNeuralAppearanceModel


def _parameter(rows: int, columns: int | None = None) -> nn.Parameter:
    shape = (rows,) if columns is None else (rows, columns)
    value = torch.empty(shape, dtype=torch.float32)
    if columns is None:
        nn.init.zeros_(value)
    else:
        nn.init.kaiming_uniform_(value, nonlinearity="relu")
    return nn.Parameter(value)


class NvidiaNeuralAppearanceLtcAdaptationModel(nn.Module):
    """包装冻结的 exact evaluator，只为 matched comparison 增加原规模 LTC-K2 head。"""

    def __init__(self, reproduction: NvidiaNeuralAppearanceModel) -> None:
        super().__init__()
        self.reproduction = reproduction
        self.ltc_sampler_w0 = _parameter(32, 11)
        self.ltc_sampler_b0 = _parameter(32)
        self.ltc_sampler_w1 = _parameter(32, 32)
        self.ltc_sampler_b1 = _parameter(32)
        self.ltc_sampler_w2 = _parameter(32, 32)
        self.ltc_sampler_b2 = _parameter(32)
        self.ltc_sampler_out_w = _parameter(13, 32)
        self.ltc_sampler_out_b = _parameter(13)
        self._session: NvidiaMatchedLtcSlangSession | None = None

    @property
    def session(self) -> NvidiaMatchedLtcSlangSession:
        if self._session is None:
            self._session = NvidiaMatchedLtcSlangSession()
        return self._session

    def forward(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        return self.reproduction(state_index, wo, wi)

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
        function = getattr(self.session.module, callable_name + role)
        result = function(
            weights[:, None, :],
            bias[:, None],
            flat[None, :, :],
        )
        return result.transpose(0, 1).reshape(*shape, weights.shape[0])

    def sampler_raw(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.reproduction.latent[state_index.long()].detach()
        role = "Training" if torch.is_grad_enabled() else "Inference"
        inputs = getattr(
            self.session.module, "nclsNvidiaMatchedLtcInput" + role
        )(latent, wo)
        hidden = self._linear(
            "nclsNvidiaMatchedLtcDot11",
            self.ltc_sampler_w0,
            self.ltc_sampler_b0,
            inputs,
        )
        relu = getattr(
            self.session.module, "nclsNvidiaMatchedLtcRelu" + role
        )
        hidden = relu(hidden)
        hidden = self._linear(
            "nclsNvidiaMatchedLtcDot32A",
            self.ltc_sampler_w1,
            self.ltc_sampler_b1,
            hidden,
        )
        hidden = relu(hidden)
        hidden = self._linear(
            "nclsNvidiaMatchedLtcDot32B",
            self.ltc_sampler_w2,
            self.ltc_sampler_b2,
            hidden,
        )
        hidden = relu(hidden)
        return self._linear(
            "nclsNvidiaMatchedLtcDot32Out",
            self.ltc_sampler_out_w,
            self.ltc_sampler_out_b,
            hidden,
        )

    def sampler_pdf_with_head(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        sampler: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sampler != "ltc-k2":
            raise ValueError("NVIDIA LTC adaptation contains only the matched LTC-K2 head")
        raw = self.sampler_raw(state_index, wo)
        direction_count = wi.shape[1]
        raw_queries = raw[:, None, :].expand(-1, direction_count, -1).contiguous()
        if torch.is_grad_enabled():
            pdf = self.session.module.nclsNvidiaMatchedLtcPdfTraining(raw_queries, wi)
        else:
            pdf = self.session.module.nclsNvidiaMatchedLtcPdf(raw_queries, wi)
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
        if sampler != "ltc-k2":
            raise ValueError("NVIDIA LTC adaptation contains only the matched LTC-K2 head")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("ltc_sampler_"))

    def reset_sampler_parameters(self, sampler: str) -> None:
        if sampler != "ltc-k2":
            raise ValueError("NVIDIA LTC adaptation contains only the matched LTC-K2 head")
        for name, parameter in self.named_parameters():
            if name.startswith("ltc_sampler_"):
                if parameter.ndim == 1:
                    nn.init.zeros_(parameter)
                else:
                    nn.init.kaiming_uniform_(parameter, nonlinearity="relu")

    def sampler_parameter_count(self, sampler: str) -> int:
        if sampler != "ltc-k2":
            raise ValueError("NVIDIA LTC adaptation contains only the matched LTC-K2 head")
        return 2925


def adapt_nvidia_model_for_sampler(
    model: NvidiaNeuralAppearanceModel,
    sampler: str,
) -> NvidiaNeuralAppearanceModel | NvidiaNeuralAppearanceLtcAdaptationModel:
    if sampler == "nvidia-diffuse-ggx9":
        return model
    if sampler == "ltc-k2":
        return NvidiaNeuralAppearanceLtcAdaptationModel(model)
    raise ValueError("unsupported NVIDIA matched sampler")


__all__ = [
    "NvidiaNeuralAppearanceLtcAdaptationModel",
    "adapt_nvidia_model_for_sampler",
]
