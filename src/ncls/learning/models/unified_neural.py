from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ncls.learning.slang import UnifiedSlangSession


UNIFIED_TOP_FLOAT_FIELDS = (
    "alphaX", "alphaY", "relativeIor", "etaR", "etaG", "etaB",
    "kR", "kG", "kB", "colorR", "colorG", "colorB",
    "tangentRotation", "reserved",
)


def _parameter(rows: int, columns: int | None = None) -> nn.Parameter:
    shape = (rows,) if columns is None else (rows, columns)
    value = torch.empty(shape, dtype=torch.float32)
    if columns is None:
        nn.init.zeros_(value)
    else:
        nn.init.kaiming_uniform_(value, a=0.1)
    return nn.Parameter(value)


class UnifiedNeuralModel(nn.Module):
    """仅持有FP32 master参数；所有方向相关前向均由同一Slang core执行。"""

    def __init__(
        self,
        *,
        state_count: int,
        response_scale: Sequence[Sequence[float]],
        top_rows: Sequence[Mapping[str, Any]],
        evaluator: str,
        runtime_class: str,
        initial_output_ratio: float = 0.01,
    ) -> None:
        super().__init__()
        if evaluator not in {"nvidia-frame-two-lobe-v1", "core-frame-neural-v1"}:
            raise ValueError("unsupported unified evaluator")
        if runtime_class not in {"realtime", "diagnostic"}:
            raise ValueError("unsupported unified runtime class")
        if not 0.0 < initial_output_ratio <= 0.25:
            raise ValueError("unified initial output ratio must lie in (0, 0.25]")
        scale = torch.as_tensor(response_scale, dtype=torch.float32)
        if scale.shape != (state_count, 3) or torch.any(scale <= 0.0):
            raise ValueError("unified response scale must be positive RGB per state")
        if len(top_rows) != state_count:
            raise ValueError("unified top-interface rows disagree with state count")
        self.evaluator = evaluator
        self.runtime_class = runtime_class
        self.latent = nn.Parameter(torch.zeros((state_count, 16), dtype=torch.float32))
        nn.init.normal_(self.latent, std=0.02)
        self.prepare_w0 = _parameter(64, 23)
        self.prepare_b0 = _parameter(64)
        self.prepare_w1 = _parameter(64, 64)
        self.prepare_b1 = _parameter(64)
        self.evaluator_state_w = _parameter(14, 64)
        self.evaluator_state_b = _parameter(14)
        self.nvidia_sampler_w = _parameter(9, 64)
        self.nvidia_sampler_b = _parameter(9)
        self.ltc_sampler_w = _parameter(13, 64)
        self.ltc_sampler_b = _parameter(13)
        self.set_sampler_training(None)
        width = 64 if runtime_class == "diagnostic" else 32
        self.evaluate_w0 = _parameter(width, 17)
        self.evaluate_b0 = _parameter(width)
        self.evaluate_w1 = _parameter(width, width)
        self.evaluate_b1 = _parameter(width)
        if runtime_class == "diagnostic":
            self.evaluate_w2 = _parameter(64, 64)
            self.evaluate_b2 = _parameter(64)
        else:
            self.register_parameter("evaluate_w2", None)
            self.register_parameter("evaluate_b2", None)
        self.evaluate_out_w = _parameter(3, width)
        self.evaluate_out_b = _parameter(3)
        initial_bias = math.log(math.expm1(initial_output_ratio))
        nn.init.constant_(self.evaluate_out_b, initial_bias)
        self.register_buffer("response_scale", scale)
        self.register_buffer(
            "top_kind",
            torch.as_tensor([int(row["interface_kind"]) for row in top_rows], dtype=torch.int32),
        )
        for field in UNIFIED_TOP_FLOAT_FIELDS:
            source = {
                "alphaX": [row["alpha"][0] for row in top_rows],
                "alphaY": [row["alpha"][1] for row in top_rows],
                "relativeIor": [row["relative_ior"] for row in top_rows],
                "etaR": [row["eta"][0] for row in top_rows],
                "etaG": [row["eta"][1] for row in top_rows],
                "etaB": [row["eta"][2] for row in top_rows],
                "kR": [row["k"][0] for row in top_rows],
                "kG": [row["k"][1] for row in top_rows],
                "kB": [row["k"][2] for row in top_rows],
                "colorR": [row["color"][0] for row in top_rows],
                "colorG": [row["color"][1] for row in top_rows],
                "colorB": [row["color"][2] for row in top_rows],
                "tangentRotation": [row["tangent_rotation"] for row in top_rows],
                "reserved": [0.0 for _ in top_rows],
            }[field]
            self.register_buffer(f"top_{field}", torch.as_tensor(source, dtype=torch.float32))
        self._session: UnifiedSlangSession | None = None

    @property
    def session(self) -> UnifiedSlangSession:
        if self._session is None:
            self._session = UnifiedSlangSession()
        return self._session

    @staticmethod
    def _role(*values: torch.Tensor) -> str:
        return "Training" if torch.is_grad_enabled() and any(
            value.requires_grad for value in values
        ) else "Inference"

    def _linear(
        self,
        name: str,
        weights: torch.Tensor,
        bias: torch.Tensor,
        value: torch.Tensor,
        *,
        role_specific: bool = False,
    ) -> torch.Tensor:
        """用Slang标量row primitive执行线性层；Torch只排列广播维。"""
        shape = value.shape[:-1]
        flat = value.reshape(-1, value.shape[-1])
        rows = weights.shape[0]
        callable_name = name + self._role(weights, bias, value) if role_specific else name
        result = getattr(self.session.module, callable_name)(
            weights[:, None, :],
            bias[:, None],
            flat[None, :, :],
        )
        return result.transpose(0, 1).reshape(*shape, rows)

    def _top(self, state_index: torch.Tensor, direction_count: int) -> dict[str, torch.Tensor]:
        states = state_index.long()
        result: dict[str, torch.Tensor] = {
            "kind": self.top_kind[states],
            "flags": torch.zeros_like(self.top_kind[states]),
        }
        for field in UNIFIED_TOP_FLOAT_FIELDS:
            result[field] = getattr(self, f"top_{field}")[states]
        return {
            name: value[:, None].expand(-1, direction_count).contiguous()
            for name, value in result.items()
        }

    def prepare_hidden(self, state_index: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        module = self.session.module
        latent = self.latent[state_index.long()]
        inputs = getattr(module, "nclsUnifiedPrepareInput" + self._role(latent))(
            latent, wo
        )
        hidden = self._linear(
            "nclsUnifiedPrepareDot23",
            self.prepare_w0,
            self.prepare_b0,
            inputs,
            role_specific=True,
        )
        hidden = getattr(
            module, "nclsUnifiedPrepareRelu" + self._role(hidden)
        )(hidden)
        hidden = self._linear(
            "nclsUnifiedPrepareDot64",
            self.prepare_w1,
            self.prepare_b1,
            hidden,
            role_specific=True,
        )
        return getattr(
            module, "nclsUnifiedPrepareRelu" + self._role(hidden)
        )(hidden)

    def set_sampler_training(self, sampler: str | None) -> None:
        """冻结 evaluator stage 的两个 head，或只打开指定 sampler head。"""
        if sampler not in {None, "nvidia-diffuse-ggx9", "ltc-k2"}:
            raise ValueError("unsupported unified sampler")
        selected = {
            "nvidia_sampler_w",
            "nvidia_sampler_b",
        } if sampler == "nvidia-diffuse-ggx9" else {
            "ltc_sampler_w",
            "ltc_sampler_b",
        } if sampler == "ltc-k2" else set()
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in selected if sampler is not None else not name.startswith(("nvidia_sampler_", "ltc_sampler_")))

    def reset_sampler_parameters(self, sampler: str) -> None:
        if sampler == "nvidia-diffuse-ggx9":
            weight, bias = self.nvidia_sampler_w, self.nvidia_sampler_b
        elif sampler == "ltc-k2":
            weight, bias = self.ltc_sampler_w, self.ltc_sampler_b
        else:
            raise ValueError("unsupported unified sampler")
        nn.init.kaiming_uniform_(weight, a=0.1)
        nn.init.zeros_(bias)

    def _prepared_with_head(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        sampler: str,
        *,
        detach_shared: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        module = self.session.module
        hidden = self.prepare_hidden(state_index, wo)
        if detach_shared:
            hidden = hidden.detach()
        evaluator = self._linear(
            "nclsUnifiedEvaluatorStateDot64",
            self.evaluator_state_w,
            self.evaluator_state_b,
            hidden,
        )
        if sampler == "nvidia-diffuse-ggx9":
            head = self._linear(
                "nclsUnifiedNvidiaSamplerHeadDot64",
                self.nvidia_sampler_w,
                self.nvidia_sampler_b,
                hidden,
                role_specific=True,
            )
            role = self._role(head)
            prepared = getattr(
                module, "nclsUnifiedJoinNvidiaSamplerState" + role
            )(evaluator, head)
            return prepared, head
        if sampler == "ltc-k2":
            head = self._linear(
                "nclsUnifiedLtcSamplerHeadDot64",
                self.ltc_sampler_w,
                self.ltc_sampler_b,
                hidden,
                role_specific=True,
            )
            role = self._role(head)
            prepared = getattr(
                module, "nclsUnifiedJoinLtcSamplerState" + role
            )(evaluator, head)
            return prepared, head
        raise ValueError("unsupported unified sampler")

    def prepared(self, state_index: torch.Tensor, wo: torch.Tensor, sampler: str) -> torch.Tensor:
        return self._prepared_with_head(state_index, wo, sampler)[0]

    def prepared_evaluator(self, state_index: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        """Evaluator stage不调用冻结sampler callable，避免污染其首次active mask。"""
        hidden = self.prepare_hidden(state_index, wo)
        evaluator = self._linear(
            "nclsUnifiedEvaluatorStateDot64",
            self.evaluator_state_w,
            self.evaluator_state_b,
            hidden,
            role_specific=True,
        )
        return getattr(
            self.session.module,
            "nclsUnifiedJoinEvaluatorState" + self._role(evaluator),
        )(evaluator)

    def forward(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        *,
        sampler: str = "nvidia-diffuse-ggx9",
    ) -> torch.Tensor:
        module = self.session.module
        group_count, direction_count = wi.shape[:2]
        prepared = self.prepared_evaluator(state_index, wo)
        prepared_queries = prepared[:, None, :].expand(-1, direction_count, -1).contiguous()
        wo_queries = wo[:, None, :].expand(-1, direction_count, -1).contiguous()
        scales = self.response_scale[state_index.long()]
        scale_queries = scales[:, None, :].expand(-1, direction_count, -1).contiguous()
        if self.evaluator == "core-frame-neural-v1":
            top = self._top(state_index, direction_count)
            core = module.nclsUnifiedTopInterface(top, wo_queries, wi)
            features = getattr(
                module,
                "nclsUnifiedResidualFeatures" + self._role(prepared_queries),
            )(
                prepared_queries,
                scale_queries,
                core,
                wo_queries,
                wi,
            )
        else:
            features = module.nclsUnifiedDirectFeatures(prepared_queries, wo_queries, wi)
        if self.runtime_class == "diagnostic":
            hidden = self._linear("nclsUnifiedDot17", self.evaluate_w0, self.evaluate_b0, features)
            hidden = module.nclsUnifiedLeakyRelu(hidden)
            hidden = self._linear("nclsUnifiedPaperDot64A", self.evaluate_w1, self.evaluate_b1, hidden)
            hidden = module.nclsUnifiedLeakyRelu(hidden)
            assert self.evaluate_w2 is not None and self.evaluate_b2 is not None
            hidden = self._linear("nclsUnifiedPaperDot64B", self.evaluate_w2, self.evaluate_b2, hidden)
            hidden = module.nclsUnifiedLeakyRelu(hidden)
            raw = self._linear("nclsUnifiedPaperDot64Out", self.evaluate_out_w, self.evaluate_out_b, hidden)
        else:
            hidden = self._linear(
                "nclsUnifiedEvaluateDot17",
                self.evaluate_w0,
                self.evaluate_b0,
                features,
                role_specific=True,
            )
            hidden = getattr(
                module, "nclsUnifiedEvaluateRelu" + self._role(hidden)
            )(hidden)
            hidden = self._linear(
                "nclsUnifiedEvaluateDot32A",
                self.evaluate_w1,
                self.evaluate_b1,
                hidden,
                role_specific=True,
            )
            hidden = getattr(
                module, "nclsUnifiedEvaluateRelu" + self._role(hidden)
            )(hidden)
            raw = self._linear(
                "nclsUnifiedEvaluateDot32Out",
                self.evaluate_out_w,
                self.evaluate_out_b,
                hidden,
                role_specific=True,
            )
        if self.evaluator == "core-frame-neural-v1":
            return getattr(
                module,
                "nclsUnifiedDecodeResidual" + self._role(raw, prepared_queries),
            )(
                raw,
                prepared_queries,
                scale_queries,
                core,
            )
        return module.nclsUnifiedDecodeDirect(raw, scale_queries)

    def top_interface_f(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> torch.Tensor:
        """用生产Slang数学单独求exact top core，供能量覆盖审计。"""

        direction_count = wi.shape[1]
        views = wo[:, None, :].expand(-1, direction_count, -1).contiguous()
        return self.session.module.nclsUnifiedTopInterface(
            self._top(state_index, direction_count), views, wi
        )

    def sampler_pdf(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        sampler: str,
    ) -> torch.Tensor:
        return self.sampler_pdf_with_head(state_index, wo, wi, sampler)[0]

    def sampler_pdf_with_head(
        self,
        state_index: torch.Tensor,
        wo: torch.Tensor,
        wi: torch.Tensor,
        sampler: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        module = self.session.module
        prepared, head = self._prepared_with_head(
            state_index, wo, sampler, detach_shared=True
        )
        directions = wi.shape[1]
        payload = prepared[:, None, :].expand(-1, directions, -1).contiguous()
        if sampler == "nvidia-diffuse-ggx9":
            views = wo[:, None, :].expand(-1, directions, -1).contiguous()
            pdf = (
                module.nclsUnifiedNvidiaPreparedPdfTraining(payload, views, wi)
                if head.requires_grad
                else module.nclsUnifiedNvidiaPreparedPdf(payload, views, wi)
            )
            return pdf, head
        if sampler == "ltc-k2":
            pdf = (
                module.nclsUnifiedLtcPreparedPdfTraining(payload, wi)
                if head.requires_grad
                else module.nclsUnifiedLtcPreparedPdf(payload, wi)
            )
            return pdf, head
        raise ValueError("unsupported unified sampler")

    def sampler_parameter_count(self, sampler: str) -> int:
        if sampler == "nvidia-diffuse-ggx9":
            return 9 * 64 + 9
        if sampler == "ltc-k2":
            return 13 * 64 + 13
        raise ValueError("unsupported unified sampler")
