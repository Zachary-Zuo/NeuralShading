from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.core.identity import sha256_bytes, sha256_json
from ncls.learning.models.metal_budgeted_compiler import MetalBudgetedProgramState
from ncls.learning.models.metal_budgeted_profile import MetalBudgetedProfile


@dataclass(frozen=True)
class MetalBudgetedAssetSample:
    detail: torch.Tensor
    context: torch.Tensor
    mip_choice: torch.Tensor
    valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


MetalBudgetedAssetCookMode = Literal[
    "encoder-only@1", "bounded-refinement@1", "direct-control@1"
]


@dataclass(frozen=True)
class MetalBudgetedCookedAsset:
    mode: MetalBudgetedAssetCookMode
    values_snorm8: torch.Tensor
    refinement_steps: int
    refinement_bound: float

    def __post_init__(self) -> None:
        if (
            self.mode
            not in {"encoder-only@1", "bounded-refinement@1", "direct-control@1"}
            or self.values_snorm8.dtype != torch.int8
            or self.values_snorm8.ndim < 2
            or self.values_snorm8.shape[-2:] != (2, 4)
            or self.refinement_steps < 0
            or not 0.0 <= self.refinement_bound <= 0.5
        ):
            raise ValueError("Metal budgeted cooked asset payload is invalid")
        if self.mode == "encoder-only@1" and (
            self.refinement_steps != 0 or self.refinement_bound != 0.0
        ):
            raise ValueError("encoder-only asset cook cannot contain refinement")

    @property
    def values(self) -> torch.Tensor:
        return self.values_snorm8.to(torch.float32) / 127.0

    @property
    def identity(self) -> str:
        payload = self.values_snorm8.detach().cpu().contiguous().numpy().tobytes()
        return sha256_json(
            {
                "schema": "ncls.metal-budgeted-cooked-asset@1",
                "mode": self.mode,
                "shape": list(self.values_snorm8.shape),
                "values_sha256": sha256_bytes(payload),
                "refinement_steps": self.refinement_steps,
                "refinement_bound": self.refinement_bound,
            }
        )


class MetalBudgetedAssetCooker:
    """三种 cook 策略共享同一个两平面 RGBA8 SNORM 部署形状。"""

    @staticmethod
    def cook(
        encoded_values: torch.Tensor,
        *,
        mode: MetalBudgetedAssetCookMode,
        objective: Callable[[torch.Tensor], torch.Tensor] | None = None,
        refinement_steps: int = 0,
        refinement_bound: float = 0.25,
        learning_rate: float = 0.05,
    ) -> MetalBudgetedCookedAsset:
        if (
            encoded_values.ndim < 2
            or encoded_values.shape[-2:] != (2, 4)
            or not encoded_values.is_floating_point()
            or not bool(torch.isfinite(encoded_values).all())
        ):
            raise ValueError(
                "Metal budgeted encoder output must be finite [...,2,4]"
            )
        if mode == "encoder-only@1":
            if objective is not None or refinement_steps != 0:
                raise ValueError("encoder-only asset cook cannot optimize hidden state")
            result = encoded_values.detach()
            bound = 0.0
        else:
            if (
                mode not in {"bounded-refinement@1", "direct-control@1"}
                or objective is None
                or not 1 <= refinement_steps <= 4096
                or not 0.0 < learning_rate <= 0.25
            ):
                raise ValueError("optimized Metal budgeted asset cook is invalid")
            if mode == "bounded-refinement@1" and not 0.0 < refinement_bound <= 0.5:
                raise ValueError("bounded asset refinement must lie in (0,0.5]")
            initial = encoded_values.detach()
            parameter = nn.Parameter(
                initial.clone()
                if mode == "bounded-refinement@1"
                else torch.zeros_like(initial)
            )
            optimizer = torch.optim.Adam(
                (parameter,), lr=learning_rate, fused=parameter.is_cuda
            )
            for _ in range(refinement_steps):
                loss = objective(parameter)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("Metal budgeted asset cook objective must be finite scalar")
                optimizer.zero_grad(set_to_none=True)
                (gradient,) = torch.autograd.grad(loss, (parameter,))
                parameter.grad = gradient
                optimizer.step()
                with torch.no_grad():
                    if mode == "bounded-refinement@1":
                        parameter.copy_(
                            initial
                            + torch.clamp(parameter - initial, -refinement_bound, refinement_bound)
                        )
                    parameter.clamp_(-1.0, 1.0)
            result = parameter.detach()
            bound = refinement_bound if mode == "bounded-refinement@1" else 0.0
        packed = torch.round(torch.clamp(result, -1.0, 1.0) * 127.0).to(
            dtype=torch.int8, device="cpu"
        )
        return MetalBudgetedCookedAsset(
            mode,
            packed.contiguous(),
            refinement_steps,
            bound,
        )


def _snorm8_ste(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bounded = torch.clamp(value, -1.0, 1.0)
    rounded = torch.round(bounded * 127.0) / 127.0
    quantized = bounded + (rounded - bounded).detach()
    return quantized, (rounded - bounded).square().mean()


class MetalBudgetedTwoReadAsset(nn.Module):
    """离线编码源纹理，运行时只暴露 Detail/Context 两个 RGBA read。"""

    def __init__(
        self, profile: MetalBudgetedProfile, *, asset_variant_count: int = 52
    ) -> None:
        super().__init__()
        if asset_variant_count <= 0:
            raise ValueError("Metal budgeted asset variant table cannot be empty")
        self.profile = profile
        self.asset_variant_count = asset_variant_count
        self.role_embedding = nn.Embedding(4, 8)
        self.slot_score = nn.Linear(20, 1)
        self.detail_encoder = nn.Sequential(
            nn.Linear(20, 16), nn.SiLU(), nn.Linear(16, 4), nn.Tanh()
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(20, 16), nn.SiLU(), nn.Linear(16, 4), nn.Tanh()
        )
        self.variant_scale_bias = nn.Embedding(asset_variant_count, 16)
        nn.init.zeros_(self.variant_scale_bias.weight)

    @staticmethod
    def _mip_choice(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        fraction = tensors["metal_mip_fraction"]
        uv = tensors["uv"]
        if fraction.ndim != 1 or uv.shape != (fraction.shape[0], 2):
            raise ValueError("Metal budgeted stochastic mip inputs have invalid shapes")
        # 固定 hash 只选择相邻 mip，不引入第三次纹理读取。
        selector = torch.remainder(
            torch.sin(uv[:, 0] * 12.9898 + uv[:, 1] * 78.233) * 43758.5453,
            1.0,
        )
        return (selector < torch.clamp(fraction, 0.0, 1.0)).to(torch.int64)

    def _encode_source_patches(
        self, tensors: Mapping[str, torch.Tensor], mip_choice: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {
            "metal_texture_patches",
            "metal_texture_slot_mask",
            "metal_texture_role_class",
        }
        missing = required - set(tensors)
        if missing:
            raise ValueError(
                f"Metal budgeted asset encoding is missing tensors: {sorted(missing)}"
            )
        patches = tensors["metal_texture_patches"]
        mask = tensors["metal_texture_slot_mask"].to(torch.bool)
        roles = tensors["metal_texture_role_class"]
        if (
            patches.ndim != 6
            or patches.shape[2] != 2
            or patches.shape[3] != 4
            or patches.shape[1] > self.profile.maximum_texture_slots
        ):
            raise ValueError(
                "metal_texture_patches must have shape [batch,slot<=9,2,4,height,width]"
            )
        batch, slots, _, _, height, width = patches.shape
        if mask.shape != (batch, slots) or roles.shape != (batch, slots):
            raise ValueError("Metal budgeted slot mask/roles disagree with texture patches")
        gather = mip_choice[:, None, None, None, None, None].expand(
            -1, slots, 1, 4, height, width
        )
        selected = torch.gather(patches, 2, gather).squeeze(2)
        y0, x0 = max(0, (height - 1) // 2), max(0, (width - 1) // 2)
        y1, x1 = min(height, y0 + 2), min(width, x0 + 2)
        center = selected[..., y0:y1, x0:x1].mean(dim=(-2, -1))
        context = selected.mean(dim=(-2, -1))
        high_pass = center - context
        features = torch.cat(
            (
                center,
                high_pass,
                context,
                self.role_embedding(torch.clamp(roles, 0, 3)),
            ),
            dim=-1,
        )
        logits = self.slot_score(features)[..., 0]
        logits = torch.where(mask, logits, torch.full_like(logits, -1e4))
        weights = torch.softmax(logits.float(), dim=1).to(features.dtype)
        any_slot = torch.any(mask, dim=1)
        weights = torch.where(any_slot[:, None], weights, 0.0)
        detail = torch.sum(weights[..., None] * self.detail_encoder(features), dim=1)
        context_value = torch.sum(
            weights[..., None] * self.context_encoder(features), dim=1
        )
        return detail, context_value, any_slot

    def _read_planes(
        self, tensors: Mapping[str, torch.Tensor], mip_choice: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_detail = "metal_budgeted_detail" in tensors
        has_context = "metal_budgeted_context" in tensors
        if has_detail != has_context:
            raise ValueError("Metal budgeted Detail/Context planes must be supplied together")
        if not has_detail:
            return self._encode_source_patches(tensors, mip_choice)
        detail = tensors["metal_budgeted_detail"]
        context = tensors["metal_budgeted_context"]
        if detail.ndim != 2 or detail.shape[1] != 4 or context.shape != detail.shape:
            raise ValueError("Metal budgeted runtime planes must both have shape [batch,4]")
        return detail, context, torch.ones(
            detail.shape[0], dtype=torch.bool, device=detail.device
        )

    def forward(
        self,
        tensors: Mapping[str, torch.Tensor],
        program: MetalBudgetedProgramState,
        *,
        qat: bool = True,
    ) -> MetalBudgetedAssetSample:
        mip_choice = self._mip_choice(tensors)
        detail, context, source_valid = self._read_planes(tensors, mip_choice)
        if detail.shape[0] != program.resource_variant.shape[0]:
            raise ValueError("Metal budgeted asset/program batch size mismatch")
        variants = program.resource_variant
        if torch.any(variants < 0) or torch.any(variants >= self.asset_variant_count):
            raise ValueError("Metal budgeted resource variant index is out of range")
        detail_q, detail_error = _snorm8_ste(detail)
        context_q, context_error = _snorm8_ste(context)
        if not qat:
            detail_q, context_q = detail, context
            detail_error = detail.new_zeros(())
            context_error = context.new_zeros(())
        table = self.variant_scale_bias(variants)
        program_scale = program.spatial_scale_bias[:, :4]
        program_bias = program.spatial_scale_bias[:, 4:]
        detail_scale = torch.exp(
            0.25 * torch.tanh(table[:, 0:4]) + 0.25 * program_scale
        )
        detail_bias = 0.25 * torch.tanh(table[:, 4:8]) + 0.1 * program_bias
        context_scale = torch.exp(
            0.25 * torch.tanh(table[:, 8:12]) + 0.25 * program_scale
        )
        context_bias = 0.25 * torch.tanh(table[:, 12:16]) + 0.1 * program_bias
        decoded_detail = detail_q * detail_scale + detail_bias
        decoded_context = context_q * context_scale + context_bias
        valid = (
            source_valid
            & torch.isfinite(decoded_detail).all(dim=1)
            & torch.isfinite(decoded_context).all(dim=1)
        )
        return MetalBudgetedAssetSample(
            detail=decoded_detail,
            context=decoded_context,
            mip_choice=mip_choice,
            valid=valid,
            trace={
                "asset_detail": decoded_detail.square().mean(),
                "asset_context": decoded_context.square().mean(),
                "asset_detail_qat_error": detail_error,
                "asset_context_qat_error": context_error,
                "asset_slot_selection": source_valid.to(detail.dtype).mean(),
            },
        )


__all__ = [
    "MetalBudgetedAssetCookMode",
    "MetalBudgetedAssetCooker",
    "MetalBudgetedAssetSample",
    "MetalBudgetedCookedAsset",
    "MetalBudgetedTwoReadAsset",
]
