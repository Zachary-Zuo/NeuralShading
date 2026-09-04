from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.models.metal_budgeted_profile import MetalBudgetedProfile


METAL_BUDGETED_TYPED_TENSOR_NAMES = (
    "metal_graph_index",
    "metal_schema_index",
    "metal_recipe_index",
    "metal_identity_index",
    "metal_finish_index",
    "metal_asset_index",
    "metal_typed_semantic_id",
    "metal_typed_type_id",
    "metal_typed_responsibility_id",
    "metal_typed_discrete",
    "metal_typed_continuous",
    "metal_typed_presence",
    "metal_canonical_optical",
    "metal_access_state",
    "metal_frame_state",
    "metal_distribution_id",
)


@dataclass(frozen=True)
class MetalBudgetedProgramState:
    compiler_condition: torch.Tensor
    primary_lobe: torch.Tensor
    secondary_lobe: torch.Tensor
    spatial_scale_bias: torch.Tensor
    proposal_prior: torch.Tensor
    resource_variant: torch.Tensor
    resource_and_flags: torch.Tensor
    access_state: torch.Tensor
    frame_state: torch.Tensor
    trace: Mapping[str, torch.Tensor]

    @property
    def analytic_lobes(self) -> torch.Tensor:
        return torch.stack((self.primary_lobe, self.secondary_lobe), dim=1)


class MetalBudgetedTypedCompiler(nn.Module):
    """小型 typed compiler；参数职责分组池化，确定性状态原样旁路。"""

    def __init__(self, profile: MetalBudgetedProfile) -> None:
        super().__init__()
        self.profile = profile
        width = profile.typed_token_width
        self.semantic_embedding = nn.Embedding(192, width)
        self.type_embedding = nn.Embedding(8, width)
        self.responsibility_embedding = nn.Embedding(
            profile.responsibility_count, width
        )
        self.discrete_embedding = nn.Embedding(64, width)
        self.value_projection = nn.Linear(4, width)
        self.graph_embedding = nn.Embedding(178, width)
        self.schema_embedding = nn.Embedding(64, width)
        self.recipe_embedding = nn.Embedding(36, width)
        self.metal_embedding = nn.Embedding(22, width)
        self.finish_embedding = nn.Embedding(36, width)
        self.optical_projection = nn.Linear(16, width)
        self.fusion = nn.Sequential(
            nn.Linear((profile.responsibility_count + 1) * width, 32),
            nn.SiLU(),
            nn.Linear(32, width),
            nn.LayerNorm(width),
        )
        self.condition_head = nn.Linear(width, 8)
        self.primary_head = nn.Linear(width, 8)
        self.secondary_head = nn.Linear(width, 8)
        self.spatial_head = nn.Linear(width, 8)
        self.proposal_head = nn.Linear(width, profile.proposal_component_count)

    def _validate(self, tensors: Mapping[str, torch.Tensor]) -> int:
        missing = set(METAL_BUDGETED_TYPED_TENSOR_NAMES) - set(tensors)
        if missing:
            raise ValueError(
                f"Metal budgeted compiler is missing tensors: {sorted(missing)}"
            )
        semantic = tensors["metal_typed_semantic_id"]
        batch = semantic.shape[0]
        token_shape = (batch, self.profile.maximum_typed_tokens)
        for name in (
            "metal_typed_semantic_id",
            "metal_typed_type_id",
            "metal_typed_responsibility_id",
            "metal_typed_discrete",
            "metal_typed_presence",
        ):
            if tensors[name].shape != token_shape:
                raise ValueError(f"{name} disagrees with the Metal budgeted token bound")
        if tensors["metal_typed_continuous"].shape != (*token_shape, 4):
            raise ValueError("metal_typed_continuous must have shape [batch,32,4]")
        expected_vectors = {
            "metal_canonical_optical": 16,
            "metal_access_state": 16,
            "metal_frame_state": 8,
        }
        for name, width in expected_vectors.items():
            if tensors[name].shape != (batch, width):
                raise ValueError(f"{name} must have shape [batch,{width}]")
        for name in (
            "metal_graph_index",
            "metal_schema_index",
            "metal_recipe_index",
            "metal_identity_index",
            "metal_finish_index",
            "metal_asset_index",
            "metal_distribution_id",
        ):
            if tensors[name].shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        distribution = tensors["metal_distribution_id"]
        known_distribution = (distribution == 0) | (distribution == 1)
        if known_distribution.device.type == "cuda":
            torch._assert_async(known_distribution.all())
        elif not bool(known_distribution.all()):
            raise ValueError("metal_distribution_id must be GGX=0 or Beckmann=1")
        return batch

    def _global(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.graph_embedding(tensors["metal_graph_index"])
            + self.schema_embedding(tensors["metal_schema_index"])
            + self.recipe_embedding(tensors["metal_recipe_index"])
            + self.metal_embedding(tensors["metal_identity_index"])
            + self.finish_embedding(tensors["metal_finish_index"])
            + self.optical_projection(tensors["metal_canonical_optical"])
        )

    @staticmethod
    def _decode_lobe(raw: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                torch.sigmoid(raw[..., :3]),
                0.015 + 0.985 * torch.sigmoid(raw[..., 3:5]),
                F.softplus(raw[..., 5:6]),
                torch.pi * torch.tanh(raw[..., 6:7]),
                torch.sigmoid(raw[..., 7:8]),
            ),
            dim=-1,
        )

    def forward(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> MetalBudgetedProgramState:
        batch = self._validate(tensors)
        presence = tensors["metal_typed_presence"].to(torch.bool)
        responsibility = tensors["metal_typed_responsibility_id"]
        token = (
            self.semantic_embedding(tensors["metal_typed_semantic_id"])
            + self.type_embedding(tensors["metal_typed_type_id"])
            + self.responsibility_embedding(responsibility)
            + self.discrete_embedding(
                torch.remainder(tensors["metal_typed_discrete"], 64)
            )
            + self.value_projection(tensors["metal_typed_continuous"])
        )
        token = F.silu(token)
        pooled = []
        responsibility_energy = []
        for index in range(self.profile.responsibility_count):
            mask = presence & (responsibility == index)
            weight = mask.to(token.dtype)[..., None]
            value = torch.sum(token * weight, dim=1) / torch.clamp(
                torch.sum(weight, dim=1), min=1.0
            )
            pooled.append(value)
            responsibility_energy.append(value.square().mean())
        global_token = self._global(tensors)
        latent = self.fusion(torch.cat((*pooled, global_token), dim=1))
        primary = self._decode_lobe(self.primary_head(latent))
        secondary = self._decode_lobe(self.secondary_head(latent))
        proposal_prior = torch.softmax(self.proposal_head(latent).float(), dim=1).to(
            latent.dtype
        )
        if proposal_prior.shape != (batch, self.profile.proposal_component_count):
            raise RuntimeError("Metal budgeted proposal head shape drifted")
        proposal_condition = torch.nn.functional.pad(
            proposal_prior, (0, 8 - self.profile.proposal_component_count)
        )
        return MetalBudgetedProgramState(
            compiler_condition=torch.tanh(self.condition_head(latent))
            + 0.05 * proposal_condition,
            primary_lobe=primary,
            secondary_lobe=secondary,
            spatial_scale_bias=torch.tanh(self.spatial_head(latent)),
            proposal_prior=proposal_prior,
            resource_variant=tensors["metal_asset_index"],
            resource_and_flags=torch.stack(
                (
                    tensors["metal_graph_index"],
                    tensors["metal_schema_index"],
                    tensors["metal_recipe_index"],
                    tensors["metal_identity_index"],
                    tensors["metal_finish_index"],
                    tensors["metal_asset_index"],
                    tensors["metal_distribution_id"],
                    torch.ones_like(tensors["metal_asset_index"]),
                ),
                dim=1,
            ),
            access_state=tensors["metal_access_state"],
            frame_state=tensors["metal_frame_state"],
            trace={
                "compiler_responsibility_groups": torch.stack(
                    responsibility_energy
                ).mean(),
                "compiler_global": global_token.square().mean(),
                "compiler_latent": latent.square().mean(),
                "compiler_primary_lobe": primary.square().mean(),
                "compiler_secondary_lobe": secondary.square().mean(),
            },
        )


class MetalBudgetedOptimizedProgramStateControl(nn.Module):
    """只用于报告 compiler gap；确定性 access/frame/resource 不能被优化。"""

    def __init__(self, initial: MetalBudgetedProgramState) -> None:
        super().__init__()
        self.compiler_condition = nn.Parameter(initial.compiler_condition.detach().clone())
        self.primary_lobe = nn.Parameter(initial.primary_lobe.detach().clone())
        self.secondary_lobe = nn.Parameter(initial.secondary_lobe.detach().clone())
        self.spatial_scale_bias = nn.Parameter(initial.spatial_scale_bias.detach().clone())
        self.proposal_logits = nn.Parameter(
            torch.log(torch.clamp(initial.proposal_prior.detach(), min=1.0e-8))
        )
        self.register_buffer(
            "resource_variant", initial.resource_variant.detach().clone()
        )
        self.register_buffer(
            "resource_and_flags", initial.resource_and_flags.detach().clone()
        )
        self.register_buffer("access_state", initial.access_state.detach().clone())
        self.register_buffer("frame_state", initial.frame_state.detach().clone())

    @staticmethod
    def _bounded_lobe(value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                torch.clamp(value[..., :3], 0.0, 1.0),
                torch.clamp(value[..., 3:5], 0.015, 1.0),
                torch.clamp(value[..., 5:6], min=0.0),
                torch.remainder(value[..., 6:7] + torch.pi, 2.0 * torch.pi)
                - torch.pi,
                torch.clamp(value[..., 7:8], 0.0, 1.0),
            ),
            dim=-1,
        )

    def forward(self) -> MetalBudgetedProgramState:
        primary = self._bounded_lobe(self.primary_lobe)
        secondary = self._bounded_lobe(self.secondary_lobe)
        proposal = torch.softmax(self.proposal_logits.float(), dim=1).to(
            self.proposal_logits.dtype
        )
        return MetalBudgetedProgramState(
            compiler_condition=torch.tanh(self.compiler_condition),
            primary_lobe=primary,
            secondary_lobe=secondary,
            spatial_scale_bias=torch.tanh(self.spatial_scale_bias),
            proposal_prior=proposal,
            resource_variant=self.resource_variant,
            resource_and_flags=self.resource_and_flags,
            access_state=self.access_state,
            frame_state=self.frame_state,
            trace={
                "optimized_program_state_control": (
                    self.compiler_condition.square().mean()
                    + primary.square().mean()
                    + secondary.square().mean()
                ),
            },
        )


__all__ = [
    "METAL_BUDGETED_TYPED_TENSOR_NAMES",
    "MetalBudgetedOptimizedProgramStateControl",
    "MetalBudgetedProgramState",
    "MetalBudgetedTypedCompiler",
]
