from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.models.metal_fused_profile import MetalFusedProfile


METAL_TYPED_TENSOR_NAMES = (
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
)


@dataclass(frozen=True)
class MetalMaterialProgramState:
    compiler_latent: torch.Tensor
    spatial_modulation: torch.Tensor
    core_state: torch.Tensor
    residual_state: torch.Tensor
    block_condition: torch.Tensor
    proposal_logits: torch.Tensor
    proposal_modulation: torch.Tensor
    correction_bound: torch.Tensor
    tail_scale: torch.Tensor
    frame_strength: torch.Tensor
    trace: Mapping[str, torch.Tensor]


class MetalTypedCompiler(nn.Module):
    """Pure family-local typed token set compiler; never reads reference responses."""

    def __init__(self, profile: MetalFusedProfile) -> None:
        super().__init__()
        width = profile.typed_token_width
        self.profile = profile
        self.semantic_embedding = nn.Embedding(192, width)
        self.type_embedding = nn.Embedding(8, width)
        self.responsibility_embedding = nn.Embedding(6, width)
        self.discrete_embedding = nn.Embedding(64, width)
        self.value_projection = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.presence_embedding = nn.Embedding(2, width)
        self.graph_embedding = nn.Embedding(178, width)
        self.schema_embedding = nn.Embedding(64, width)
        self.recipe_embedding = nn.Embedding(36, width)
        self.metal_embedding = nn.Embedding(22, width)
        self.finish_embedding = nn.Embedding(36, width)
        self.optical_projection = nn.Sequential(
            nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                profile.typed_attention_heads,
                dim_feedforward=4 * width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(profile.typed_attention_blocks)
        )
        self.output_norm = nn.LayerNorm(width)
        self.spatial_head = nn.Linear(width, profile.structured_width)
        self.core_head = nn.Linear(width, profile.core_lobe_count * 9)
        self.residual_head = nn.Linear(width, profile.residual_lobe_count * 7)
        self.block_head = nn.Linear(
            width, profile.evaluator_blocks * profile.asset_adapter_rank
        )
        self.proposal_head = nn.Linear(
            width, (profile.core_lobe_count + profile.residual_lobe_count + 1) * 4
        )
        self.bounds_head = nn.Linear(width, 3)

    def _global_token(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.graph_embedding(tensors["metal_graph_index"])
            + self.schema_embedding(tensors["metal_schema_index"])
            + self.recipe_embedding(tensors["metal_recipe_index"])
            + self.metal_embedding(tensors["metal_identity_index"])
            + self.finish_embedding(tensors["metal_finish_index"])
            + self.optical_projection(tensors["metal_canonical_optical"])
        )

    def encode(self, tensors: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        missing = set(METAL_TYPED_TENSOR_NAMES) - set(tensors)
        if missing:
            raise ValueError(f"Metal typed compiler is missing tensors: {sorted(missing)}")
        semantic = tensors["metal_typed_semantic_id"]
        type_id = tensors["metal_typed_type_id"]
        responsibility = tensors["metal_typed_responsibility_id"]
        discrete = tensors["metal_typed_discrete"]
        continuous = tensors["metal_typed_continuous"]
        presence = tensors["metal_typed_presence"]
        batch = semantic.shape[0]
        expected_tokens = (batch, self.profile.maximum_typed_tokens)
        if (
            semantic.shape != expected_tokens
            or type_id.shape != expected_tokens
            or responsibility.shape != expected_tokens
            or discrete.shape != expected_tokens
            or presence.shape != expected_tokens
            or continuous.shape != (*expected_tokens, 4)
        ):
            raise ValueError("Metal typed token tensors disagree with the full profile")
        token = (
            self.semantic_embedding(semantic)
            + self.type_embedding(type_id)
            + self.responsibility_embedding(responsibility)
            + self.discrete_embedding(torch.remainder(discrete, 64))
            + self.value_projection(continuous)
            + self.presence_embedding(presence.to(torch.int64))
        )
        global_token = self._global_token(tensors)
        token = token + global_token[:, None, :]
        active = presence.to(torch.bool)
        empty = ~torch.any(active, dim=1)
        # A schema with no authored parameters still has a valid family/global
        # state.  Give those rows one synthetic global token so attention never
        # sees an all-masked row, while every absent payload remains irrelevant.
        first = torch.arange(
            self.profile.maximum_typed_tokens, device=presence.device
        )[None, :] == 0
        synthetic = empty[:, None] & first
        active = active | synthetic
        token = torch.where(synthetic[:, :, None], global_token[:, None, :], token)
        padding = ~active
        hidden = token
        block_energy = []
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=padding)
            block_energy.append(hidden.square().mean())
        mask = active.to(hidden.dtype)[:, :, None]
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        latent = self.output_norm(pooled + global_token)
        return latent, {
            "typed_tokens": token.square().mean(),
            "typed_attention": torch.stack(block_energy).mean(),
            "pure_compiler": latent.square().mean(),
        }

    def decode(self, latent: torch.Tensor, trace: Mapping[str, torch.Tensor]) -> MetalMaterialProgramState:
        batch = latent.shape[0]
        core_raw = self.core_head(latent).reshape(
            batch, self.profile.core_lobe_count, 9
        )
        residual_raw = self.residual_head(latent).reshape(
            batch, self.profile.residual_lobe_count, 7
        )
        # color/F0, roughness, anisotropy, energy and active probability are bounded.
        core_state = torch.cat(
            (
                torch.sigmoid(core_raw[..., :3]),
                0.015 + 0.985 * torch.sigmoid(core_raw[..., 3:5]),
                torch.tanh(core_raw[..., 5:6]),
                F.softplus(core_raw[..., 6:7]),
                torch.sigmoid(core_raw[..., 7:8]),
                torch.sigmoid(core_raw[..., 8:9]),
            ),
            dim=-1,
        )
        residual_state = torch.cat(
            (
                F.softplus(residual_raw[..., :3]),
                0.02 + 0.98 * torch.sigmoid(residual_raw[..., 3:5]),
                torch.sigmoid(residual_raw[..., 5:6]),
                torch.tanh(residual_raw[..., 6:7]),
            ),
            dim=-1,
        )
        bounds = self.bounds_head(latent)
        proposal = self.proposal_head(latent).reshape(
            batch,
            self.profile.core_lobe_count + self.profile.residual_lobe_count + 1,
            4,
        )
        return MetalMaterialProgramState(
            compiler_latent=latent,
            spatial_modulation=torch.tanh(self.spatial_head(latent)),
            core_state=core_state,
            residual_state=residual_state,
            block_condition=torch.tanh(self.block_head(latent)).reshape(
                batch,
                self.profile.evaluator_blocks,
                self.profile.asset_adapter_rank,
            ),
            proposal_logits=proposal[..., 0],
            proposal_modulation=torch.tanh(proposal[..., 1:4]),
            correction_bound=0.25 + 2.75 * torch.sigmoid(bounds[:, 0:1]),
            tail_scale=0.01 + 0.49 * torch.sigmoid(bounds[:, 1:2]),
            frame_strength=torch.sigmoid(bounds[:, 2:3]),
            trace={
                **trace,
                "compiler_core_state": core_state.square().mean(),
                "compiler_residual_state": residual_state.square().mean(),
            },
        )

    def forward(self, tensors: Mapping[str, torch.Tensor]) -> MetalMaterialProgramState:
        latent, trace = self.encode(tensors)
        return self.decode(latent, trace)


class MetalOptimizedStateTeacher(nn.Module):
    """Target-visible optimized-state control; never used by the product compiler path."""

    def __init__(self, width: int, capacity: int = 4096) -> None:
        super().__init__()
        if capacity != 4096:
            raise ValueError("Metal full teacher capacity is frozen at 4096 states")
        self.capacity = capacity
        self.state = nn.Embedding(capacity, width)
        nn.init.normal_(self.state.weight, std=0.02)

    def forward(self, source_index: torch.Tensor) -> torch.Tensor:
        if torch.any(source_index < 0) or torch.any(source_index >= self.capacity):
            raise ValueError("Metal optimized teacher source index exceeds its frozen capacity")
        return self.state(source_index)


__all__ = [
    "METAL_TYPED_TENSOR_NAMES",
    "MetalMaterialProgramState",
    "MetalOptimizedStateTeacher",
    "MetalTypedCompiler",
]
