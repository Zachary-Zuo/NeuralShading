from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.batches import AssetTileBatch
from ncls.learning.models.metal_directional_evaluator import (
    MetalDirectionalRepresentation,
    MetalEvaluation,
    MetalHybridEvaluator,
    _orthonormal_frame,
    _safe_normalize,
)
from ncls.learning.models.metal_fused_profile import (
    METAL_FUSED_FULL_PROFILE,
    MetalFusedProfile,
)
from ncls.learning.models.metal_sampler import (
    METAL_PROPOSAL_COMPONENT_COUNT,
    METAL_PROPOSAL_DISTRIBUTION_IDS,
    METAL_PROPOSAL_FRAME_INDICES,
    MetalProposalPdf,
    MetalProposalSample,
    metal_proposal_pdf,
    metal_sample_proposal,
)
from ncls.learning.models.metal_texture_codec import (
    MetalTextureCodec,
    semantic_role_class,
)
from ncls.learning.models.metal_typed_compiler import (
    MetalMaterialProgramState,
    MetalOptimizedStateTeacher,
    MetalTypedCompiler,
)


METAL_FUSED_REQUIRED_CONTEXT: Mapping[str, Any] = {
    "profile_id": "metal_fused_full_v1",
    "asset_count": 52,
    "graph_count": 178,
    "schema_count": 64,
    "recipe_count": 36,
    "metal_count": 22,
    "finish_count": 36,
    "source_state_capacity": 4096,
    "required_route_kinds": [
        "asset-tile",
        "reference-evaluator",
        "method-sampler",
    ],
}


@dataclass(frozen=True)
class MetalSpatialState:
    structured: torch.Tensor
    access_uv: torch.Tensor
    access_dx: torch.Tensor
    access_dy: torch.Tensor
    mip_fraction: torch.Tensor
    valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class MetalPreparedState:
    program: MetalMaterialProgramState
    spatial: torch.Tensor
    view_token: torch.Tensor
    frames: torch.Tensor
    core_state: torch.Tensor
    residual_state: torch.Tensor
    proposal_state: torch.Tensor
    valid: torch.Tensor
    trace: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class MetalScatteringSample:
    wi: torch.Tensor
    f: torch.Tensor
    forward_pdf: torch.Tensor
    reverse_pdf: torch.Tensor
    weight: torch.Tensor
    valid: torch.Tensor
    component: torch.Tensor
    component_pdfs: torch.Tensor


class MetalPreparedModel(nn.Module):
    def __init__(self, profile: MetalFusedProfile) -> None:
        super().__init__()
        self.profile = profile
        condition_width = 2 * profile.structured_width
        self.frame_head = nn.Sequential(
            nn.Linear(condition_width + 8, 128),
            nn.SiLU(),
            nn.Linear(128, profile.learned_frame_count * 6),
        )
        self.core_spatial_head = nn.Sequential(
            nn.Linear(condition_width, 128),
            nn.SiLU(),
            nn.Linear(128, profile.core_lobe_count * 9),
        )
        self.residual_spatial_head = nn.Sequential(
            nn.Linear(condition_width, 128),
            nn.SiLU(),
            nn.Linear(128, profile.residual_lobe_count * 7),
        )
        self.view_encoder = nn.Sequential(
            nn.Linear(3 + condition_width, 128),
            nn.SiLU(),
            nn.Linear(128, profile.structured_width),
            nn.SiLU(),
        )
        self.proposal_spatial_head = nn.Linear(
            condition_width, METAL_PROPOSAL_COMPONENT_COUNT * 4
        )

    @staticmethod
    def execute_spatial_access(
        uv: torch.Tensor,
        uv_dx: torch.Tensor,
        uv_dy: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.shape != (uv.shape[0], 16):
            raise ValueError("Metal spatial access state must have shape [batch,16]")
        scale = state[:, 0:2]
        translate = state[:, 2:4]
        cosine = state[:, 4:5]
        sine = state[:, 5:6]
        rotation = torch.stack(
            (
                torch.cat((cosine, -sine), dim=1),
                torch.cat((sine, cosine), dim=1),
            ),
            dim=1,
        )
        scaled = uv * scale
        transformed = torch.einsum("bij,bj->bi", rotation, scaled) + translate
        transformed_dx = torch.einsum("bij,bj->bi", rotation, uv_dx * scale)
        transformed_dy = torch.einsum("bij,bj->bi", rotation, uv_dy * scale)
        wrap = state[:, 6:7] > 0.5
        transformed = torch.where(wrap, torch.frac(transformed), transformed)
        valid = (
            torch.isfinite(transformed).all(dim=1)
            & torch.isfinite(transformed_dx).all(dim=1)
            & torch.isfinite(transformed_dy).all(dim=1)
        )
        return transformed, transformed_dx, transformed_dy, valid

    def forward(
        self,
        program: MetalMaterialProgramState,
        spatial: MetalSpatialState,
        wo: torch.Tensor,
        frame_state: torch.Tensor,
    ) -> MetalPreparedState:
        condition = torch.cat((spatial.structured, program.spatial_modulation), dim=1)
        frame_raw = self.frame_head(torch.cat((condition, frame_state), dim=1)).reshape(
            wo.shape[0], self.profile.learned_frame_count, 2, 3
        )
        frame_normal, normal_valid = _safe_normalize(
            frame_raw[:, :, 1, :],
            torch.tensor((0.0, 0.0, 1.0), dtype=wo.dtype, device=wo.device),
        )
        tangent_seed = frame_raw[:, :, 0, :]
        tangent_seed = tangent_seed - torch.sum(
            tangent_seed * frame_normal, dim=-1, keepdim=True
        ) * frame_normal
        frame_tangent, tangent_valid = _safe_normalize(
            tangent_seed,
            torch.tensor((1.0, 0.0, 0.0), dtype=wo.dtype, device=wo.device),
        )
        base_normal = torch.tensor(
            (0.0, 0.0, 1.0), dtype=wo.dtype, device=wo.device
        ).expand_as(wo)
        base_frame = _orthonormal_frame(base_normal)[:, None, :, :]
        # ``torch.lerp`` requires the interpolation weight to match the promoted
        # input dtype.  Keep frame blending in the renderer-frame precision when
        # autocast has produced a BF16 compiler state.
        strength = program.frame_strength[:, None, None, :].to(base_frame.dtype)
        base_learned = base_frame.expand(
            -1, self.profile.learned_frame_count, -1, -1
        )
        mixed_normal, mixed_normal_valid = _safe_normalize(
            torch.lerp(base_learned[:, :, 2, :], frame_normal, strength[:, :, 0, :]),
            base_learned[:, :, 2, :],
        )
        mixed_tangent_seed = torch.lerp(
            base_learned[:, :, 0, :], frame_tangent, strength[:, :, 0, :]
        )
        mixed_tangent_seed = mixed_tangent_seed - torch.sum(
            mixed_tangent_seed * mixed_normal, dim=-1, keepdim=True
        ) * mixed_normal
        mixed_tangent, mixed_tangent_valid = _safe_normalize(
            mixed_tangent_seed, base_learned[:, :, 0, :]
        )
        mixed_bitangent = torch.cross(mixed_normal, mixed_tangent, dim=-1)
        learned_frames = torch.stack(
            (mixed_tangent, mixed_bitangent, mixed_normal), dim=-2
        )
        frames = torch.cat((base_frame, learned_frames), dim=1)

        core_delta = self.core_spatial_head(condition).reshape(
            wo.shape[0], self.profile.core_lobe_count, 9
        )
        core = program.core_state
        core_state = torch.cat(
            (
                torch.clamp(core[..., :3] * torch.exp(0.25 * torch.tanh(core_delta[..., :3])), 0.0, 4.0),
                torch.clamp(core[..., 3:5] * torch.exp(0.35 * torch.tanh(core_delta[..., 3:5])), 0.01, 1.0),
                torch.clamp(core[..., 5:6] + 0.25 * torch.tanh(core_delta[..., 5:6]), -1.0, 1.0),
                core[..., 6:7] * F.softplus(core_delta[..., 6:7] + 1.0),
                torch.clamp(core[..., 7:8] * torch.sigmoid(core_delta[..., 7:8] + 2.0), 0.0, 1.0),
                torch.clamp(core[..., 8:9] + 0.25 * torch.tanh(core_delta[..., 8:9]), 0.0, 1.0),
            ),
            dim=-1,
        )
        residual_delta = self.residual_spatial_head(condition).reshape(
            wo.shape[0], self.profile.residual_lobe_count, 7
        )
        residual = program.residual_state
        residual_state = torch.cat(
            (
                residual[..., :3] * F.softplus(residual_delta[..., :3] + 1.0),
                torch.clamp(residual[..., 3:5] * torch.exp(0.35 * torch.tanh(residual_delta[..., 3:5])), 0.01, 1.0),
                torch.clamp(residual[..., 5:6] * torch.sigmoid(residual_delta[..., 5:6] + 2.0), 0.0, 1.0),
                torch.clamp(residual[..., 6:7] + 0.25 * torch.tanh(residual_delta[..., 6:7]), -1.0, 1.0),
            ),
            dim=-1,
        )
        view_token = self.view_encoder(torch.cat((wo, condition), dim=1))
        proposal_delta = self.proposal_spatial_head(condition).reshape(
            wo.shape[0], METAL_PROPOSAL_COMPONENT_COUNT, 4
        )
        proposal_logits = program.proposal_logits + proposal_delta[..., 0]
        core_alpha = core_state[..., 3:5]
        core_alpha = core_alpha.clone()
        core_alpha[:, 4, :] = torch.clamp(
            torch.sqrt(core_alpha[:, 4, :]), min=0.01, max=1.0
        )
        core_alpha[:, 5, :] = torch.clamp(
            torch.sqrt(core_alpha[:, 5, :]), min=0.01, max=1.0
        )
        residual_skew = residual_state[..., 6:7]
        residual_alpha = residual_state[..., 3:5] * torch.cat(
            (torch.exp(0.35 * residual_skew), torch.exp(-0.35 * residual_skew)),
            dim=-1,
        )
        base_alpha = torch.cat(
            (
                core_alpha,
                residual_alpha,
                torch.ones((wo.shape[0], 1, 2), dtype=wo.dtype, device=wo.device),
            ),
            dim=1,
        )
        proposal_modulation = program.proposal_modulation + proposal_delta[..., 1:4]
        alpha = torch.clamp(
            base_alpha
            * torch.exp(0.5 * torch.tanh(proposal_modulation[..., 0:2])),
            min=0.01,
            max=1.0,
        )
        base_rotation = torch.cat(
            (
                torch.pi * core_state[..., 5],
                0.5 * torch.pi * residual_state[..., 6],
                torch.zeros((wo.shape[0], 1), dtype=wo.dtype, device=wo.device),
            ),
            dim=1,
        )
        rotation = base_rotation + torch.pi * torch.tanh(
            proposal_modulation[..., 2]
        )
        luminance = wo.new_tensor((0.2126, 0.7152, 0.0722))
        core_active = core_state[..., 7]
        residual_active = residual_state[..., 5]
        activity = torch.cat(
            (
                core_active,
                residual_active,
                torch.ones((wo.shape[0], 1), dtype=wo.dtype, device=wo.device),
            ),
            dim=1,
        )
        # The proposal ABI carries a binary topology mask separately from the
        # differentiable lobe activity used to shape mixture mass.  Compiler
        # gates are continuous, so serializing them directly as an ``active``
        # flag would make every positive gate below 0.5 fail validation while
        # still assigning it non-zero probability.
        active = (activity > 0.0).to(wo.dtype)
        core_clue = (
            torch.sum(core_state[..., :3] * luminance, dim=-1)
            * core_state[..., 6]
            * core_active
        )
        residual_clue = (
            torch.sum(residual_state[..., :3] * luminance, dim=-1)
            * residual_active
        )
        fallback_clue = 0.05 + program.tail_scale.mean(dim=1, keepdim=True)
        clue = torch.cat((core_clue, residual_clue, fallback_clue), dim=1)
        raw_weight = (
            activity.float()
            * torch.clamp(clue.float(), min=1e-6)
            * torch.exp(torch.clamp(proposal_logits.float(), min=-8.0, max=8.0))
        )
        normalized = raw_weight / torch.clamp(
            torch.sum(raw_weight, dim=1, keepdim=True), min=1e-12
        )
        fallback_floor = 0.02
        proposal_weights = (1.0 - fallback_floor) * normalized
        proposal_weights = torch.cat(
            (
                proposal_weights[:, :-1],
                proposal_weights[:, -1:] + fallback_floor,
            ),
            dim=1,
        ).to(wo.dtype)
        frame_index = torch.tensor(
            METAL_PROPOSAL_FRAME_INDICES, dtype=wo.dtype, device=wo.device
        )[None, :].expand(wo.shape[0], -1)
        distribution = torch.tensor(
            METAL_PROPOSAL_DISTRIBUTION_IDS, dtype=wo.dtype, device=wo.device
        )[None, :].expand(wo.shape[0], -1)
        proposal_state = torch.stack(
            (
                proposal_weights,
                alpha[..., 0],
                alpha[..., 1],
                rotation,
                active,
                frame_index,
                distribution,
                clue,
            ),
            dim=-1,
        )
        valid = (
            spatial.valid
            & normal_valid.all(dim=1)
            & tangent_valid.all(dim=1)
            & mixed_normal_valid.all(dim=1)
            & mixed_tangent_valid.all(dim=1)
            & torch.isfinite(condition).all(dim=1)
        )
        return MetalPreparedState(
            program,
            torch.tanh(spatial.structured + program.spatial_modulation),
            view_token,
            frames,
            core_state,
            residual_state,
            proposal_state,
            valid,
            {
                **spatial.trace,
                **program.trace,
                "prepared_frames": (learned_frames - base_learned).square().mean(),
                "prepared_view": view_token.square().mean(),
                "proposal_state": proposal_weights.square().mean(),
            },
        )


class MetalModel(nn.Module):
    def __init__(
        self,
        profile: MetalFusedProfile = METAL_FUSED_FULL_PROFILE,
        *,
        source_state_capacity: int = 4096,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.texture_codec = MetalTextureCodec(profile, asset_count=52)
        self.typed_compiler = MetalTypedCompiler(profile)
        self.optimized_teacher = MetalOptimizedStateTeacher(
            profile.typed_token_width, source_state_capacity
        )
        self.prepared_model = MetalPreparedModel(profile)
        self.directional = MetalDirectionalRepresentation(profile)
        directional_width = (
            10
            + 8
            + profile.learned_frame_count * 6
            + profile.angular_levels * profile.angular_channels
            + profile.angular_difference_rank
            + 3 * profile.structured_width
        )
        self.hybrid = MetalHybridEvaluator(profile, directional_width)

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "MetalModel":
        if dict(context) != dict(METAL_FUSED_REQUIRED_CONTEXT):
            raise ValueError(
                "Metal fused trainable requires the exact quality-first full profile context"
            )
        return cls(source_state_capacity=int(context["source_state_capacity"]))

    @staticmethod
    def _center(value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        y0 = max(0, (height - 1) // 2)
        x0 = max(0, (width - 1) // 2)
        y1 = min(height, y0 + 2)
        x1 = min(width, x0 + 2)
        return value[..., y0:y1, x0:x1].mean(dim=(-2, -1))

    def spatial_state(self, tensors: Mapping[str, torch.Tensor]) -> MetalSpatialState:
        required = {
            "metal_texture_patches",
            "metal_texture_slot_mask",
            "metal_texture_role_class",
            "metal_asset_index",
            "metal_mip_fraction",
            "mip_level",
            "uv",
            "uv_dx",
            "uv_dy",
            "metal_access_state",
        }
        missing = required - set(tensors)
        if missing:
            raise ValueError(f"Metal spatial preparation is missing tensors: {sorted(missing)}")
        patches = tensors["metal_texture_patches"]
        if patches.ndim != 6 or patches.shape[2:4] != (2, 4):
            raise ValueError(
                "metal_texture_patches must have shape [batch,slot,2,4,height,width]"
            )
        fraction = tensors["metal_mip_fraction"]
        levels = []
        traces: dict[str, torch.Tensor] = {}
        for level_index in range(2):
            level = self.texture_codec.forward_level(
                patches[:, :, level_index],
                tensors["metal_texture_slot_mask"].to(torch.bool),
                tensors["metal_texture_role_class"],
                tensors["metal_asset_index"],
                torch.floor(tensors["mip_level"]) + float(level_index),
            )
            levels.append(self._center(level.structured))
            for name, value in level.trace.items():
                traces[f"codec_level{level_index}_{name}"] = value
        structured = torch.lerp(levels[0], levels[1], fraction[:, None])
        access_uv, access_dx, access_dy, access_valid = (
            self.prepared_model.execute_spatial_access(
                tensors["uv"],
                tensors["uv_dx"],
                tensors["uv_dy"],
                tensors["metal_access_state"],
            )
        )
        patch_valid = tensors["metal_texture_slot_mask"].to(torch.bool).any(dim=1)
        return MetalSpatialState(
            structured,
            access_uv,
            access_dx,
            access_dy,
            fraction,
            access_valid & patch_valid,
            {
                **traces,
                "adjacent_mip_interpolation": (levels[1] - levels[0]).square().mean(),
                "spatial_access": access_uv.square().mean()
                + access_dx.square().mean()
                + access_dy.square().mean(),
            },
        )

    def compile_program_states(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> tuple[MetalMaterialProgramState, MetalMaterialProgramState]:
        pure = self.typed_compiler(tensors)
        teacher_latent = self.optimized_teacher(tensors["source_index"])
        teacher = self.typed_compiler.decode(
            teacher_latent,
            {"optimized_teacher": teacher_latent.square().mean()},
        )
        return pure, teacher

    def prepare_from_components(
        self,
        program: MetalMaterialProgramState,
        spatial: MetalSpatialState,
        tensors: Mapping[str, torch.Tensor],
        *,
        wo: torch.Tensor | None = None,
    ) -> MetalPreparedState:
        view = tensors["wo"] if wo is None else wo
        return self.prepared_model(
            program, spatial, view, tensors["metal_frame_state"]
        )

    def evaluate_prepared(
        self, prepared: MetalPreparedState, wo: torch.Tensor, wi: torch.Tensor
    ) -> MetalEvaluation:
        features = self.directional(
            wo,
            wi,
            prepared.frames[:, 1:, :, :],
            prepared.spatial,
            prepared.program.compiler_latent,
            prepared.view_token,
        )
        return self.hybrid(
            features,
            wo,
            wi,
            prepared.program,
            prepared.core_state,
            prepared.residual_state,
            prepared.valid,
        )

    def pdf_prepared(
        self,
        prepared: MetalPreparedState,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> MetalProposalPdf:
        return metal_proposal_pdf(
            prepared.proposal_state,
            prepared.frames,
            prepared.valid,
            wo,
            wi,
        )

    def sample_prepared(
        self,
        prepared: MetalPreparedState,
        wo: torch.Tensor,
        sample_u: torch.Tensor,
    ) -> MetalScatteringSample:
        proposal = self.sample_proposal_prepared(prepared, wo, sample_u)
        # Runtime identity: exactly one directional evaluator invocation after
        # the proposal has produced a valid candidate direction.
        evaluated = self.evaluate_prepared(prepared, wo, proposal.wi)
        cosine = torch.clamp(proposal.wi[..., 2:3], min=0.0)
        safe_pdf = torch.clamp(proposal.forward_pdf[..., None], min=1e-12)
        weight = evaluated.f * cosine / safe_pdf
        valid = (
            proposal.valid
            & evaluated.valid
            & torch.isfinite(weight).all(dim=-1)
            & torch.isfinite(evaluated.f).all(dim=-1)
            & (proposal.forward_pdf > 0.0)
        )
        return MetalScatteringSample(
            proposal.wi,
            torch.where(valid[..., None], evaluated.f, 0.0),
            torch.where(valid, proposal.forward_pdf, 0.0),
            torch.where(valid, proposal.reverse_pdf, 0.0),
            torch.where(valid[..., None], weight, 0.0),
            valid,
            proposal.component,
            proposal.component_pdfs,
        )

    @staticmethod
    def sample_proposal_prepared(
        prepared: MetalPreparedState,
        wo: torch.Tensor,
        sample_u: torch.Tensor,
    ) -> MetalProposalSample:
        return metal_sample_proposal(
            prepared.proposal_state,
            prepared.frames,
            prepared.valid,
            wo,
            sample_u,
        )

    @staticmethod
    def _structured_target(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = target * mask[:, :, None, None, None]
        denominator = torch.clamp(mask.sum(dim=1), min=1.0)[:, None, None, None]
        base = masked.sum(dim=1) / denominator
        local = F.avg_pool2d(base, 3, stride=1, padding=1)
        dx = F.pad(base[..., 1:] - base[..., :-1], (0, 1, 0, 0))
        dy = F.pad(base[..., 1:, :] - base[..., :-1, :], (0, 0, 0, 1))
        return torch.cat((base, local, dx.abs(), dy.abs()), dim=1).repeat(1, 4, 1, 1)

    def codec_objective(
        self, batch: AssetTileBatch
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        losses = []
        normal_losses = []
        qat_losses = []
        structured_losses = []
        mip_losses = []
        traces: dict[str, list[torch.Tensor]] = {}
        for tile in batch.tiles:
            roles = tile.roles
            if len(roles) > self.profile.maximum_texture_slots:
                raise ValueError("one Metal source domain exceeds the codec slot bound")
            height, width = tile.values.shape[:2]
            target = torch.zeros(
                (1, len(roles), 4, height, width),
                dtype=tile.values.dtype,
                device=tile.values.device,
            )
            role_class = []
            for role_index, role in enumerate(roles):
                source = tile.role_values(role.role_id).permute(2, 0, 1)
                target[0, role_index, : source.shape[0]] = source
                role_class.append(semantic_role_class(role.semantic, role.channel_count))
            slot_mask = torch.ones(
                (1, len(roles)), dtype=torch.bool, device=tile.values.device
            )
            role_tensor = torch.tensor(
                role_class, dtype=torch.int64, device=tile.values.device
            )[None, :]
            level = self.texture_codec.forward_level(
                target,
                slot_mask,
                role_tensor,
                torch.tensor([tile.asset_index], dtype=torch.int64, device=tile.values.device),
                torch.tensor([float(tile.mip_level)], dtype=tile.values.dtype, device=tile.values.device),
            )
            prediction = torch.sigmoid(level.semantic)
            for role_index, role in enumerate(roles):
                channels = role.channel_count
                losses.append(
                    F.smooth_l1_loss(
                        prediction[:, role_index, :channels],
                        target[:, role_index, :channels],
                    )
                )
                if role_class[role_index] == 1 and channels >= 3:
                    predicted_normal = F.normalize(
                        2.0 * prediction[:, role_index, :3] - 1.0, dim=1, eps=1e-6
                    )
                    target_normal = F.normalize(
                        2.0 * target[:, role_index, :3] - 1.0, dim=1, eps=1e-6
                    )
                    normal_losses.append(
                        (1.0 - torch.sum(predicted_normal * target_normal, dim=1)).mean()
                    )
            target_structured = self._structured_target(
                target, slot_mask.to(target.dtype)
            )
            structured_losses.append(
                F.smooth_l1_loss(level.structured, target_structured)
            )
            high_up = F.interpolate(
                level.high_quantized,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            low_up = F.interpolate(
                level.low_quantized,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            mip_losses.append(
                F.smooth_l1_loss(
                    high_up.mean(dim=1), low_up.mean(dim=1)
                )
            )
            qat_losses.append(level.qat_error)
            for name, value in level.trace.items():
                traces.setdefault(name, []).append(value)
        semantic_loss = torch.stack(losses).mean()
        normal_loss = (
            torch.stack(normal_losses).mean()
            if normal_losses
            else semantic_loss * 0.0
        )
        structured_loss = torch.stack(structured_losses).mean()
        mip_loss = torch.stack(mip_losses).mean()
        qat_loss = torch.stack(qat_losses).mean()
        total = (
            semantic_loss
            + 0.5 * normal_loss
            + 0.25 * structured_loss
            + 0.05 * mip_loss
            + 0.01 * qat_loss
        )
        metrics = {
            "codec_semantic_loss": semantic_loss,
            "codec_normal_loss": normal_loss,
            "codec_structured_loss": structured_loss,
            "codec_mip_loss": mip_loss,
            "codec_qat_loss": qat_loss,
            **{f"codec_{name}_trace": torch.stack(values).mean() for name, values in traces.items()},
        }
        return total, metrics

    @staticmethod
    def parameter_groups_for_model(
        model: "MetalModel",
    ) -> Mapping[str, tuple[str, ...]]:
        classified: dict[str, list[str]] = {
            "codec_role_stems": [],
            "codec_encoder": [],
            "codec_decoder": [],
            "codec_semantic_heads": [],
            "asset_adapter": [],
            "quantization": [],
            "typed_compiler": [],
            "optimized_state_teacher": [],
            "prepared_model": [],
            "angular_bank": [],
            "analytic_core": [],
            "hybrid_evaluator": [],
            "proposal_sampler": [],
        }
        for name, _ in model.named_parameters():
            if name.startswith("texture_codec.role_stems") or name.startswith(
                "texture_codec.role_embedding"
            ):
                group = "codec_role_stems"
            elif name in {
                "texture_codec.high_log_scale",
                "texture_codec.low_log_scale",
            }:
                group = "quantization"
            elif name.startswith("texture_codec.adapter_head") or ".adapter_" in name or name.startswith(
                "texture_codec.asset_embedding"
            ):
                group = "asset_adapter"
            elif name.startswith("texture_codec.semantic_heads"):
                group = "codec_semantic_heads"
            elif name.startswith("texture_codec.decoder") or name.startswith(
                "texture_codec.structured_head"
            ):
                group = "codec_decoder"
            elif name.startswith("texture_codec"):
                group = "codec_encoder"
            elif name.startswith("typed_compiler.proposal_head") or name.startswith(
                "prepared_model.proposal_spatial_head"
            ):
                group = "proposal_sampler"
            elif name.startswith("typed_compiler"):
                group = "typed_compiler"
            elif name.startswith("optimized_teacher"):
                group = "optimized_state_teacher"
            elif name.startswith("prepared_model"):
                group = "prepared_model"
            elif name.startswith("directional.angular_bank"):
                group = "angular_bank"
            elif name == "hybrid.analytic_gain_log":
                group = "analytic_core"
            elif name.startswith("hybrid"):
                group = "hybrid_evaluator"
            else:
                raise ValueError(f"orphan Metal fused parameter {name!r}")
            classified[group].append(name)
        if any(not values for values in classified.values()):
            raise ValueError("Metal full model produced an empty required parameter group")
        return {name: tuple(values) for name, values in classified.items()}


def metal_fused_parameter_groups() -> Mapping[str, tuple[str, ...]]:
    with torch.device("meta"):
        model = MetalModel.from_context(
            METAL_FUSED_REQUIRED_CONTEXT
        )
    return MetalModel.parameter_groups_for_model(model)


__all__ = [
    "METAL_FUSED_REQUIRED_CONTEXT",
    "MetalModel",
    "MetalPreparedState",
    "MetalScatteringSample",
    "MetalSpatialState",
    "metal_fused_parameter_groups",
]
