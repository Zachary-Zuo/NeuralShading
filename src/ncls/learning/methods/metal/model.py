from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import nn

from ncls.learning.methods.metal.asset import (
    MetalBudgetedAssetSample,
    MetalBudgetedTwoReadAsset,
)
from ncls.learning.methods.metal.compiler import (
    MetalBudgetedProgramState,
    MetalBudgetedTypedCompiler,
)
from ncls.learning.methods.metal.evaluator import (
    MetalBudgetedDirectionalRepresentation,
    MetalBudgetedEvaluation,
    MetalBudgetedEvaluator,
    MetalBudgetedPrepare,
    MetalBudgetedPreparedState,
    _local_frames,
)
from ncls.learning.methods.metal.profile import (
    METAL_BUDGETED_HYBRID_PROFILE,
    METAL_BUDGETED_HYBRID_PROFILE_ID,
    MetalBudgetedProfile,
    metal_budgeted_profile,
    METAL_SPATIAL_PROFILE_ID,
    METAL_SPATIAL_SUMMARY_PROFILE_ID,
)
from ncls.learning.conditioning_resources import ConditioningResources
from ncls.learning.methods.metal.asset_read import fp16_ste
from ncls.learning.methods.metal.spatial_asset import MetalSpatialAsset
from ncls.learning.methods.metal.sampler import (
    MetalBudgetedProposalPdf,
    MetalBudgetedProposalSample,
    metal_budgeted_proposal_pdf,
    metal_budgeted_sample_proposal,
)


METAL_BUDGETED_REQUIRED_CONTEXT = {
    "profile_id": METAL_SPATIAL_PROFILE_ID,
    "maximum_texture_slots": 9,
    "maximum_typed_tokens": 32,
    "maximum_uv_groups": 9,
    "runtime_asset_reads": 54,
}


@dataclass(frozen=True)
class MetalBudgetedScatteringSample:
    wi: torch.Tensor
    f: torch.Tensor
    forward_pdf: torch.Tensor
    reverse_pdf: torch.Tensor
    weight: torch.Tensor
    valid: torch.Tensor
    component: torch.Tensor
    component_pdfs: torch.Tensor


def pack_metal_budgeted_program_state(
    program: MetalBudgetedProgramState,
) -> tuple[torch.Tensor, torch.Tensor]:
    floats = torch.cat(
        (
            program.compiler_condition,
            program.primary_lobe,
            program.secondary_lobe,
            program.spatial_scale_bias,
            program.proposal_prior,
            torch.zeros(
                (program.compiler_condition.shape[0], 5),
                dtype=program.compiler_condition.dtype,
                device=program.compiler_condition.device,
            ),
            program.access_state,
            program.frame_state,
        ),
        dim=1,
    )
    if floats.shape[1] != 64 or program.resource_and_flags.shape[1] != 8:
        raise RuntimeError("Metal budgeted ProgramState packing width drifted")
    return floats.to(torch.float16), program.resource_and_flags.to(torch.uint32)


def pack_metal_budgeted_prepared_state(
    prepared: MetalBudgetedPreparedState,
) -> tuple[torch.Tensor, torch.Tensor]:
    floats = torch.cat(
        (
            prepared.semantic_state,
            prepared.view_state,
            prepared.compact_frame_state,
            prepared.analytic_lobes.flatten(start_dim=1),
            prepared.proposal_state.flatten(start_dim=1),
            prepared.access_state,
        ),
        dim=1,
    )
    if prepared.compact_proposal_frame_state is not None:
        floats = torch.cat((floats, prepared.compact_proposal_frame_state), dim=1)
    expected = 80 if prepared.compact_proposal_frame_state is not None else 72
    if floats.shape[1] != expected or prepared.identity_and_flags.shape[1] != 4:
        raise RuntimeError("Metal budgeted PreparedState packing width drifted")
    return floats.to(torch.float16), prepared.identity_and_flags.to(torch.uint32)


class MetalBudgetedModel(nn.Module):
    def __init__(
        self,
        profile: MetalBudgetedProfile = METAL_BUDGETED_HYBRID_PROFILE,
        *,
        asset_variant_count: int = 52,
    ) -> None:
        super().__init__()
        if profile.profile_id == METAL_SPATIAL_SUMMARY_PROFILE_ID:
            raise ValueError("matched summary encoder 尚未实现；不能用 raw CNN 冒充 summary control")
        self.profile = profile
        self.typed_compiler = MetalBudgetedTypedCompiler(profile)
        self.asset = MetalSpatialAsset() if profile.is_spatial else MetalBudgetedTwoReadAsset(profile, asset_variant_count=asset_variant_count)
        self.prepared_model = MetalBudgetedPrepare(profile)
        self.directional = MetalBudgetedDirectionalRepresentation(profile)
        self.evaluator = MetalBudgetedEvaluator(profile)
        self.register_buffer(
            "appearance_scale_rgb", torch.ones(3, dtype=torch.float32)
        )
        self.register_buffer(
            "appearance_peak_rgb", torch.ones(3, dtype=torch.float32)
        )
        self.register_buffer(
            "appearance_energy_epsilon", torch.full((1,), 1.0e-6)
        )
        self.register_buffer(
            "appearance_calibration_identity", torch.zeros(32, dtype=torch.uint8)
        )
        self.register_buffer(
            "appearance_calibrated", torch.zeros(1, dtype=torch.uint8)
        )

    def set_appearance_calibration(
        self,
        scale_rgb: torch.Tensor,
        peak_rgb: torch.Tensor,
        energy_epsilon: float,
        identity: str,
    ) -> None:
        try:
            identity_bytes = bytes.fromhex(identity)
        except ValueError as error:
            raise ValueError("appearance calibration identity must be SHA-256") from error
        if (
            scale_rgb.shape != (3,)
            or peak_rgb.shape != (3,)
            or len(identity_bytes) != 32
            or not torch.isfinite(scale_rgb).all()
            or not torch.isfinite(peak_rgb).all()
            or torch.any(scale_rgb <= 0.0)
            or torch.any(peak_rgb < 0.0)
            or not 0.0 < float(energy_epsilon) < float("inf")
        ):
            raise ValueError("appearance calibration payload is invalid")
        with torch.no_grad():
            self.appearance_scale_rgb.copy_(
                scale_rgb.to(
                    device=self.appearance_scale_rgb.device,
                    dtype=self.appearance_scale_rgb.dtype,
                )
            )
            self.appearance_peak_rgb.copy_(
                peak_rgb.to(
                    device=self.appearance_peak_rgb.device,
                    dtype=self.appearance_peak_rgb.dtype,
                )
            )
            self.appearance_energy_epsilon.fill_(float(energy_epsilon))
            self.appearance_calibration_identity.copy_(
                torch.tensor(
                    tuple(identity_bytes),
                    dtype=torch.uint8,
                    device=self.appearance_calibration_identity.device,
                )
            )
            self.appearance_calibrated.fill_(1)

    @property
    def appearance_calibration_identity_hex(self) -> str:
        if int(self.appearance_calibrated.item()) != 1:
            raise RuntimeError("appearance calibration has not been initialized")
        return bytes(
            self.appearance_calibration_identity.detach().cpu().tolist()
        ).hex()

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "MetalBudgetedModel":
        expected = dict(METAL_BUDGETED_REQUIRED_CONTEXT)
        profile_id = str(context.get("profile_id", expected["profile_id"]))
        if not metal_budgeted_profile(profile_id).is_spatial:
            raise ValueError("当前 Metal 入口需要 spatial profile；历史配置不能作为新实验运行")
        expected["profile_id"] = profile_id
        if dict(context) != expected:
            raise ValueError(
                "Metal spatial trainable requires the exact native UV context"
            )
        return cls(metal_budgeted_profile(profile_id))

    def compile_program_state(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> MetalBudgetedProgramState:
        return self.typed_compiler(tensors)

    def sample_asset(
        self,
        tensors: Mapping[str, torch.Tensor],
        program: MetalBudgetedProgramState,
        *,
        qat: bool = True,
        resources: ConditioningResources | None = None,
        binding: torch.Tensor | None = None,
        encoded=None,
    ) -> MetalBudgetedAssetSample:
        if self.profile.is_spatial:
            if resources is None or binding is None:
                raise ValueError("spatial asset requires its conditioning resource binding")
            return self.asset(tensors, program, qat=qat, resources=resources, binding=binding, encoded=encoded)
        return self.asset(tensors, program, qat=qat)

    def prepare_from_components(
        self,
        program: MetalBudgetedProgramState,
        asset: MetalBudgetedAssetSample,
        wo: torch.Tensor,
        *, qat: bool = True,
    ) -> MetalBudgetedPreparedState:
        if self.profile.is_spatial and asset.global_condition is not None:
            program = replace(program, compiler_condition=program.compiler_condition + 0.25 * asset.global_condition)
        if self.profile.is_spatial and qat:
            program = replace(program, **{name: fp16_ste(getattr(program, name)) for name in (
                "compiler_condition", "primary_lobe", "secondary_lobe", "spatial_scale_bias", "proposal_prior", "access_state", "frame_state")})
        prepared = self.prepared_model(program, asset, wo)
        if self.profile.is_spatial and qat:
            updates = {name: fp16_ste(getattr(prepared, name)) for name in (
                "semantic_state", "view_state", "compact_frame_state", "analytic_lobes", "proposal_state", "access_state", "compact_proposal_frame_state")}
            proposal = updates["proposal_state"]
            proposal = torch.cat((proposal[..., :1], proposal[..., 1:3].clamp_min(0.015), proposal[..., 3:]), dim=-1)
            updates["proposal_state"] = proposal
            updates["frames"] = _local_frames(updates["compact_frame_state"][:, :4].reshape(-1, 2, 2), updates["analytic_lobes"][..., 6])
            compact = updates["compact_proposal_frame_state"]
            updates["proposal_frames"] = _local_frames(compact[:, :4].reshape(-1, 2, 2), torch.atan2(compact[:, 6:8], compact[:, 4:6]))
            valid = prepared.valid
            for value in updates.values():
                valid = valid & torch.isfinite(value).flatten(start_dim=1).all(dim=1)
            updates["valid"] = valid
            prepared = replace(prepared, **updates)
        return prepared

    def prepare(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        wo: torch.Tensor | None = None,
        qat: bool = True,
        resources: ConditioningResources | None = None,
        binding: torch.Tensor | None = None,
        encoded=None,
    ) -> MetalBudgetedPreparedState:
        program = self.compile_program_state(tensors)
        asset = self.sample_asset(tensors, program, qat=qat, resources=resources, binding=binding, encoded=encoded)
        return self.prepare_from_components(
            program, asset, tensors["wo"] if wo is None else wo, qat=qat
        )

    def prepare_paired(
        self,
        tensors: Mapping[str, torch.Tensor],
        program: MetalBudgetedProgramState,
        *,
        qat: bool = True,
        resources: ConditioningResources | None = None,
        binding: torch.Tensor | None = None,
        encoded=None,
    ) -> MetalBudgetedPreparedState:
        required = {
            "paired_uv",
            "paired_uv_dx",
            "paired_uv_dy",
            "metal_paired_texture_patches",
        }
        if self.profile.is_spatial:
            required.remove("metal_paired_texture_patches")
        missing = required - set(tensors)
        if missing:
            raise ValueError(
                f"Metal budgeted paired preparation is missing tensors: {sorted(missing)}"
            )
        paired = {
            **tensors,
            "uv": tensors["paired_uv"],
            "uv_dx": tensors["paired_uv_dx"],
            "uv_dy": tensors["paired_uv_dy"],
        }
        if not self.profile.is_spatial:
            paired["metal_texture_patches"] = tensors["metal_paired_texture_patches"]
        asset = self.sample_asset(paired, program, qat=qat, resources=resources, binding=binding, encoded=encoded)
        return self.prepare_from_components(program, asset, tensors["wo"], qat=qat)

    def evaluate_prepared(
        self,
        prepared: MetalBudgetedPreparedState,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> MetalBudgetedEvaluation:
        features = self.directional(prepared, wo, wi)
        return self.evaluator(prepared, features, wo, wi)

    @staticmethod
    def pdf_prepared(
        prepared: MetalBudgetedPreparedState,
        wo: torch.Tensor,
        wi: torch.Tensor,
    ) -> MetalBudgetedProposalPdf:
        return metal_budgeted_proposal_pdf(
            prepared.proposal_state,
            prepared.proposal_frames if prepared.proposal_frames is not None else prepared.frames,
            prepared.valid,
            wo,
            wi,
        )

    @staticmethod
    def sample_proposal_prepared(
        prepared: MetalBudgetedPreparedState,
        wo: torch.Tensor,
        sample_u: torch.Tensor,
    ) -> MetalBudgetedProposalSample:
        return metal_budgeted_sample_proposal(
            prepared.proposal_state,
            prepared.proposal_frames if prepared.proposal_frames is not None else prepared.frames,
            prepared.valid,
            wo,
            sample_u,
        )

    def sample_prepared(
        self,
        prepared: MetalBudgetedPreparedState,
        wo: torch.Tensor,
        sample_u: torch.Tensor,
    ) -> MetalBudgetedScatteringSample:
        proposal = self.sample_proposal_prepared(prepared, wo, sample_u)
        evaluated = self.evaluate_prepared(prepared, wo, proposal.wi)
        cosine = torch.clamp(proposal.wi[..., 2:3], min=0.0)
        safe_pdf = torch.clamp(proposal.forward_pdf[..., None], min=1e-12)
        weight = evaluated.f * cosine / safe_pdf
        valid = (
            proposal.valid
            & evaluated.valid
            & torch.isfinite(weight).all(dim=-1)
            & (proposal.forward_pdf > 0.0)
        )
        return MetalBudgetedScatteringSample(
            wi=proposal.wi,
            f=torch.where(valid[..., None], evaluated.f, 0.0),
            forward_pdf=torch.where(valid, proposal.forward_pdf, 0.0),
            reverse_pdf=torch.where(valid, proposal.reverse_pdf, 0.0),
            weight=torch.where(valid[..., None], weight, 0.0),
            valid=valid,
            component=proposal.component,
            component_pdfs=proposal.component_pdfs,
        )


__all__ = [
    "METAL_BUDGETED_REQUIRED_CONTEXT",
    "MetalBudgetedModel",
    "MetalBudgetedScatteringSample",
    "pack_metal_budgeted_prepared_state",
    "pack_metal_budgeted_program_state",
]
