from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from ncls.learning.models.metal_budgeted_asset import (
    MetalBudgetedAssetSample,
    MetalBudgetedTwoReadAsset,
)
from ncls.learning.models.metal_budgeted_compiler import (
    MetalBudgetedProgramState,
    MetalBudgetedTypedCompiler,
)
from ncls.learning.models.metal_budgeted_evaluator import (
    MetalBudgetedDirectionalRepresentation,
    MetalBudgetedEvaluation,
    MetalBudgetedEvaluator,
    MetalBudgetedPrepare,
    MetalBudgetedPreparedState,
)
from ncls.learning.models.metal_budgeted_profile import (
    METAL_BUDGETED_HYBRID_PROFILE,
    MetalBudgetedProfile,
    metal_budgeted_profile,
)
from ncls.learning.models.metal_budgeted_sampler import (
    MetalBudgetedProposalPdf,
    MetalBudgetedProposalSample,
    metal_budgeted_proposal_pdf,
    metal_budgeted_sample_proposal,
)


METAL_BUDGETED_REQUIRED_CONTEXT = {
    "profile_id": "metal_budgeted_hybrid_v1",
    "asset_variant_count": 52,
    "maximum_texture_slots": 9,
    "maximum_typed_tokens": 32,
    "runtime_asset_reads": 2,
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
    if floats.shape[1] != 72 or prepared.identity_and_flags.shape[1] != 4:
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
        self.profile = profile
        self.typed_compiler = MetalBudgetedTypedCompiler(profile)
        self.asset = MetalBudgetedTwoReadAsset(
            profile, asset_variant_count=asset_variant_count
        )
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
        expected["profile_id"] = profile_id
        if dict(context) != expected:
            raise ValueError(
                "Metal budgeted trainable requires the exact NVIDIA-class context"
            )
        return cls(
            metal_budgeted_profile(profile_id),
            asset_variant_count=int(context["asset_variant_count"]),
        )

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
    ) -> MetalBudgetedAssetSample:
        return self.asset(tensors, program, qat=qat)

    def prepare_from_components(
        self,
        program: MetalBudgetedProgramState,
        asset: MetalBudgetedAssetSample,
        wo: torch.Tensor,
    ) -> MetalBudgetedPreparedState:
        return self.prepared_model(program, asset, wo)

    def prepare(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        wo: torch.Tensor | None = None,
        qat: bool = True,
    ) -> MetalBudgetedPreparedState:
        program = self.compile_program_state(tensors)
        asset = self.sample_asset(tensors, program, qat=qat)
        return self.prepare_from_components(
            program, asset, tensors["wo"] if wo is None else wo
        )

    def prepare_paired(
        self,
        tensors: Mapping[str, torch.Tensor],
        program: MetalBudgetedProgramState,
        *,
        qat: bool = True,
    ) -> MetalBudgetedPreparedState:
        required = {
            "paired_uv",
            "paired_uv_dx",
            "paired_uv_dy",
            "metal_paired_texture_patches",
        }
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
            "metal_texture_patches": tensors["metal_paired_texture_patches"],
        }
        asset = self.sample_asset(paired, program, qat=qat)
        return self.prepare_from_components(program, asset, tensors["wo"])

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
            prepared.frames,
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
            prepared.frames,
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
