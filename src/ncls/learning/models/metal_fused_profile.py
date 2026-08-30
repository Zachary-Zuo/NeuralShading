from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_json
from ncls.learning.models.metal_sampler import (
    METAL_PROPOSAL_COMPONENTS,
    METAL_PROPOSAL_DISTRIBUTION_IDS,
    METAL_PROPOSAL_FRAME_INDICES,
    METAL_PROPOSAL_SPECULAR_FLAGS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
METAL_FUSED_LAYOUT_PATH = (
    PROJECT_ROOT / "src/ncls/learning/abi/metal_fused_layout_v1.json"
)


@dataclass(frozen=True)
class MetalFusedProfile:
    profile_id: str
    maximum_texture_slots: int
    maximum_typed_tokens: int
    typed_token_width: int
    typed_attention_blocks: int
    typed_attention_heads: int
    encoder_role_width: int
    encoder_widths: tuple[int, ...]
    grid_high_channels: int
    grid_low_channels: int
    asset_adapter_rank: int
    decoder_width: int
    decoder_blocks: int
    structured_width: int
    learned_frame_count: int
    core_lobe_count: int
    residual_lobe_count: int
    angular_levels: int
    angular_channels: int
    angular_difference_rank: int
    evaluator_width: int
    evaluator_blocks: int
    maximum_sample_steps: int
    maximum_pdf_steps: int
    maximum_sample_random_values: int
    maximum_sample_evaluator_calls: int
    maximum_reads: int
    maximum_state_bytes: int

    def __post_init__(self) -> None:
        if self.profile_id != "metal_fused_full_v1":
            raise ValueError("unsupported Metal fused profile identity")
        if self.maximum_texture_slots != 9 or self.maximum_typed_tokens != 32:
            raise ValueError("Metal full profile must preserve audited source bounds")
        if self.encoder_widths != (64, 128, 192, 256):
            raise ValueError("Metal full profile encoder shape drifted")
        if (
            self.grid_high_channels != 8
            or self.grid_low_channels != 8
            or self.asset_adapter_rank != 8
            or self.structured_width != 64
            or self.learned_frame_count != 3
            or self.core_lobe_count != 6
            or self.residual_lobe_count != 4
            or self.maximum_sample_random_values != 2
            or self.maximum_sample_evaluator_calls != 1
        ):
            raise ValueError("Metal full profile required branch shape drifted")
        if self.maximum_reads > 192 or self.maximum_state_bytes > 4096:
            raise ValueError("Metal full profile exceeds its deployment envelope")


def load_metal_fused_layout(
    path: Path = METAL_FUSED_LAYOUT_PATH,
) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "ncls.metal-fused-layout@1":
        raise ValueError("unsupported Metal fused layout schema")
    identity = str(value.get("identity", ""))
    canonical = dict(value)
    canonical.pop("identity", None)
    if sha256_json(canonical) != identity:
        raise ValueError("Metal fused layout identity mismatch")
    return value


def full_metal_fused_profile() -> MetalFusedProfile:
    value = load_metal_fused_layout()
    profile = value["profile"]
    result = MetalFusedProfile(
        profile_id=str(profile["profile_id"]),
        maximum_texture_slots=int(profile["maximum_texture_slots"]),
        maximum_typed_tokens=int(profile["maximum_typed_tokens"]),
        typed_token_width=int(profile["typed_token_width"]),
        typed_attention_blocks=int(profile["typed_attention_blocks"]),
        typed_attention_heads=int(profile["typed_attention_heads"]),
        encoder_role_width=int(profile["encoder_role_width"]),
        encoder_widths=tuple(int(item) for item in profile["encoder_widths"]),
        grid_high_channels=int(profile["grid_high_channels"]),
        grid_low_channels=int(profile["grid_low_channels"]),
        asset_adapter_rank=int(profile["asset_adapter_rank"]),
        decoder_width=int(profile["decoder_width"]),
        decoder_blocks=int(profile["decoder_blocks"]),
        structured_width=int(profile["structured_width"]),
        learned_frame_count=int(profile["learned_frame_count"]),
        core_lobe_count=int(profile["core_lobe_count"]),
        residual_lobe_count=int(profile["residual_lobe_count"]),
        angular_levels=int(profile["angular_levels"]),
        angular_channels=int(profile["angular_channels"]),
        angular_difference_rank=int(profile["angular_difference_rank"]),
        evaluator_width=int(profile["evaluator_width"]),
        evaluator_blocks=int(profile["evaluator_blocks"]),
        maximum_sample_steps=int(value["bounded_execution"]["maximum_sample_steps"]),
        maximum_pdf_steps=int(value["bounded_execution"]["maximum_pdf_steps"]),
        maximum_sample_random_values=int(
            value["bounded_execution"]["maximum_sample_random_values"]
        ),
        maximum_sample_evaluator_calls=int(
            value["bounded_execution"]["maximum_sample_evaluator_calls"]
        ),
        maximum_reads=int(value["bounded_execution"]["maximum_reads"]),
        maximum_state_bytes=int(
            value["bounded_execution"]["maximum_prepared_state_bytes"]
        ),
    )
    if int(value["prepared_state"]["total_bytes"]) > result.maximum_state_bytes:
        raise ValueError("Metal prepared-state layout exceeds the registered bound")
    computed_reads = (
        result.maximum_texture_slots * 2 * (4 + 1)
        + result.angular_levels * 4
    )
    if computed_reads != result.maximum_reads:
        raise ValueError("Metal fused layout read accounting drifted")
    proposal = value["proposal_reservation"]
    if (
        tuple(proposal["components"]) != METAL_PROPOSAL_COMPONENTS
        or tuple(int(item) for item in proposal["component_frame_indices"])
        != METAL_PROPOSAL_FRAME_INDICES
        or tuple(int(item) for item in proposal["component_distribution_ids"])
        != METAL_PROPOSAL_DISTRIBUTION_IDS
        or tuple(bool(item) for item in proposal["component_specular_flags"])
        != METAL_PROPOSAL_SPECULAR_FLAGS
        or proposal["random_tuple"].get("shape") != [2]
        or float(proposal["fallback_weight_floor"]) != 0.02
    ):
        raise ValueError("Metal proposal layout disagrees with its generated component ABI")
    return result


METAL_FUSED_FULL_PROFILE = full_metal_fused_profile()


__all__ = [
    "METAL_FUSED_FULL_PROFILE",
    "METAL_FUSED_LAYOUT_PATH",
    "MetalFusedProfile",
    "full_metal_fused_profile",
    "load_metal_fused_layout",
]
