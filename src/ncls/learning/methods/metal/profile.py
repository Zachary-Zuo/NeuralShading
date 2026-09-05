from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_json


from ncls.paths import PROJECT_ROOT
METAL_BUDGETED_LAYOUT_PATH = (
    PROJECT_ROOT / "src/ncls/learning/abi/metal_budgeted_layout_v2.json"
)
METAL_SPATIAL_PROFILE_ID = "metal_spatial_hybrid_v1"
METAL_SPATIAL_SUMMARY_PROFILE_ID = "metal_spatial_summary_control_v1"
METAL_BUDGETED_HYBRID_PROFILE_ID = "metal_budgeted_hybrid_v3"
METAL_BUDGETED_DIRECT_PROFILE_ID = "metal_budgeted_direct_control_v3"
METAL_BUDGETED_ROLE_DETAIL_PROFILE_ID = "metal_budgeted_hybrid_role_detail_v4"
METAL_BUDGETED_CENTER_DETAIL_PROFILE_ID = "metal_budgeted_hybrid_center_detail_v5"
METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID = "metal_budgeted_hybrid_dual_local_v6"

_DTYPE_BYTES = {"float16": 2, "float32": 4, "uint32": 4}


def _dense_macs(layers: tuple[int, ...]) -> int:
    if len(layers) < 2 or any(width <= 0 for width in layers):
        raise ValueError("dense layer widths must be positive")
    return sum(left * right for left, right in zip(layers, layers[1:]))


def _field_bytes(field: Mapping[str, Any]) -> int:
    try:
        item_bytes = _DTYPE_BYTES[str(field["dtype"])]
    except KeyError as error:
        raise ValueError(f"unsupported Metal budgeted state dtype: {field.get('dtype')}") from error
    return item_bytes * math.prod(int(value) for value in field["shape"])


def _validate_packed_state(name: str, state: Mapping[str, Any]) -> int:
    total = int(state["total_bytes"])
    alignment = int(state["alignment"])
    if total <= 0 or alignment <= 0 or total % alignment:
        raise ValueError(f"{name} total size must satisfy its alignment")
    occupied: list[tuple[int, int, str]] = []
    for field in state["fields"]:
        offset = int(field["offset"])
        end = offset + _field_bytes(field)
        if offset < 0 or end > total:
            raise ValueError(f"{name}.{field['name']} exceeds the packed state")
        occupied.append((offset, end, str(field["name"])))
    occupied.sort()
    for left, right in zip(occupied, occupied[1:], strict=False):
        if left[1] > right[0]:
            raise ValueError(f"{name} fields {left[2]} and {right[2]} overlap")
    return total


@dataclass(frozen=True)
class MetalBudgetedProfile:
    profile_id: str
    evaluator_mode: str
    asset_detail_aggregation: str
    asset_detail_center: str
    asset_spatial_features: str
    asset_context_resolution_divisor: int
    maximum_texture_slots: int
    maximum_typed_tokens: int
    typed_token_width: int
    responsibility_count: int
    asset_plane_channels: int
    asset_plane_count: int
    semantic_decoder_layers: tuple[int, ...]
    directional_width: int
    evaluator_layers: tuple[int, ...]
    analytic_lobe_count: int
    proposal_component_count: int
    maximum_texture_reads: int
    maximum_evaluate_dense_macs: int
    maximum_prepared_state_bytes: int
    program_state_bytes: int
    prepared_state_bytes: int

    @property
    def prepare_dense_macs(self) -> int:
        return _dense_macs(self.semantic_decoder_layers)

    @property
    def is_spatial(self) -> bool:
        return self.profile_id in {METAL_SPATIAL_PROFILE_ID, METAL_SPATIAL_SUMMARY_PROFILE_ID}

    @property
    def runtime_prepare_dense_macs(self) -> int:
        """完整部署包含语义解码与 proposal adapter；训练 profile 身份保持冻结。"""
        return self.prepare_dense_macs + (1488 if self.is_spatial else self.semantic_decoder_layers[-1] * self.proposal_component_count)

    @property
    def evaluate_dense_macs(self) -> int:
        return _dense_macs(self.evaluator_layers)

    def __post_init__(self) -> None:
        if self.profile_id not in {
            METAL_BUDGETED_HYBRID_PROFILE_ID,
            METAL_BUDGETED_DIRECT_PROFILE_ID,
            METAL_BUDGETED_ROLE_DETAIL_PROFILE_ID,
            METAL_BUDGETED_CENTER_DETAIL_PROFILE_ID,
            METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID,
            METAL_SPATIAL_PROFILE_ID,
            METAL_SPATIAL_SUMMARY_PROFILE_ID,
        }:
            raise ValueError("unsupported Metal budgeted profile identity")
        expected_mode = (
            "direct"
            if self.profile_id == METAL_BUDGETED_DIRECT_PROFILE_ID
            else "hybrid"
        )
        if self.evaluator_mode != expected_mode:
            raise ValueError("Metal budgeted profile mode disagrees with its identity")
        expected_aggregation = (
            "role-separated-slot-softmax@1"
            if self.profile_id == METAL_BUDGETED_ROLE_DETAIL_PROFILE_ID
            else "shared-slot-softmax@1"
        )
        if self.is_spatial:
            expected_aggregation = "uv-group-fusion@1"
        if self.asset_detail_aggregation != expected_aggregation:
            raise ValueError(
                "Metal budgeted profile Detail aggregation disagrees with its identity"
            )
        expected_center = (
            "request-texel@1"
            if self.profile_id
            in {
                METAL_BUDGETED_CENTER_DETAIL_PROFILE_ID,
                METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID,
            }
            else "two-by-two-mean@1"
        )
        if self.is_spatial:
            expected_center = "full-native-grid@1"
        if self.asset_detail_center != expected_center:
            raise ValueError(
                "Metal budgeted profile Detail center disagrees with its identity"
            )
        expected_spatial = (
            "signed-cross-summary@1"
            if self.profile_id == METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID
            else "isotropic-summary@1"
        )
        if self.is_spatial:
            expected_spatial = "semantic-cnn@1" if self.profile_id == METAL_SPATIAL_PROFILE_ID else "native-summary-control@1"
        if self.asset_spatial_features != expected_spatial:
            raise ValueError(
                "Metal budgeted profile spatial feature recipe disagrees with its identity"
            )
        expected_context_divisor = (
            1 if self.profile_id == METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID else 4
        )
        if self.asset_context_resolution_divisor != expected_context_divisor:
            raise ValueError(
                "Metal budgeted profile Context resolution disagrees with its identity"
            )
        if (
            self.maximum_texture_slots != 9
            or self.maximum_typed_tokens != 32
            or self.typed_token_width != 16
            or self.responsibility_count != 6
            or self.asset_plane_channels != 4
            or self.asset_plane_count != 2
            or self.semantic_decoder_layers != ((137, 32, 32, 24) if self.is_spatial else (24, 32, 32, 24))
            or self.directional_width != 44
            or self.evaluator_layers != (44, 64, 64, 64, 6)
            or self.analytic_lobe_count != 2
            or self.proposal_component_count != 3
        ):
            raise ValueError("Metal budgeted NVIDIA-class shape drifted")
        if self.maximum_texture_reads != (54 if self.is_spatial else 2):
            raise ValueError("Metal prepare texture-read bound disagrees with its native UV profile")
        if self.evaluate_dense_macs > self.maximum_evaluate_dense_macs:
            raise ValueError("Metal budgeted evaluator exceeds the 20k dense-MAC hard bound")
        if self.prepared_state_bytes > self.maximum_prepared_state_bytes:
            raise ValueError("Metal budgeted PreparedState exceeds the 192-byte hard bound")


def load_metal_budgeted_layout(
    path: Path = METAL_BUDGETED_LAYOUT_PATH,
) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") not in {"ncls.metal-budgeted-layout@1", "ncls.metal-budgeted-layout@2"}:
        raise ValueError("unsupported Metal budgeted layout schema")
    identity = str(value.get("identity", ""))
    canonical = dict(value)
    canonical.pop("identity", None)
    if sha256_json(canonical) != identity:
        raise ValueError("Metal budgeted layout identity mismatch")

    shape = value["shape"]
    semantic_layers = tuple(int(item) for item in shape["semantic_decoder_layers"])
    evaluator_layers = tuple(int(item) for item in shape["evaluator_layers"])
    if _dense_macs(semantic_layers) != int(shape["semantic_decoder_macs"]):
        raise ValueError("Metal budgeted semantic decoder MAC accounting drifted")
    if _dense_macs(evaluator_layers) != int(shape["evaluate_dense_macs"]):
        raise ValueError("Metal budgeted evaluator MAC accounting drifted")
    directional_width = sum(
        int(field["width"]) for field in value["directional_features"]["fields"]
    )
    if directional_width != int(shape["directional_width"]):
        raise ValueError("Metal budgeted directional feature width drifted")
    expected_reads = int(shape["asset_plane_count"]) * int(shape.get("maximum_uv_groups", 1)) * int(shape.get("maximum_native_lookups", 1))
    if expected_reads != int(value["bounded_execution"]["maximum_texture_reads"]):
        raise ValueError("Metal budgeted asset reads disagree with its plane count")
    _validate_packed_state("material_program_state", value["material_program_state"])
    _validate_packed_state("prepared_state", value["prepared_state"])
    if len(value["proposal"]["components"]) != int(shape["proposal_component_count"]):
        raise ValueError("Metal budgeted proposal component count drifted")
    return value


def metal_budgeted_profile(
    profile_id: str = METAL_BUDGETED_HYBRID_PROFILE_ID,
    *,
    layout: Mapping[str, Any] | None = None,
) -> MetalBudgetedProfile:
    if layout is None:
        spatial = profile_id in {METAL_SPATIAL_PROFILE_ID, METAL_SPATIAL_SUMMARY_PROFILE_ID}
        value = load_metal_budgeted_layout(METAL_BUDGETED_LAYOUT_PATH if spatial else METAL_BUDGETED_LAYOUT_PATH.with_name("metal_budgeted_layout_v1.json"))
    else:
        value = layout
    try:
        selected = value["profiles"][profile_id]
    except KeyError as error:
        raise ValueError(f"unsupported Metal budgeted profile: {profile_id}") from error
    shape = value["shape"]
    bounded = value["bounded_execution"]
    result = MetalBudgetedProfile(
        profile_id=profile_id,
        evaluator_mode=str(selected["evaluator_mode"]),
        asset_detail_aggregation=str(selected["asset_detail_aggregation"]),
        asset_detail_center=str(selected["asset_detail_center"]),
        asset_spatial_features=str(selected["asset_spatial_features"]),
        asset_context_resolution_divisor=int(
            selected["asset_context_resolution_divisor"]
        ),
        maximum_texture_slots=int(shape["maximum_texture_slots"]),
        maximum_typed_tokens=int(shape["maximum_typed_tokens"]),
        typed_token_width=int(shape["typed_token_width"]),
        responsibility_count=int(shape["responsibility_count"]),
        asset_plane_channels=int(shape["asset_plane_channels"]),
        asset_plane_count=int(shape["asset_plane_count"]),
        semantic_decoder_layers=tuple(int(item) for item in shape["semantic_decoder_layers"]),
        directional_width=int(shape["directional_width"]),
        evaluator_layers=tuple(int(item) for item in shape["evaluator_layers"]),
        analytic_lobe_count=int(shape["analytic_lobe_count"]),
        proposal_component_count=int(shape["proposal_component_count"]),
        maximum_texture_reads=int(bounded["maximum_texture_reads"]),
        maximum_evaluate_dense_macs=int(bounded["maximum_evaluate_dense_macs"]),
        maximum_prepared_state_bytes=int(bounded["maximum_prepared_state_bytes"]),
        program_state_bytes=_validate_packed_state(
            "material_program_state", value["material_program_state"]
        ),
        prepared_state_bytes=_validate_packed_state(
            "prepared_state", value["prepared_state"]
        ),
    )
    if result.prepare_dense_macs != int(shape["semantic_decoder_macs"]):
        raise ValueError("Metal budgeted profile prepare cost disagrees with layout")
    if result.evaluate_dense_macs != int(shape["evaluate_dense_macs"]):
        raise ValueError("Metal budgeted profile evaluator cost disagrees with layout")
    return result


METAL_BUDGETED_HYBRID_PROFILE = metal_budgeted_profile()
METAL_BUDGETED_DIRECT_PROFILE = metal_budgeted_profile(
    METAL_BUDGETED_DIRECT_PROFILE_ID
)
METAL_BUDGETED_ROLE_DETAIL_PROFILE = metal_budgeted_profile(
    METAL_BUDGETED_ROLE_DETAIL_PROFILE_ID
)
METAL_BUDGETED_CENTER_DETAIL_PROFILE = metal_budgeted_profile(
    METAL_BUDGETED_CENTER_DETAIL_PROFILE_ID
)
METAL_BUDGETED_DUAL_LOCAL_PROFILE = metal_budgeted_profile(
    METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID
)
METAL_SPATIAL_PROFILE = metal_budgeted_profile(METAL_SPATIAL_PROFILE_ID)


__all__ = [
    "METAL_BUDGETED_DIRECT_PROFILE",
    "METAL_BUDGETED_DIRECT_PROFILE_ID",
    "METAL_BUDGETED_DUAL_LOCAL_PROFILE",
    "METAL_BUDGETED_DUAL_LOCAL_PROFILE_ID",
    "METAL_BUDGETED_CENTER_DETAIL_PROFILE",
    "METAL_BUDGETED_CENTER_DETAIL_PROFILE_ID",
    "METAL_BUDGETED_HYBRID_PROFILE",
    "METAL_BUDGETED_HYBRID_PROFILE_ID",
    "METAL_BUDGETED_LAYOUT_PATH",
    "METAL_BUDGETED_ROLE_DETAIL_PROFILE",
    "METAL_BUDGETED_ROLE_DETAIL_PROFILE_ID",
    "MetalBudgetedProfile",
    "load_metal_budgeted_layout",
    "metal_budgeted_profile",
]
