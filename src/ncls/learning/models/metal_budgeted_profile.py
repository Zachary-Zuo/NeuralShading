from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_json


PROJECT_ROOT = Path(__file__).resolve().parents[4]
METAL_BUDGETED_LAYOUT_PATH = (
    PROJECT_ROOT / "src/ncls/learning/abi/metal_budgeted_layout_v1.json"
)
METAL_BUDGETED_HYBRID_PROFILE_ID = "metal_budgeted_hybrid_v3"
METAL_BUDGETED_DIRECT_PROFILE_ID = "metal_budgeted_direct_control_v3"

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
    def evaluate_dense_macs(self) -> int:
        return _dense_macs(self.evaluator_layers)

    def __post_init__(self) -> None:
        if self.profile_id not in {
            METAL_BUDGETED_HYBRID_PROFILE_ID,
            METAL_BUDGETED_DIRECT_PROFILE_ID,
        }:
            raise ValueError("unsupported Metal budgeted profile identity")
        expected_mode = (
            "hybrid"
            if self.profile_id == METAL_BUDGETED_HYBRID_PROFILE_ID
            else "direct"
        )
        if self.evaluator_mode != expected_mode:
            raise ValueError("Metal budgeted profile mode disagrees with its identity")
        if (
            self.maximum_texture_slots != 9
            or self.maximum_typed_tokens != 32
            or self.typed_token_width != 16
            or self.responsibility_count != 6
            or self.asset_plane_channels != 4
            or self.asset_plane_count != 2
            or self.semantic_decoder_layers != (24, 32, 32, 24)
            or self.directional_width != 44
            or self.evaluator_layers != (44, 64, 64, 64, 6)
            or self.analytic_lobe_count != 2
            or self.proposal_component_count != 3
        ):
            raise ValueError("Metal budgeted NVIDIA-class shape drifted")
        if self.maximum_texture_reads != 2:
            raise ValueError("Metal budgeted prepare must perform exactly two texture reads")
        if self.evaluate_dense_macs > self.maximum_evaluate_dense_macs:
            raise ValueError("Metal budgeted evaluator exceeds the 20k dense-MAC hard bound")
        if self.prepared_state_bytes > self.maximum_prepared_state_bytes:
            raise ValueError("Metal budgeted PreparedState exceeds the 192-byte hard bound")


def load_metal_budgeted_layout(
    path: Path = METAL_BUDGETED_LAYOUT_PATH,
) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "ncls.metal-budgeted-layout@1":
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
    if int(shape["asset_plane_count"]) != int(
        value["bounded_execution"]["maximum_texture_reads"]
    ):
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
    value = load_metal_budgeted_layout() if layout is None else layout
    try:
        selected = value["profiles"][profile_id]
    except KeyError as error:
        raise ValueError(f"unsupported Metal budgeted profile: {profile_id}") from error
    shape = value["shape"]
    bounded = value["bounded_execution"]
    result = MetalBudgetedProfile(
        profile_id=profile_id,
        evaluator_mode=str(selected["evaluator_mode"]),
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


__all__ = [
    "METAL_BUDGETED_DIRECT_PROFILE",
    "METAL_BUDGETED_DIRECT_PROFILE_ID",
    "METAL_BUDGETED_HYBRID_PROFILE",
    "METAL_BUDGETED_HYBRID_PROFILE_ID",
    "METAL_BUDGETED_LAYOUT_PATH",
    "MetalBudgetedProfile",
    "load_metal_budgeted_layout",
    "metal_budgeted_profile",
]
