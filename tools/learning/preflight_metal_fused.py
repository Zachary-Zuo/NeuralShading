from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

from ncls.core.identity import sha256_json, write_json_atomic
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.models.metal_fused_profile import load_metal_fused_layout
from ncls.learning.models.metal_sampler import (
    METAL_PROPOSAL_COMPONENTS,
    METAL_PROPOSAL_COMPONENT_COUNT,
    METAL_PROPOSAL_DISTRIBUTION_IDS,
    METAL_PROPOSAL_FRAME_INDICES,
    METAL_PROPOSAL_SPECULAR_FLAGS,
    METAL_PROPOSAL_STATE_WIDTH,
)
from ncls.learning.models.metal_texture_codec import semantic_role_class
from ncls.source_materials.mdl_metal import (
    MDL_METAL_EXPECTED_COUNTS,
    MdlMetalExport,
    MdlMetalRegistry,
    PARAMETER_RESPONSIBILITIES,
)


def _slot_role_classes(slot: Mapping[str, Any]) -> set[int]:
    roles = tuple(str(value) for value in slot["roles"])
    if len(roles) > 1:
        return {3}
    channels = slot["channels"]
    count = 3 if "RGB" in channels else len(channels)
    return {semantic_role_class(roles[0], count)}


def _features(
    registry: MdlMetalRegistry, record: MdlMetalExport
) -> set[str]:
    texture_set = registry.texture_sets[record.texture_set_id]
    role_classes = {
        role_class
        for slot in texture_set["slots"]
        for role_class in _slot_role_classes(slot)
    }
    responsibilities = {
        str(parameter["responsibility"]) for parameter in record.parameters
    }
    types = {str(parameter["type"]) for parameter in record.parameters}
    return {
        *(f"role:{value}" for value in role_classes),
        *(f"responsibility:{value}" for value in responsibilities),
        *(f"type:{value}" for value in types),
        f"recipe-class:{'standard' if record.finish in {'base', 'brushed', 'foil', 'hammered', 'knurling', 'scratched', 'sheet'} else 'special'}",
    }


def _activation_set(registry: MdlMetalRegistry) -> tuple[MdlMetalExport, ...]:
    universe = {
        *(f"role:{value}" for value in range(4)),
        *(f"responsibility:{value}" for value in PARAMETER_RESPONSIBILITIES),
        *(f"type:{value}" for value in ("float", "bool", "color", "enum", "float2", "int")),
        "recipe-class:standard",
        "recipe-class:special",
    }
    candidates = {record.export_id: _features(registry, record) for record in registry.exports}
    selected = []
    remaining = set(universe)
    while remaining:
        record = max(
            registry.exports,
            key=lambda item: (
                len(candidates[item.export_id] & remaining),
                -int(item.export_id, 16),
            ),
        )
        gain = candidates[record.export_id] & remaining
        if not gain:
            raise RuntimeError(f"Metal activation universe is unreachable: {sorted(remaining)}")
        selected.append(record)
        remaining -= gain
    return tuple(selected)


def build_preflight_report(registry: MdlMetalRegistry) -> Mapping[str, Any]:
    graph_ids = set()
    schema_ids = set()
    texture_set_ids = set()
    recipe_ids = set()
    maximum_tokens = 0
    types = set()
    responsibilities = set()
    for record in tqdm(registry.exports, desc="metal-full-preflight", unit="export"):
        graph_ids.add(record.graph_id)
        schema_ids.add(record.parameter_schema_id)
        texture_set_ids.add(record.texture_set_id)
        recipe_ids.add(record.recipe_id)
        maximum_tokens = max(maximum_tokens, len(record.parameters))
        types.update(str(value["type"]) for value in record.parameters)
        responsibilities.update(
            str(value["responsibility"]) for value in record.parameters
        )
    role_classes = {
        role_class
        for texture_set in registry.texture_sets.values()
        for slot in texture_set["slots"]
        for role_class in _slot_role_classes(slot)
    }
    closure = {
        "exports": len(registry.exports),
        "graphs": len(graph_ids),
        "parameter_schema_table": len(registry.parameter_schemas),
        "opaque_reachable_parameter_schemas": len(schema_ids),
        "texture_sets": len(texture_set_ids),
        "recipes": len(recipe_ids),
        "maximum_export_typed_tokens": maximum_tokens,
        "maximum_schema_typed_tokens": max(
            len(value["parameters"])
            for value in registry.parameter_schemas.values()
        ),
        "maximum_texture_slots": max(
            len(value["slots"]) for value in registry.texture_sets.values()
        ),
        "role_classes": sorted(role_classes),
        "parameter_types": sorted(types),
        "responsibilities": sorted(responsibilities),
    }
    if closure != {
        "exports": MDL_METAL_EXPECTED_COUNTS["opaque_exports"],
        "graphs": MDL_METAL_EXPECTED_COUNTS["opaque_graphs"],
        "parameter_schema_table": MDL_METAL_EXPECTED_COUNTS["parameter_schemas"],
        "opaque_reachable_parameter_schemas": 59,
        "texture_sets": MDL_METAL_EXPECTED_COUNTS["opaque_texture_sets"],
        "recipes": len(registry.recipes),
        "maximum_export_typed_tokens": 22,
        "maximum_schema_typed_tokens": 31,
        "maximum_texture_slots": 9,
        "role_classes": [0, 1, 2, 3],
        "parameter_types": ["bool", "color", "enum", "float", "float2", "int"],
        "responsibilities": sorted(PARAMETER_RESPONSIBILITIES),
    }:
        raise RuntimeError(f"Metal full-cohort closure drifted: {closure}")
    activation = _activation_set(registry)
    layout = load_metal_fused_layout()
    proposal = layout["proposal_reservation"]
    proposal_closure = {
        "components": list(proposal["components"]),
        "component_count": len(proposal["components"]),
        "state_width": len(proposal["state_fields"]),
        "state_fields": list(proposal["state_fields"]),
        "component_frame_indices": list(proposal["component_frame_indices"]),
        "component_distribution_ids": list(
            proposal["component_distribution_ids"]
        ),
        "component_specular_flags": list(proposal["component_specular_flags"]),
        "random_tuple_width": int(proposal["random_tuple"]["shape"][0]),
        "fallback_weight_floor": float(proposal["fallback_weight_floor"]),
        "hemisphere_convention": str(proposal["hemisphere_convention"]),
        "sample_evaluator_calls": int(
            layout["bounded_execution"]["maximum_sample_evaluator_calls"]
        ),
    }
    expected_proposal = {
        "components": list(METAL_PROPOSAL_COMPONENTS),
        "component_count": METAL_PROPOSAL_COMPONENT_COUNT,
        "state_width": METAL_PROPOSAL_STATE_WIDTH,
        "state_fields": [
            "normalized_weight",
            "alpha_x",
            "alpha_y",
            "rotation_radians",
            "active",
            "frame_index",
            "distribution_id",
            "energy_clue",
        ],
        "component_frame_indices": list(METAL_PROPOSAL_FRAME_INDICES),
        "component_distribution_ids": list(METAL_PROPOSAL_DISTRIBUTION_IDS),
        "component_specular_flags": list(METAL_PROPOSAL_SPECULAR_FLAGS),
        "random_tuple_width": 2,
        "fallback_weight_floor": 0.02,
        "hemisphere_convention": "renderer-shading-z-positive-folded-preimage@1",
        "sample_evaluator_calls": 1,
    }
    if proposal_closure != expected_proposal:
        raise RuntimeError(f"Metal proposal ABI closure drifted: {proposal_closure}")
    component_evidence = {
        component.component_id: {
            "required": component.required,
            "parameter_groups": list(component.parameter_groups),
            "python_outputs": list(component.python_outputs),
            "activation_export_ids": [record.export_id for record in activation],
        }
        for component in METHOD_DEFINITION.descriptor.components
    }
    report = {
        "schema": "ncls.metal-fused-full-cohort-preflight@1",
        "registry_identity": registry.identity,
        "method_descriptor_identity": METHOD_DEFINITION.descriptor.descriptor_sha256,
        "profile_id": "metal_fused_full_v1",
        "full_cohort_closure": closure,
        "proposal_closure": proposal_closure,
        "activation_set": [
            {
                "export_id": record.export_id,
                "exact_locator": dict(record.exact_locator),
                "graph_id": record.graph_id,
                "schema_id": record.parameter_schema_id,
                "texture_set_id": record.texture_set_id,
                "recipe_id": record.recipe_id,
                "features": sorted(_features(registry, record)),
            }
            for record in activation
        ],
        "component_evidence": component_evidence,
    }
    return {**report, "identity": sha256_json(report)}


def main() -> int:
    parser = argparse.ArgumentParser(description="验证Metal full cohort与组件激活闭包")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("references/mdl-vmaterials2-v1/metal-opaque-v1.json"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = build_preflight_report(MdlMetalRegistry.load(arguments.registry))
    if arguments.output is not None:
        write_json_atomic(arguments.output, report)
    print(report["identity"])
    print(
        f"activation_exports={len(report['activation_set'])} "
        f"components={len(report['component_evidence'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
