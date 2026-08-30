from __future__ import annotations

from pathlib import Path

from ncls.source_materials.mdl_metal import MdlMetalRegistry
from tools.learning.preflight_metal_fused import build_preflight_report


def test_metal_full_cohort_preflight_closes_every_registered_component() -> None:
    registry = MdlMetalRegistry.load(
        Path("references/mdl-vmaterials2-v1/metal-opaque-v1.json")
    )
    report = build_preflight_report(registry)
    assert report["full_cohort_closure"]["exports"] == 692
    assert report["full_cohort_closure"]["graphs"] == 178
    assert report["full_cohort_closure"]["texture_sets"] == 52
    assert report["full_cohort_closure"]["role_classes"] == [0, 1, 2, 3]
    assert len(report["component_evidence"]) == 20
    assert report["proposal_closure"] == {
        "components": [
            "core-conductor-ggx",
            "core-conductor-beckmann",
            "core-coat-specular",
            "core-diffuse-contamination",
            "core-broad-scatter",
            "core-secondary-specular",
            "positive-residual-0",
            "positive-residual-1",
            "positive-residual-2",
            "positive-residual-3",
            "full-hemisphere-fallback",
        ],
        "component_count": 11,
        "state_width": 8,
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
        "component_frame_indices": [0, 0, 1, 0, 2, 3, 0, 1, 2, 3, 0],
        "component_distribution_ids": [0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3],
        "component_specular_flags": [
            True, True, True, False, False, True,
            True, True, True, True, False,
        ],
        "random_tuple_width": 2,
        "fallback_weight_floor": 0.02,
        "hemisphere_convention": "renderer-shading-z-positive-folded-preimage@1",
        "sample_evaluator_calls": 1,
    }
    assert all(
        value["activation_export_ids"]
        for value in report["component_evidence"].values()
    )
