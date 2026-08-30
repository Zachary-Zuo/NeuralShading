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
    assert len(report["component_evidence"]) == 18
    assert all(
        value["activation_export_ids"]
        for value in report["component_evidence"].values()
    )
