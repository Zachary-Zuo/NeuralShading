from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.references import (
    deterministic_directional_metrics,
    linear_hdr_image_metrics,
    load_reference_acceptance,
    load_reference_registry,
    validate_reference_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_reference_registry_and_packages_are_consistent() -> None:
    packages = validate_reference_tree(PROJECT_ROOT / "references")
    assert {package.reference_id for package in packages} == {
        "ncls.layer-stack-random-walk@1",
        "ncls.pbrt-coated-crosscheck@1",
        "ncls.openpbr@1.1.1",
        "ncls.merl-brdf@1",
        "ncls.materialx-polyhaven@1",
    }
    by_id = {package.reference_id: package for package in packages}
    assert by_id["ncls.openpbr@1.1.1"].dependencies[0]["dependency_id"] == "glm@1.0.1"
    assert by_id["ncls.materialx-polyhaven@1"].source_assets[0]["license"] == "CC0-1.0"


def test_registry_tracks_each_integration_capability_independently() -> None:
    entries = {entry.reference_id: entry for entry in load_reference_registry(PROJECT_ROOT / "references")}
    assert entries["ncls.openpbr@1.1.1"].status == "active"
    assert entries["ncls.openpbr@1.1.1"].capabilities["falcor_runtime"] == "ready"
    assert entries["ncls.merl-brdf@1"].capabilities["viewer_integration"] == "ready"
    assert entries["ncls.materialx-polyhaven@1"].capabilities["numerical_parity"] == "not-applicable"
    assert entries["ncls.materialx-polyhaven@1"].capabilities["image_parity"] == "ready"


def test_deterministic_directional_acceptance_is_executable() -> None:
    acceptance = load_reference_acceptance(PROJECT_ROOT / "references" / "acceptance.json")
    native = np.asarray([[0.2, 0.1, 0.05], [0.01, 0.02, 0.03]])
    passing = deterministic_directional_metrics(native, native + 1e-6, acceptance.deterministic_directional)
    failing = deterministic_directional_metrics(native, native * 0.8, acceptance.deterministic_directional)
    assert passing.passed
    assert not failing.passed


def test_linear_hdr_acceptance_is_executable() -> None:
    acceptance = load_reference_acceptance(PROJECT_ROOT / "references" / "acceptance.json")
    native = np.full((8, 8, 3), 0.25, dtype=np.float64)
    passing = linear_hdr_image_metrics(native, native + 1e-6, acceptance.linear_hdr_image)
    failing = linear_hdr_image_metrics(native, native * 0.8, acceptance.linear_hdr_image)
    textured = linear_hdr_image_metrics(native, native * 0.995, acceptance.linear_hdr_textured_image)
    assert passing.passed
    assert not failing.passed
    assert textured.passed
