from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from ncls.data import E0_FOOTPRINT_PROFILE_ID, ReferenceDataset
from ncls.learning.audit import audit_supervision
from ncls.learning.gates import evaluate_supervision_gate, load_supervision_gate
from ncls.paths import PROJECT_ROOT, SOURCE_MATERIAL_ROOT
from ncls.source_materials.identity import materialx_asset_sha256
from ncls.source_materials.materialx import MaterialXAssetCatalog


pytest.importorskip("falcor")
pytestmark = pytest.mark.falcor


def test_all_current_reference_materials_export_one_fixed_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "all-current-materials.h5"
    subprocess.run(
        [
            sys.executable, "-m", "ncls.cli", "data", "collect-reference",
            "--provider", "all",
            "--families", "1",
            "--local-states", "1",
            "--views", "1",
            "--lights", "2",
            "--spatial-samples", "1",
            "--samples-per-replica", "1",
            "--max-depth", "2",
            "--query-group-batch", "1",
            "--output", str(path),
        ],
        check=True,
        timeout=300,
    )

    with ReferenceDataset.open(path) as dataset:
        families, counts = np.unique(dataset.state_strings("family_id"), return_counts=True)
        assert dict(zip(families.tolist(), counts.tolist(), strict=True)) == {
            "materialx.textured-surface@1": 8,
            "merl.measured-brdf@1": 100,
            "ncls.layer-stack@1": 1,
            "openpbr.surface@1.1.1": 83,
        }
        assert dataset.manifest.counts == {
            "state_count": 192,
            "query_group_count": 192,
            "direction_count": 2,
        }
        responses = np.asarray(dataset.stream["responses/mean"], dtype=np.float32)
        assert np.all(np.isfinite(responses))
        assert np.all(np.asarray(dataset.stream["responses/valid"], dtype=np.uint8) == 1)
        family_rows = dataset.state_strings("family_id")
        state_indices = np.asarray(dataset.stream["queries/state_index"], dtype=np.uint32)
        rng_seed = np.asarray(dataset.stream["queries/rng_seed"], dtype=np.uint64)
        layer_rows = family_rows[state_indices] == "ncls.layer-stack@1"
        assert np.all(rng_seed[~layer_rows] == 0)
        assert np.all(rng_seed[layer_rows] > 0)
        assert np.all(rng_seed[layer_rows, 0] == rng_seed[layer_rows, 1])

        openpbr_index = int(np.flatnonzero(family_rows == "openpbr.surface@1.1.1")[0])
        openpbr_payload = json.loads(dataset.state_payload(openpbr_index))
        assert not Path(str(openpbr_payload["source_document"])).is_absolute()

        materialx_index = int(np.flatnonzero(family_rows == "materialx.textured-surface@1")[0])
        asset_id = str(dataset.state_strings("asset_id")[materialx_index])
        catalog = MaterialXAssetCatalog(PROJECT_ROOT / "references/materialx-polyhaven-v1/assets.json")
        source = catalog.source_material(asset_id)
        records = {str(record["semantic"]): record for record in catalog.records(asset_id)}
        asset_root = SOURCE_MATERIAL_ROOT / "materialx-polyhaven/v1"
        texture = lambda semantic: asset_root / str(records[semantic]["path"]) if semantic in records else None
        expected_source_hash = materialx_asset_sha256(
            asset_root / source.document_uri,
            (
                texture("base-color-srgb"), texture("roughness-raw"), texture("metalness-raw"),
                texture("normal-opengl-tangent-raw"), texture("normalized-displacement-raw"),
            ),
        )
        assert dataset.state_strings("source_sha256")[materialx_index] == expected_source_hash


def test_materialx_e0_surface_profile_persists_scale_rotation_and_seam_queries(tmp_path: Path) -> None:
    path = tmp_path / "materialx-e0-surface-profile.h5"
    subprocess.run(
        [
            sys.executable, "-m", "ncls.cli", "data", "collect-reference",
            "--provider", "materialx",
            "--material-id", "american_walnut_veneer",
            "--views", "1",
            "--validation-views", "1",
            "--test-views", "1",
            "--adversarial-views", "1",
            "--lights", "16",
            "--spatial-samples", "20",
            "--surface-profile", E0_FOOTPRINT_PROFILE_ID,
            "--query-profile", "ncls.e0-peak-grazing-mixture@2",
            "--output", str(path),
        ],
        check=True,
        timeout=300,
    )

    with ReferenceDataset.open(path) as dataset:
        assert dataset.manifest.generation_config["surface_profile_id"] == E0_FOOTPRINT_PROFILE_ID
        uv = np.asarray(dataset.stream["queries/uv"], dtype=np.float64)
        dx = np.asarray(dataset.stream["queries/uv_dx"], dtype=np.float64)
        dy = np.asarray(dataset.stream["queries/uv_dy"], dtype=np.float64)
        scales = {
            (
                round(float(np.log2(np.linalg.norm(x))), 4),
                round(float(np.log2(np.linalg.norm(y))), 4),
            )
            for x, y in zip(dx, dy, strict=True)
        }
        rotations = {
            round(float(np.mod(np.arctan2(x[1], x[0]), np.pi)), 6)
            for x in dx
        }
        assert len(scales) >= 4
        assert len(rotations) >= 4
        assert np.any(uv[:, 0] <= 0.01) and np.any(uv[:, 0] >= 0.99)
        assert np.any(uv[:, 1] <= 0.01) and np.any(uv[:, 1] >= 0.99)

    audit = audit_supervision(path, tmp_path / "audit")
    assert audit["coverage"]["adversarial_profile_presence"]["spatial_footprint_rotation"]
    assert audit["coverage"]["unique_footprint_scale_count"] >= 4
    assert audit["coverage"]["unique_footprint_rotation_count"] >= 4
    assert audit["coverage"]["uv_seam"]["opposite_edge_pair_axis_count"] == 2
    gate = load_supervision_gate(PROJECT_ROOT / "configs/research/e0-supervision-gates-v5.json")
    checks = {
        item["name"]: item["passed"]
        for item in evaluate_supervision_gate(audit, gate)["checks"]
    }
    assert checks["dataset.query_profile_id"]
    assert checks["family.materialx.textured-surface@1.surface_profile_id"]
    assert checks["family.materialx.textured-surface@1.footprint_scales"]
    assert checks["family.materialx.textured-surface@1.footprint_rotations"]
    assert checks["family.materialx.textured-surface@1.uv_seam_opposite_edge_pair_axes"]
