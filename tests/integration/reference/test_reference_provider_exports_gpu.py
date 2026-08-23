from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from ncls.data import ReferenceDataset
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
