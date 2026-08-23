from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ncls.data import ReferenceDataset


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
