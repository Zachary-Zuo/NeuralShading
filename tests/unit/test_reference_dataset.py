from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ncls.cli import main as cli_main
from ncls.data import (
    FORMAT_NAME,
    FORMAT_VERSION,
    CollectionConfig,
    EvaluatedBlock,
    QueryPlan,
    ReferenceDataset,
    ReferenceDescriptor,
    ReplicaMoments,
    SourceState,
    SurfaceSample,
    collect_reference_dataset,
    combine_replica_moments,
    equal_area_hemisphere,
    make_state_id,
    stratified_view_directions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeProvider:
    descriptor = ReferenceDescriptor(
        "test.surface@1",
        "test.reference@1",
        "test.native-json@1",
        implementation_sha256="1" * 64,
    )

    def __init__(self) -> None:
        states = []
        for index, split in enumerate((0, 1, 2)):
            payload = json.dumps({"value": index + 1}, sort_keys=True).encode("utf-8")
            source_hash = hashlib.sha256(payload).hexdigest()
            states.append(SourceState(
                make_state_id(self.descriptor.family_id, self.descriptor.native_schema_id, payload, source_hash),
                self.descriptor.family_id,
                self.descriptor.reference_id,
                f"asset-{index}",
                f"group-{index}",
                self.descriptor.native_schema_id,
                payload,
                f"test://asset-{index}",
                source_hash,
                split,
                index + 1,
            ))
        self.states = tuple(states)

    def source_states(self):
        return self.states

    def surface_samples(self, state):
        return (SurfaceSample(),)

    def query_plan(self, state):
        lights, weights = equal_area_hemisphere(4)
        return QueryPlan(
            stratified_view_directions(2),
            lights,
            weights,
            np.full(4, 1.0 / (2.0 * np.pi), dtype=np.float32),
            "test-uniform@1",
            17,
        )

    def evaluate(self, state, surfaces, plan):
        shape = (len(surfaces), len(plan.view_directions), len(plan.light_directions), 3)
        value = np.full(shape, state.runtime_state, dtype=np.float32)
        return EvaluatedBlock.deterministic(value)

    def metadata(self):
        return {"family_id": self.descriptor.family_id, "reference_id": self.descriptor.reference_id}

    def close(self):
        pass


def _collect(path: Path):
    return collect_reference_dataset(
        path,
        [_FakeProvider()],
        CollectionConfig(view_count=2, light_count=4),
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )


def test_replica_moment_merge_uses_stable_parallel_variance() -> None:
    shape = (1, 2, 3)
    a = ReplicaMoments(np.zeros(shape), np.ones(shape), np.asarray([2], dtype=np.uint32))
    b = ReplicaMoments(np.full(shape, 2.0), np.ones(shape), np.asarray([2], dtype=np.uint32))
    combined = combine_replica_moments(a, b)
    assert combined.mean == pytest.approx(1.0)
    assert combined.variance == pytest.approx(2.0)
    assert combined.sample_count.tolist() == [4]
    assert combined.standard_error == pytest.approx(np.sqrt(0.5))


def test_equal_area_directions_integrate_lambert_response() -> None:
    directions, weights = equal_area_hemisphere(128)
    albedo = np.asarray([0.2, 0.5, 0.9], dtype=np.float64)
    response_cos = albedo[None, :] * directions[:, 2:3] / np.pi
    integrated = np.sum(response_cos * weights[:, None], axis=0)
    assert integrated == pytest.approx(albedo, rel=1e-6, abs=1e-6)


def test_hdf5_contract_preserves_native_state_queries_and_responses(tmp_path: Path) -> None:
    path = tmp_path / "reference.h5"
    manifest = _collect(path)
    assert manifest.format_name == FORMAT_NAME
    assert manifest.format_version == FORMAT_VERSION
    assert manifest.counts == {"state_count": 3, "query_group_count": 6, "direction_count": 4}

    with ReferenceDataset.open(path) as dataset:
        assert json.loads(dataset.state_payload(1)) == {"value": 2}
        assert {name: len(dataset.group_indices(name)) for name in ("train", "validation", "test")} == {
            "train": 2,
            "validation": 2,
            "test": 2,
        }
        batch = dataset.group_batch((0, 3, 5))
        assert batch["wo"].shape == (3, 3)
        assert batch["wi"].shape == (3, 4, 3)
        assert batch["proposal_pdf"].shape == (3, 4)
        assert batch["mean"].shape == (3, 4, 3)
        np.testing.assert_allclose(batch["standard_error"], 0.0)
        np.testing.assert_allclose(batch["mean"][:, 0, 0], [1.0, 2.0, 3.0])


def test_dataset_cli_and_semantic_hash_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "reference.h5"
    _collect(path)
    assert cli_main(["data", "validate", str(path)]) == 0
    assert "ReferenceDataset OK" in capsys.readouterr().out
    with h5py.File(path, "r+") as stream:
        stream["responses/mean"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="semantic content hash mismatch"):
        ReferenceDataset.open(path)


def test_reference_dataset_layout_tracks_runtime_constants() -> None:
    layout_path = PROJECT_ROOT / "src" / "ncls" / "data" / "schemas" / "reference_dataset_v3.layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    assert layout["format_name"] == FORMAT_NAME
    assert layout["format_version"] == FORMAT_VERSION
    assert "payload_blob" in layout["required_groups"]["states"]
    assert layout["legacy_reader"] is False
