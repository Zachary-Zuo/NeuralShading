from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ncls.cli import main as cli_main
from ncls.data import (
    FORMAT_NAME,
    FORMAT_VERSION,
    ReferenceDataset,
    ReplicaMoments,
    combine_replica_moments,
    equal_area_hemisphere,
)
from ncls.data.legacy_v0 import convert_legacy_v0_dataset
from ncls.data.generator import ReferenceGenerationConfig, generate_reference_dataset
from ncls.core.material.legacy_v0 import (
    LegacyLayerInterface,
    LegacyLayerStack,
    LegacyLayerType,
    pack_legacy_stack,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _make_legacy_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    root.mkdir()
    stack = LegacyLayerStack(
        (
            LegacyLayerInterface(
                LegacyLayerType.DIFFUSE,
                0.5,
                0.5,
                albedo=(0.3, 0.5, 0.7),
            ),
        ),
        (),
    )
    (root / "stacks.bin").write_bytes(pack_legacy_stack(stack))
    views = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0]], dtype=np.float32)
    lights, weights = equal_area_hemisphere(4)
    np.save(root / "views.npy", views)
    np.save(root / "light_directions.npy", lights)
    np.save(root / "solid_angle_weights.npy", weights)
    states = np.zeros(
        1,
        dtype=np.dtype(
            [("family_index", "<u4"), ("local_state", "<u2"), ("split", "u1"), ("reserved", "u1")]
        ),
    )
    np.save(root / "states.npy", states)
    np.save(root / "family_splits.npy", np.zeros(1, dtype=np.uint8))

    response_dtype = np.dtype(
        [
            ("mean_a", "<f2", (4, 3)),
            ("mean_b", "<f2", (4, 3)),
            ("count", "<u2", (4,)),
        ]
    )
    response = np.zeros(2, dtype=response_dtype)
    response["mean_a"] = 1.0
    response["mean_b"] = 3.0
    response["count"] = 8
    index = np.asarray([[0, 0], [0, 1]], dtype=np.uint32)
    np.save(root / "tiles-00000.npy", response)
    np.save(root / "index-00000.npy", index)
    metadata = {
        "format": "ncls-direction-tiles",
        "format_version": 1,
        "prior_version": "v0.2",
        "teacher_source_sha256": "a" * 64,
        "seed": 17,
        "direction_parameterization": "equal-area-fibonacci-hemisphere",
        "shards": [
            {
                "tiles": "tiles-00000.npy",
                "index": "index-00000.npy",
                "tile_start": 0,
                "tile_count": 2,
            }
        ],
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return lights, weights


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


def test_legacy_dataset_conversion_preserves_response_and_marks_uncertainty(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    output = tmp_path / "converted"
    _make_legacy_dataset(source)
    manifest = convert_legacy_v0_dataset(
        source,
        output,
        created_at="2026-08-23T00:00:00+00:00",
        generator_git_commit="test",
    )
    assert manifest.format_name == FORMAT_NAME
    assert manifest.format_version == FORMAT_VERSION
    assert manifest.statistics_encoding["uncertainty_kind"] == "replica-mean-variance"

    dataset = ReferenceDataset.open(output)
    statistics = dataset.statistics(0)
    assert statistics.mean == pytest.approx(2.0)
    assert statistics.variance == pytest.approx(2.0)
    assert statistics.standard_error == pytest.approx(np.sqrt(2.0))
    assert statistics.sample_count.tolist() == [16, 16, 16, 16]
    assert statistics.replica_mean_a == pytest.approx(1.0)
    assert statistics.replica_mean_b == pytest.approx(3.0)
    assert dataset.tile_index(1)["view_index"] == 1
    assert dataset.material_programs[0].metadata["legacy_state_index"] == 0
    assert len(dataset.canonical_material_ir(0).interfaces) == 1

    resumed = convert_legacy_v0_dataset(source, output, resume=True, created_at="later")
    assert resumed.dataset_id == manifest.dataset_id


def test_dataset_cli_and_hash_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "legacy"
    output = tmp_path / "converted"
    _make_legacy_dataset(source)
    assert cli_main(["data", "convert-legacy-v0", str(source), str(output)]) == 0
    capsys.readouterr()
    assert cli_main(["data", "validate", str(output)]) == 0
    assert "ReferenceDataset OK" in capsys.readouterr().out

    light_path = output / "light_directions.npy"
    light_path.write_bytes(light_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="content hash mismatch"):
        ReferenceDataset.open(output)


def test_reference_dataset_schema_tracks_runtime_constants() -> None:
    schema_path = PROJECT_ROOT / "src" / "ncls" / "data" / "schemas" / "reference_dataset_v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["format_name"]["const"] == FORMAT_NAME
    assert schema["properties"]["format_version"]["const"] == FORMAT_VERSION


class _FakeReferenceEvaluator:
    def __init__(self, light_count: int, *, fail_on_call: bool = False) -> None:
        self.light_count = light_count
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def evaluate_tiles(
        self,
        materials,
        view_directions,
        *,
        sample_count_per_replica: int,
        tile_seeds: np.ndarray,
        sample_offset: int = 0,
    ):
        if self.fail_on_call:
            raise AssertionError("completed shards should be resumed before evaluation")
        self.call_count += 1
        shape = (len(materials), self.light_count, 3)
        mean_a = np.ones(shape, dtype=np.float32)
        mean_b = np.full(shape, 3.0, dtype=np.float32)
        return mean_a, np.full(shape, 2.0, dtype=np.float32), mean_b, np.full(shape, 10.0, dtype=np.float32)


def test_v2_generator_writes_true_variance_and_resumes_completed_shards(tmp_path: Path) -> None:
    output = tmp_path / "reference-v2"
    config = ReferenceGenerationConfig(
        family_count=1,
        local_state_count=1,
        view_count=2,
        light_count=4,
        samples_per_replica=8,
        tile_batch=2,
        shard_tiles=1,
        seed=19,
        max_depth=4,
    )
    evaluator = _FakeReferenceEvaluator(config.light_count)
    manifest = generate_reference_dataset(
        output,
        config,
        evaluator=evaluator,
        created_at="2026-08-23T00:00:00+00:00",
        generator_git_commit="test",
    )
    assert evaluator.call_count == 2
    assert len(manifest.shards) == 2
    dataset = ReferenceDataset.open(output)
    statistics = dataset.statistics(0)
    assert statistics.mean == pytest.approx(2.0)
    assert statistics.variance == pytest.approx(2.0)
    assert statistics.standard_error == pytest.approx(np.sqrt(2.0 / 16.0))
    assert statistics.sample_count.tolist() == [16, 16, 16, 16]

    resumed = generate_reference_dataset(
        output,
        config,
        resume=True,
        evaluator=_FakeReferenceEvaluator(config.light_count, fail_on_call=True),
        created_at="later",
        generator_git_commit="test",
    )
    assert resumed.dataset_id == manifest.dataset_id
