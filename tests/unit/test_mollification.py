"""锁定 directional mollification 的冻结协议、方向序列、近邻重建和 tamper gate。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial import cKDTree

from ncls.data.mollification import (
    MollificationProtocol,
    _TrainNeighborhood,
    _audit_report_payload,
    _reference_relative_standard_error,
    _reconstruct_neighborhood,
    _select_dense_lights,
    load_mollification_anchor_lock,
    mollification_cone_directions,
)
from ncls.paths import PROJECT_ROOT
from ncls.data.dataset import RESPONSE_MEASURE
from ncls.data.reference import evaluate_reference_batched_fixed
from ncls.data.mollification_collection import (
    MollificationBudgetExhausted,
    MollificationSupplementBudgetPlan,
    _mollified_shard_semantic_hash,
    _validate_mollified_shard,
    _write_mollified_shard,
    _write_mollification_failure_report,
)
from ncls.learning.data import MollificationCurriculumStore


PROTOCOL_PATH = PROJECT_ROOT / "configs/corpus/layer-stack-p1-mollification-adequacy-v1.json"
PROTOCOL_SHA256 = "7160bd6f210038a6d2f39cac3ec513287b0cacab3ffb94423737c8e4d0cbbec2"
V8_BUDGET_SHA256 = "d430305f92dbf619d51ee3fa6634fc05f900437131ccf48efd2fd56b9e64abfd"


def test_protocol_hash_and_nested_schema_are_frozen() -> None:
    protocol = MollificationProtocol.load(PROTOCOL_PATH)
    assert protocol.sha256 == PROTOCOL_SHA256
    schema = json.loads(
        (PROJECT_ROOT / "src/ncls/data/schemas/mollification_protocol_v1.schema.json")
        .read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    for name in (
        "anchor_selection", "curriculum", "reference_budget", "reconstruction",
        "adequacy_gates", "supplement",
    ):
        nested = schema["properties"][name]
        assert nested["additionalProperties"] is False
        assert set(nested["required"]) == set(nested["properties"])
        assert all("const" in item for item in nested["properties"].values())


def test_mollification_schemas_match_runtime_contract() -> None:
    schema_root = PROJECT_ROOT / "src/ncls/data/schemas"
    corpus_schema = json.loads(
        (schema_root / "mollified_reference_corpus_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert corpus_schema["properties"]["response_measure"]["const"] == RESPONSE_MEASURE

    protocol = MollificationProtocol.load(PROTOCOL_PATH)
    entry_schema = json.loads(
        (schema_root / "mollification_training_data_entry_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    reconstruction = entry_schema["$defs"]["reconstruction"]
    assert set(reconstruction["required"]) == set(protocol.document["reconstruction"])
    assert {
        name: field["const"] for name, field in reconstruction["properties"].items()
    } == protocol.document["reconstruction"]

    for filename, definitions in (
        (
            "mollification_anchor_lock_v1.schema.json",
            ("baseIdentity", "baseShard", "anchor", "source", "light"),
        ),
        (
            "mollification_supplement_anchor_lock_v1.schema.json",
            ("state", "source", "view"),
        ),
    ):
        schema = json.loads((schema_root / filename).read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        for name in definitions:
            nested = schema["$defs"][name]
            assert nested["additionalProperties"] is False
            assert set(nested["required"]) == set(nested["properties"])


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("curriculum", "jitter_count", 255),
        ("reference_budget", "audit_paths_per_jitter_per_replica", 256),
        ("reconstruction", "neighbors", 31),
        ("adequacy_gates", "support_fraction_minimum", 0.9),
        ("supplement", "view_count", 7),
    ),
)
def test_protocol_parser_rejects_threshold_and_budget_tamper(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    document[section][field] = value
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="do(?:es|) not match|must use|frozen"):
        MollificationProtocol.load(target)


def test_protocol_parser_rejects_nested_additional_field(tmp_path: Path) -> None:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    document["curriculum"]["result_driven_override"] = 1
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="fields do not match"):
        MollificationProtocol.load(target)


def test_protocol_parser_accepts_reordered_gate_keys(tmp_path: Path) -> None:
    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    document["adequacy_gates"] = dict(
        reversed(tuple(document["adequacy_gates"].items()))
    )
    target = tmp_path / "reordered.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    assert MollificationProtocol.load(target).sha256 == PROTOCOL_SHA256


def test_anchor_lock_rerun_timestamp_changes_hash_without_known_hash_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ncls.data.mollification as mollification

    protocol = MollificationProtocol.load(PROTOCOL_PATH)
    base_identity = {"corpus_id": "b" * 64}
    fake_manifest = object()
    monkeypatch.setattr(
        mollification,
        "_base_identity",
        lambda _: (fake_manifest, base_identity),
    )
    hashes = []
    for index, timestamp in enumerate(
        ("2026-08-25T00:00:00+00:00", "2026-08-25T00:00:01+00:00")
    ):
        payload = {
            "schema": {"name": "mollification-anchor-lock", "version": 1},
            "protocol_uri": "configs/corpus/layer-stack-p1-mollification-adequacy-v1.json",
            "protocol_sha256": protocol.sha256,
            "base_corpus_uri": protocol.document["base_corpus_uri"],
            "base_identity": base_identity,
            "anchors": [{} for _ in range(24)],
            "frozen_at": timestamp,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload["anchor_lock_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        path = tmp_path / f"lock-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        _, loaded, manifest = load_mollification_anchor_lock(PROTOCOL_PATH, path)
        hashes.append(loaded["anchor_lock_sha256"])
        assert manifest is fake_manifest
    assert hashes[0] != hashes[1]


def test_v8_budget_freezes_multi_source_state_promotion() -> None:
    plan = MollificationSupplementBudgetPlan.load(
        PROJECT_ROOT / "configs/corpus/layer-stack-p1-mollification-reference-se-v8.json"
    )
    assert plan.sha256 == V8_BUDGET_SHA256
    assert plan.document["reference_budget"]["maximum_paths_per_jitter_per_replica"] == 524288
    reuse = plan.document["reuse_policy"]
    assert reuse["name"] == "verified-multi-source-state-shard-reuse-v1"
    assert len(reuse["sources"]) == 2
    assert reuse["promoted_state_ids"] == [
        "1796065779d0932fe7ded3cc2c40b84a8a19190dd7e68732f06bc518ae7fe54a"
    ]


def test_upper_cap_hammersley_is_deterministic_and_inside_measure() -> None:
    center = np.asarray((-0.36717224, -0.91983861, 0.13813573), dtype=np.float64)
    center /= np.linalg.norm(center)
    first = mollification_cone_directions(center, 10.0, 256, 91)
    second = mollification_cone_directions(center, 10.0, 256, 91)
    assert np.array_equal(first, second)
    assert np.all(first[:, 2] > 0.0)
    assert np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0)) < 2e-7
    angles = np.degrees(np.arccos(np.clip(first @ center, -1.0, 1.0)))
    assert np.max(angles) <= 10.0 + 2e-5
    assert len(np.unique(first, axis=0)) == 256


def test_dense_light_selector_uses_frozen_roles_and_tie_break() -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    z = np.linspace(0.02, 0.98, 128)
    wi = np.column_stack((np.sqrt(1.0 - z * z) * np.cos(phi), np.sqrt(1.0 - z * z) * np.sin(phi), z))
    response = np.repeat(np.arange(128, dtype=np.float64)[:, None], 3, axis=1)
    response[20] = response[21]
    protocol = MollificationProtocol.load(PROTOCOL_PATH)
    choices = _select_dense_lights(wi, response, protocol.document["anchor_selection"])
    assert [item["role"] for item in choices] == ["peak", "shoulder", "grazing-light", "background"]
    assert choices[0]["direction_index"] == 127
    assert all(0 <= int(item["direction_index"]) < 128 for item in choices)


def test_knn_reconstruction_exact_sample_and_support_boundary() -> None:
    wo = np.asarray(((0.0, 0.0, 1.0), (0.1, 0.0, math.sqrt(0.99))), dtype=np.float64)
    wi = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.1, math.sqrt(0.99))), dtype=np.float64)
    response = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float64)
    scale_wo = 1.0 / (2.0 * math.sin(math.radians(2.0) / 2.0))
    scale_wi = 1.0 / (2.0 * math.sin(math.radians(1.0) / 2.0))
    embeddings = np.column_stack((scale_wo * wo, scale_wi * wi))
    neighborhood = _TrainNeighborhood(
        cKDTree(embeddings), embeddings, wo, wi, response,
        np.asarray((2, 3)), np.asarray((7, 8)), scale_wo, scale_wi,
    )
    reconstruction = copy.deepcopy(MollificationProtocol.load(PROTOCOL_PATH).document["reconstruction"])
    reconstruction["neighbors"] = 2
    result, support, nearest_wo, nearest_wi, effective = _reconstruct_neighborhood(
        neighborhood, wo[:1], wi[:1], reconstruction
    )
    assert np.array_equal(result[0], response[0].astype(np.float32))
    assert support.tolist() == [1]
    assert nearest_wo[0] == pytest.approx(0.0, abs=1e-12)
    assert nearest_wi[0] == pytest.approx(0.0, abs=1e-12)
    assert effective[0] == pytest.approx(1.0)


def test_relative_se_uses_v5_group_peak_floor() -> None:
    mean = np.asarray(((10.0, 1e-9, 1.0), (2.0, 0.5, 0.25)), dtype=np.float64)
    standard_error = np.full_like(mean, 0.01)
    relative = _reference_relative_standard_error(
        mean, standard_error, group_axes=(0, 1), absolute_floor=1e-6
    )
    assert relative[0, 0] == pytest.approx(0.001)
    assert relative[0, 1] == pytest.approx(0.2)
    assert relative[1, 1] == pytest.approx(0.02)


def test_audit_support_noise_and_numeric_gates_are_independent(tmp_path: Path) -> None:
    protocol = MollificationProtocol.load(PROTOCOL_PATH)
    anchors = [
        {
            "state_id": f"{anchor_index:064x}",
            "representative_role": "control",
            "view_index": anchor_index % 4,
            "lights": [
                {"role": role}
                for role in ("peak", "shoulder", "grazing-light", "background")
            ],
        }
        for anchor_index in range(24)
    ]
    lock = {
        "anchor_lock_sha256": "1" * 64,
        "base_identity": {"corpus_id": "2" * 64},
        "frozen_at": "2026-08-25T00:00:00+00:00",
        "anchors": anchors,
    }
    shape = (24, 4, 2, 4, 3)
    raw = {
        "fresh_mean": np.ones(shape, dtype=np.float32),
        "reconstructed_mean": np.ones(shape, dtype=np.float32),
        "fresh_replica_mean_a": np.ones(shape, dtype=np.float32),
        "fresh_replica_mean_b": np.ones(shape, dtype=np.float32),
        "support": np.ones(shape[:-1], dtype=np.uint8),
        "first_reference_result_at": "2026-08-25T00:00:01+00:00",
        "raw_id": "3" * 64,
        "identity": {
            "reference": {
                "family_id": "ncls.layer-stack@1",
                "reference_id": "ncls.layer-stack-random-walk@1",
                "implementation_sha256": "4" * 64,
            }
        },
    }

    passed = _audit_report_payload(protocol, lock, raw, tmp_path / "raw.h5")
    assert passed["gate_results"] == {
        "support": True,
        "noise": True,
        "numeric": True,
        "repeat": True,
    }
    assert passed["decision"] == "reuse-v5"

    support_raw = {**raw, "support": np.asarray(raw["support"]).copy()}
    support_raw["support"][0, 0, :, 0] = 0
    support_failed = _audit_report_payload(
        protocol, lock, support_raw, tmp_path / "raw.h5"
    )
    assert support_failed["gate_results"] == {
        "support": False,
        "noise": True,
        "numeric": True,
        "repeat": True,
    }

    noise_raw = {
        **raw,
        "fresh_replica_mean_a": np.full(shape, 0.9, dtype=np.float32),
        "fresh_replica_mean_b": np.full(shape, 1.1, dtype=np.float32),
    }
    noise_failed = _audit_report_payload(protocol, lock, noise_raw, tmp_path / "raw.h5")
    assert noise_failed["gate_results"] == {
        "support": True,
        "noise": False,
        "numeric": True,
        "repeat": True,
    }

    numeric_raw = {
        **raw,
        "reconstructed_mean": np.full(shape, 2.0, dtype=np.float32),
    }
    numeric_failed = _audit_report_payload(
        protocol, lock, numeric_raw, tmp_path / "raw.h5"
    )
    assert numeric_failed["gate_results"] == {
        "support": True,
        "noise": True,
        "numeric": False,
        "repeat": True,
    }


class _FakeReferenceEvaluator:
    light_count = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def evaluate_query_groups(
        self,
        materials: list[object],
        view_directions: np.ndarray,
        *,
        sample_count_per_replica: int,
        query_group_seeds: np.ndarray,
        light_directions: np.ndarray | None = None,
        sample_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del materials, view_directions, query_group_seeds, light_directions
        self.calls.append((sample_offset, sample_count_per_replica))
        samples = np.arange(
            sample_offset + 1,
            sample_offset + sample_count_per_replica + 1,
            dtype=np.float64,
        )
        base = np.broadcast_to(samples[:, None, None, None], (len(samples), 1, 2, 3))
        replica_a = base + np.asarray((0.0, 0.25, 0.5))[None, None, None, :]
        replica_b = replica_a + 1.0
        return (
            np.mean(replica_a, axis=0),
            np.mean(replica_a * replica_a, axis=0),
            np.mean(replica_b, axis=0),
            np.mean(replica_b * replica_b, axis=0),
        )


def test_batched_fixed_reference_uses_contiguous_offsets_and_float64_merge() -> None:
    evaluator = _FakeReferenceEvaluator()
    result = evaluate_reference_batched_fixed(
        evaluator,
        [object()],
        np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32),
        query_group_seeds=np.asarray((7,), dtype=np.uint32),
        samples_per_replica=10,
        batch_samples_per_replica=4,
    )
    assert evaluator.calls == [(0, 4), (4, 4), (8, 2)]
    expected_a = np.asarray((5.5, 5.75, 6.0))
    assert np.allclose(result.replica_mean_a[0, 0], expected_a, atol=1e-12)
    assert np.allclose(result.replica_mean_b[0, 1], expected_a + 1.0, atol=1e-12)
    assert result.sample_count.tolist() == [20]
    assert np.all(np.isfinite(result.variance))


def test_curriculum_reader_routes_stored_levels_and_base_v5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ncls.learning.data as learning_data

    state_id = "1" * 64
    shard_path = tmp_path / "mollified.h5"
    wo = np.zeros((8, 3), dtype=np.float32)
    wo[:, 2] = 1.0
    wi = np.zeros((8, 64, 3), dtype=np.float32)
    wi[..., 2] = 1.0
    source = np.full((8, 64, 3), 9.0, dtype=np.float32)
    means = np.empty((8, 4, 64, 3), dtype=np.float32)
    for level_index in range(4):
        means[:, level_index] = float(level_index + 1)
    with h5py.File(shard_path, "w") as stream:
        stream.create_dataset("anchors/wo", data=wo)
        stream.create_dataset("anchors/wi", data=wi)
        stream.create_dataset("anchors/source_response", data=source)
        stream.create_dataset("responses/mean", data=means)
    entry = {
        "variant": "base-v5-plus-mollification-v1",
        "entry_id": "2" * 64,
        "supplement_corpus_uri": "manifest.json",
        "curriculum": {
            "stored_progress": [0.0, 0.25, 0.5, 0.75],
            "stored_radius_degrees": [10.0, 8.535533905932738, 5.0, 1.4644660940672627],
            "zero_radius_switch_progress": 0.875,
        },
    }
    manifest = {"shards": [{"state_id": state_id, "uri": shard_path.name}]}
    monkeypatch.setattr(learning_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(learning_data, "load_mollification_training_data_entry", lambda _: entry)
    monkeypatch.setattr(learning_data, "validate_mollification_supplement", lambda _: manifest)
    with MollificationCurriculumStore(tmp_path / "entry.json") as store:
        assert store.state_count == 1
        mollified = store.batch(
            [state_id], [2], [3], training_progress=0.3
        )
        assert mollified["target_source"].tolist() == ["mollified-reference"]
        assert mollified["mollification_level_progress"].tolist() == pytest.approx([0.25])
        assert np.allclose(mollified["response"], 2.0)
        last_positive = store.batch(
            [state_id], [2], [3], training_progress=0.874
        )
        assert last_positive["target_source"].tolist() == ["mollified-reference"]
        assert last_positive["mollification_radius_degrees"][0] > 0.0
        base = store.batch(
            [state_id], [2], [3], training_progress=0.875
        )
        assert base["target_source"].tolist() == ["base-v5"]
        assert base["mollification_radius_degrees"].tolist() == [0.0]
        assert np.allclose(base["response"], 9.0)


def test_budget_failure_report_preserves_target_attempts(tmp_path: Path) -> None:
    diagnostic = {
        "state_id": "5" * 64,
        "view_index": 6,
        "level_index": 0,
        "radius_degrees": 10.0,
        "status": "failed",
        "attempts": [
            {
                "paths_per_jitter_per_replica": 524288,
                "combined_reference_samples_per_target": 268435456,
                "relative_se_p95": 0.07,
                "relative_se_max": 0.2,
                "passed": False,
            }
        ],
    }
    error = MollificationBudgetExhausted("frozen budget exhausted", [diagnostic])
    path = tmp_path / "failure.json"
    report = _write_mollification_failure_report(
        path,
        protocol_sha256="1" * 64,
        supplement_anchor_lock_sha256="2" * 64,
        budget_plan_sha256="3" * 64,
        collection_lock_sha256="4" * 64,
        state_id="5" * 64,
        error=error,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == report
    assert loaded["targets"] == [diagnostic]
    payload = dict(loaded)
    stored = payload.pop("report_sha256")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == stored


def test_mollified_shard_layout_identity_and_tamper_gate(tmp_path: Path) -> None:
    path = tmp_path / "shard.h5"
    wo = np.zeros((8, 3), dtype=np.float32)
    wo[:, 2] = 1.0
    wi = np.zeros((8, 64, 3), dtype=np.float32)
    wi[..., 2] = 1.0
    identity = {
        "protocol_sha256": "1" * 64,
        "anchor_lock_sha256": "2" * 64,
        "supplement_anchor_lock_sha256": "3" * 64,
        "supplement_budget_plan_sha256": "4" * 64,
        "collection_lock_sha256": "5" * 64,
        "base_corpus_id": "6" * 64,
        "selection_sha256": "7" * 64,
        "state_id": "8" * 64,
        "structure_family_id": "layers-01-test",
        "source_dataset_id": "9" * 64,
        "reference": {
            "family_id": "ncls.layer-stack@1",
            "reference_id": "ncls.layer-stack-random-walk@1",
            "implementation_sha256": "a" * 64,
        },
        "response_measure": "rgb-bsdf-times-absolute-shading-normal-light-cosine",
        "jitter": {
            "progress": [0.0, 0.25, 0.5, 0.75],
            "radius_degrees": [10.0, 8.535533905932738, 5.0, 1.4644660940672627],
            "jitter_count": 256,
            "mollified_direction": "wo",
            "sequence": "scrambled-hammersley-upper-cap-v1",
            "zero_radius_switch_progress": 0.875,
        },
        "reference_budget": {
            "initial_paths_per_jitter_per_replica": 64,
            "maximum_paths_per_jitter_per_replica": 512,
            "target_relative_se_p95": 0.06,
            "maximum_group_relative_se": 0.25,
            "jitter_count": 256,
            "replica_count": 2,
            "maximum_path_depth": 64,
        },
    }
    response_shape = (8, 4, 64, 3)
    result = _write_mollified_shard(
        path,
        identity,
        {
            "wo": wo,
            "wi": wi,
            "source_response": np.ones((8, 64, 3), dtype=np.float32),
            "source_group_index": np.arange(8, dtype=np.uint32),
            "source_direction_index": np.zeros((8, 64), dtype=np.uint32),
        },
        {
            "progress": np.asarray((0.0, 0.25, 0.5, 0.75), dtype=np.float32),
            "radius_degrees": np.asarray(
                (10.0, 8.535533905932738, 5.0, 1.4644660940672627),
                dtype=np.float32,
            ),
        },
        {
            "mean": np.ones(response_shape, dtype=np.float32),
            "variance": np.full(response_shape, 0.0025, dtype=np.float32),
            "replica_mean_a": np.full(response_shape, 0.95, dtype=np.float32),
            "replica_mean_b": np.full(response_shape, 1.05, dtype=np.float32),
            "sample_count": np.full((8, 4, 64), 32768, dtype=np.uint32),
            "relative_se_p95": np.full((8, 4), 0.05, dtype=np.float32),
            "relative_se_max": np.full((8, 4), 0.05, dtype=np.float32),
        },
    )
    assert result["target_count"] == 2048
    assert _validate_mollified_shard(path, identity)["dataset_id"] == result["dataset_id"]
    with h5py.File(path, "r+") as stream:
        stream["responses/mean"][0, 0, 0, 0] = np.float32(2.0)
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        _validate_mollified_shard(path, identity)


def test_mollified_shard_recomputes_noise_summary_after_semantic_rehash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.h5"
    wo = np.zeros((8, 3), dtype=np.float32)
    wo[:, 2] = 1.0
    wi = np.zeros((8, 64, 3), dtype=np.float32)
    wi[..., 2] = 1.0
    identity = {
        "protocol_sha256": "1" * 64,
        "anchor_lock_sha256": "2" * 64,
        "supplement_anchor_lock_sha256": "3" * 64,
        "supplement_budget_plan_sha256": "4" * 64,
        "collection_lock_sha256": "5" * 64,
        "base_corpus_id": "6" * 64,
        "selection_sha256": "7" * 64,
        "state_id": "8" * 64,
        "structure_family_id": "layers-01-test",
        "source_dataset_id": "9" * 64,
        "reference": {
            "family_id": "ncls.layer-stack@1",
            "reference_id": "ncls.layer-stack-random-walk@1",
            "implementation_sha256": "a" * 64,
        },
        "response_measure": "rgb-bsdf-times-absolute-shading-normal-light-cosine",
        "jitter": {
            "progress": [0.0, 0.25, 0.5, 0.75],
            "radius_degrees": [10.0, 8.535533905932738, 5.0, 1.4644660940672627],
            "jitter_count": 256,
            "mollified_direction": "wo",
            "sequence": "scrambled-hammersley-upper-cap-v1",
            "zero_radius_switch_progress": 0.875,
        },
        "reference_budget": {
            "initial_paths_per_jitter_per_replica": 64,
            "maximum_paths_per_jitter_per_replica": 512,
            "target_relative_se_p95": 0.06,
            "maximum_group_relative_se": 0.25,
            "jitter_count": 256,
            "replica_count": 2,
            "maximum_path_depth": 64,
        },
    }
    response_shape = (8, 4, 64, 3)
    _write_mollified_shard(
        path,
        identity,
        {
            "wo": wo,
            "wi": wi,
            "source_response": np.ones((8, 64, 3), dtype=np.float32),
            "source_group_index": np.arange(8, dtype=np.uint32),
            "source_direction_index": np.zeros((8, 64), dtype=np.uint32),
        },
        {
            "progress": np.asarray((0.0, 0.25, 0.5, 0.75), dtype=np.float32),
            "radius_degrees": np.asarray(
                (10.0, 8.535533905932738, 5.0, 1.4644660940672627),
                dtype=np.float32,
            ),
        },
        {
            "mean": np.ones(response_shape, dtype=np.float32),
            "variance": np.full(response_shape, 0.0025, dtype=np.float32),
            "replica_mean_a": np.full(response_shape, 0.95, dtype=np.float32),
            "replica_mean_b": np.full(response_shape, 1.05, dtype=np.float32),
            "sample_count": np.full((8, 4, 64), 32768, dtype=np.uint32),
            "relative_se_p95": np.full((8, 4), 0.05, dtype=np.float32),
            "relative_se_max": np.full((8, 4), 0.05, dtype=np.float32),
        },
    )
    with h5py.File(path, "r+") as stream:
        stream["responses/relative_se_p95"][...] = np.float32(0.0)
        stream.attrs["dataset_id"] = _mollified_shard_semantic_hash(stream)
    with pytest.raises(ValueError, match="relative SE p95 summary mismatch"):
        _validate_mollified_shard(path, identity)
