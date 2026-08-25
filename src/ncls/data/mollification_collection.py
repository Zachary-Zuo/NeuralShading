from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .corpus import ReferenceCorpusManifest, validate_reference_corpus
from .mollification import (
    MOLLIFICATION_RESPONSE_MEASURE,
    MOLLIFIED_CORPUS_FORMAT,
    MOLLIFIED_CORPUS_VERSION,
    MOLLIFIED_SHARD_FORMAT,
    MOLLIFIED_SHARD_VERSION,
    SUPPLEMENT_BUDGET_SCHEMA_NAME,
    SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V1,
    SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2,
    SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3,
    MollificationProtocol,
    _canonical_json,
    _make_layer_stack_provider,
    _project_uri,
    _reference_relative_standard_error,
    _require_exact_fields,
    _require_sha256,
    _resolve_uri,
    _seed32,
    _sha256_file,
    _sha256_json,
    _write_json_atomic,
    load_mollification_supplement_anchor_lock,
    mollification_cone_directions,
)
@dataclass(frozen=True)
class MollificationSupplementBudgetPlan:
    document: Mapping[str, Any]
    source_path: Path

    def __post_init__(self) -> None:
        value = dict(self.document)
        schema_value = value.get("schema")
        schema_version = (
            int(schema_value.get("version", -1))
            if isinstance(schema_value, Mapping) else -1
        )
        expected_fields = {
            "schema", "name", "protocol_sha256", "audit_report_sha256",
            "supplement_anchor_lock_sha256", "predecessor_failure",
            "derivation", "reference_budget",
        }
        if schema_version >= 5:
            expected_fields.add("accumulator")
        if schema_version >= 6:
            expected_fields.add("reuse_policy")
        _require_exact_fields(
            "MollificationSupplementBudgetPlan",
            value,
            expected_fields,
        )
        schema = value["schema"]
        if (
            not isinstance(schema, Mapping)
            or set(schema) != {"name", "version"}
            or schema["name"] != SUPPLEMENT_BUDGET_SCHEMA_NAME
            or int(schema["version"]) not in {1, 2, 3, 4, 5, 6, 7}
            or not str(value["name"]).startswith(
            "layer-stack-p1-mollification-reference-se-v"
            )
        ):
            raise ValueError("unsupported mollification supplement budget plan")
        for name in (
            "protocol_sha256", "audit_report_sha256", "supplement_anchor_lock_sha256"
        ):
            _require_sha256(name, value[name])
        failure = value["predecessor_failure"]
        _require_exact_fields(
            "predecessor_failure",
            failure,
            {
                "name", "maximum_paths_per_jitter_per_replica", "state_id",
                "view_index", "level_index", "observed_relative_se_p95",
                "observed_relative_se_max", "published", "directory",
            },
        )
        _require_sha256("predecessor failure state_id", failure["state_id"])
        if (
            int(failure["maximum_paths_per_jitter_per_replica"]) < 1
            or int(failure["view_index"]) < 0
            or int(failure["level_index"]) < 0
            or float(failure["observed_relative_se_p95"]) <= 0.0
            or float(failure["observed_relative_se_max"]) <= 0.0
            or bool(failure["published"])
            or not str(failure["directory"])
        ):
            raise ValueError("mollification predecessor failure evidence is invalid")
        budget = value["reference_budget"]
        _require_exact_fields(
            "supplement reference_budget",
            budget,
            {
                "initial_paths_per_jitter_per_replica", "maximum_paths_per_jitter_per_replica",
                "target_relative_se_p95", "maximum_group_relative_se", "jitter_count",
                "replica_count",
            },
        )
        if (
            int(budget["initial_paths_per_jitter_per_replica"]) != 64
            or float(budget["target_relative_se_p95"]) != 0.06
            or float(budget["maximum_group_relative_se"]) != 0.25
            or int(budget["jitter_count"]) != 256
            or int(budget["replica_count"]) != 2
        ):
            raise ValueError("mollification supplement budget may only refine the sample cap")
        derivation = value["derivation"]
        _require_exact_fields(
            "supplement budget derivation",
            derivation,
            {"rule", "estimated_required_paths", "rounded_power_of_two"},
        )
        rule = str(derivation["rule"])
        if rule not in {
            "ceil-power-of-two((observed_p95/target_p95)^2*previous_cap)",
            "ceil-power-of-two((observed_max/hard_max)^2*previous_cap)",
            "replace-full-dispatch-with-batched-fixed-without-changing-cap",
        }:
            raise ValueError("unsupported mollification budget derivation rule")
        if rule == "replace-full-dispatch-with-batched-fixed-without-changing-cap":
            estimated = int(failure["maximum_paths_per_jitter_per_replica"])
            rounded = estimated
        else:
            observed = (
                float(failure["observed_relative_se_p95"])
                if "observed_p95" in rule
                else float(failure["observed_relative_se_max"])
            )
            target = (
                float(budget["target_relative_se_p95"])
                if "target_p95" in rule
                else float(budget["maximum_group_relative_se"])
            )
            estimated = math.ceil(
                (observed / target) ** 2
                * int(failure["maximum_paths_per_jitter_per_replica"])
            )
            rounded = 1 << (estimated - 1).bit_length()
        if (
            int(derivation["estimated_required_paths"]) != estimated
            or int(derivation["rounded_power_of_two"]) != rounded
            or int(budget["maximum_paths_per_jitter_per_replica"]) != rounded
        ):
            raise ValueError("mollification supplement budget derivation does not match its evidence")
        if schema_version >= 5:
            accumulator = value["accumulator"]
            _require_exact_fields(
                "supplement budget accumulator",
                accumulator,
                {
                    "name", "gpu_batch_samples_per_replica", "sample_offset",
                    "cpu_accumulator", "implementation_uri", "implementation_sha256",
                },
            )
            if (
                accumulator["name"] != "batched-fixed-float64-welford-v1"
                or int(accumulator["gpu_batch_samples_per_replica"]) != 256
                or accumulator["sample_offset"] != "contiguous-zero-based-v1"
                or accumulator["cpu_accumulator"] != "float64-parallel-welford-v1"
            ):
                raise ValueError("unsupported mollification supplement accumulator")
            implementation_path = _resolve_uri(str(accumulator["implementation_uri"]))
            if _sha256_file(implementation_path) != _require_sha256(
                "accumulator implementation_sha256", accumulator["implementation_sha256"]
            ):
                raise ValueError("mollification supplement accumulator implementation hash mismatch")
        if schema_version >= 6:
            reuse = value["reuse_policy"]
            if schema_version == 6:
                _require_exact_fields(
                    "supplement budget reuse_policy",
                    reuse,
                    {
                        "name", "source_budget_plan_uri", "source_budget_plan_sha256",
                        "source_collection_lock_uri", "source_collection_lock_sha256",
                        "source_shard_root", "promoted_state_ids",
                    },
                )
                if reuse["name"] != "verified-state-shard-reuse-v1":
                    raise ValueError("unsupported mollification supplement reuse policy")
                sources = (reuse,)
            else:
                _require_exact_fields(
                    "supplement budget reuse_policy",
                    reuse,
                    {"name", "promoted_state_ids", "sources"},
                )
                if reuse["name"] != "verified-multi-source-state-shard-reuse-v1":
                    raise ValueError("unsupported mollification supplement reuse policy")
                sources = tuple(reuse["sources"])
                if len(sources) < 2:
                    raise ValueError("multi-source reuse requires at least two frozen sources")
            promoted = tuple(map(str, reuse["promoted_state_ids"]))
            if not promoted or len(set(promoted)) != len(promoted):
                raise ValueError("mollification promoted state IDs must be unique and non-empty")
            for state_id in promoted:
                _require_sha256("promoted state_id", state_id)
            if str(failure["state_id"]) not in promoted:
                raise ValueError("predecessor failure state must be explicitly promoted")
            source_keys: set[tuple[str, str]] = set()
            for source in sources:
                source_fields = {
                    "source_budget_plan_uri", "source_budget_plan_sha256",
                    "source_collection_lock_uri", "source_collection_lock_sha256",
                    "source_shard_root",
                }
                if schema_version == 6:
                    source_fields |= {"name", "promoted_state_ids"}
                _require_exact_fields("supplement reuse source", source, source_fields)
                for name in ("source_budget_plan_sha256", "source_collection_lock_sha256"):
                    _require_sha256(name, source[name])
                source_budget_path = _resolve_uri(str(source["source_budget_plan_uri"]))
                source_collection_path = _resolve_uri(str(source["source_collection_lock_uri"]))
                source_shard_root = _resolve_uri(str(source["source_shard_root"]))
                key = (str(source_budget_path.resolve()), str(source_shard_root.resolve()))
                if key in source_keys:
                    raise ValueError("mollification reuse sources must be unique")
                source_keys.add(key)
                if (
                    _sha256_json(json.loads(source_budget_path.read_text(encoding="utf-8")))
                    != source["source_budget_plan_sha256"]
                    or not source_collection_path.is_file()
                    or not source_shard_root.is_dir()
                ):
                    raise ValueError("mollification reuse source identity mismatch")
                source_collection = json.loads(source_collection_path.read_text(encoding="utf-8"))
                source_collection_hash = source_collection.get("collection_lock_sha256", "")
                source_collection_payload = dict(source_collection)
                source_collection_payload.pop("collection_lock_sha256", None)
                if (
                    source_collection_hash != source["source_collection_lock_sha256"]
                    or _sha256_json(source_collection_payload) != source_collection_hash
                ):
                    raise ValueError("mollification reuse source collection lock mismatch")

    @property
    def sha256(self) -> str:
        return _sha256_json(self.document)

    @classmethod
    def load(cls, path: Path | str) -> "MollificationSupplementBudgetPlan":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("MollificationSupplementBudgetPlan root must be an object")
        return cls(dict(value), source.resolve())


class MollificationBudgetExhausted(RuntimeError):
    """携带一次 frozen state collection 的逐 target 自适应预算证据。"""

    def __init__(self, message: str, diagnostics: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.diagnostics = tuple(dict(item) for item in diagnostics)


def freeze_mollification_supplement_collection(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    audit_report_path: Path | str,
    supplement_lock_path: Path | str,
    budget_plan_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    protocol, anchor_lock, manifest, supplement_lock = load_mollification_supplement_anchor_lock(
        protocol_path, anchor_lock_path, audit_report_path, supplement_lock_path
    )
    budget = MollificationSupplementBudgetPlan.load(budget_plan_path)
    if (
        budget.document["protocol_sha256"] != protocol.sha256
        or budget.document["audit_report_sha256"] != supplement_lock["audit_report_sha256"]
        or budget.document["supplement_anchor_lock_sha256"]
        != supplement_lock["supplement_anchor_lock_sha256"]
    ):
        raise ValueError("mollification supplement budget plan provenance mismatch")
    failure_directory = _resolve_uri(str(budget.document["predecessor_failure"]["directory"]))
    failure_files = tuple(sorted(path for path in failure_directory.glob("*.h5") if path.is_file()))
    if not failure_files:
        raise ValueError("mollification supplement budget plan has no predecessor failure evidence")
    states = tuple(supplement_lock["states"])
    state_ids = tuple(item["state_id"] for item in states)
    provider, _ = _make_layer_stack_provider(
        protocol,
        manifest,
        state_ids,
        fixed_samples_per_replica=int(
            budget.document["reference_budget"]["maximum_paths_per_jitter_per_replica"]
        ),
    )
    reference_implementation_sha256 = provider.descriptor.implementation_sha256
    reused_shards: list[dict[str, Any]] = []
    collection_state_ids = list(state_ids)
    promoted_state_ids: list[str] = []
    schema = SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V1
    if int(budget.document["schema"]["version"]) >= 6:
        schema = (
            SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3
            if int(budget.document["schema"]["version"]) >= 7
            else SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2
        )
        reused_shards = _validated_reuse_entries(
            protocol,
            anchor_lock,
            manifest,
            supplement_lock,
            budget,
            provider,
        )
        reused_state_ids = {item["state_id"] for item in reused_shards}
        collection_state_ids = [state_id for state_id in state_ids if state_id not in reused_state_ids]
        promoted_state_ids = list(
            map(str, budget.document["reuse_policy"]["promoted_state_ids"])
        )
        if not set(promoted_state_ids).issubset(collection_state_ids):
            raise ValueError("promoted states must not have a reusable passing shard")
        if not collection_state_ids:
            raise ValueError("mollification reuse policy has no states left to collect")
    provider.close()
    value: dict[str, Any] = {
        "schema": schema,
        "budget_plan_uri": _project_uri(budget.source_path),
        "budget_plan_sha256": budget.sha256,
        "protocol_sha256": protocol.sha256,
        "audit_report_sha256": supplement_lock["audit_report_sha256"],
        "supplement_anchor_lock_sha256": supplement_lock["supplement_anchor_lock_sha256"],
        "base_corpus_id": manifest.corpus_id,
        "state_ids": list(state_ids),
        "reference_implementation_sha256": reference_implementation_sha256,
        "predecessor_failure": {
            "directory": _project_uri(failure_directory),
            "file_count": len(failure_files),
            "total_bytes": int(sum(path.stat().st_size for path in failure_files)),
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    if schema in (SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2, SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3):
        value["collection_state_ids"] = collection_state_ids
        value["promoted_state_ids"] = promoted_state_ids
        value["reused_shards"] = reused_shards
    value["collection_lock_sha256"] = _sha256_json(value)
    _write_json_atomic(Path(output_path), value)
    return value


def load_mollification_supplement_collection_lock(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    audit_report_path: Path | str,
    supplement_lock_path: Path | str,
    budget_plan_path: Path | str,
    collection_lock_path: Path | str,
) -> tuple[
    MollificationProtocol,
    dict[str, Any],
    ReferenceCorpusManifest,
    dict[str, Any],
    MollificationSupplementBudgetPlan,
    dict[str, Any],
]:
    protocol, anchor_lock, manifest, supplement_lock = load_mollification_supplement_anchor_lock(
        protocol_path, anchor_lock_path, audit_report_path, supplement_lock_path
    )
    budget = MollificationSupplementBudgetPlan.load(budget_plan_path)
    value = json.loads(Path(collection_lock_path).read_text(encoding="utf-8"))
    stored = _require_sha256("collection_lock_sha256", value.get("collection_lock_sha256", ""))
    payload = dict(value)
    payload.pop("collection_lock_sha256")
    if _sha256_json(payload) != stored:
        raise ValueError("mollification supplement collection lock hash mismatch")
    schema = value.get("schema")
    common_fields = {
        "schema", "budget_plan_uri", "budget_plan_sha256", "protocol_sha256",
        "audit_report_sha256", "supplement_anchor_lock_sha256", "base_corpus_id",
        "state_ids", "reference_implementation_sha256", "predecessor_failure",
        "frozen_at", "collection_lock_sha256",
    }
    if schema == SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V1:
        _require_exact_fields("mollification supplement collection lock", value, common_fields)
    elif schema in (SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2, SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3):
        _require_exact_fields(
            "mollification supplement collection lock",
            value,
            common_fields | {"collection_state_ids", "promoted_state_ids", "reused_shards"},
        )
    else:
        raise ValueError("unsupported mollification supplement collection lock")
    if (
        value.get("budget_plan_sha256") != budget.sha256
        or value.get("protocol_sha256") != protocol.sha256
        or value.get("audit_report_sha256") != supplement_lock["audit_report_sha256"]
        or value.get("supplement_anchor_lock_sha256")
        != supplement_lock["supplement_anchor_lock_sha256"]
        or value.get("base_corpus_id") != manifest.corpus_id
        or value.get("state_ids") != [item["state_id"] for item in supplement_lock["states"]]
    ):
        raise ValueError("mollification supplement collection lock provenance mismatch")
    collection_state_ids = list(value["state_ids"])
    if schema in (SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2, SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3):
        collection_state_ids = list(map(str, value["collection_state_ids"]))
        promoted_state_ids = list(map(str, value["promoted_state_ids"]))
        reused_state_ids = [str(item.get("state_id", "")) for item in value["reused_shards"]]
        if (
            len(set(collection_state_ids)) != len(collection_state_ids)
            or len(set(reused_state_ids)) != len(reused_state_ids)
            or set(collection_state_ids).intersection(reused_state_ids)
            or set(collection_state_ids).union(reused_state_ids) != set(value["state_ids"])
            or promoted_state_ids != list(
                map(str, budget.document["reuse_policy"]["promoted_state_ids"])
            )
            or not set(promoted_state_ids).issubset(collection_state_ids)
        ):
            raise ValueError("mollification collection/reuse state partition mismatch")
    provider, _ = _make_layer_stack_provider(
        protocol,
        manifest,
        tuple(collection_state_ids),
        fixed_samples_per_replica=int(
            budget.document["reference_budget"]["maximum_paths_per_jitter_per_replica"]
        ),
    )
    current_implementation = provider.descriptor.implementation_sha256
    provider.close()
    if value.get("reference_implementation_sha256") != current_implementation:
        raise ValueError("mollification supplement collection lock reference implementation mismatch")
    if schema in (SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2, SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3):
        verification_provider, _ = _make_layer_stack_provider(
            protocol,
            manifest,
            tuple(value["state_ids"]),
            fixed_samples_per_replica=int(
                budget.document["reference_budget"]["maximum_paths_per_jitter_per_replica"]
            ),
        )
        try:
            verified = _validated_reuse_entries(
                protocol,
                anchor_lock,
                manifest,
                supplement_lock,
                budget,
                verification_provider,
            )
        finally:
            verification_provider.close()
        if verified != value["reused_shards"]:
            raise ValueError("mollification reused shard evidence changed after collection freeze")
    return protocol, anchor_lock, manifest, supplement_lock, budget, dict(value)


_MOLLIFIED_DATASETS = (
    "anchors/wo", "anchors/wi", "anchors/source_response",
    "anchors/source_group_index", "anchors/source_direction_index",
    "curriculum/progress", "curriculum/radius_degrees",
    "responses/mean", "responses/variance", "responses/replica_mean_a",
    "responses/replica_mean_b", "responses/sample_count",
    "responses/relative_se_p95", "responses/relative_se_max",
)

_MOLLIFICATION_RELATIVE_SE_ABSOLUTE_FLOOR = 1e-6


def _mollified_shard_semantic_hash(stream: h5py.File) -> str:
    digest = hashlib.sha256()
    digest.update(str(stream.attrs["identity_json"]).encode("utf-8"))
    for name in _MOLLIFIED_DATASETS:
        dataset = stream[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(dataset.dtype).encode("ascii"))
        digest.update(_canonical_json(dataset.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(dataset[...]).tobytes())
    return digest.hexdigest()


def _validate_mollified_shard(
    path: Path,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as stream:
        if (
            str(stream.attrs.get("format_name", "")) != MOLLIFIED_SHARD_FORMAT
            or int(stream.attrs.get("format_version", -1)) != MOLLIFIED_SHARD_VERSION
            or str(stream.attrs.get("response_measure", "")) != MOLLIFICATION_RESPONSE_MEASURE
        ):
            raise ValueError("unsupported mollified-reference-shard")
        identity = json.loads(str(stream.attrs.get("identity_json", "{}")))
        _require_exact_fields(
            "mollified-reference-shard identity",
            identity,
            {
                "protocol_sha256", "anchor_lock_sha256",
                "supplement_anchor_lock_sha256", "supplement_budget_plan_sha256",
                "collection_lock_sha256", "base_corpus_id", "selection_sha256",
                "state_id", "structure_family_id", "source_dataset_id", "reference",
                "response_measure", "jitter", "reference_budget",
            },
        )
        if identity["response_measure"] != MOLLIFICATION_RESPONSE_MEASURE:
            raise ValueError("mollified-reference-shard identity measure mismatch")
        expected_jitter = {
            "progress": [0.0, 0.25, 0.5, 0.75],
            "radius_degrees": [10.0, 8.535533905932738, 5.0, 1.4644660940672627],
            "jitter_count": 256,
            "mollified_direction": "wo",
            "sequence": "scrambled-hammersley-upper-cap-v1",
            "zero_radius_switch_progress": 0.875,
        }
        if identity["jitter"] != expected_jitter:
            raise ValueError("mollified-reference-shard jitter contract mismatch")
        reference_budget = identity["reference_budget"]
        _require_exact_fields(
            "mollified-reference-shard reference budget",
            reference_budget,
            {
                "initial_paths_per_jitter_per_replica",
                "maximum_paths_per_jitter_per_replica", "target_relative_se_p95",
                "maximum_group_relative_se", "jitter_count", "replica_count",
                "maximum_path_depth",
            },
        )
        if (
            int(reference_budget["initial_paths_per_jitter_per_replica"]) != 64
            or float(reference_budget["target_relative_se_p95"]) != 0.06
            or float(reference_budget["maximum_group_relative_se"]) != 0.25
            or int(reference_budget["jitter_count"]) != 256
            or int(reference_budget["replica_count"]) != 2
            or int(reference_budget["maximum_path_depth"]) != 64
        ):
            raise ValueError("mollified-reference-shard reference budget contract mismatch")
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("existing mollified-reference-shard has a different identity")
        if any(name not in stream for name in _MOLLIFIED_DATASETS):
            raise ValueError("mollified-reference-shard is incomplete")
        dataset_id = _require_sha256("mollified shard dataset_id", stream.attrs.get("dataset_id", ""))
        if _mollified_shard_semantic_hash(stream) != dataset_id:
            raise ValueError("mollified-reference-shard semantic hash mismatch")
        expected_shapes = {
            "anchors/wo": (8, 3),
            "anchors/wi": (8, 64, 3),
            "anchors/source_response": (8, 64, 3),
            "anchors/source_group_index": (8,),
            "anchors/source_direction_index": (8, 64),
            "curriculum/progress": (4,),
            "curriculum/radius_degrees": (4,),
            "responses/mean": (8, 4, 64, 3),
            "responses/variance": (8, 4, 64, 3),
            "responses/replica_mean_a": (8, 4, 64, 3),
            "responses/replica_mean_b": (8, 4, 64, 3),
            "responses/sample_count": (8, 4, 64),
            "responses/relative_se_p95": (8, 4),
            "responses/relative_se_max": (8, 4),
        }
        for name, shape in expected_shapes.items():
            if stream[name].shape != shape:
                raise ValueError(f"mollified-reference-shard {name} shape mismatch")
        if not np.array_equal(
            np.asarray(stream["curriculum/progress"]),
            np.asarray(expected_jitter["progress"], dtype=np.float32),
        ) or not np.array_equal(
            np.asarray(stream["curriculum/radius_degrees"]),
            np.asarray(expected_jitter["radius_degrees"], dtype=np.float32),
        ):
            raise ValueError("mollified-reference-shard curriculum levels mismatch")
        for name in _MOLLIFIED_DATASETS:
            values = np.asarray(stream[name])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"mollified-reference-shard {name} contains non-finite values")
        for name in ("anchors/wo", "anchors/wi"):
            lengths = np.linalg.norm(np.asarray(stream[name], dtype=np.float64), axis=-1)
            if np.max(np.abs(lengths - 1.0)) > 2e-4:
                raise ValueError(f"mollified-reference-shard {name} contains non-unit directions")
        if np.any(np.asarray(stream["responses/variance"]) < 0.0):
            raise ValueError("mollified-reference-shard variance must be nonnegative")
        sample_count = np.asarray(stream["responses/sample_count"], dtype=np.uint64)
        if np.any(sample_count < 1):
            raise ValueError("mollified-reference-shard sample_count must be positive")
        mean = np.asarray(stream["responses/mean"], dtype=np.float64)
        replica_a = np.asarray(stream["responses/replica_mean_a"], dtype=np.float64)
        replica_b = np.asarray(stream["responses/replica_mean_b"], dtype=np.float64)
        reconstructed_mean = 0.5 * (replica_a + replica_b)
        if not np.allclose(mean, reconstructed_mean, rtol=2e-6, atol=2e-8):
            raise ValueError("mollified-reference-shard mean disagrees with replica means")
        reconstructed_variance = 0.5 * (
            (replica_a - reconstructed_mean) ** 2
            + (replica_b - reconstructed_mean) ** 2
        )
        if not np.allclose(
            np.asarray(stream["responses/variance"], dtype=np.float64),
            reconstructed_variance,
            rtol=1e-4,
            atol=3e-7,
        ):
            raise ValueError("mollified-reference-shard variance disagrees with replica means")
        relative_se = _reference_relative_standard_error(
            reconstructed_mean,
            0.5 * np.abs(replica_a - replica_b),
            group_axes=(2, 3),
            absolute_floor=_MOLLIFICATION_RELATIVE_SE_ABSOLUTE_FLOOR,
        )
        reconstructed_p95 = np.quantile(relative_se, 0.95, axis=(2, 3))
        reconstructed_max = np.max(relative_se, axis=(2, 3))
        stored_p95 = np.asarray(stream["responses/relative_se_p95"], dtype=np.float64)
        stored_max = np.asarray(stream["responses/relative_se_max"], dtype=np.float64)
        if not np.allclose(stored_p95, reconstructed_p95, rtol=3e-5, atol=2e-7):
            raise ValueError("mollified-reference-shard relative SE p95 summary mismatch")
        if not np.allclose(stored_max, reconstructed_max, rtol=3e-5, atol=2e-7):
            raise ValueError("mollified-reference-shard relative SE max summary mismatch")
        return {
            "dataset_id": dataset_id,
            "identity": identity,
            "created_at": str(stream.attrs["created_at"]),
            "target_count": int(np.prod(stream["responses/sample_count"].shape)),
            "combined_reference_samples": int(np.sum(stream["responses/sample_count"], dtype=np.uint64)),
            "maximum_relative_se_p95": float(np.max(reconstructed_p95)),
            "maximum_relative_se": float(np.max(reconstructed_max)),
        }


def _validated_reuse_entries(
    protocol: MollificationProtocol,
    anchor_lock: Mapping[str, Any],
    manifest: ReferenceCorpusManifest,
    supplement_lock: Mapping[str, Any],
    budget: MollificationSupplementBudgetPlan,
    provider: Any,
) -> list[dict[str, Any]]:
    """逐 state 验证 predecessor shard；缺失 state 留给新 collection。"""

    reuse = budget.document["reuse_policy"]
    states = tuple(supplement_lock["states"])
    sources = (
        tuple(reuse["sources"])
        if reuse["name"] == "verified-multi-source-state-shard-reuse-v1"
        else (reuse,)
    )
    entries_by_state: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_budget_path = _resolve_uri(str(source["source_budget_plan_uri"]))
        source_collection_path = _resolve_uri(str(source["source_collection_lock_uri"]))
        source_budget = MollificationSupplementBudgetPlan.load(source_budget_path)
        if source_budget.sha256 != source["source_budget_plan_sha256"]:
            raise ValueError("mollification reuse source budget hash mismatch")
        source_collection = json.loads(source_collection_path.read_text(encoding="utf-8"))
        source_collection_hash = _require_sha256(
            "source collection_lock_sha256",
            source_collection.get("collection_lock_sha256", ""),
        )
        source_collection_payload = dict(source_collection)
        source_collection_payload.pop("collection_lock_sha256")
        if (
            _sha256_json(source_collection_payload) != source_collection_hash
            or source_collection.get("schema") not in (
                SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V1,
                SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2,
                SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3,
            )
            or source_collection.get("budget_plan_sha256") != source_budget.sha256
            or source_collection.get("protocol_sha256") != protocol.sha256
            or source_collection.get("audit_report_sha256")
            != supplement_lock["audit_report_sha256"]
            or source_collection.get("supplement_anchor_lock_sha256")
            != supplement_lock["supplement_anchor_lock_sha256"]
            or source_collection.get("base_corpus_id") != manifest.corpus_id
            or source_collection.get("state_ids")
            != [item["state_id"] for item in states]
            or source_collection.get("reference_implementation_sha256")
            != provider.descriptor.implementation_sha256
            or source_collection_hash != source["source_collection_lock_sha256"]
        ):
            raise ValueError("mollification reuse provenance disagrees with current protocol")
        root = _resolve_uri(str(source["source_shard_root"]))
        expected_paths = {
            root / f"{state['structure_family_id']}-{str(state['state_id'])[:16]}.h5"
            for state in states
        }
        actual_paths = {path for path in root.glob("*.h5") if path.is_file()}
        if not actual_paths.issubset(expected_paths):
            raise ValueError("mollification reuse directory contains an unexpected shard")
        target_p95 = float(
            source_budget.document["reference_budget"]["target_relative_se_p95"]
        )
        hard_limit = float(
            source_budget.document["reference_budget"]["maximum_group_relative_se"]
        )
        for state in states:
            state_id = str(state["state_id"])
            path = root / f"{state['structure_family_id']}-{state_id[:16]}.h5"
            if not path.is_file():
                continue
            if state_id in entries_by_state:
                raise ValueError(f"mollification reuse sources overlap at state: {state_id}")
            identity = _supplement_shard_identity(
                protocol,
                anchor_lock,
                supplement_lock,
                source_budget,
                source_collection,
                provider,
                state,
            )
            result = _validate_mollified_shard(path, identity)
            if (
                result["maximum_relative_se_p95"] > target_p95
                or result["maximum_relative_se"] > hard_limit
            ):
                raise ValueError(
                    f"mollification predecessor shard failed its own gate: {state_id}"
                )
            entries_by_state[state_id] = {
                "source": "reused",
                "state_id": state_id,
                "structure_family_id": state["structure_family_id"],
                "uri": _project_uri(path),
                "dataset_id": result["dataset_id"],
                "sha256": _sha256_file(path),
                "target_count": result["target_count"],
                "combined_reference_samples": result["combined_reference_samples"],
                "maximum_relative_se_p95": result["maximum_relative_se_p95"],
                "maximum_relative_se": result["maximum_relative_se"],
                "budget_plan_uri": _project_uri(source_budget_path),
                "budget_plan_sha256": source_budget.sha256,
                "collection_lock_uri": _project_uri(source_collection_path),
                "collection_lock_sha256": source_collection["collection_lock_sha256"],
            }
    return [entries_by_state[state["state_id"]] for state in states if state["state_id"] in entries_by_state]


def _write_mollified_shard(
    path: Path,
    identity: Mapping[str, Any],
    source: Mapping[str, np.ndarray],
    curriculum: Mapping[str, np.ndarray],
    responses: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as stream:
        stream.attrs["format_name"] = MOLLIFIED_SHARD_FORMAT
        stream.attrs["format_version"] = MOLLIFIED_SHARD_VERSION
        stream.attrs["response_measure"] = MOLLIFICATION_RESPONSE_MEASURE
        stream.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        stream.attrs["identity_json"] = _canonical_json(identity)
        for group_name, values in (
            ("anchors", source), ("curriculum", curriculum), ("responses", responses)
        ):
            group = stream.create_group(group_name)
            for name, value in values.items():
                group.create_dataset(name, data=np.asarray(value), compression="gzip", shuffle=True)
        stream.flush()
        stream.attrs["dataset_id"] = _mollified_shard_semantic_hash(stream)
        stream.flush()
    os.replace(temporary, path)
    return _validate_mollified_shard(path, identity)


def _supplement_shard_identity(
    protocol: MollificationProtocol,
    anchor_lock: Mapping[str, Any],
    supplement_lock: Mapping[str, Any],
    budget: MollificationSupplementBudgetPlan,
    collection_lock: Mapping[str, Any],
    provider: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_sha256": protocol.sha256,
        "anchor_lock_sha256": anchor_lock["anchor_lock_sha256"],
        "supplement_anchor_lock_sha256": supplement_lock["supplement_anchor_lock_sha256"],
        "supplement_budget_plan_sha256": budget.sha256,
        "collection_lock_sha256": collection_lock["collection_lock_sha256"],
        "base_corpus_id": supplement_lock["base_corpus_id"],
        "selection_sha256": supplement_lock["selection_sha256"],
        "state_id": state["state_id"],
        "structure_family_id": state["structure_family_id"],
        "source_dataset_id": state["source"]["dataset_id"],
        "reference": {
            "family_id": provider.descriptor.family_id,
            "reference_id": provider.descriptor.reference_id,
            "implementation_sha256": provider.descriptor.implementation_sha256,
        },
        "response_measure": MOLLIFICATION_RESPONSE_MEASURE,
        "jitter": dict(protocol.document["curriculum"]),
        "reference_budget": {
            **dict(budget.document["reference_budget"]),
            "maximum_path_depth": protocol.document["reference_budget"]["maximum_path_depth"],
        },
    }


def _collect_one_mollified_shard(
    path: Path,
    protocol: MollificationProtocol,
    anchor_lock: Mapping[str, Any],
    supplement_lock: Mapping[str, Any],
    budget: MollificationSupplementBudgetPlan,
    collection_lock: Mapping[str, Any],
    provider: Any,
    runtime_state: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    from .contract import QueryPlan, QueryRole, SurfaceSample

    identity = _supplement_shard_identity(
        protocol, anchor_lock, supplement_lock, budget, collection_lock, provider, state
    )
    if path.exists():
        return _validate_mollified_shard(path, identity)
    source_wo = np.asarray([view["wo"] for view in state["views"]], dtype=np.float32)
    source_wi = np.asarray([view["wi"] for view in state["views"]], dtype=np.float32)
    source_response = np.asarray(
        [view["source_response"] for view in state["views"]], dtype=np.float32
    )
    source_groups = np.asarray(
        [view["source_group_index"] for view in state["views"]], dtype=np.uint32
    )
    source_directions = np.asarray(
        [view["light_direction_indices"] for view in state["views"]], dtype=np.uint32
    )
    progress = np.asarray(protocol.document["curriculum"]["progress"], dtype=np.float32)
    radii = np.asarray(protocol.document["curriculum"]["radius_degrees"], dtype=np.float32)
    mean = np.empty((8, 4, 64, 3), dtype=np.float32)
    variance = np.empty_like(mean)
    replica_a = np.empty_like(mean)
    replica_b = np.empty_like(mean)
    sample_count = np.empty((8, 4, 64), dtype=np.uint32)
    relative_p95 = np.empty((8, 4), dtype=np.float32)
    relative_max = np.empty((8, 4), dtype=np.float32)
    supplement = protocol.document["supplement"]
    reference_budget = budget.document["reference_budget"]
    initial_paths = int(reference_budget["initial_paths_per_jitter_per_replica"])
    maximum_paths = int(reference_budget["maximum_paths_per_jitter_per_replica"])
    target_p95 = float(reference_budget["target_relative_se_p95"])
    hard_limit = float(reference_budget["maximum_group_relative_se"])
    floor = float(supplement["normalization_floor"])
    jitter_count = int(reference_budget["jitter_count"])
    diagnostics: list[dict[str, Any]] = []
    for view_index in range(8):
        lights = source_wi[view_index]
        for level_index, radius in enumerate(radii.tolist()):
            jitter = mollification_cone_directions(
                source_wo[view_index],
                float(radius),
                jitter_count,
                _seed32(
                    int(protocol.document["seed"]), state["state_id"],
                    view_index, level_index, "supplement-cone",
                ),
            )
            plan = QueryPlan(
                jitter,
                np.broadcast_to(lights[None, :, :], (jitter_count, 64, 3)).copy(),
                np.ones((jitter_count, 64), dtype=np.float32),
                np.ones((jitter_count, 64), dtype=np.float32),
                "mollification-supplement-upper-cap-v1",
                _seed32(
                    int(protocol.document["seed"]), state["state_id"],
                    view_index, level_index, "supplement-reference",
                ),
                np.full(jitter_count, int(QueryRole.TRAIN), dtype=np.uint8),
            )
            paths = initial_paths
            attempts: list[dict[str, Any]] = []
            while True:
                evaluated = provider.evaluate_batched_fixed(
                    runtime_state,
                    (SurfaceSample(),),
                    plan,
                    samples_per_replica=paths,
                )
                target_a = np.mean(evaluated.replica_mean_a[0], axis=0, dtype=np.float64)
                target_b = np.mean(evaluated.replica_mean_b[0], axis=0, dtype=np.float64)
                target_mean = 0.5 * (target_a + target_b)
                target_se = 0.5 * np.abs(target_a - target_b)
                relative = _reference_relative_standard_error(
                    target_mean,
                    target_se,
                    group_axes=(0, 1),
                    absolute_floor=floor,
                )
                score_p95 = float(np.quantile(relative, 0.95))
                score_max = float(np.max(relative))
                passed = score_p95 <= target_p95 and score_max <= hard_limit
                attempts.append(
                    {
                        "paths_per_jitter_per_replica": paths,
                        "combined_reference_samples_per_target": 2 * paths * jitter_count,
                        "relative_se_p95": score_p95,
                        "relative_se_max": score_max,
                        "passed": passed,
                    }
                )
                if passed:
                    diagnostics.append(
                        {
                            "state_id": state["state_id"],
                            "view_index": view_index,
                            "level_index": level_index,
                            "radius_degrees": float(radius),
                            "status": "passed",
                            "attempts": attempts,
                        }
                    )
                    break
                if paths >= maximum_paths:
                    diagnostics.append(
                        {
                            "state_id": state["state_id"],
                            "view_index": view_index,
                            "level_index": level_index,
                            "radius_degrees": float(radius),
                            "status": "failed",
                            "attempts": attempts,
                        }
                    )
                    raise MollificationBudgetExhausted(
                        "mollification supplement exhausted its frozen reference budget: "
                        f"state={state['state_id']} view={view_index} level={level_index} "
                        f"p95={score_p95:.6g} max={score_max:.6g}",
                        diagnostics,
                    )
                paths = min(paths * 2, maximum_paths)
            mean[view_index, level_index] = target_mean.astype(np.float32)
            replica_a[view_index, level_index] = target_a.astype(np.float32)
            replica_b[view_index, level_index] = target_b.astype(np.float32)
            variance[view_index, level_index] = (
                0.5 * ((target_a - target_mean) ** 2 + (target_b - target_mean) ** 2)
            ).astype(np.float32)
            sample_count[view_index, level_index] = np.uint32(2 * paths * jitter_count)
            relative_p95[view_index, level_index] = np.float32(score_p95)
            relative_max[view_index, level_index] = np.float32(score_max)
    return _write_mollified_shard(
        path,
        identity,
        {
            "wo": source_wo,
            "wi": source_wi,
            "source_response": source_response,
            "source_group_index": source_groups,
            "source_direction_index": source_directions,
        },
        {"progress": progress, "radius_degrees": radii},
        {
            "mean": mean,
            "variance": variance,
            "replica_mean_a": replica_a,
            "replica_mean_b": replica_b,
            "sample_count": sample_count,
            "relative_se_p95": relative_p95,
            "relative_se_max": relative_max,
        },
    )


def _supplement_manifest_payload(value: Mapping[str, Any], *, include_identity: bool) -> dict[str, Any]:
    result = json.loads(_canonical_json(value))
    if not include_identity:
        result.pop("created_at", None)
        result.pop("corpus_id", None)
        for shard in result["shards"]:
            shard.pop("uri", None)
            shard.pop("sha256", None)
    return result


def _write_mollification_failure_report(
    path: Path | str,
    *,
    protocol_sha256: str,
    supplement_anchor_lock_sha256: str,
    budget_plan_sha256: str,
    collection_lock_sha256: str,
    state_id: str,
    error: MollificationBudgetExhausted,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "schema": {"name": "mollification-supplement-failure-report", "version": 1},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _require_sha256("protocol_sha256", protocol_sha256),
        "supplement_anchor_lock_sha256": _require_sha256(
            "supplement_anchor_lock_sha256", supplement_anchor_lock_sha256
        ),
        "budget_plan_sha256": _require_sha256("budget_plan_sha256", budget_plan_sha256),
        "collection_lock_sha256": _require_sha256(
            "collection_lock_sha256", collection_lock_sha256
        ),
        "state_id": _require_sha256("state_id", state_id),
        "message": str(error),
        "targets": list(error.diagnostics),
    }
    failure["report_sha256"] = _sha256_json(failure)
    _write_json_atomic(Path(path), failure)
    return failure


def _collect_locked_mollified_shards(
    protocol: MollificationProtocol,
    anchor_lock: Mapping[str, Any],
    manifest: ReferenceCorpusManifest,
    supplement_lock: Mapping[str, Any],
    budget: MollificationSupplementBudgetPlan,
    collection_lock: Mapping[str, Any],
    collection_lock_path: Path | str,
    shard_root: Path | str,
    state_ids: Sequence[str],
    failure_report_path: Path | str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    maximum_paths = int(
        budget.document["reference_budget"]["maximum_paths_per_jitter_per_replica"]
    )
    provider, runtime_states = _make_layer_stack_provider(
        protocol, manifest, tuple(state_ids), fixed_samples_per_replica=maximum_paths
    )
    state_by_id = {str(item["state_id"]): item for item in supplement_lock["states"]}
    root = Path(shard_root)
    entries: dict[str, dict[str, Any]] = {}
    try:
        reference = {
            "family_id": provider.descriptor.family_id,
            "reference_id": provider.descriptor.reference_id,
            "implementation_sha256": provider.descriptor.implementation_sha256,
        }
        for state_id in state_ids:
            state = state_by_id[str(state_id)]
            filename = f"{state['structure_family_id']}-{str(state_id)[:16]}.h5"
            path = root / filename
            try:
                result = _collect_one_mollified_shard(
                    path,
                    protocol,
                    anchor_lock,
                    supplement_lock,
                    budget,
                    collection_lock,
                    provider,
                    runtime_states[str(state_id)],
                    state,
                )
            except MollificationBudgetExhausted as error:
                if failure_report_path is not None:
                    _write_mollification_failure_report(
                        failure_report_path,
                        protocol_sha256=protocol.sha256,
                        supplement_anchor_lock_sha256=supplement_lock[
                            "supplement_anchor_lock_sha256"
                        ],
                        budget_plan_sha256=budget.sha256,
                        collection_lock_sha256=collection_lock[
                            "collection_lock_sha256"
                        ],
                        state_id=str(state_id),
                        error=error,
                    )
                raise
            entries[str(state_id)] = {
                "source": "collected",
                "state_id": str(state_id),
                "structure_family_id": state["structure_family_id"],
                "uri": _project_uri(path),
                "dataset_id": result["dataset_id"],
                "sha256": _sha256_file(path),
                "target_count": result["target_count"],
                "combined_reference_samples": result["combined_reference_samples"],
                "maximum_relative_se_p95": result["maximum_relative_se_p95"],
                "maximum_relative_se": result["maximum_relative_se"],
                "budget_plan_uri": _project_uri(budget.source_path),
                "budget_plan_sha256": budget.sha256,
                "collection_lock_uri": _project_uri(Path(collection_lock_path)),
                "collection_lock_sha256": collection_lock["collection_lock_sha256"],
            }
    finally:
        provider.close()
    return reference, entries


def collect_mollification_supplement_states(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    audit_report_path: Path | str,
    supplement_lock_path: Path | str,
    budget_plan_path: Path | str,
    collection_lock_path: Path | str,
    shard_root: Path | str,
    state_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """按已冻结 collection lock 续采指定 state，不提前发布 manifest。"""

    (
        protocol, anchor_lock, manifest, supplement_lock, budget, collection_lock,
    ) = load_mollification_supplement_collection_lock(
        protocol_path,
        anchor_lock_path,
        audit_report_path,
        supplement_lock_path,
        budget_plan_path,
        collection_lock_path,
    )
    requested = tuple(map(str, state_ids))
    allowed = tuple(map(str, collection_lock.get("collection_state_ids", collection_lock["state_ids"])))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("mollification state continuation requires unique state IDs")
    if not set(requested).issubset(allowed):
        raise ValueError("mollification state continuation is outside the frozen collection lock")
    _, entries = _collect_locked_mollified_shards(
        protocol,
        anchor_lock,
        manifest,
        supplement_lock,
        budget,
        collection_lock,
        collection_lock_path,
        shard_root,
        requested,
    )
    return [entries[state_id] for state_id in requested]


def collect_mollification_supplement(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    audit_report_path: Path | str,
    supplement_lock_path: Path | str,
    budget_plan_path: Path | str,
    collection_lock_path: Path | str,
    shard_root: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """采集完整 30-state mollification supplement；任一 noise gate 失败则不发布 manifest。"""

    (
        protocol, anchor_lock, manifest, supplement_lock, budget, collection_lock,
    ) = load_mollification_supplement_collection_lock(
        protocol_path,
        anchor_lock_path,
        audit_report_path,
        supplement_lock_path,
        budget_plan_path,
        collection_lock_path,
    )
    state_ids = tuple(item["state_id"] for item in supplement_lock["states"])
    collection_state_ids = tuple(
        map(str, collection_lock.get("collection_state_ids", state_ids))
    )
    shards_by_state = {
        str(item["state_id"]): dict(item)
        for item in collection_lock.get("reused_shards", ())
    }
    reference, collected = _collect_locked_mollified_shards(
        protocol,
        anchor_lock,
        manifest,
        supplement_lock,
        budget,
        collection_lock,
        collection_lock_path,
        shard_root,
        collection_state_ids,
        Path(output_path).parent / f"{budget.document['name']}-failure.json",
    )
    shards_by_state.update(collected)
    if set(shards_by_state) != set(state_ids):
        raise RuntimeError("mollification supplement did not produce all 30 state shards")
    shards = [shards_by_state[state_id] for state_id in state_ids]
    value: dict[str, Any] = {
        "format_name": MOLLIFIED_CORPUS_FORMAT,
        "format_version": MOLLIFIED_CORPUS_VERSION,
        "name": protocol.document["supplement"]["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_corpus_uri": _project_uri(_resolve_uri(str(protocol.document["base_corpus_uri"]))),
        "base_corpus_id": manifest.corpus_id,
        "protocol_sha256": protocol.sha256,
        "anchor_selection_sha256": supplement_lock["supplement_anchor_lock_sha256"],
        "supplement_budget_plan_sha256": budget.sha256,
        "collection_lock_sha256": collection_lock["collection_lock_sha256"],
        "reference": reference,
        "response_measure": MOLLIFICATION_RESPONSE_MEASURE,
        "state_ids": list(state_ids),
        "shards": shards,
        "totals": {
            "state_count": len(shards),
            "target_count": int(sum(item["target_count"] for item in shards)),
            "combined_reference_samples": int(sum(item["combined_reference_samples"] for item in shards)),
        },
    }
    value["corpus_id"] = _sha256_json(_supplement_manifest_payload(value, include_identity=False))
    _write_json_atomic(Path(output_path), value)
    return validate_mollification_supplement(output_path)


def validate_mollification_supplement(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    required = {
        "format_name", "format_version", "name", "created_at", "base_corpus_uri",
        "base_corpus_id", "protocol_sha256", "anchor_selection_sha256", "reference",
        "supplement_budget_plan_sha256", "collection_lock_sha256",
        "response_measure", "state_ids", "shards", "totals", "corpus_id",
    }
    _require_exact_fields("mollified-reference-corpus", value, required)
    if (
        value["format_name"] != MOLLIFIED_CORPUS_FORMAT
        or int(value["format_version"]) != 1
        or value["name"] != "layer-stack-p1-mollification-v1"
        or value["response_measure"] != MOLLIFICATION_RESPONSE_MEASURE
    ):
        raise ValueError("unsupported mollified-reference-corpus")
    for name in (
        "base_corpus_id", "protocol_sha256", "anchor_selection_sha256",
        "supplement_budget_plan_sha256", "collection_lock_sha256", "corpus_id",
    ):
        _require_sha256(name, value[name])
    if _sha256_json(_supplement_manifest_payload(value, include_identity=False)) != value["corpus_id"]:
        raise ValueError("mollified-reference-corpus identity mismatch")
    _require_exact_fields(
        "mollified-reference-corpus reference",
        value["reference"],
        {"family_id", "reference_id", "implementation_sha256"},
    )
    if (
        value["reference"]["family_id"] != "ncls.layer-stack@1"
        or value["reference"]["reference_id"] != "ncls.layer-stack-random-walk@1"
    ):
        raise ValueError("mollified-reference-corpus reference identity mismatch")
    _require_sha256(
        "reference implementation_sha256", value["reference"]["implementation_sha256"]
    )
    state_ids = tuple(map(str, value["state_ids"]))
    if len(state_ids) != 30 or len(set(state_ids)) != 30 or len(value["shards"]) != 30:
        raise ValueError("mollified-reference-corpus must contain exactly 30 unique states and shards")
    for state_id in state_ids:
        _require_sha256("mollified corpus state_id", state_id)
    base_manifest = validate_reference_corpus(_resolve_uri(str(value["base_corpus_uri"])))
    if (
        base_manifest.corpus_id != value["base_corpus_id"]
        or base_manifest.selection is None
        or tuple(map(str, base_manifest.selection["state_ids"])) != state_ids
    ):
        raise ValueError("mollified-reference-corpus base selection provenance mismatch")
    base_sources = {
        str(shard.dataset_id): (set(shard.state_ids), shard.structure_family_id)
        for shard in base_manifest.shards
        if shard.dataset_id is not None
    }
    if tuple(item["state_id"] for item in value["shards"]) != state_ids:
        raise ValueError("mollified-reference-corpus shard order disagrees with state selection")
    totals = {"state_count": 0, "target_count": 0, "combined_reference_samples": 0}
    for shard in value["shards"]:
        _require_exact_fields(
            "mollified-reference-corpus shard",
            shard,
            {
                "source", "state_id", "structure_family_id", "uri", "dataset_id",
                "sha256", "target_count", "combined_reference_samples",
                "maximum_relative_se_p95", "maximum_relative_se", "budget_plan_uri",
                "budget_plan_sha256", "collection_lock_uri", "collection_lock_sha256",
            },
        )
        if shard["source"] not in {"reused", "collected"}:
            raise ValueError("mollified-reference-corpus shard source is invalid")
        for name in (
            "state_id", "dataset_id", "sha256", "budget_plan_sha256",
            "collection_lock_sha256",
        ):
            _require_sha256(name, shard[name])
        shard_budget = MollificationSupplementBudgetPlan.load(
            _resolve_uri(str(shard["budget_plan_uri"]))
        )
        if shard_budget.sha256 != shard["budget_plan_sha256"]:
            raise ValueError("mollified-reference-corpus shard budget plan mismatch")
        shard_lock_path = _resolve_uri(str(shard["collection_lock_uri"]))
        shard_lock = json.loads(shard_lock_path.read_text(encoding="utf-8"))
        shard_lock_payload = dict(shard_lock)
        shard_lock_hash = shard_lock_payload.pop("collection_lock_sha256", "")
        if (
            shard_lock_hash != shard["collection_lock_sha256"]
            or _sha256_json(shard_lock_payload) != shard_lock_hash
            or shard_lock.get("budget_plan_sha256") != shard_budget.sha256
            or shard_lock.get("base_corpus_id") != value["base_corpus_id"]
            or shard_lock.get("protocol_sha256") != value["protocol_sha256"]
            or shard_lock.get("supplement_anchor_lock_sha256")
            != value["anchor_selection_sha256"]
            or shard_lock.get("reference_implementation_sha256")
            != value["reference"]["implementation_sha256"]
            or shard_lock.get("state_ids") != list(state_ids)
        ):
            raise ValueError("mollified-reference-corpus shard collection lock mismatch")
        shard_path = _resolve_uri(str(shard["uri"]))
        if _sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"mollified-reference-corpus file hash mismatch: {shard['state_id']}")
        result = _validate_mollified_shard(shard_path)
        if result["dataset_id"] != shard["dataset_id"]:
            raise ValueError(f"mollified-reference-corpus dataset identity mismatch: {shard['state_id']}")
        if result["identity"]["state_id"] != shard["state_id"]:
            raise ValueError("mollified-reference-corpus state provenance mismatch")
        source_dataset_id = str(result["identity"]["source_dataset_id"])
        source_states, source_structure = base_sources.get(source_dataset_id, (set(), ""))
        if (
            shard["state_id"] not in source_states
            or result["identity"]["structure_family_id"] != source_structure
            or shard["structure_family_id"] != source_structure
        ):
            raise ValueError("mollified-reference-corpus source v5 shard provenance mismatch")
        if (
            result["identity"]["base_corpus_id"] != value["base_corpus_id"]
            or result["identity"]["protocol_sha256"] != value["protocol_sha256"]
            or result["identity"]["supplement_anchor_lock_sha256"] != value["anchor_selection_sha256"]
            or result["identity"]["supplement_budget_plan_sha256"]
            != shard["budget_plan_sha256"]
            or result["identity"]["collection_lock_sha256"]
            != shard["collection_lock_sha256"]
            or result["identity"]["reference"] != value["reference"]
        ):
            raise ValueError("mollified-reference-corpus shard provenance mismatch")
        for field in ("target_count", "combined_reference_samples"):
            if shard[field] != result[field]:
                raise ValueError(f"mollified-reference-corpus shard {field} mismatch")
        for field in ("maximum_relative_se_p95", "maximum_relative_se"):
            if not math.isclose(
                float(shard[field]),
                float(result[field]),
                rel_tol=3e-5,
                abs_tol=2e-7,
            ):
                raise ValueError(f"mollified-reference-corpus shard {field} mismatch")
        shard_budget_values = shard_budget.document["reference_budget"]
        if (
            result["maximum_relative_se_p95"]
            > float(shard_budget_values["target_relative_se_p95"])
            or result["maximum_relative_se"]
            > float(shard_budget_values["maximum_group_relative_se"])
        ):
            raise ValueError("mollified-reference-corpus shard failed its own noise gate")
        totals["state_count"] += 1
        totals["target_count"] += int(result["target_count"])
        totals["combined_reference_samples"] += int(result["combined_reference_samples"])
    _require_exact_fields(
        "mollified-reference-corpus totals",
        value["totals"],
        {"state_count", "target_count", "combined_reference_samples"},
    )
    if value["totals"] != totals:
        raise ValueError("mollified-reference-corpus totals mismatch")
    if not any(
        item["budget_plan_sha256"] == value["supplement_budget_plan_sha256"]
        and item["collection_lock_sha256"] == value["collection_lock_sha256"]
        for item in value["shards"]
    ):
        raise ValueError("mollified-reference-corpus has no shard from its composing collection")
    return value
