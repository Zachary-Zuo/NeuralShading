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

from ncls.paths import PROJECT_ROOT

from .corpus import ReferenceCorpusManifest, validate_reference_corpus
from .dataset import RESPONSE_MEASURE, ReferenceDataset


PROTOCOL_SCHEMA = {"name": "mollification-protocol", "version": 1}
ANCHOR_LOCK_SCHEMA = {"name": "mollification-anchor-lock", "version": 1}
AUDIT_REPORT_SCHEMA = {"name": "mollification-adequacy-report", "version": 1}
TRAINING_DATA_ENTRY_SCHEMA = {"name": "mollification-training-data-entry", "version": 1}
SUPPLEMENT_ANCHOR_LOCK_SCHEMA = {"name": "mollification-supplement-anchor-lock", "version": 1}
SUPPLEMENT_BUDGET_SCHEMA_NAME = "mollification-supplement-budget"
SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V1 = {
    "name": "mollification-supplement-collection-lock", "version": 1,
}
SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V2 = {
    "name": "mollification-supplement-collection-lock", "version": 2,
}
SUPPLEMENT_COLLECTION_LOCK_SCHEMA_V3 = {
    "name": "mollification-supplement-collection-lock", "version": 3,
}
MOLLIFIED_SHARD_FORMAT = "mollified-reference-shard"
MOLLIFIED_SHARD_VERSION = 1
MOLLIFIED_CORPUS_FORMAT = "mollified-reference-corpus"
MOLLIFIED_CORPUS_VERSION = 1
MOLLIFICATION_RESPONSE_MEASURE = RESPONSE_MEASURE


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _project_uri(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_uri(uri: str) -> Path:
    path = Path(uri)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_sha256(name: str, value: Any) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _require_exact_fields(name: str, value: Mapping[str, Any], fields: set[str]) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields do not match v1")


@dataclass(frozen=True)
class MollificationProtocol:
    document: Mapping[str, Any]
    source_path: Path

    def __post_init__(self) -> None:
        root = dict(self.document)
        _require_exact_fields(
            "MollificationProtocol",
            root,
            {
                "schema", "name", "seed", "base_corpus_uri", "representative_states",
                "anchor_selection", "curriculum", "reference_budget", "reconstruction",
                "adequacy_gates", "supplement",
            },
        )
        if root["schema"] != PROTOCOL_SCHEMA or root["name"] != "layer-stack-p1-mollification-adequacy-v1":
            raise ValueError("unsupported mollification protocol")
        if int(root["seed"]) < 0 or not str(root["base_corpus_uri"]):
            raise ValueError("mollification protocol requires a nonnegative seed and base corpus URI")
        representatives = tuple(root["representative_states"])
        if len(representatives) != 6:
            raise ValueError("mollification protocol requires exactly six representative states")
        roles: set[str] = set()
        state_ids: set[str] = set()
        for item in representatives:
            if not isinstance(item, Mapping):
                raise ValueError("representative state entries must be objects")
            _require_exact_fields("representative state", item, {"role", "state_id"})
            role = str(item["role"])
            state_id = _require_sha256("representative state_id", item["state_id"])
            if not role or role in roles or state_id in state_ids:
                raise ValueError("representative state roles and IDs must be unique")
            roles.add(role)
            state_ids.add(state_id)
        anchors = root["anchor_selection"]
        _require_exact_fields(
            "anchor_selection",
            anchors,
            {
                "dense_view_count", "light_roles", "shoulder_degrees",
                "grazing_light_z", "background_light_z", "fallback",
            },
        )
        if int(anchors["dense_view_count"]) != 4 or tuple(anchors["light_roles"]) != (
            "peak", "shoulder", "grazing-light", "background"
        ):
            raise ValueError("mollification anchor roles are frozen at four dense views and four light roles")
        if tuple(map(float, anchors["shoulder_degrees"])) != (2.0, 5.0):
            raise ValueError("mollification shoulder band is frozen at 2..5 degrees")
        if tuple(map(float, anchors["grazing_light_z"])) != (0.02, 0.15):
            raise ValueError("mollification grazing-light band is frozen at z=0.02..0.15")
        if tuple(map(float, anchors["background_light_z"])) != (0.4, 0.8):
            raise ValueError("mollification background band is frozen at z=0.4..0.8")
        if anchors["fallback"] != "nearest-band-center-then-source-index-v1":
            raise ValueError("unsupported mollification anchor fallback")
        curriculum = root["curriculum"]
        _require_exact_fields(
            "curriculum",
            curriculum,
            {
                "progress", "radius_degrees", "zero_radius_switch_progress",
                "jitter_count", "sequence", "mollified_direction",
            },
        )
        progress = tuple(map(float, curriculum["progress"]))
        radii = tuple(map(float, curriculum["radius_degrees"]))
        expected_radii = tuple(5.0 * (1.0 + math.cos(math.pi * value)) for value in progress)
        if progress != (0.0, 0.25, 0.5, 0.75) or not np.allclose(radii, expected_radii, atol=1e-12, rtol=0.0):
            raise ValueError("mollification curriculum must use the frozen four NVIDIA schedule levels")
        if (
            int(curriculum["jitter_count"]) != 256
            or curriculum["sequence"] != "scrambled-hammersley-upper-cap-v1"
            or curriculum["mollified_direction"] != "wo"
            or float(curriculum["zero_radius_switch_progress"]) != 0.875
        ):
            raise ValueError("mollification curriculum contract does not match v1")
        budget = root["reference_budget"]
        _require_exact_fields(
            "reference_budget",
            budget,
            {
                "replica_count", "audit_paths_per_jitter_per_replica",
                "maximum_path_depth", "maximum_dispatch_queries",
            },
        )
        if tuple(int(budget[name]) for name in (
            "replica_count", "audit_paths_per_jitter_per_replica",
            "maximum_path_depth", "maximum_dispatch_queries",
        )) != (2, 512, 64, 4096):
            raise ValueError("mollification audit reference budget does not match v1")
        reconstruction = root["reconstruction"]
        _require_exact_fields(
            "reconstruction",
            reconstruction,
            {
                "source_role", "embedding", "neighbors", "shepard_power",
                "support_wo_degrees", "support_wi_degrees",
            },
        )
        if (
            reconstruction["source_role"] != "train"
            or reconstruction["embedding"] != "scaled-wo-wi-chord-v1"
            or int(reconstruction["neighbors"]) != 32
            or float(reconstruction["shepard_power"]) != 2.0
            or float(reconstruction["support_wo_degrees"]) != 2.0
            or float(reconstruction["support_wi_degrees"]) != 1.0
        ):
            raise ValueError("mollification reconstruction contract does not match v1")
        gates = root["adequacy_gates"]
        _require_exact_fields(
            "adequacy_gates",
            gates,
            {
                "support_fraction_minimum", "fresh_relative_se_p95_maximum",
                "normalized_rgb_l1_median_maximum", "normalized_rgb_l1_p95_maximum",
                "normalized_rgb_l1_worst_maximum", "normalization_floor",
                "repeat_float_tolerance",
            },
        )
        gate_names = (
            "support_fraction_minimum",
            "fresh_relative_se_p95_maximum",
            "normalized_rgb_l1_median_maximum",
            "normalized_rgb_l1_p95_maximum",
            "normalized_rgb_l1_worst_maximum",
            "normalization_floor",
            "repeat_float_tolerance",
        )
        expected_gates = (0.95, 0.04, 0.025, 0.05, 0.10, 1e-6, 1e-7)
        if tuple(float(gates[name]) for name in gate_names) != expected_gates:
            raise ValueError("mollification adequacy gates do not match v1")
        supplement = root["supplement"]
        _require_exact_fields(
            "supplement",
            supplement,
            {
                "name", "view_count", "light_count", "peak_light_count",
                "log_response_light_count", "proposal_index_light_count",
                "initial_paths_per_jitter_per_replica", "maximum_paths_per_jitter_per_replica",
                "target_relative_se_p95", "maximum_group_relative_se",
                "normalization_floor", "shard_format", "corpus_format",
            },
        )
        if (
            supplement["name"] != "layer-stack-p1-mollification-v1"
            or tuple(int(supplement[name]) for name in (
                "view_count", "light_count", "peak_light_count",
                "log_response_light_count", "proposal_index_light_count",
                "initial_paths_per_jitter_per_replica", "maximum_paths_per_jitter_per_replica",
            )) != (8, 64, 16, 16, 32, 64, 512)
            or tuple(float(supplement[name]) for name in (
                "target_relative_se_p95", "maximum_group_relative_se", "normalization_floor"
            )) != (0.06, 0.25, 1e-6)
            or supplement["shard_format"] != {"name": MOLLIFIED_SHARD_FORMAT, "version": 1}
            or supplement["corpus_format"] != {"name": MOLLIFIED_CORPUS_FORMAT, "version": 1}
        ):
            raise ValueError("mollification supplement contract does not match v1")

    @property
    def sha256(self) -> str:
        return _sha256_json(self.document)

    @classmethod
    def load(cls, path: Path | str) -> "MollificationProtocol":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("MollificationProtocol root must be an object")
        return cls(dict(value), source.resolve())


def _base_identity(path: Path) -> tuple[ReferenceCorpusManifest, dict[str, Any]]:
    manifest = validate_reference_corpus(path)
    if manifest.format_version != 2 or manifest.selection is None or manifest.corpus_id is None:
        raise ValueError("mollification requires a selected reference-corpus v2")
    shards = []
    for shard in manifest.shards:
        if shard.status != "complete" or shard.dataset_id is None or shard.sha256 is None:
            raise ValueError("mollification base corpus must contain only complete shards")
        shards.append(
            {
                "shard_id": shard.shard_id,
                "uri": shard.uri,
                "role": shard.role,
                "state_ids": list(shard.state_ids),
                "dataset_id": shard.dataset_id,
                "sha256": shard.sha256,
            }
        )
    identity = {
        "corpus_id": manifest.corpus_id,
        "plan_sha256": manifest.plan_sha256,
        "selection_sha256": manifest.selection_sha256,
        "manifest_sha256": _sha256_file(path),
        "shards": shards,
    }
    return manifest, identity


def _state_shard(
    manifest: ReferenceCorpusManifest,
    state_id: str,
    role: str,
) -> Any:
    matches = [
        shard for shard in manifest.shards
        if shard.role == role and state_id in shard.state_ids and shard.status == "complete"
    ]
    if len(matches) != 1:
        raise ValueError(f"base corpus must contain exactly one {role} shard for state {state_id}")
    return matches[0]


def _angular_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _argmax_then_index(values: np.ndarray, indices: np.ndarray) -> int:
    maximum = float(np.max(values[indices]))
    return int(indices[np.flatnonzero(values[indices] == maximum)[0]])


def _select_dense_lights(
    wi: np.ndarray,
    response: np.ndarray,
    anchors: Mapping[str, Any],
) -> list[dict[str, Any]]:
    magnitude = np.sum(np.abs(response.astype(np.float64)), axis=-1)
    all_indices = np.arange(len(wi), dtype=np.int64)
    peak = _argmax_then_index(magnitude, all_indices)
    angles = _angular_degrees(wi.astype(np.float64), wi[peak][None, :].astype(np.float64))
    shoulder_band = tuple(map(float, anchors["shoulder_degrees"]))
    candidates = np.flatnonzero((angles >= shoulder_band[0]) & (angles <= shoulder_band[1]))
    if len(candidates):
        shoulder = _argmax_then_index(magnitude, candidates)
        shoulder_fallback = False
    else:
        center = 0.5 * sum(shoulder_band)
        shoulder = int(np.lexsort((all_indices, np.abs(angles - center)))[0])
        shoulder_fallback = True
    grazing_band = tuple(map(float, anchors["grazing_light_z"]))
    candidates = np.flatnonzero((wi[:, 2] >= grazing_band[0]) & (wi[:, 2] <= grazing_band[1]))
    if len(candidates):
        grazing = _argmax_then_index(magnitude, candidates)
        grazing_fallback = False
    else:
        center = 0.5 * sum(grazing_band)
        grazing = int(np.lexsort((all_indices, np.abs(wi[:, 2] - center)))[0])
        grazing_fallback = True
    background_band = tuple(map(float, anchors["background_light_z"]))
    candidates = np.flatnonzero((wi[:, 2] >= background_band[0]) & (wi[:, 2] <= background_band[1]))
    if len(candidates):
        median = float(np.median(magnitude[candidates]))
        background = int(candidates[np.lexsort((candidates, np.abs(magnitude[candidates] - median)))[0]])
        background_fallback = False
    else:
        center = 0.5 * sum(background_band)
        background = int(np.lexsort((all_indices, np.abs(wi[:, 2] - center)))[0])
        background_fallback = True
    return [
        {"role": "peak", "direction_index": peak, "fallback_used": False},
        {"role": "shoulder", "direction_index": shoulder, "fallback_used": shoulder_fallback},
        {"role": "grazing-light", "direction_index": grazing, "fallback_used": grazing_fallback},
        {"role": "background", "direction_index": background, "fallback_used": background_fallback},
    ]


def freeze_mollification_anchors(
    protocol_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """只读 v5 dense slice 并锁定 matched audit 的精确方向。"""

    protocol = MollificationProtocol.load(protocol_path)
    base_path = _resolve_uri(str(protocol.document["base_corpus_uri"]))
    manifest, base_identity = _base_identity(base_path)
    selection_ids = tuple(map(str, manifest.selection["state_ids"]))
    locked: list[dict[str, Any]] = []
    for representative in protocol.document["representative_states"]:
        state_id = str(representative["state_id"])
        if state_id not in selection_ids:
            raise ValueError(f"representative state is absent from base selection: {state_id}")
        dense_shard = _state_shard(manifest, state_id, "dense_slice")
        with ReferenceDataset.open(_resolve_uri(dense_shard.uri), verify_hashes=True) as dataset:
            state_ids = dataset.state_strings("state_id").tolist()
            local_state = state_ids.index(state_id)
            groups = np.flatnonzero(
                (np.asarray(dataset.stream["queries/state_index"], dtype=np.int64) == local_state)
                & (dataset.query_roles == 4)
            )
            expected_views = int(protocol.document["anchor_selection"]["dense_view_count"])
            if len(groups) != expected_views:
                raise ValueError(f"dense slice for {state_id} must contain exactly {expected_views} views")
            batch = dataset.group_batch(groups, fields=("wo", "wi", "mean"))
            for local_view, group_index in enumerate(groups.tolist()):
                choices = _select_dense_lights(
                    batch["wi"][local_view],
                    batch["mean"][local_view],
                    protocol.document["anchor_selection"],
                )
                for choice in choices:
                    direction_index = int(choice["direction_index"])
                    choice["wi"] = [float(value) for value in batch["wi"][local_view, direction_index]]
                    choice["source_response"] = [
                        float(value) for value in batch["mean"][local_view, direction_index]
                    ]
                locked.append(
                    {
                        "representative_role": str(representative["role"]),
                        "state_id": state_id,
                        "view_index": local_view,
                        "wo": [float(value) for value in batch["wo"][local_view]],
                        "source": {
                            "shard_id": dense_shard.shard_id,
                            "dataset_id": dense_shard.dataset_id,
                            "group_index": int(group_index),
                        },
                        "lights": choices,
                    }
                )
    if len(locked) != 24:
        raise AssertionError("mollification anchor lock must contain six states by four views")
    value: dict[str, Any] = {
        "schema": ANCHOR_LOCK_SCHEMA,
        "protocol_uri": _project_uri(protocol.source_path),
        "protocol_sha256": protocol.sha256,
        "base_corpus_uri": _project_uri(base_path),
        "base_identity": base_identity,
        "anchors": locked,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    value["anchor_lock_sha256"] = _sha256_json(value)
    _write_json_atomic(Path(output_path), value)
    return value


def load_mollification_anchor_lock(
    protocol_path: Path | str,
    lock_path: Path | str,
) -> tuple[MollificationProtocol, dict[str, Any], ReferenceCorpusManifest]:
    """在创建 GPU evaluator 前验证 protocol、lock 与 base corpus 的完整绑定。"""

    protocol = MollificationProtocol.load(protocol_path)
    value = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("mollification anchor lock root must be an object")
    _require_exact_fields(
        "mollification anchor lock",
        value,
        {
            "schema", "protocol_uri", "protocol_sha256", "base_corpus_uri",
            "base_identity", "anchors", "frozen_at", "anchor_lock_sha256",
        },
    )
    if value["schema"] != ANCHOR_LOCK_SCHEMA:
        raise ValueError("unsupported mollification anchor lock")
    stored_hash = _require_sha256("anchor_lock_sha256", value["anchor_lock_sha256"])
    payload = dict(value)
    payload.pop("anchor_lock_sha256")
    if _sha256_json(payload) != stored_hash:
        raise ValueError("mollification anchor lock hash mismatch")
    if value["protocol_sha256"] != protocol.sha256:
        raise ValueError("mollification anchor lock belongs to a different protocol")
    base_path = _resolve_uri(str(protocol.document["base_corpus_uri"]))
    if value["base_corpus_uri"] != _project_uri(base_path):
        raise ValueError("mollification anchor lock points to a different base corpus")
    manifest, identity = _base_identity(base_path)
    if value["base_identity"] != identity:
        raise ValueError("mollification anchor lock base corpus identity mismatch")
    if len(value["anchors"]) != 24:
        raise ValueError("mollification anchor lock must contain exactly 24 view anchors")
    return protocol, dict(value), manifest


def _radical_inverse_base2(value: int) -> float:
    result = 0.0
    scale = 0.5
    while value:
        result += scale * (value & 1)
        value >>= 1
        scale *= 0.5
    return result


def mollification_cone_directions(
    center: Sequence[float],
    radius_degrees: float,
    count: int,
    seed: int,
) -> np.ndarray:
    """生成 spherical cap 与 upper hemisphere 交集上的确定性均匀方向。"""

    axis = np.asarray(center, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("mollification cone center must be a finite direction")
    length = float(np.linalg.norm(axis))
    if abs(length - 1.0) > 2e-4 or not 0.0 < radius_degrees <= 10.0 or count < 1 or seed < 0:
        raise ValueError("mollification cone arguments are outside the v1 domain")
    axis /= length
    helper = np.asarray((0.0, 0.0, 1.0) if abs(axis[2]) < 0.999 else (1.0, 0.0, 0.0))
    tangent = np.cross(helper, axis)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    digest = hashlib.sha256(f"mollification-cone-v1:{seed}".encode("ascii")).digest()
    scramble_u = int.from_bytes(digest[:8], "little") / float(1 << 64)
    scramble_bits = int.from_bytes(digest[8:16], "little")
    cosine_min = math.cos(math.radians(radius_degrees))
    accepted: list[np.ndarray] = []
    index = 0
    maximum_attempts = count * 64
    while len(accepted) < count and index < maximum_attempts:
        u = ((index + 0.5) / count + scramble_u) % 1.0
        v = _radical_inverse_base2(index ^ scramble_bits)
        cosine = 1.0 - u * (1.0 - cosine_min)
        sine = math.sqrt(max(1.0 - cosine * cosine, 0.0))
        phi = 2.0 * math.pi * v
        direction = (
            cosine * axis
            + sine * math.cos(phi) * tangent
            + sine * math.sin(phi) * bitangent
        )
        direction /= np.linalg.norm(direction)
        if direction[2] > 0.0:
            accepted.append(direction)
        index += 1
    if len(accepted) != count:
        raise RuntimeError("mollification cone rejection sequence did not produce enough upper-hemisphere directions")
    return np.asarray(accepted, dtype=np.float32)


def _seed32(*parts: Any) -> int:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _make_layer_stack_provider(
    protocol: MollificationProtocol,
    manifest: ReferenceCorpusManifest,
    state_ids: tuple[str, ...],
    *,
    fixed_samples_per_replica: int,
):
    from .collector import CollectionConfig
    from .profiles import CorpusPlan
    from .providers.layer_stack import LayerStackProvider, LayerStackProviderConfig

    plan = CorpusPlan.from_dict(manifest.plan)
    collection = CollectionConfig(
        name="mollification-reference-v1",
        query_role="train",
        view_count=1,
        light_count=1,
        proposal="uniform",
        # SourceState identity belongs to the base CorpusPlan. The independent
        # mollification seed is applied only when constructing cone/reference queries.
        seed=int(plan.document["seed"]),
    )
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=int(plan.provider["family_count"]),
            states_per_family=int(plan.provider["states_per_family"]),
            heldout_family_count=int(plan.split["heldout_family_count"]),
            fixed_samples_per_replica=fixed_samples_per_replica,
            max_dispatch_queries=int(protocol.document["reference_budget"]["maximum_dispatch_queries"]),
            max_depth=int(protocol.document["reference_budget"]["maximum_path_depth"]),
            adaptive=False,
            selected_state_ids=state_ids,
            state_profile=str(plan.provider["state_profile"]),
            peak_calibration_directions=int(
                plan.document["sampling"]["moving_peak"]["probe_directions"]
            ),
            peak_calibration_samples_per_replica=int(
                plan.document["sampling"]["moving_peak"]["samples_per_replica"]
            ),
        ),
    )
    states = {state.state_id: state for state in provider.source_states()}
    if set(states) != set(state_ids):
        raise ValueError("current LayerStack state generator disagrees with the locked state IDs")
    return provider, states


@dataclass(frozen=True)
class _TrainNeighborhood:
    tree: Any
    embeddings: np.ndarray
    wo: np.ndarray
    wi: np.ndarray
    response: np.ndarray
    group_index: np.ndarray
    direction_index: np.ndarray
    scale_wo: float
    scale_wi: float


def _load_train_neighborhood(
    manifest: ReferenceCorpusManifest,
    state_id: str,
    reconstruction: Mapping[str, Any],
) -> _TrainNeighborhood:
    from scipy.spatial import cKDTree

    shard = _state_shard(manifest, state_id, str(reconstruction["source_role"]))
    with ReferenceDataset.open(_resolve_uri(shard.uri), verify_hashes=True) as dataset:
        state_ids = dataset.state_strings("state_id").tolist()
        local_state = state_ids.index(state_id)
        groups = np.flatnonzero(
            (np.asarray(dataset.stream["queries/state_index"], dtype=np.int64) == local_state)
            & (dataset.query_roles == 0)
        )
        if not len(groups):
            raise ValueError(f"train role is empty for state {state_id}")
        batch = dataset.group_batch(groups, fields=("wo", "wi", "mean"))
    direction_count = batch["wi"].shape[1]
    wo = np.repeat(batch["wo"], direction_count, axis=0).astype(np.float64)
    wi = batch["wi"].reshape(-1, 3).astype(np.float64)
    response = batch["mean"].reshape(-1, 3).astype(np.float64)
    group_index = np.repeat(groups.astype(np.int64), direction_count)
    direction_index = np.tile(np.arange(direction_count, dtype=np.int64), len(groups))
    scale_wo = 1.0 / (2.0 * math.sin(math.radians(float(reconstruction["support_wo_degrees"])) / 2.0))
    scale_wi = 1.0 / (2.0 * math.sin(math.radians(float(reconstruction["support_wi_degrees"])) / 2.0))
    embeddings = np.column_stack((scale_wo * wo, scale_wi * wi))
    return _TrainNeighborhood(
        cKDTree(embeddings, compact_nodes=True, balanced_tree=True),
        embeddings,
        wo,
        wi,
        response,
        group_index,
        direction_index,
        scale_wo,
        scale_wi,
    )


def _reconstruct_neighborhood(
    neighborhood: _TrainNeighborhood,
    wo: np.ndarray,
    wi: np.ndarray,
    reconstruction: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries_wo = np.asarray(wo, dtype=np.float64)
    queries_wi = np.asarray(wi, dtype=np.float64)
    if queries_wo.shape != queries_wi.shape or queries_wo.ndim != 2 or queries_wo.shape[1] != 3:
        raise ValueError("mollification reconstruction queries must have matching [N,3] shapes")
    query_embedding = np.column_stack((
        neighborhood.scale_wo * queries_wo,
        neighborhood.scale_wi * queries_wi,
    ))
    neighbors = int(reconstruction["neighbors"])
    distances, indices = neighborhood.tree.query(
        query_embedding,
        k=neighbors,
        p=2.0,
        workers=1,
    )
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    ordered_indices = np.empty_like(indices)
    ordered_distances = np.empty_like(distances)
    for row in range(len(indices)):
        order = np.lexsort((
            neighborhood.direction_index[indices[row]],
            neighborhood.group_index[indices[row]],
            distances[row],
        ))
        ordered_indices[row] = indices[row, order]
        ordered_distances[row] = distances[row, order]
    exact = ordered_distances[:, 0] <= 1e-14
    power = float(reconstruction["shepard_power"])
    weights = np.zeros_like(ordered_distances)
    weights[~exact] = 1.0 / np.maximum(ordered_distances[~exact], 1e-14) ** power
    weights[exact, 0] = 1.0
    weights /= np.sum(weights, axis=1, keepdims=True)
    response = np.sum(
        weights[..., None] * neighborhood.response[ordered_indices],
        axis=1,
    )
    nearest = ordered_indices[:, 0]
    nearest_wo = _angular_degrees(queries_wo, neighborhood.wo[nearest])
    nearest_wi = _angular_degrees(queries_wi, neighborhood.wi[nearest])
    support_candidates = neighborhood.tree.query_ball_point(
        query_embedding,
        r=math.sqrt(2.0) + 1e-12,
        p=2.0,
        workers=1,
        return_sorted=True,
    )
    maximum_wo = float(reconstruction["support_wo_degrees"])
    maximum_wi = float(reconstruction["support_wi_degrees"])
    support = np.zeros(len(query_embedding), dtype=bool)
    for row, candidates in enumerate(support_candidates):
        if not candidates:
            continue
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        wo_angles = _angular_degrees(
            np.broadcast_to(queries_wo[row], (len(candidate_indices), 3)),
            neighborhood.wo[candidate_indices],
        )
        wi_angles = _angular_degrees(
            np.broadcast_to(queries_wi[row], (len(candidate_indices), 3)),
            neighborhood.wi[candidate_indices],
        )
        support[row] = bool(np.any((wo_angles <= maximum_wo) & (wi_angles <= maximum_wi)))
    effective_count = 1.0 / np.sum(weights * weights, axis=1)
    return (
        response.astype(np.float32),
        support.astype(np.uint8),
        nearest_wo.astype(np.float32),
        nearest_wi.astype(np.float32),
        effective_count.astype(np.float32),
    )


def _audit_raw_identity(
    protocol: MollificationProtocol,
    lock: Mapping[str, Any],
    provider: Any,
) -> dict[str, Any]:
    return {
        "format": {"name": "mollification-matched-reference", "version": 1},
        "protocol_sha256": protocol.sha256,
        "anchor_lock_sha256": lock["anchor_lock_sha256"],
        "base_corpus_id": lock["base_identity"]["corpus_id"],
        "reference": {
            "family_id": provider.descriptor.family_id,
            "reference_id": provider.descriptor.reference_id,
            "implementation_sha256": provider.descriptor.implementation_sha256,
        },
        "response_measure": MOLLIFICATION_RESPONSE_MEASURE,
        "layout": {
            "anchor_count": 24,
            "level_count": 4,
            "jitter_count": 256,
            "light_count": 4,
        },
    }


_AUDIT_RAW_DATASETS = (
    "jitter_wo", "fresh_mean", "fresh_variance", "fresh_replica_mean_a",
    "fresh_replica_mean_b", "fresh_sample_count", "reconstructed_mean",
    "support", "nearest_wo_degrees", "nearest_wi_degrees", "effective_neighbor_count",
)


def _audit_raw_semantic_hash(stream: h5py.File) -> str:
    digest = hashlib.sha256()
    digest.update(str(stream.attrs["identity_json"]).encode("utf-8"))
    for name in _AUDIT_RAW_DATASETS:
        dataset = stream[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(dataset.dtype).encode("ascii"))
        digest.update(_canonical_json(dataset.shape).encode("ascii"))
        for start in range(0, len(dataset), 4):
            digest.update(np.ascontiguousarray(dataset[start : start + 4]).tobytes())
    return digest.hexdigest()


def _read_audit_raw(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as stream:
        identity = json.loads(str(stream.attrs.get("identity_json", "{}")))
        if identity != expected_identity:
            raise ValueError("existing mollification raw shard has a different identity")
        stored = _require_sha256("raw_id", stream.attrs.get("raw_id", ""))
        if _audit_raw_semantic_hash(stream) != stored:
            raise ValueError("mollification raw shard semantic hash mismatch")
        if any(name not in stream for name in _AUDIT_RAW_DATASETS):
            raise ValueError("mollification raw shard is incomplete")
        return {
            "identity": identity,
            "raw_id": stored,
            "first_reference_result_at": str(stream.attrs["first_reference_result_at"]),
            **{name: np.asarray(stream[name]) for name in _AUDIT_RAW_DATASETS},
        }


def _write_audit_raw(
    path: Path,
    identity: Mapping[str, Any],
    first_reference_result_at: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as stream:
        stream.attrs["identity_json"] = _canonical_json(identity)
        stream.attrs["first_reference_result_at"] = first_reference_result_at
        for name in _AUDIT_RAW_DATASETS:
            stream.create_dataset(name, data=np.asarray(arrays[name]), compression="gzip", shuffle=True)
        stream.flush()
        stream.attrs["raw_id"] = _audit_raw_semantic_hash(stream)
        stream.flush()
    os.replace(temporary, path)
    return _read_audit_raw(path, identity)


def _reference_relative_standard_error(
    mean: np.ndarray,
    standard_error: np.ndarray,
    *,
    group_axes: tuple[int, ...],
    absolute_floor: float,
) -> np.ndarray:
    """复用 v5 adaptive reference 的 0.5% group-peak denominator。"""

    mean_value = np.asarray(mean, dtype=np.float64)
    error_value = np.asarray(standard_error, dtype=np.float64)
    if mean_value.shape != error_value.shape or not group_axes or absolute_floor <= 0.0:
        raise ValueError("reference relative-SE inputs are invalid")
    peak = np.max(np.abs(mean_value), axis=group_axes, keepdims=True)
    denominator_floor = np.maximum(0.005 * peak, absolute_floor)
    return error_value / np.maximum(np.abs(mean_value), denominator_floor)


def _collect_audit_raw(
    protocol: MollificationProtocol,
    lock: Mapping[str, Any],
    manifest: ReferenceCorpusManifest,
    raw_path: Path,
) -> dict[str, Any]:
    from .contract import QueryPlan, QueryRole, SurfaceSample

    state_ids = tuple(str(item["state_id"]) for item in protocol.document["representative_states"])
    paths = int(protocol.document["reference_budget"]["audit_paths_per_jitter_per_replica"])
    provider, states = _make_layer_stack_provider(
        protocol,
        manifest,
        state_ids,
        fixed_samples_per_replica=paths,
    )
    identity = _audit_raw_identity(protocol, lock, provider)
    if raw_path.exists():
        provider.close()
        return _read_audit_raw(raw_path, identity)
    layout = identity["layout"]
    shape = (
        int(layout["anchor_count"]), int(layout["level_count"]),
        int(layout["jitter_count"]), int(layout["light_count"]),
    )
    arrays = {
        "jitter_wo": np.empty(shape[:3] + (3,), dtype=np.float32),
        "fresh_mean": np.empty(shape + (3,), dtype=np.float32),
        "fresh_variance": np.empty(shape + (3,), dtype=np.float32),
        "fresh_replica_mean_a": np.empty(shape + (3,), dtype=np.float32),
        "fresh_replica_mean_b": np.empty(shape + (3,), dtype=np.float32),
        "fresh_sample_count": np.empty(shape, dtype=np.uint32),
        "reconstructed_mean": np.empty(shape + (3,), dtype=np.float32),
        "support": np.empty(shape, dtype=np.uint8),
        "nearest_wo_degrees": np.empty(shape, dtype=np.float32),
        "nearest_wi_degrees": np.empty(shape, dtype=np.float32),
        "effective_neighbor_count": np.empty(shape, dtype=np.float32),
    }
    first_reference_result_at: str | None = None
    try:
        anchors_by_state: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
        for anchor_index, anchor in enumerate(lock["anchors"]):
            anchors_by_state.setdefault(str(anchor["state_id"]), []).append((anchor_index, anchor))
        radii = tuple(map(float, protocol.document["curriculum"]["radius_degrees"]))
        for state_id in state_ids:
            neighborhood = _load_train_neighborhood(
                manifest,
                state_id,
                protocol.document["reconstruction"],
            )
            query_wo_rows: list[np.ndarray] = []
            query_wi_rows: list[np.ndarray] = []
            query_regions: list[tuple[int, int]] = []
            for anchor_index, anchor in anchors_by_state[state_id]:
                lights = np.asarray([item["wi"] for item in anchor["lights"]], dtype=np.float32)
                for level_index, radius in enumerate(radii):
                    cone_seed = _seed32(
                        int(protocol.document["seed"]), state_id,
                        int(anchor["view_index"]), level_index, "cone",
                    )
                    jitter = mollification_cone_directions(
                        anchor["wo"], radius, int(protocol.document["curriculum"]["jitter_count"]), cone_seed
                    )
                    arrays["jitter_wo"][anchor_index, level_index] = jitter
                    repeated_wo = np.repeat(jitter, len(lights), axis=0)
                    repeated_wi = np.tile(lights, (len(jitter), 1))
                    query_regions.append((anchor_index, level_index))
                    query_wo_rows.append(repeated_wo)
                    query_wi_rows.append(repeated_wi)
                    plan = QueryPlan(
                        jitter,
                        np.broadcast_to(lights[None, :, :], (len(jitter), len(lights), 3)).copy(),
                        np.ones((len(jitter), len(lights)), dtype=np.float32),
                        np.ones((len(jitter), len(lights)), dtype=np.float32),
                        "mollification-matched-upper-cap-v1",
                        _seed32(int(protocol.document["seed"]), state_id, int(anchor["view_index"]), level_index, "reference"),
                        np.full(len(jitter), int(QueryRole.TRAIN), dtype=np.uint8),
                    )
                    evaluated = provider.evaluate(states[state_id], (SurfaceSample(),), plan)
                    if first_reference_result_at is None:
                        first_reference_result_at = datetime.now(timezone.utc).isoformat()
                    region = np.s_[anchor_index, level_index]
                    arrays["fresh_mean"][region] = evaluated.mean[0]
                    arrays["fresh_variance"][region] = evaluated.variance[0]
                    arrays["fresh_replica_mean_a"][region] = evaluated.replica_mean_a[0]
                    arrays["fresh_replica_mean_b"][region] = evaluated.replica_mean_b[0]
                    arrays["fresh_sample_count"][region] = evaluated.sample_count[0]
            reconstructed = _reconstruct_neighborhood(
                neighborhood,
                np.concatenate(query_wo_rows, axis=0),
                np.concatenate(query_wi_rows, axis=0),
                protocol.document["reconstruction"],
            )
            offset = 0
            block_count = int(protocol.document["curriculum"]["jitter_count"]) * 4
            for anchor_index, level_index in query_regions:
                region = np.s_[anchor_index, level_index]
                end = offset + block_count
                for name, values in zip(
                    (
                        "reconstructed_mean", "support", "nearest_wo_degrees",
                        "nearest_wi_degrees", "effective_neighbor_count",
                    ),
                    reconstructed,
                    strict=True,
                ):
                    arrays[name][region] = values[offset:end].reshape(shape[2], shape[3], *values.shape[1:])
                offset = end
    finally:
        provider.close()
    if first_reference_result_at is None:
        raise AssertionError("mollification audit produced no reference results")
    frozen_at = datetime.fromisoformat(str(lock["frozen_at"]))
    first_at = datetime.fromisoformat(first_reference_result_at)
    if not frozen_at < first_at:
        raise RuntimeError("mollification protocol was not frozen before the first reference result")
    return _write_audit_raw(raw_path, identity, first_reference_result_at, arrays)


def _audit_report_payload(
    protocol: MollificationProtocol,
    lock: Mapping[str, Any],
    raw: Mapping[str, Any],
    raw_path: Path,
) -> dict[str, Any]:
    fresh = np.asarray(raw["fresh_mean"], dtype=np.float64)
    reconstructed = np.asarray(raw["reconstructed_mean"], dtype=np.float64)
    replica_a = np.asarray(raw["fresh_replica_mean_a"], dtype=np.float64)
    replica_b = np.asarray(raw["fresh_replica_mean_b"], dtype=np.float64)
    support = np.asarray(raw["support"], dtype=bool)
    reference_target = np.mean(fresh, axis=2)
    reconstructed_target = np.mean(reconstructed, axis=2)
    replica_target_a = np.mean(replica_a, axis=2)
    replica_target_b = np.mean(replica_b, axis=2)
    gates = protocol.document["adequacy_gates"]
    floor = float(gates["normalization_floor"])
    denominator = np.maximum(np.mean(np.abs(reference_target), axis=-1), floor)
    normalized_l1 = np.mean(np.abs(reconstructed_target - reference_target), axis=-1) / denominator
    target_se = 0.5 * np.abs(replica_target_a - replica_target_b)
    relative_se = _reference_relative_standard_error(
        reference_target,
        target_se,
        group_axes=(2, 3),
        absolute_floor=floor,
    )
    support_fraction = np.mean(support, axis=2)
    numeric = {
        "median": float(np.median(normalized_l1)),
        "p95": float(np.quantile(normalized_l1, 0.95)),
        "worst": float(np.max(normalized_l1)),
    }
    noise_p95 = float(np.quantile(relative_se, 0.95))
    support_worst = float(np.min(support_fraction))
    gate_results = {
        "support": support_worst >= float(gates["support_fraction_minimum"]),
        "noise": noise_p95 <= float(gates["fresh_relative_se_p95_maximum"]),
        "numeric": (
            numeric["median"] <= float(gates["normalized_rgb_l1_median_maximum"])
            and numeric["p95"] <= float(gates["normalized_rgb_l1_p95_maximum"])
            and numeric["worst"] <= float(gates["normalized_rgb_l1_worst_maximum"])
        ),
        "repeat": True,
    }
    target_rows: list[dict[str, Any]] = []
    radii = tuple(map(float, protocol.document["curriculum"]["radius_degrees"]))
    for anchor_index, anchor in enumerate(lock["anchors"]):
        for level_index, radius in enumerate(radii):
            for light_index, light in enumerate(anchor["lights"]):
                target_rows.append(
                    {
                        "state_id": anchor["state_id"],
                        "representative_role": anchor["representative_role"],
                        "view_index": int(anchor["view_index"]),
                        "light_role": light["role"],
                        "level_index": level_index,
                        "radius_degrees": radius,
                        "support_fraction": float(support_fraction[anchor_index, level_index, light_index]),
                        "relative_se_p95_rgb": float(np.quantile(relative_se[anchor_index, level_index, light_index], 0.95)),
                        "normalized_rgb_l1": float(normalized_l1[anchor_index, level_index, light_index]),
                        "reference_mean": reference_target[anchor_index, level_index, light_index].tolist(),
                        "reconstructed_mean": reconstructed_target[anchor_index, level_index, light_index].tolist(),
                    }
                )
    decision = "reuse-v5" if all(gate_results.values()) else "use-mollification-supplement-v1"
    return {
        "schema": AUDIT_REPORT_SCHEMA,
        "protocol_sha256": protocol.sha256,
        "anchor_lock_sha256": lock["anchor_lock_sha256"],
        "base_corpus_id": lock["base_identity"]["corpus_id"],
        "protocol_frozen_at": lock["frozen_at"],
        "first_reference_result_at": raw["first_reference_result_at"],
        "raw_uri": _project_uri(raw_path),
        "raw_id": raw["raw_id"],
        "reference": raw["identity"]["reference"],
        "response_measure": MOLLIFICATION_RESPONSE_MEASURE,
        "thresholds": dict(gates),
        "metrics": {
            "support_fraction_worst": support_worst,
            "fresh_relative_se_p95": noise_p95,
            "normalized_rgb_l1": numeric,
            "target_count": len(target_rows),
        },
        "gate_results": gate_results,
        "decision": decision,
        "targets": target_rows,
    }


def run_mollification_audit(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    raw_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """执行或复用 matched raw reference，并派生唯一二选一决定。"""

    protocol, lock, manifest = load_mollification_anchor_lock(protocol_path, anchor_lock_path)
    raw_target = Path(raw_path)
    raw = _collect_audit_raw(protocol, lock, manifest, raw_target)
    payload = _audit_report_payload(protocol, lock, raw, raw_target)
    repeated = _audit_report_payload(protocol, lock, raw, raw_target)
    if _canonical_json(payload) != _canonical_json(repeated):
        raise RuntimeError("mollification audit report is not deterministically repeatable")
    payload["report_sha256"] = _sha256_json(payload)
    _write_json_atomic(Path(output_path), payload)
    return payload


def _select_farthest_views(wo: np.ndarray, count: int) -> np.ndarray:
    views = np.asarray(wo, dtype=np.float64)
    if views.ndim != 2 or views.shape[1] != 3 or len(views) < count:
        raise ValueError("supplement view selection requires enough [N,3] directions")
    selected = [int(np.lexsort((np.arange(len(views)), views[:, 2]))[0])]
    minimum_angles = np.full(len(views), np.inf, dtype=np.float64)
    while len(selected) < count:
        newest = selected[-1]
        angles = _angular_degrees(views, np.broadcast_to(views[newest], views.shape))
        minimum_angles = np.minimum(minimum_angles, angles)
        minimum_angles[np.asarray(selected, dtype=np.int64)] = -np.inf
        maximum = float(np.max(minimum_angles))
        candidate = int(np.flatnonzero(minimum_angles == maximum)[0])
        selected.append(candidate)
    return np.asarray(selected, dtype=np.int64)


def _take_unique_ordered(candidates: Sequence[int], selected: set[int], count: int) -> list[int]:
    result: list[int] = []
    for raw in candidates:
        index = int(raw)
        if index in selected:
            continue
        selected.add(index)
        result.append(index)
        if len(result) == count:
            break
    return result


def _select_supplement_lights(
    response: np.ndarray,
    light_count: int,
    supplement: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    magnitude = np.sum(np.abs(np.asarray(response, dtype=np.float64)), axis=-1)
    source_indices = np.arange(len(magnitude), dtype=np.int64)
    selected: set[int] = set()
    peak_order = np.lexsort((source_indices, -magnitude))
    peak = _take_unique_ordered(
        peak_order,
        selected,
        int(supplement["peak_light_count"]),
    )
    log_response = np.log(np.maximum(magnitude, float(supplement["normalization_floor"])))
    response_order = np.lexsort((source_indices, log_response))
    log_candidates: list[int] = []
    requested_log = int(supplement["log_response_light_count"])
    for index in range(requested_log):
        position = int(round((index + 0.5) * len(response_order) / requested_log - 0.5))
        position = min(max(position, 0), len(response_order) - 1)
        for delta in range(len(response_order)):
            for candidate_position in (position - delta, position + delta):
                if 0 <= candidate_position < len(response_order):
                    candidate = int(response_order[candidate_position])
                    if candidate not in selected:
                        log_candidates.append(candidate)
                        break
            else:
                continue
            break
    log_rows = _take_unique_ordered(log_candidates, selected, requested_log)
    proposal_candidates = [
        min(int((index + 0.5) * len(source_indices) / int(supplement["proposal_index_light_count"])), len(source_indices) - 1)
        for index in range(int(supplement["proposal_index_light_count"]))
    ]
    proposal_candidates.extend(source_indices.tolist())
    proposal = _take_unique_ordered(
        proposal_candidates,
        selected,
        int(supplement["proposal_index_light_count"]),
    )
    result = peak + log_rows + proposal
    if len(result) != light_count or len(set(result)) != light_count:
        raise RuntimeError("supplement light selector did not produce the frozen 16+16+32 layout")
    roles = (
        ("peak",) * len(peak)
        + ("log-response",) * len(log_rows)
        + ("proposal-index",) * len(proposal)
    )
    return np.asarray(result, dtype=np.int64), roles


def freeze_mollification_supplement_anchors(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    audit_report_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """在 supplement fresh query 前锁定全部 30-state 的 8×64 来源方向。"""

    protocol, anchor_lock, manifest = load_mollification_anchor_lock(protocol_path, anchor_lock_path)
    report = json.loads(Path(audit_report_path).read_text(encoding="utf-8"))
    report_hash = _require_sha256("audit report_sha256", report.get("report_sha256", ""))
    report_payload = dict(report)
    report_payload.pop("report_sha256")
    if _sha256_json(report_payload) != report_hash:
        raise ValueError("mollification audit report hash mismatch")
    if (
        report.get("schema") != AUDIT_REPORT_SCHEMA
        or report.get("protocol_sha256") != protocol.sha256
        or report.get("anchor_lock_sha256") != anchor_lock["anchor_lock_sha256"]
        or report.get("decision") != "use-mollification-supplement-v1"
    ):
        raise ValueError("supplement freeze requires the matching failed adequacy report")
    state_ids = tuple(map(str, manifest.selection["state_ids"]))
    if len(state_ids) != 30 or len(set(state_ids)) != 30:
        raise ValueError("mollification supplement requires the frozen 30-state P1 selection")
    supplement = protocol.document["supplement"]
    state_rows: list[dict[str, Any]] = []
    for state_id in state_ids:
        shard = _state_shard(manifest, state_id, "train")
        with ReferenceDataset.open(_resolve_uri(shard.uri), verify_hashes=True) as dataset:
            local_state = dataset.state_strings("state_id").tolist().index(state_id)
            groups = np.flatnonzero(
                (np.asarray(dataset.stream["queries/state_index"], dtype=np.int64) == local_state)
                & (dataset.query_roles == 0)
            )
            batch = dataset.group_batch(groups, fields=("wo", "wi", "mean"))
            selected_views = _select_farthest_views(
                batch["wo"], int(supplement["view_count"])
            )
            views: list[dict[str, Any]] = []
            for output_view_index, local_group_index in enumerate(selected_views.tolist()):
                light_indices, light_roles = _select_supplement_lights(
                    batch["mean"][local_group_index],
                    int(supplement["light_count"]),
                    supplement,
                )
                views.append(
                    {
                        "view_index": output_view_index,
                        "source_group_index": int(groups[local_group_index]),
                        "wo": [float(value) for value in batch["wo"][local_group_index]],
                        "light_direction_indices": light_indices.tolist(),
                        "light_roles": list(light_roles),
                        "wi": batch["wi"][local_group_index, light_indices].astype(float).tolist(),
                        "source_response": batch["mean"][local_group_index, light_indices].astype(float).tolist(),
                    }
                )
            state_rows.append(
                {
                    "state_id": state_id,
                    "structure_family_id": shard.structure_family_id,
                    "difficulty_class": shard.difficulty_class,
                    "difficulty_tags": list(shard.difficulty_tags),
                    "source": {
                        "shard_id": shard.shard_id,
                        "dataset_id": shard.dataset_id,
                    },
                    "views": views,
                }
            )
    value: dict[str, Any] = {
        "schema": SUPPLEMENT_ANCHOR_LOCK_SCHEMA,
        "protocol_sha256": protocol.sha256,
        "anchor_lock_sha256": anchor_lock["anchor_lock_sha256"],
        "audit_report_sha256": report_hash,
        "base_corpus_id": manifest.corpus_id,
        "selection_sha256": manifest.selection_sha256,
        "states": state_rows,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    value["supplement_anchor_lock_sha256"] = _sha256_json(value)
    _write_json_atomic(Path(output_path), value)
    return value


def load_mollification_supplement_anchor_lock(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    audit_report_path: Path | str,
    supplement_lock_path: Path | str,
) -> tuple[MollificationProtocol, dict[str, Any], ReferenceCorpusManifest, dict[str, Any]]:
    protocol, anchor_lock, manifest = load_mollification_anchor_lock(protocol_path, anchor_lock_path)
    report = json.loads(Path(audit_report_path).read_text(encoding="utf-8"))
    report_hash = _require_sha256("audit report_sha256", report.get("report_sha256", ""))
    report_payload = dict(report)
    report_payload.pop("report_sha256")
    if _sha256_json(report_payload) != report_hash or report.get("decision") != "use-mollification-supplement-v1":
        raise ValueError("mollification supplement requires an intact failed audit report")
    value = json.loads(Path(supplement_lock_path).read_text(encoding="utf-8"))
    stored = _require_sha256(
        "supplement_anchor_lock_sha256", value.get("supplement_anchor_lock_sha256", "")
    )
    payload = dict(value)
    payload.pop("supplement_anchor_lock_sha256")
    if _sha256_json(payload) != stored:
        raise ValueError("mollification supplement anchor lock hash mismatch")
    if (
        value.get("schema") != SUPPLEMENT_ANCHOR_LOCK_SCHEMA
        or value.get("protocol_sha256") != protocol.sha256
        or value.get("anchor_lock_sha256") != anchor_lock["anchor_lock_sha256"]
        or value.get("audit_report_sha256") != report_hash
        or value.get("base_corpus_id") != manifest.corpus_id
        or value.get("selection_sha256") != manifest.selection_sha256
        or tuple(item["state_id"] for item in value.get("states", ()))
        != tuple(map(str, manifest.selection["state_ids"]))
    ):
        raise ValueError("mollification supplement anchor lock provenance mismatch")
    for state in value["states"]:
        if len(state["views"]) != 8 or any(
            len(view["wi"]) != 64 or len(view["light_direction_indices"]) != 64
            for view in state["views"]
        ):
            raise ValueError("mollification supplement anchor lock is incomplete")
    return protocol, anchor_lock, manifest, dict(value)


def write_mollification_training_data_entry(
    protocol_path: Path | str,
    anchor_lock_path: Path | str,
    audit_report_path: Path | str,
    output_path: Path | str,
    *,
    supplement_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """把 audit 唯一决定发布成 03 可消费的冻结数据入口。"""

    from .mollification_collection import validate_mollification_supplement

    protocol, anchor_lock, manifest = load_mollification_anchor_lock(
        protocol_path, anchor_lock_path
    )
    report = json.loads(Path(audit_report_path).read_text(encoding="utf-8"))
    report_hash = _require_sha256("audit report_sha256", report.get("report_sha256", ""))
    report_payload = dict(report)
    report_payload.pop("report_sha256")
    if (
        _sha256_json(report_payload) != report_hash
        or report.get("protocol_sha256") != protocol.sha256
        or report.get("anchor_lock_sha256") != anchor_lock["anchor_lock_sha256"]
    ):
        raise ValueError("mollification data entry requires the matching intact audit report")
    decision = str(report.get("decision"))
    value: dict[str, Any] = {
        "schema": TRAINING_DATA_ENTRY_SCHEMA,
        "variant": (
            "base-v5-neighborhood-v1"
            if decision == "reuse-v5"
            else "base-v5-plus-mollification-v1"
        ),
        "base_corpus_uri": _project_uri(_resolve_uri(str(protocol.document["base_corpus_uri"]))),
        "base_corpus_id": manifest.corpus_id,
        "protocol_sha256": protocol.sha256,
        "anchor_lock_sha256": anchor_lock["anchor_lock_sha256"],
        "audit_report_sha256": report_hash,
        "curriculum": {
            "stored_progress": list(protocol.document["curriculum"]["progress"]),
            "stored_radius_degrees": list(protocol.document["curriculum"]["radius_degrees"]),
            "zero_radius_switch_progress": protocol.document["curriculum"]["zero_radius_switch_progress"],
            "selection": "nearest-stored-level-v1",
            "zero_radius_source": "base-v5",
        },
    }
    if decision == "reuse-v5":
        if supplement_manifest_path is not None:
            raise ValueError("reuse-v5 data entry cannot include a supplement manifest")
        value["reconstruction"] = dict(protocol.document["reconstruction"])
    elif decision == "use-mollification-supplement-v1":
        if supplement_manifest_path is None:
            raise ValueError("failed adequacy audit requires a complete supplement manifest")
        supplement = validate_mollification_supplement(supplement_manifest_path)
        if (
            supplement["base_corpus_id"] != manifest.corpus_id
            or supplement["protocol_sha256"] != protocol.sha256
        ):
            raise ValueError("mollification supplement disagrees with the data entry provenance")
        value["supplement_corpus_uri"] = _project_uri(Path(supplement_manifest_path))
        value["supplement_corpus_id"] = supplement["corpus_id"]
    else:
        raise ValueError("mollification audit report has no supported binary decision")
    value["entry_id"] = _sha256_json(value)
    _write_json_atomic(Path(output_path), value)
    return value


def load_mollification_training_data_entry(path: Path | str) -> dict[str, Any]:
    from .mollification_collection import validate_mollification_supplement

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = _require_sha256("mollification data entry_id", value.get("entry_id", ""))
    payload = dict(value)
    payload.pop("entry_id")
    if _sha256_json(payload) != stored:
        raise ValueError("mollification training data entry identity mismatch")
    if value.get("schema") != TRAINING_DATA_ENTRY_SCHEMA:
        raise ValueError("unsupported mollification training data entry")
    base = validate_reference_corpus(_resolve_uri(str(value.get("base_corpus_uri", ""))))
    if base.corpus_id != value.get("base_corpus_id"):
        raise ValueError("mollification data entry base corpus identity mismatch")
    variant = value.get("variant")
    if variant == "base-v5-neighborhood-v1":
        _require_exact_fields(
            "base-v5-neighborhood mollification data entry",
            value,
            {
                "schema", "variant", "base_corpus_uri", "base_corpus_id",
                "protocol_sha256", "anchor_lock_sha256", "audit_report_sha256",
                "curriculum", "reconstruction", "entry_id",
            },
        )
        if "supplement_corpus_uri" in value or "supplement_corpus_id" in value:
            raise ValueError("base-v5-neighborhood data entry cannot contain a supplement")
    elif variant == "base-v5-plus-mollification-v1":
        _require_exact_fields(
            "base-v5-plus-mollification data entry",
            value,
            {
                "schema", "variant", "base_corpus_uri", "base_corpus_id",
                "protocol_sha256", "anchor_lock_sha256", "audit_report_sha256",
                "curriculum", "supplement_corpus_uri", "supplement_corpus_id",
                "entry_id",
            },
        )
        supplement = validate_mollification_supplement(
            _resolve_uri(str(value.get("supplement_corpus_uri", "")))
        )
        if (
            supplement["corpus_id"] != value.get("supplement_corpus_id")
            or supplement["base_corpus_id"] != base.corpus_id
            or supplement["protocol_sha256"] != value.get("protocol_sha256")
        ):
            raise ValueError("mollification data entry supplement identity mismatch")
    else:
        raise ValueError("unsupported mollification training data entry variant")
    curriculum = value.get("curriculum", {})
    if curriculum != {
        "stored_progress": [0.0, 0.25, 0.5, 0.75],
        "stored_radius_degrees": [10.0, 8.535533905932738, 5.0, 1.4644660940672627],
        "zero_radius_switch_progress": 0.875,
        "selection": "nearest-stored-level-v1",
        "zero_radius_source": "base-v5",
    }:
        raise ValueError("mollification data entry curriculum contract mismatch")
    return value
