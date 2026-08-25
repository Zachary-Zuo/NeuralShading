from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np

from ncls.data import (
    QUERY_ROLE_NAMES,
    ReferenceCorpusManifest,
    ReferenceDataset,
    SPLIT_NAMES,
    load_mollification_training_data_entry,
    validate_mollification_supplement,
    validate_reference_corpus,
)
from ncls.paths import PROJECT_ROOT


PARTITION_POLICY_IDS = (
    "parametric-v1",
    "target-visible-v1",
    "workflow-v1",
)


def select_mollification_curriculum_target(
    training_progress: float,
    stored_progress: Sequence[float],
    stored_radius_degrees: Sequence[float],
    zero_radius_switch_progress: float,
) -> dict[str, Any]:
    progress = float(training_progress)
    levels = np.asarray(stored_progress, dtype=np.float64)
    radii = np.asarray(stored_radius_degrees, dtype=np.float64)
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("training progress must be finite and in [0, 1]")
    if levels.shape != (4,) or radii.shape != (4,):
        raise ValueError("mollification curriculum requires four stored levels")
    if progress >= float(zero_radius_switch_progress):
        return {
            "target_source": "base-v5",
            "level_index": None,
            "level_progress": 1.0,
            "radius_degrees": 0.0,
        }
    level_index = int(np.argmin(np.abs(levels - progress)))
    return {
        "target_source": "mollified-reference",
        "level_index": level_index,
        "level_progress": float(levels[level_index]),
        "radius_degrees": float(radii[level_index]),
    }


class MollificationCurriculumStore:
    """读取冻结 supplement，并显式返回当前 curriculum target 来源。"""

    def __init__(self, data_entry_path: Path | str) -> None:
        self.data_entry_path = Path(data_entry_path)
        self.entry = load_mollification_training_data_entry(self.data_entry_path)
        if self.entry["variant"] != "base-v5-plus-mollification-v1":
            raise ValueError("mollification curriculum store requires the supplement variant")
        manifest_path = Path(str(self.entry["supplement_corpus_uri"]))
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        self.manifest = validate_mollification_supplement(manifest_path)
        self._streams: dict[str, h5py.File] = {}
        try:
            for shard in self.manifest["shards"]:
                state_id = str(shard["state_id"])
                path = Path(str(shard["uri"]))
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                self._streams[state_id] = h5py.File(path, "r")
        except Exception:
            self.close()
            raise
        self._progress = np.asarray(
            self.entry["curriculum"]["stored_progress"], dtype=np.float64
        )
        self._radii = np.asarray(
            self.entry["curriculum"]["stored_radius_degrees"], dtype=np.float64
        )
        self._switch_progress = float(
            self.entry["curriculum"]["zero_radius_switch_progress"]
        )

    @property
    def data_id(self) -> str:
        return str(self.entry["entry_id"])

    @property
    def state_count(self) -> int:
        return len(self._streams)

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()

    def __enter__(self) -> "MollificationCurriculumStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def select_target(self, training_progress: float) -> dict[str, Any]:
        return select_mollification_curriculum_target(
            training_progress,
            self._progress,
            self._radii,
            self._switch_progress,
        )

    def batch(
        self,
        state_ids: Sequence[str],
        view_indices: Sequence[int],
        light_indices: Sequence[int],
        *,
        training_progress: float,
    ) -> dict[str, np.ndarray]:
        requested_states = tuple(map(str, state_ids))
        views = np.asarray(view_indices, dtype=np.int64)
        lights = np.asarray(light_indices, dtype=np.int64)
        if len(requested_states) != len(views) or len(views) != len(lights):
            raise ValueError("mollification batch fields must have equal length")
        if not requested_states:
            raise ValueError("mollification batch must be non-empty")
        if np.any((views < 0) | (views >= 8)) or np.any((lights < 0) | (lights >= 64)):
            raise ValueError("mollification batch indices are outside the frozen 8x64 layout")
        unknown = set(requested_states) - set(self._streams)
        if unknown:
            raise ValueError(f"mollification batch contains unknown states: {sorted(unknown)}")
        target = self.select_target(training_progress)
        response = np.empty((len(views), 3), dtype=np.float32)
        wo = np.empty_like(response)
        wi = np.empty_like(response)
        for index, (state_id, view_index, light_index) in enumerate(
            zip(requested_states, views.tolist(), lights.tolist(), strict=True)
        ):
            stream = self._streams[state_id]
            wo[index] = stream["anchors/wo"][view_index]
            wi[index] = stream["anchors/wi"][view_index, light_index]
            if target["level_index"] is None:
                response[index] = stream["anchors/source_response"][view_index, light_index]
            else:
                response[index] = stream["responses/mean"][
                    view_index, int(target["level_index"]), light_index
                ]
        return {
            "state_id": np.asarray(requested_states, dtype=object),
            "view_index": views,
            "light_index": lights,
            "wo": wo,
            "wi": wi,
            "response": response,
            "mollification_progress": np.full(len(views), training_progress, dtype=np.float32),
            "mollification_level_progress": np.full(
                len(views), target["level_progress"], dtype=np.float32
            ),
            "mollification_radius_degrees": np.full(
                len(views), target["radius_degrees"], dtype=np.float32
            ),
            "target_source": np.full(
                len(views), target["target_source"], dtype=object
            ),
        }


class ReferenceQueryStore:
    """只解释公共 query/response 合同，不依赖任何源材质表示。"""

    def __init__(self, dataset_path: Path | str, *, verify_hashes: bool = True) -> None:
        self.dataset = ReferenceDataset.open(dataset_path, verify_hashes=verify_hashes)
        self.query_group_count = self.dataset.query_group_count
        self.source_split_indices = {
            name: self.dataset.group_indices(source_split=name)
            for name in SPLIT_NAMES
        }
        self.query_role_indices = {
            name: self.dataset.group_indices(query_role=name)
            for name in QUERY_ROLE_NAMES
        }

    def close(self) -> None:
        self.dataset.close()

    @property
    def data_id(self) -> str:
        return self.dataset.manifest.dataset_id

    @property
    def state_count(self) -> int:
        return self.dataset.state_count

    def state_strings(self, field: str) -> np.ndarray:
        return self.dataset.state_strings(field)

    @property
    def state_splits(self) -> np.ndarray:
        return self.dataset.state_splits

    def state_payload(self, index: int) -> bytes:
        return self.dataset.state_payload(index)

    def sanity_checks(self) -> dict[str, bool]:
        split_groups = self.state_strings("split_group_id")
        source_hashes = self.state_strings("source_sha256")

        def leak_free(values: np.ndarray) -> bool:
            seen: dict[str, set[int]] = {}
            for value, split in zip(values, self.state_splits, strict=True):
                seen.setdefault(str(value), set()).add(int(split))
            return all(len(splits) == 1 for splits in seen.values())

        return {
            "split_group_leak_free": leak_free(split_groups),
            "source_hash_leak_free": leak_free(source_hashes),
        }

    def __enter__(self) -> "ReferenceQueryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def partition_indices(self, policy_id: str, lifecycle_role: str) -> np.ndarray:
        if policy_id not in PARTITION_POLICY_IDS:
            raise ValueError(f"unsupported dataset partition policy: {policy_id}")
        if lifecycle_role not in SPLIT_NAMES:
            raise ValueError(f"lifecycle role must be one of {SPLIT_NAMES}")
        source_split = lifecycle_role if policy_id != "target-visible-v1" else None
        query_role = lifecycle_role
        return self.dataset.group_indices(source_split=source_split, query_role=query_role)

    def indices_for_query_role(self, role: str) -> np.ndarray:
        return self.dataset.group_indices(query_role=role)

    def select_indices(
        self,
        indices: np.ndarray,
        selection: Mapping[str, Any],
    ) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("selection indices must be one-dimensional")
        if not selection:
            return requested
        allowed = {"state_ids", "asset_ids", "family_ids"}
        unknown = set(selection) - allowed
        if unknown:
            raise ValueError(f"unsupported dataset selection fields: {sorted(unknown)}")
        query_states = np.asarray(self.dataset.stream["queries/state_index"], dtype=np.int64)[requested]
        keep = np.ones(len(requested), dtype=bool)
        for field, state_field in (
            ("state_ids", "state_id"),
            ("asset_ids", "asset_id"),
            ("family_ids", "family_id"),
        ):
            if field not in selection:
                continue
            available = self.dataset.state_strings(state_field)
            accepted = set(map(str, selection[field]))
            unknown_values = accepted - set(map(str, available.tolist()))
            if unknown_values:
                raise ValueError(f"dataset selection contains unknown {field}: {sorted(unknown_values)}")
            keep &= np.asarray([str(available[state]) in accepted for state in query_states])
        return requested[keep]

    @staticmethod
    def sample_batch_indices(candidates: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if not len(candidates):
            raise ValueError("dataset lifecycle partition is empty")
        return np.asarray(rng.choice(candidates, size=batch_size, replace=True), dtype=np.int64)

    def batch(
        self,
        query_group_indices: np.ndarray,
        *,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        requested = np.asarray(query_group_indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("query_group_indices must be one-dimensional")
        result = self.dataset.group_batch(requested, fields=fields)
        if fields is None:
            result["view"] = result["wo"]
            result["lights"] = result["wi"]
        return result

    def iter_batches(
        self,
        query_group_indices: np.ndarray,
        batch_size: int,
        *,
        fields: tuple[str, ...] | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        for start in range(0, len(query_group_indices), batch_size):
            yield self.batch(
                query_group_indices[start : start + batch_size], fields=fields
            )

    def query_state_indices(self, query_group_indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.dataset.stream["queries/state_index"], dtype=np.int64
        )[np.asarray(query_group_indices, dtype=np.int64)]


class ReferenceCorpusStore:
    """把矩形 HDF5 shard 暴露为一个语料；每个实际 batch 只来自一个 shard。"""

    def __init__(self, manifest_path: Path | str) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest: ReferenceCorpusManifest = validate_reference_corpus(self.manifest_path)
        self.shards: list[ReferenceQueryStore] = []
        for shard in self.manifest.shards:
            path = Path(shard.uri)
            resolved = path if path.is_absolute() else PROJECT_ROOT / path
            self.shards.append(ReferenceQueryStore(resolved))
        metadata: dict[str, dict[str, Any]] = {}
        for store in self.shards:
            fields = {
                name: store.state_strings(name)
                for name in (
                    "state_id", "family_id", "asset_id", "structure_family_id",
                    "difficulty_class", "difficulty_tags_json", "evaluation_cohort",
                )
            }
            state_ids = fields["state_id"]
            for local_index, state_id in enumerate(state_ids):
                record = {
                    "state_id": str(state_id),
                    "family_id": str(fields["family_id"][local_index]),
                    "asset_id": str(fields["asset_id"][local_index]),
                    "structure_family_id": str(fields["structure_family_id"][local_index]),
                    "difficulty_class": str(fields["difficulty_class"][local_index]),
                    "difficulty_tags_json": str(fields["difficulty_tags_json"][local_index]),
                    "evaluation_cohort": str(fields["evaluation_cohort"][local_index]),
                    "split": int(store.state_splits[local_index]),
                }
                previous = metadata.setdefault(str(state_id), record)
                if previous != record:
                    raise ValueError(f"state metadata changes across corpus shards: {state_id}")
        self._state_ids = np.asarray(sorted(metadata), dtype=object)
        self._state_index = {
            str(state_id): index for index, state_id in enumerate(self._state_ids)
        }
        self._payload_sources: dict[str, tuple[int, int]] = {}
        for shard_index, store in enumerate(self.shards):
            for local_index, state_id in enumerate(store.state_strings("state_id")):
                self._payload_sources.setdefault(str(state_id), (shard_index, local_index))
        self._state_fields = {
            field: np.asarray([metadata[str(state_id)][field] for state_id in self._state_ids], dtype=object)
            for field in (
                "state_id", "family_id", "asset_id", "structure_family_id",
                "difficulty_class", "difficulty_tags_json", "evaluation_cohort",
            )
        }
        self._state_splits = np.asarray(
            [metadata[str(state_id)]["split"] for state_id in self._state_ids],
            dtype=np.uint8,
        )

    @property
    def data_id(self) -> str:
        assert self.manifest.corpus_id is not None
        return self.manifest.corpus_id

    @property
    def state_count(self) -> int:
        return len(self._state_ids)

    def state_strings(self, field: str) -> np.ndarray:
        try:
            return self._state_fields[field]
        except KeyError as error:
            raise KeyError(f"unsupported corpus state string field {field!r}") from error

    @property
    def state_splits(self) -> np.ndarray:
        return self._state_splits

    def state_payload(self, index: int) -> bytes:
        if index < 0 or index >= self.state_count:
            raise IndexError("corpus state index is out of range")
        state_id = str(self._state_ids[index])
        shard_index, local_index = self._payload_sources[state_id]
        return self.shards[shard_index].dataset.state_payload(local_index)

    def sanity_checks(self) -> dict[str, bool]:
        return {
            "split_group_leak_free": True,
            "source_hash_leak_free": True,
            "corpus_roles_complete": True,
        }

    def close(self) -> None:
        for store in self.shards:
            store.close()

    def __enter__(self) -> "ReferenceCorpusStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _join(self, rows: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(rows, axis=0) if rows else np.empty((0, 2), dtype=np.int64)

    def partition_indices(self, policy_id: str, lifecycle_role: str) -> np.ndarray:
        rows = []
        for shard_index, store in enumerate(self.shards):
            local = store.partition_indices(policy_id, lifecycle_role)
            if len(local):
                rows.append(np.column_stack((np.full(len(local), shard_index), local)))
        return self._join(rows)

    def indices_for_query_role(self, role: str) -> np.ndarray:
        rows = []
        for shard_index, store in enumerate(self.shards):
            local = store.indices_for_query_role(role)
            if len(local):
                rows.append(np.column_stack((np.full(len(local), shard_index), local)))
        return self._join(rows)

    def select_indices(
        self,
        indices: np.ndarray,
        selection: Mapping[str, Any],
    ) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 2 or requested.shape[1] != 2:
            raise ValueError("corpus query references must have [shard, group] shape")
        if not selection:
            return requested
        allowed = {"state_ids", "asset_ids", "family_ids"}
        if set(selection) - allowed:
            raise ValueError("dataset selection contains unsupported fields")
        state_indices = self.query_state_indices(requested)
        keep = np.ones(len(requested), dtype=bool)
        for name, field in (
            ("state_ids", "state_id"),
            ("asset_ids", "asset_id"),
            ("family_ids", "family_id"),
        ):
            if name in selection:
                accepted = set(map(str, selection[name]))
                available = self.state_strings(field)
                unknown = accepted - set(map(str, available.tolist()))
                if unknown:
                    raise ValueError(f"dataset selection contains unknown {name}: {sorted(unknown)}")
                keep &= np.asarray([str(available[index]) in accepted for index in state_indices])
        return requested[keep]

    @staticmethod
    def sample_batch_indices(
        candidates: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        values = np.asarray(candidates, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values):
            raise ValueError("corpus sampling requires nonempty [shard, group] candidates")
        shard_ids, counts = np.unique(values[:, 0], return_counts=True)
        shard = int(rng.choice(shard_ids, p=counts / np.sum(counts)))
        local = values[values[:, 0] == shard]
        selected = rng.choice(len(local), size=batch_size, replace=True)
        return local[selected]

    def batch(
        self,
        query_references: np.ndarray,
        *,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        requested = np.asarray(query_references, dtype=np.int64)
        if requested.ndim != 2 or requested.shape[1] != 2 or not len(requested):
            raise ValueError("corpus batch requires nonempty [shard, group] references")
        if len(np.unique(requested[:, 0])) != 1:
            raise ValueError("one rectangular batch may only read one shard")
        shard_index = int(requested[0, 0])
        raw = self.shards[shard_index].batch(requested[:, 1], fields=fields)
        local_state_index = raw["state_index"].copy()
        if fields is None:
            raw["shard_state_index"] = local_state_index
        state_ids = self.shards[shard_index].state_strings("state_id")
        raw["state_index"] = np.asarray(
            [self._state_index[str(state_ids[index])] for index in local_state_index],
            dtype=np.int64,
        )
        if fields is None:
            raw["shard_index"] = np.full(len(requested), shard_index, dtype=np.int64)
        return raw

    def iter_batches(
        self,
        query_references: np.ndarray,
        batch_size: int,
        *,
        fields: tuple[str, ...] | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        values = np.asarray(query_references, dtype=np.int64)
        for shard_index in sorted(set(map(int, values[:, 0].tolist()))):
            selected = values[values[:, 0] == shard_index]
            for start in range(0, len(selected), batch_size):
                yield self.batch(
                    selected[start : start + batch_size], fields=fields
                )

    def query_state_indices(self, query_references: np.ndarray) -> np.ndarray:
        values = np.asarray(query_references, dtype=np.int64)
        result = np.empty(len(values), dtype=np.int64)
        for shard_index in sorted(set(map(int, values[:, 0].tolist()))):
            selected = np.flatnonzero(values[:, 0] == shard_index)
            local_states = self.shards[shard_index].query_state_indices(values[selected, 1])
            state_ids = self.shards[shard_index].state_strings("state_id")
            result[selected] = [self._state_index[str(state_ids[index])] for index in local_states]
        return result


def open_reference_store(path: Path | str) -> ReferenceQueryStore | ReferenceCorpusStore:
    target = Path(path)
    if target.suffix.lower() == ".json":
        return ReferenceCorpusStore(target)
    return ReferenceQueryStore(target)
