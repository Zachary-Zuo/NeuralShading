from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from ncls.paths import PROJECT_ROOT

from .contract import QUERY_ROLE_NAMES, SPLIT_NAMES
from .corpus import ReferenceCorpusManifest, validate_reference_corpus
from .dataset import ReferenceDataset


PARTITION_POLICY_IDS = ("parametric-v1", "target-visible-v1", "workflow-v1")


class ReferenceQueryStore:
    """只解释公共 query/response 合同，不依赖 source family 或方法。"""

    def __init__(self, dataset_path: Path | str, *, verify_hashes: bool = True) -> None:
        self.dataset = ReferenceDataset.open(dataset_path, verify_hashes=verify_hashes)

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> "ReferenceQueryStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

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

    def partition_indices(self, policy_id: str, lifecycle_role: str) -> np.ndarray:
        if policy_id not in PARTITION_POLICY_IDS:
            raise ValueError(f"unsupported dataset partition policy: {policy_id}")
        if lifecycle_role not in SPLIT_NAMES:
            raise ValueError(f"lifecycle role must be one of {SPLIT_NAMES}")
        source_split = lifecycle_role if policy_id != "target-visible-v1" else None
        return self.dataset.group_indices(source_split=source_split, query_role=lifecycle_role)

    def indices_for_query_role(self, role: str) -> np.ndarray:
        if role not in QUERY_ROLE_NAMES:
            raise ValueError(f"query role must be one of {QUERY_ROLE_NAMES}")
        return self.dataset.group_indices(query_role=role)

    def select_indices(self, indices: np.ndarray, selection: Mapping[str, Any]) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("selection indices must be one-dimensional")
        if not selection:
            return requested
        fields = {"state_ids": "state_id", "asset_ids": "asset_id", "family_ids": "family_id"}
        if set(selection) - set(fields):
            raise ValueError("dataset selection contains unsupported fields")
        states = np.asarray(self.dataset.stream["queries/state_index"], dtype=np.int64)[requested]
        keep = np.ones(len(requested), dtype=bool)
        for name, field in fields.items():
            if name not in selection:
                continue
            available = self.state_strings(field)
            accepted = set(map(str, selection[name]))
            unknown = accepted - set(map(str, available.tolist()))
            if unknown:
                raise ValueError(f"dataset selection contains unknown {name}: {sorted(unknown)}")
            keep &= np.asarray([str(available[index]) in accepted for index in states])
        return requested[keep]

    @staticmethod
    def sample_batch_indices(candidates: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if batch_size < 1 or not len(candidates):
            raise ValueError("dataset sampling requires positive batch size and nonempty candidates")
        return np.asarray(rng.choice(candidates, size=batch_size, replace=True), dtype=np.int64)

    def batch(self, query_group_indices: np.ndarray, *, fields: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        requested = np.asarray(query_group_indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("query_group_indices must be one-dimensional")
        result = self.dataset.group_batch(requested, fields=fields)
        if fields is None:
            result["view"], result["lights"] = result["wo"], result["wi"]
        return result

    def iter_batches(self, indices: np.ndarray, batch_size: int, *, fields: tuple[str, ...] | None = None) -> Iterator[dict[str, np.ndarray]]:
        for start in range(0, len(indices), batch_size):
            yield self.batch(indices[start : start + batch_size], fields=fields)

    def query_state_indices(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.dataset.stream["queries/state_index"], dtype=np.int64)[np.asarray(indices, dtype=np.int64)]


class ReferenceCorpusStore:
    """把矩形 HDF5 shards 暴露为一个统一 offline query source。"""

    def __init__(self, manifest_path: Path | str) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest: ReferenceCorpusManifest = validate_reference_corpus(self.manifest_path)
        self.shards = []
        for shard in self.manifest.shards:
            path = Path(shard.uri)
            self.shards.append(ReferenceQueryStore(path if path.is_absolute() else PROJECT_ROOT / path))
        metadata: dict[str, dict[str, Any]] = {}
        self._payload_sources: dict[str, tuple[int, int]] = {}
        fields = ("state_id", "family_id", "asset_id", "structure_family_id", "difficulty_class", "difficulty_tags_json", "evaluation_cohort")
        for shard_index, store in enumerate(self.shards):
            values = {name: store.state_strings(name) for name in fields}
            for local_index, state_id in enumerate(values["state_id"]):
                record = {name: str(values[name][local_index]) for name in fields}
                record["split"] = int(store.state_splits[local_index])
                previous = metadata.setdefault(str(state_id), record)
                if previous != record:
                    raise ValueError(f"state metadata changes across corpus shards: {state_id}")
                self._payload_sources.setdefault(str(state_id), (shard_index, local_index))
        self._state_ids = np.asarray(sorted(metadata), dtype=object)
        self._state_index = {str(value): index for index, value in enumerate(self._state_ids)}
        self._state_fields = {
            name: np.asarray([metadata[str(state_id)][name] for state_id in self._state_ids], dtype=object)
            for name in fields
        }
        self._state_splits = np.asarray([metadata[str(value)]["split"] for value in self._state_ids], dtype=np.uint8)

    @property
    def data_id(self) -> str:
        if self.manifest.corpus_id is None:
            raise ValueError("ReferenceCorpus manifest has no identity")
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
        state_id = str(self._state_ids[index])
        shard, local = self._payload_sources[state_id]
        return self.shards[shard].state_payload(local)

    def close(self) -> None:
        for store in self.shards:
            store.close()

    def __enter__(self) -> "ReferenceCorpusStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _join(rows: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(rows, axis=0) if rows else np.empty((0, 2), dtype=np.int64)

    def partition_indices(self, policy_id: str, lifecycle_role: str) -> np.ndarray:
        rows = []
        for index, store in enumerate(self.shards):
            local = store.partition_indices(policy_id, lifecycle_role)
            if len(local):
                rows.append(np.column_stack((np.full(len(local), index), local)))
        return self._join(rows)

    def indices_for_query_role(self, role: str) -> np.ndarray:
        rows = []
        for index, store in enumerate(self.shards):
            local = store.indices_for_query_role(role)
            if len(local):
                rows.append(np.column_stack((np.full(len(local), index), local)))
        return self._join(rows)

    def select_indices(self, indices: np.ndarray, selection: Mapping[str, Any]) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("corpus query references must have [shard, group] shape")
        if not selection:
            return values
        fields = {"state_ids": "state_id", "asset_ids": "asset_id", "family_ids": "family_id"}
        if set(selection) - set(fields):
            raise ValueError("dataset selection contains unsupported fields")
        state_indices = self.query_state_indices(values)
        keep = np.ones(len(values), dtype=bool)
        for name, field in fields.items():
            if name in selection:
                accepted = set(map(str, selection[name]))
                available = self.state_strings(field)
                keep &= np.asarray([str(available[index]) in accepted for index in state_indices])
        return values[keep]

    @staticmethod
    def sample_batch_indices(candidates: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        values = np.asarray(candidates, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values) or batch_size < 1:
            raise ValueError("corpus sampling requires nonempty [shard, group] candidates")
        shard_ids, counts = np.unique(values[:, 0], return_counts=True)
        shard = int(rng.choice(shard_ids, p=counts / np.sum(counts)))
        local = values[values[:, 0] == shard]
        return local[rng.choice(len(local), size=batch_size, replace=True)]

    def batch(self, references: np.ndarray, *, fields: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        values = np.asarray(references, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values) or len(np.unique(values[:, 0])) != 1:
            raise ValueError("one corpus batch must contain [shard, group] rows from one shard")
        shard_index = int(values[0, 0])
        raw = self.shards[shard_index].batch(values[:, 1], fields=fields)
        local_states = raw["state_index"].copy()
        state_ids = self.shards[shard_index].state_strings("state_id")
        raw["state_index"] = np.asarray([self._state_index[str(state_ids[index])] for index in local_states], dtype=np.int64)
        return raw

    def iter_batches(self, references: np.ndarray, batch_size: int, *, fields: tuple[str, ...] | None = None) -> Iterator[dict[str, np.ndarray]]:
        values = np.asarray(references, dtype=np.int64)
        for shard in sorted(set(map(int, values[:, 0].tolist()))):
            selected = values[values[:, 0] == shard]
            for start in range(0, len(selected), batch_size):
                yield self.batch(selected[start : start + batch_size], fields=fields)

    def query_state_indices(self, references: np.ndarray) -> np.ndarray:
        values = np.asarray(references, dtype=np.int64)
        result = np.empty(len(values), dtype=np.int64)
        for shard in sorted(set(map(int, values[:, 0].tolist()))):
            selected = np.flatnonzero(values[:, 0] == shard)
            local = self.shards[shard].query_state_indices(values[selected, 1])
            ids = self.shards[shard].state_strings("state_id")
            result[selected] = [self._state_index[str(ids[index])] for index in local]
        return result


def open_reference_store(path: Path | str) -> ReferenceQueryStore | ReferenceCorpusStore:
    target = Path(path)
    return ReferenceCorpusStore(target) if target.suffix.lower() == ".json" else ReferenceQueryStore(target)
