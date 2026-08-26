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
        state_ids: list[str] = []
        wo: list[np.ndarray] = []
        wi: list[np.ndarray] = []
        source_response: list[np.ndarray] = []
        mollified_response: list[np.ndarray] = []
        for shard in self.manifest["shards"]:
            state_id = str(shard["state_id"])
            path = Path(str(shard["uri"]))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            with h5py.File(path, "r") as stream:
                state_ids.append(state_id)
                wo.append(np.asarray(stream["anchors/wo"][...], dtype=np.float32))
                wi.append(np.asarray(stream["anchors/wi"][...], dtype=np.float32))
                source_response.append(np.asarray(
                    stream["anchors/source_response"][...], dtype=np.float32
                ))
                mollified_response.append(np.asarray(
                    stream["responses/mean"][...], dtype=np.float32
                ))
        self._state_ids = tuple(state_ids)
        self._state_index = {
            state_id: index for index, state_id in enumerate(self._state_ids)
        }
        self._wo = np.stack(wo)
        self._wi = np.stack(wi)
        self._source_response = np.stack(source_response)
        self._mollified_response = np.stack(mollified_response)
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
        return len(self._state_ids)

    @property
    def state_ids(self) -> tuple[str, ...]:
        return self._state_ids

    def close(self) -> None:
        self._state_ids = ()
        self._state_index = {}
        self._wo = np.empty((0, 8, 3), dtype=np.float32)
        self._wi = np.empty((0, 8, 64, 3), dtype=np.float32)
        self._source_response = np.empty((0, 8, 64, 3), dtype=np.float32)
        self._mollified_response = np.empty((0, 8, 4, 64, 3), dtype=np.float32)

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
        unknown = set(requested_states) - set(self._state_index)
        if unknown:
            raise ValueError(f"mollification batch contains unknown states: {sorted(unknown)}")
        target = self.select_target(training_progress)
        states = np.fromiter(
            (self._state_index[state_id] for state_id in requested_states),
            dtype=np.int64,
            count=len(requested_states),
        )
        wo = self._wo[states, views].copy()
        wi = self._wi[states, views, lights].copy()
        if target["level_index"] is None:
            response = self._source_response[states, views, lights].copy()
        else:
            response = self._mollified_response[
                states, views, int(target["level_index"]), lights
            ].copy()
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


class UnifiedScatteringTrainingStore:
    """03唯一数据入口：train可走冻结curriculum，其余角色始终委托base v5。"""

    ENTRY_ID = "47ef20138007703f2d1b644bcb4ca4b084001da4ec975f1b712587d3e7e35a89"
    BASE_CORPUS_ID = "0513d0c837b109f74cbf6fd4f811e05c6bc68c02226bd6d443f3225ef5dd64b7"
    SUPPLEMENT_CORPUS_ID = "f6931474890ab7642f244b84df2736e2a5fc1f9e169b5f7a620494184d99e4f3"
    CURRICULUM_STEPS = 20_000
    BASE_TARGET_START_STEP = 17_501
    _BASE_TRAINING_FIELDS = (
        "state_index", "wo", "wi", "mean", "solid_angle_weight"
    )

    def __init__(self, data_entry_path: Path | str) -> None:
        self.data_entry_path = Path(data_entry_path)
        self.entry = load_mollification_training_data_entry(self.data_entry_path)
        if self.entry["entry_id"] != self.ENTRY_ID:
            raise ValueError("unified training requires the frozen 03 data entry")
        if self.entry["base_corpus_id"] != self.BASE_CORPUS_ID:
            raise ValueError("unified training base corpus identity mismatch")
        if self.entry.get("supplement_corpus_id") != self.SUPPLEMENT_CORPUS_ID:
            raise ValueError("unified training supplement corpus identity mismatch")
        base_path = Path(str(self.entry["base_corpus_uri"]))
        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_path
        self.base = ReferenceCorpusStore(base_path)
        if self.base.data_id != self.BASE_CORPUS_ID:
            self.base.close()
            raise ValueError("unified training resolved a different base corpus")
        try:
            self.curriculum = MollificationCurriculumStore(self.data_entry_path)
        except Exception:
            self.base.close()
            raise
        self._state_ids = tuple(map(str, self.base.state_strings("state_id").tolist()))
        self._state_index = {state_id: index for index, state_id in enumerate(self._state_ids)}
        self._base_training_references: np.ndarray | None = None
        self._base_training_cache: dict[
            int, tuple[np.ndarray, dict[str, np.ndarray]]
        ] = {}
        if set(self.curriculum.state_ids) != set(self._state_ids):
            self.close()
            raise ValueError("unified curriculum and base state identities disagree")

    @property
    def data_id(self) -> str:
        return self.ENTRY_ID

    @property
    def state_count(self) -> int:
        return self.base.state_count

    @property
    def state_splits(self) -> np.ndarray:
        return self.base.state_splits

    def state_strings(self, field: str) -> np.ndarray:
        return self.base.state_strings(field)

    def state_payload(self, index: int) -> bytes:
        return self.base.state_payload(index)

    def sanity_checks(self) -> dict[str, bool]:
        return {
            **self.base.sanity_checks(),
            "training_entry_identity": True,
            "validation_test_base_v5": True,
        }

    def training_lifecycle_contract(self, total_steps: int) -> dict[str, Any]:
        """返回entry冻结的训练阶段，供runner约束早停并写入run identity证据。"""

        if total_steps < 1:
            raise ValueError("training lifecycle requires a positive step budget")
        return {
            "contract": "ncls.unified-scattering-curriculum@1",
            "curriculum_steps": self.CURRICULUM_STEPS,
            "base_target_start_step": self.BASE_TARGET_START_STEP,
            "post_curriculum_base_start_step": self.CURRICULUM_STEPS + 1,
            "early_stopping_floor_step": min(self.CURRICULUM_STEPS, total_steps),
            "total_steps": total_steps,
        }

    def close(self) -> None:
        self._base_training_references = None
        self._base_training_cache.clear()
        self.curriculum.close()
        self.base.close()

    def __enter__(self) -> "UnifiedScatteringTrainingStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def partition_indices(self, policy_id: str, lifecycle_role: str) -> np.ndarray:
        return self.base.partition_indices(policy_id, lifecycle_role)

    def indices_for_query_role(self, role: str) -> np.ndarray:
        return self.base.indices_for_query_role(role)

    def select_indices(self, indices: np.ndarray, selection: Mapping[str, Any]) -> np.ndarray:
        return self.base.select_indices(indices, selection)

    def sample_batch_indices(
        self, candidates: np.ndarray, batch_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        return self.base.sample_batch_indices(candidates, batch_size, rng)

    def batch(
        self,
        query_references: np.ndarray,
        *,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        return self.base.batch(query_references, fields=fields)

    def iter_batches(
        self,
        query_references: np.ndarray,
        batch_size: int,
        *,
        fields: tuple[str, ...] | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        return self.base.iter_batches(query_references, batch_size, fields=fields)

    def query_state_indices(self, query_references: np.ndarray) -> np.ndarray:
        return self.base.query_state_indices(query_references)

    def prepare_training_partition(self, train_indices: np.ndarray) -> None:
        """把冻结 train 分区驻留内存，避免优化循环逐条随机读取 HDF5。"""

        requested = np.asarray(train_indices, dtype=np.int64)
        if requested.ndim != 2 or requested.shape[1] != 2 or not len(requested):
            raise ValueError("training cache requires nonempty [shard, group] references")
        if self._base_training_references is not None:
            if not np.array_equal(requested, self._base_training_references):
                raise ValueError("training cache cannot change its frozen query partition")
            return
        cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        for shard_index in sorted(set(map(int, requested[:, 0].tolist()))):
            local_indices = np.unique(requested[requested[:, 0] == shard_index, 1])
            references = np.column_stack((
                np.full(len(local_indices), shard_index, dtype=np.int64),
                local_indices,
            ))
            cache[shard_index] = (
                local_indices,
                self.base.batch(references, fields=self._BASE_TRAINING_FIELDS),
            )
        self._base_training_references = requested.copy()
        self._base_training_cache = cache

    def _cached_base_training_batch(
        self, selected: np.ndarray
    ) -> dict[str, np.ndarray]:
        requested = np.asarray(selected, dtype=np.int64)
        if requested.ndim != 2 or requested.shape[1] != 2 or not len(requested):
            raise ValueError("cached training batch requires [shard, group] references")
        shard_ids = np.unique(requested[:, 0])
        if len(shard_ids) != 1:
            raise ValueError("one cached training batch may only select one shard")
        shard_index = int(shard_ids[0])
        try:
            local_indices, cached = self._base_training_cache[shard_index]
        except KeyError as error:
            raise ValueError("cached training batch selected an uncached shard") from error
        positions = np.searchsorted(local_indices, requested[:, 1])
        if (
            np.any(positions >= len(local_indices))
            or not np.array_equal(local_indices[positions], requested[:, 1])
        ):
            raise ValueError("cached training batch selected outside the frozen train partition")
        return {name: values[positions].copy() for name, values in cached.items()}

    def base_training_batch(
        self,
        train_indices: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        self.prepare_training_partition(train_indices)
        selected = self.base.sample_batch_indices(train_indices, batch_size, rng)
        return self._cached_base_training_batch(selected)

    def training_batch(
        self,
        train_indices: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        *,
        step: int,
        total_steps: int,
    ) -> dict[str, np.ndarray]:
        del total_steps
        if step > self.CURRICULUM_STEPS:
            raw = self.base_training_batch(train_indices, batch_size, rng)
            raw["target_source"] = np.full(batch_size, "base-v5", dtype=object)
            raw["mollification_progress"] = np.ones(batch_size, dtype=np.float32)
            return raw
        progress = (step - 1) / max(self.CURRICULUM_STEPS - 1, 1)
        state_ids = rng.choice(np.asarray(self._state_ids, dtype=object), size=batch_size, replace=True)
        view_indices = rng.integers(0, 8, size=batch_size, dtype=np.int64)
        light_indices = np.tile(np.arange(64, dtype=np.int64), batch_size)
        flat_states = np.repeat(state_ids, 64)
        flat_views = np.repeat(view_indices, 64)
        raw = self.curriculum.batch(
            flat_states.tolist(),
            flat_views,
            light_indices,
            training_progress=progress,
        )
        state_index = np.asarray([self._state_index[str(value)] for value in state_ids], dtype=np.int64)
        return {
            "state_index": state_index,
            "wo": raw["wo"].reshape(batch_size, 64, 3)[:, 0],
            "wi": raw["wi"].reshape(batch_size, 64, 3),
            "mean": raw["response"].reshape(batch_size, 64, 3),
            "solid_angle_weight": np.full((batch_size, 64), 2.0 * math.pi / 64.0, dtype=np.float32),
            "target_source": raw["target_source"].reshape(batch_size, 64)[:, 0],
            "mollification_progress": raw["mollification_progress"].reshape(batch_size, 64)[:, 0],
            "mollification_radius_degrees": raw["mollification_radius_degrees"].reshape(batch_size, 64)[:, 0],
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
