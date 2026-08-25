from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .contract import QUERY_ROLE_NAMES, SPLIT_NAMES, EvaluatedBlock, QueryPlan, SourceState, SurfaceSample


FORMAT_NAME = "reference-shard"
FORMAT_VERSION = 5
RESPONSE_MEASURE = "rgb-bsdf-times-absolute-shading-normal-light-cosine"
COLOR_MODEL = "linear-srgb"
_STRING = h5py.string_dtype(encoding="utf-8")


@dataclass(frozen=True)
class ReferenceDatasetManifest:
    dataset_id: str
    created_at: str
    generator_git_commit: str
    sampling_name: str
    counts: Mapping[str, int]
    generation_config: Mapping[str, Any]
    provider_metadata: tuple[Mapping[str, Any], ...]
    format_name: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    response_measure: str = RESPONSE_MEASURE
    color_model: str = COLOR_MODEL


@dataclass(frozen=True)
class ReferenceStatistics:
    mean: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    sample_count: np.ndarray
    replica_mean_a: np.ndarray
    replica_mean_b: np.ndarray


_STATE_DATASETS = (
    "state_id", "family_id", "reference_id", "asset_id", "split_group_id",
    "structure_family_id", "difficulty_class", "difficulty_tags_json",
    "evaluation_cohort", "native_schema_id", "source_uri", "source_sha256",
    "parent_state_id", "split",
    "payload_offsets", "payload_blob",
)
_QUERY_DATASETS = (
    "state_index", "query_role", "position_kind", "position", "uv", "uv_dx", "uv_dy",
    "geometric_normal", "geometric_tangent", "wo", "wi", "proposal_code", "proposal_pdf",
    "solid_angle_weight", "rng_seed",
)
_PRIMARY_RESPONSE_DATASETS = (
    "mean", "variance", "replica_mean_a", "replica_mean_b", "sample_count",
    "valid", "event_flags", "reference_pdf",
)
_RECIPROCAL_RESPONSE_DATASETS = (
    "reciprocal_mean", "reciprocal_variance", "reciprocal_sample_count",
)
_RESPONSE_DATASETS = (*_PRIMARY_RESPONSE_DATASETS, *_RECIPROCAL_RESPONSE_DATASETS)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _semantic_hash(stream: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in (
        "format_name", "format_version", "generator_git_commit",
        "response_measure", "color_model", "sampling_name",
        "generation_config_json", "provider_metadata_json", "proposal_ids_json",
    ):
        value = stream.attrs[name]
        payload = str(value).encode("utf-8")
        digest.update(name.encode("ascii") + b"\0" + len(payload).to_bytes(8, "little") + payload)
    for group_name, dataset_names in (
        ("states", _STATE_DATASETS), ("queries", _QUERY_DATASETS), ("responses", _RESPONSE_DATASETS),
    ):
        group = stream[group_name]
        for dataset_name in dataset_names:
            dataset = group[dataset_name]
            digest.update(f"{group_name}/{dataset_name}".encode("ascii") + b"\0")
            digest.update(str(dataset.dtype).encode("ascii") + b"\0")
            digest.update(np.asarray(dataset.shape, dtype="<u8").tobytes())
            if h5py.check_string_dtype(dataset.dtype) is not None:
                for item in dataset.asstr()[...].reshape(-1):
                    payload = str(item).encode("utf-8")
                    digest.update(len(payload).to_bytes(8, "little") + payload)
                continue
            if dataset.ndim == 0:
                digest.update(np.asarray(dataset[()]).tobytes())
                continue
            step = max(1, min(dataset.shape[0], 4096))
            for start in range(0, dataset.shape[0], step):
                digest.update(np.ascontiguousarray(dataset[start : start + step]).tobytes())
    return digest.hexdigest()


def _create_extendible(group: h5py.Group, name: str, tail: tuple[int, ...], dtype: Any) -> h5py.Dataset:
    chunk_first = 64 if not tail else max(1, min(256, 65536 // max(int(np.prod(tail)), 1)))
    return group.create_dataset(
        name,
        shape=(0, *tail),
        maxshape=(None, *tail),
        chunks=(chunk_first, *tail),
        dtype=dtype,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


class ReferenceDatasetWriter:
    """原子写入唯一的 HDF5 合同；只接受公共 state/query/response 语义。"""

    def __init__(
        self,
        path: Path | str,
        states: Sequence[SourceState],
        *,
        created_at: str,
        generator_git_commit: str,
        sampling_name: str,
        generation_config: Mapping[str, Any],
        provider_metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("ReferenceDataset output must be an .h5 or .hdf5 file")
        if self.path.exists():
            raise FileExistsError(f"reference shard already exists: {self.path}")
        if not states:
            raise ValueError("ReferenceDataset requires at least one source state")
        state_ids = [state.state_id for state in states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("source state IDs must be unique")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.name + ".tmp")
        if self.temporary.exists():
            self.temporary.unlink()
        self.stream = h5py.File(self.temporary, "w")
        attrs = self.stream.attrs
        attrs["format_name"] = FORMAT_NAME
        attrs["format_version"] = FORMAT_VERSION
        attrs["created_at"] = created_at
        attrs["generator_git_commit"] = generator_git_commit
        attrs["response_measure"] = RESPONSE_MEASURE
        attrs["color_model"] = COLOR_MODEL
        if not sampling_name:
            raise ValueError("reference shard requires a sampling name")
        attrs["sampling_name"] = sampling_name
        attrs["generation_config_json"] = _json(dict(generation_config))
        attrs["provider_metadata_json"] = _json([dict(item) for item in provider_metadata])
        attrs["proposal_ids_json"] = _json({})

        state_group = self.stream.create_group("states")
        string_fields = (
            "state_id", "family_id", "reference_id", "asset_id", "split_group_id",
            "structure_family_id", "difficulty_class", "evaluation_cohort",
            "native_schema_id", "source_uri", "source_sha256", "parent_state_id",
        )
        for field in string_fields:
            state_group.create_dataset(field, data=np.asarray([getattr(state, field) for state in states], dtype=object), dtype=_STRING)
        state_group.create_dataset(
            "difficulty_tags_json",
            data=np.asarray([_json(list(state.difficulty_tags)) for state in states], dtype=object),
            dtype=_STRING,
        )
        state_group.create_dataset("split", data=np.asarray([state.split for state in states], dtype=np.uint8))
        payload_offsets = np.zeros(len(states) + 1, dtype=np.uint64)
        payload_parts = []
        for index, state in enumerate(states):
            part = np.frombuffer(state.native_payload, dtype=np.uint8)
            payload_parts.append(part)
            payload_offsets[index + 1] = payload_offsets[index] + len(part)
        state_group.create_dataset("payload_offsets", data=payload_offsets)
        state_group.create_dataset("payload_blob", data=np.concatenate(payload_parts), compression="gzip", compression_opts=4)

        query_group = self.stream.create_group("queries")
        response_group = self.stream.create_group("responses")
        self.direction_count: int | None = None
        self.query_group_count = 0
        self._query_group = query_group
        self._response_group = response_group
        self._proposal_codes: dict[str, int] = {}

    def _initialize_query_datasets(self, direction_count: int) -> None:
        self.direction_count = direction_count
        for name, tail, dtype in (
            ("state_index", (), "<u4"), ("query_role", (), "u1"), ("position_kind", (), "u1"),
            ("position", (3,), "<f4"), ("uv", (2,), "<f4"),
            ("uv_dx", (2,), "<f4"), ("uv_dy", (2,), "<f4"),
            ("geometric_normal", (3,), "<f4"), ("geometric_tangent", (3,), "<f4"),
            ("wo", (3,), "<f4"), ("wi", (direction_count, 3), "<f4"),
            ("proposal_pdf", (direction_count,), "<f4"),
            ("solid_angle_weight", (direction_count,), "<f4"),
            ("rng_seed", (direction_count,), "<u8"),
        ):
            _create_extendible(self._query_group, name, tail, dtype)
        self._query_group.create_dataset("proposal_code", shape=(0,), maxshape=(None,), chunks=(256,), dtype="<u2")
        for name, tail, dtype in (
            ("mean", (direction_count, 3), "<f4"),
            ("variance", (direction_count, 3), "<f4"),
            ("replica_mean_a", (direction_count, 3), "<f4"),
            ("replica_mean_b", (direction_count, 3), "<f4"),
            ("sample_count", (direction_count,), "<u4"),
            ("valid", (direction_count,), "u1"),
            ("event_flags", (direction_count,), "<u4"),
            ("reference_pdf", (direction_count,), "<f4"),
            ("reciprocal_mean", (direction_count, 3), "<f4"),
            ("reciprocal_variance", (direction_count, 3), "<f4"),
            ("reciprocal_sample_count", (direction_count,), "<u4"),
        ):
            _create_extendible(self._response_group, name, tail, dtype)

    def append(
        self,
        state_index: int,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
        evaluated: EvaluatedBlock,
        reciprocal: EvaluatedBlock,
    ) -> None:
        if not 0 <= state_index < len(self.stream["states/state_id"]):
            raise IndexError("state_index is outside the state table")
        if not surfaces:
            raise ValueError("each provider state requires at least one surface sample")
        if self.direction_count is None:
            self._initialize_query_datasets(plan.direction_count)
        if plan.direction_count != self.direction_count:
            raise ValueError("all query groups in one HDF5 dataset must share direction_count")
        expected = (len(surfaces), len(plan.view_directions), self.direction_count, 3)
        if evaluated.mean.shape != expected:
            raise ValueError(f"provider returned {evaluated.mean.shape}, expected {expected}")
        if reciprocal.mean.shape != expected:
            raise ValueError(f"reciprocal provider returned {reciprocal.mean.shape}, expected {expected}")
        group_count = len(surfaces) * len(plan.view_directions)
        start = self.query_group_count
        end = start + group_count
        view_proposal_codes = np.asarray([
            self._proposal_codes.setdefault(name, len(self._proposal_codes))
            for name in plan.proposal_id
        ], dtype=np.uint16)
        surface_rows = [surface for surface in surfaces for _ in plan.view_directions]
        views = np.tile(plan.view_directions, (len(surfaces), 1))
        if plan.light_directions.ndim == 3:
            light_directions = np.tile(plan.light_directions, (len(surfaces), 1, 1))
            proposal_pdf = np.tile(plan.proposal_pdf, (len(surfaces), 1))
            solid_angle_weight = np.tile(plan.solid_angle_weights, (len(surfaces), 1))
        else:
            if plan.light_directions.shape[0] != len(surfaces):
                raise ValueError("surface-dependent QueryPlan must match the appended surface count")
            light_directions = plan.light_directions.reshape(group_count, self.direction_count, 3)
            proposal_pdf = plan.proposal_pdf.reshape(group_count, self.direction_count)
            solid_angle_weight = plan.solid_angle_weights.reshape(group_count, self.direction_count)
        query_values: dict[str, np.ndarray] = {
            "state_index": np.full(group_count, state_index, dtype=np.uint32),
            "query_role": np.tile(plan.query_roles, len(surfaces)),
            "position_kind": np.asarray([surface.position_kind for surface in surface_rows], dtype=np.uint8),
            "position": np.asarray([surface.position for surface in surface_rows], dtype=np.float32),
            "uv": np.asarray([surface.uv for surface in surface_rows], dtype=np.float32),
            "uv_dx": np.asarray([surface.uv_dx for surface in surface_rows], dtype=np.float32),
            "uv_dy": np.asarray([surface.uv_dy for surface in surface_rows], dtype=np.float32),
            "geometric_normal": np.asarray([surface.geometric_normal for surface in surface_rows], dtype=np.float32),
            "geometric_tangent": np.asarray([surface.geometric_tangent for surface in surface_rows], dtype=np.float32),
            "wo": views,
            "wi": light_directions,
            "proposal_pdf": proposal_pdf,
            "solid_angle_weight": solid_angle_weight,
        }
        query_values["rng_seed"] = evaluated.rng_seed.reshape(group_count, self.direction_count)
        for name, values in query_values.items():
            dataset = self._query_group[name]
            dataset.resize(end, axis=0)
            dataset[start:end] = values
        codes = self._query_group["proposal_code"]
        codes.resize(end, axis=0)
        codes[start:end] = np.tile(view_proposal_codes, len(surfaces))
        for name in _PRIMARY_RESPONSE_DATASETS:
            values = np.asarray(getattr(evaluated, name)).reshape(group_count, *getattr(evaluated, name).shape[2:])
            dataset = self._response_group[name]
            dataset.resize(end, axis=0)
            dataset[start:end] = values
        for name, source_name in (
            ("reciprocal_mean", "mean"),
            ("reciprocal_variance", "variance"),
            ("reciprocal_sample_count", "sample_count"),
        ):
            source = np.asarray(getattr(reciprocal, source_name))
            values = source.reshape(group_count, *source.shape[2:])
            dataset = self._response_group[name]
            dataset.resize(end, axis=0)
            dataset[start:end] = values
        self.query_group_count = end

    def finalize(self) -> ReferenceDatasetManifest:
        if self.query_group_count < 1 or self.direction_count is None:
            raise ValueError("ReferenceDataset requires at least one evaluated query group")
        self.stream.attrs["proposal_ids_json"] = _json(
            {str(code): name for name, code in sorted(self._proposal_codes.items(), key=lambda item: item[1])}
        )
        self.stream.attrs["state_count"] = len(self.stream["states/state_id"])
        self.stream.attrs["query_group_count"] = self.query_group_count
        self.stream.attrs["direction_count"] = self.direction_count
        self.stream.flush()
        dataset_id = _semantic_hash(self.stream)
        self.stream.attrs["dataset_id"] = dataset_id
        self.stream.flush()
        self.stream.close()
        os.replace(self.temporary, self.path)
        with ReferenceDataset.open(self.path, verify_hashes=True) as dataset:
            return dataset.manifest

    def abort(self) -> None:
        if self.stream:
            self.stream.close()
        if self.temporary.exists():
            self.temporary.unlink()


class ReferenceDataset:
    def __init__(self, path: Path, stream: h5py.File, manifest: ReferenceDatasetManifest) -> None:
        self.path = path
        self.stream = stream
        self.manifest = manifest

    @classmethod
    def open(cls, path: Path | str, *, verify_hashes: bool = True) -> "ReferenceDataset":
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        stream = h5py.File(dataset_path, "r")
        try:
            if _decode(stream.attrs.get("format_name", "")) != FORMAT_NAME or int(stream.attrs.get("format_version", -1)) != FORMAT_VERSION:
                raise ValueError("unsupported ReferenceDataset format; collect a v5 shard from its CorpusPlan")
            if _decode(stream.attrs.get("response_measure", "")) != RESPONSE_MEASURE:
                raise ValueError("unsupported ReferenceDataset response measure")
            if _decode(stream.attrs.get("color_model", "")) != COLOR_MODEL:
                raise ValueError("unsupported ReferenceDataset color model")
            for group_name, names in (
                ("states", _STATE_DATASETS), ("queries", _QUERY_DATASETS), ("responses", _RESPONSE_DATASETS),
            ):
                if group_name not in stream or any(name not in stream[group_name] for name in names):
                    raise ValueError(f"ReferenceDataset is missing required {group_name} datasets")
            dataset_id = _decode(stream.attrs.get("dataset_id", ""))
            if verify_hashes and _semantic_hash(stream) != dataset_id:
                raise ValueError("ReferenceDataset semantic content hash mismatch")
            manifest = ReferenceDatasetManifest(
                dataset_id=dataset_id,
                created_at=_decode(stream.attrs["created_at"]),
                generator_git_commit=_decode(stream.attrs["generator_git_commit"]),
                sampling_name=_decode(stream.attrs["sampling_name"]),
                counts={
                    "state_count": int(stream.attrs["state_count"]),
                    "query_group_count": int(stream.attrs["query_group_count"]),
                    "direction_count": int(stream.attrs["direction_count"]),
                },
                generation_config=json.loads(_decode(stream.attrs["generation_config_json"])),
                provider_metadata=tuple(json.loads(_decode(stream.attrs["provider_metadata_json"]))),
            )
            result = cls(dataset_path, stream, manifest)
            result.validate_structure()
            return result
        except Exception:
            stream.close()
            raise

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "ReferenceDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def state_count(self) -> int:
        return self.manifest.counts["state_count"]

    @property
    def query_group_count(self) -> int:
        return self.manifest.counts["query_group_count"]

    @property
    def direction_count(self) -> int:
        return self.manifest.counts["direction_count"]

    def state_strings(self, field: str) -> np.ndarray:
        return np.asarray(self.stream[f"states/{field}"].asstr()[...], dtype=object)

    @property
    def state_splits(self) -> np.ndarray:
        return np.asarray(self.stream["states/split"], dtype=np.uint8)

    def state_payload(self, index: int) -> bytes:
        if not 0 <= index < self.state_count:
            raise IndexError(index)
        offsets = self.stream["states/payload_offsets"]
        start, end = int(offsets[index]), int(offsets[index + 1])
        return np.asarray(self.stream["states/payload_blob"][start:end], dtype=np.uint8).tobytes()

    @property
    def query_roles(self) -> np.ndarray:
        return np.asarray(self.stream["queries/query_role"], dtype=np.uint8)

    def group_indices(
        self,
        *,
        source_split: str | None = None,
        query_role: str | None = None,
    ) -> np.ndarray:
        if source_split is not None and source_split not in SPLIT_NAMES:
            raise ValueError(f"source_split must be one of {SPLIT_NAMES}")
        if query_role is not None and query_role not in QUERY_ROLE_NAMES:
            raise ValueError(f"query_role must be one of {QUERY_ROLE_NAMES}")
        state_indices = np.asarray(self.stream["queries/state_index"], dtype=np.int64)
        mask = np.ones(len(state_indices), dtype=bool)
        if source_split is not None:
            mask &= self.state_splits[state_indices] == SPLIT_NAMES.index(source_split)
        if query_role is not None:
            mask &= self.query_roles == QUERY_ROLE_NAMES.index(query_role)
        return np.flatnonzero(mask).astype(np.int64)

    @staticmethod
    def _read_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or np.any(requested < 0) or np.any(requested >= dataset.shape[0]):
            raise IndexError("query group index is outside the dataset")
        order = np.argsort(requested, kind="stable")
        sorted_indices = requested[order]
        unique, inverse = np.unique(sorted_indices, return_inverse=True)
        sorted_values = np.asarray(dataset[unique])[inverse]
        result = np.empty_like(sorted_values)
        result[order] = sorted_values
        return result

    def group_batch(
        self,
        indices: Iterable[int],
        *,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        requested = np.asarray(tuple(indices), dtype=np.int64)
        if fields is None:
            query_fields = _QUERY_DATASETS
            response_fields = _RESPONSE_DATASETS
        else:
            if len(fields) != len(set(fields)):
                raise ValueError("group batch fields must be unique")
            unknown = set(fields) - set(_QUERY_DATASETS) - set(_RESPONSE_DATASETS)
            if unknown:
                raise ValueError(f"unsupported raw group batch fields: {sorted(unknown)}")
            query_fields = tuple(name for name in fields if name in _QUERY_DATASETS)
            response_fields = tuple(name for name in fields if name in _RESPONSE_DATASETS)
        result = {
            name: self._read_rows(self.stream[f"queries/{name}"], requested)
            for name in query_fields
        }
        result.update({
            name: self._read_rows(self.stream[f"responses/{name}"], requested)
            for name in response_fields
        })
        if fields is None:
            state_indices = result["state_index"].astype(np.int64)
            result["source_split"] = self.state_splits[state_indices]
            result["query_group_id"] = requested
            denominator = np.maximum(
                result["sample_count"].astype(np.float32), 1.0
            )[..., None]
            result["standard_error"] = np.sqrt(
                np.maximum(result["variance"], 0.0) / denominator
            ).astype(np.float32)
            reciprocal_denominator = np.maximum(
                result["reciprocal_sample_count"].astype(np.float32), 1.0
            )[..., None]
            result["reciprocal_standard_error"] = np.sqrt(
                np.maximum(result["reciprocal_variance"], 0.0)
                / reciprocal_denominator
            ).astype(np.float32)
        return result

    def statistics(self, group_index: int) -> ReferenceStatistics:
        batch = self.group_batch((group_index,))
        return ReferenceStatistics(
            batch["mean"][0], batch["variance"][0], batch["standard_error"][0],
            batch["sample_count"][0], batch["replica_mean_a"][0], batch["replica_mean_b"][0],
        )

    def validate_structure(self) -> None:
        states = self.state_count
        groups = self.query_group_count
        directions = self.direction_count
        if len(self.stream["states/state_id"]) != states or len(self.stream["states/payload_offsets"]) != states + 1:
            raise ValueError("ReferenceDataset state table shape mismatch")
        offsets = np.asarray(self.stream["states/payload_offsets"], dtype=np.uint64)
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(self.stream["states/payload_blob"]):
            raise ValueError("ReferenceDataset state payload offsets are invalid")
        if np.any(self.state_splits >= 3):
            raise ValueError("ReferenceDataset contains an invalid split code")
        if np.any(self.query_roles >= len(QUERY_ROLE_NAMES)):
            raise ValueError("ReferenceDataset contains an invalid query role code")
        state_ids = self.state_strings("state_id")
        source_hashes = self.state_strings("source_sha256")
        difficulty_classes = self.state_strings("difficulty_class")
        evaluation_cohorts = self.state_strings("evaluation_cohort")
        if len(set(state_ids.tolist())) != states:
            raise ValueError("ReferenceDataset contains duplicate state IDs")
        for name, values in (("state_id", state_ids), ("source_sha256", source_hashes)):
            if any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in values
            ):
                raise ValueError(f"states/{name} contains an invalid SHA-256 digest")
        if any(value not in {"W", "G", "S", "unclassified"} for value in difficulty_classes):
            raise ValueError("states/difficulty_class contains an unsupported value")
        if any(value not in {"train", "validation", "g2", "g2s", "workflow"} for value in evaluation_cohorts):
            raise ValueError("states/evaluation_cohort contains an unsupported value")
        for value in self.state_strings("difficulty_tags_json"):
            tags = json.loads(str(value))
            if not isinstance(tags, list) or any(tag not in {"T", "M"} for tag in tags):
                raise ValueError("states/difficulty_tags_json contains an unsupported value")
        for name in _QUERY_DATASETS:
            if self.stream[f"queries/{name}"].shape[0] != groups:
                raise ValueError(f"queries/{name} group count mismatch")
        for name in _RESPONSE_DATASETS:
            if self.stream[f"responses/{name}"].shape[0] != groups:
                raise ValueError(f"responses/{name} group count mismatch")
        if self.stream["queries/wi"].shape != (groups, directions, 3):
            raise ValueError("queries/wi shape mismatch")
        if self.stream["responses/mean"].shape != (groups, directions, 3):
            raise ValueError("responses/mean shape mismatch")
        def chunks(path: str):
            dataset = self.stream[path]
            for start in range(0, len(dataset), 4096):
                yield np.asarray(dataset[start : start + 4096])

        if any(np.any(values >= states) for values in chunks("queries/state_index")):
            raise ValueError("query group references an invalid source state")
        proposal_ids = json.loads(_decode(self.stream.attrs["proposal_ids_json"]))
        if any(
            str(int(code)) not in proposal_ids
            for values in chunks("queries/proposal_code")
            for code in np.unique(values)
        ):
            raise ValueError("query group references an unknown proposal code")
        for path in ("queries/wo", "queries/wi", "queries/geometric_normal", "queries/geometric_tangent"):
            dataset = self.stream[path]
            for start in range(0, len(dataset), 4096):
                values = np.asarray(dataset[start : start + 4096], dtype=np.float32)
                lengths = np.linalg.norm(values, axis=-1)
                if not np.all(np.isfinite(values)) or np.any(np.abs(lengths - 1.0) > 2e-4):
                    raise ValueError(f"{path} contains an invalid direction")
        for path in ("queries/position", "queries/uv", "queries/uv_dx", "queries/uv_dy"):
            if any(not np.all(np.isfinite(values)) for values in chunks(path)):
                raise ValueError(f"{path} contains a non-finite value")
        for path in ("queries/proposal_pdf", "queries/solid_angle_weight"):
            if any(not np.all(np.isfinite(values)) or np.any(values <= 0.0) for values in chunks(path)):
                raise ValueError(f"{path} must be positive and finite")
        for path in (
            "responses/mean", "responses/variance", "responses/replica_mean_a",
            "responses/replica_mean_b", "responses/reference_pdf",
            "responses/reciprocal_mean", "responses/reciprocal_variance",
        ):
            if any(not np.all(np.isfinite(values)) for values in chunks(path)):
                raise ValueError(f"{path} contains a non-finite value")
        if any(np.any(values < 0.0) for values in chunks("responses/variance")):
            raise ValueError("responses/variance contains a negative value")
        if any(np.any(values < 0.0) for values in chunks("responses/reciprocal_variance")):
            raise ValueError("responses/reciprocal_variance contains a negative value")
        if any(np.any(values < 0.0) for values in chunks("responses/reference_pdf")):
            raise ValueError("responses/reference_pdf contains a negative value")
        if any(np.any(values < 1) for values in chunks("responses/sample_count")):
            raise ValueError("responses/sample_count contains zero")
        if any(np.any(values < 1) for values in chunks("responses/reciprocal_sample_count")):
            raise ValueError("responses/reciprocal_sample_count contains zero")
        if any(np.any(values > 1) for values in chunks("responses/valid")):
            raise ValueError("responses/valid must contain only zero or one")


def validate_reference_dataset(path: Path | str, *, verify_hashes: bool = True) -> ReferenceDataset:
    return ReferenceDataset.open(path, verify_hashes=verify_hashes)
