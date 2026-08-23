from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ncls.core.material import BINARY_SIZE, MaterialProgram, unpack_layer_stack

from .manifest import ReferenceDatasetManifest, ShardRecord, sha256_file


MATERIAL_STATE_DTYPE = np.dtype(
    [
        ("family_index", "<u4"),
        ("local_state_index", "<u4"),
        ("program_index", "<u4"),
        ("canonical_ir_index", "<u4"),
        ("split", "u1"),
        ("reserved", "u1", (15,)),
    ],
    align=False,
)

INDEX_DTYPE = np.dtype(
    [
        ("tile_id", "<u8"),
        ("material_state_index", "<u4"),
        ("view_index", "<u4"),
        ("family_index", "<u4"),
        ("split", "u1"),
        ("reserved", "u1", (11,)),
    ],
    align=False,
)


def make_response_dtype(light_count: int) -> np.dtype:
    if light_count < 1:
        raise ValueError("light_count must be positive")
    return np.dtype(
        [
            ("mean", "<f4", (light_count, 3)),
            ("variance", "<f4", (light_count, 3)),
            ("replica_mean_a", "<f4", (light_count, 3)),
            ("replica_mean_b", "<f4", (light_count, 3)),
            ("sample_count", "<u4", (light_count,)),
        ],
        align=False,
    )


@dataclass(frozen=True)
class ReferenceStatistics:
    mean: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    sample_count: np.ndarray
    replica_mean_a: np.ndarray | None
    replica_mean_b: np.ndarray | None
    uncertainty_kind: str


def _atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_common_files(
    root: Path,
    *,
    material_programs: Sequence[MaterialProgram],
    canonical_material_irs: Sequence[bytes],
    material_states: np.ndarray,
    family_splits: np.ndarray,
    view_directions: np.ndarray,
    light_directions: np.ndarray,
    solid_angle_weights: np.ndarray,
    reuse_identical: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """写入 v2 公共文件，返回逻辑名到 URI 及 URI 到哈希的映射。"""

    root.mkdir(parents=True, exist_ok=True)
    states = np.asarray(material_states)
    splits = np.asarray(family_splits, dtype=np.uint8)
    views = np.asarray(view_directions, dtype=np.float32)
    lights = np.asarray(light_directions, dtype=np.float32)
    weights = np.asarray(solid_angle_weights, dtype=np.float32)
    if states.dtype != MATERIAL_STATE_DTYPE:
        raise ValueError("material_states uses an unexpected dtype")
    if states.ndim != 1 or splits.ndim != 1:
        raise ValueError("material states and family splits must be one-dimensional")
    if views.ndim != 2 or views.shape[1] != 4 or lights.ndim != 2 or lights.shape[1] != 4:
        raise ValueError("direction arrays must have shape [count, 4]")
    if weights.shape != (len(lights),):
        raise ValueError("solid angle weights must match light directions")
    if len(material_programs) < 1 or len(canonical_material_irs) != len(material_programs):
        raise ValueError("material programs and canonical IR payloads must have the same nonzero count")
    if any(len(payload) != BINARY_SIZE for payload in canonical_material_irs):
        raise ValueError("canonical IR payload has the wrong stride")
    if np.any(states["reserved"] != 0):
        raise ValueError("material state reserved bytes must be zero")

    files = {
        "material_programs": "material_programs.jsonl",
        "canonical_material_ir": "canonical_material_ir.bin",
        "material_states": "material_states.npy",
        "family_splits": "family_splits.npy",
        "view_directions": "view_directions.npy",
        "light_directions": "light_directions.npy",
        "solid_angle_weights": "solid_angle_weights.npy",
    }
    jsonl = "".join(program.to_json(indent=None) for program in material_programs).encode("utf-8")
    binary_payloads = {
        files["material_programs"]: jsonl,
        files["canonical_material_ir"]: b"".join(canonical_material_irs),
    }
    for uri, payload in binary_payloads.items():
        path = root / uri
        if reuse_identical and path.is_file():
            if path.read_bytes() != payload:
                raise ValueError(f"existing common dataset file differs during resume: {uri}")
        else:
            _atomic_write_bytes(path, payload)
    arrays = {
        files["material_states"]: states,
        files["family_splits"]: splits,
        files["view_directions"]: views,
        files["light_directions"]: lights,
        files["solid_angle_weights"]: weights,
    }
    for uri, value in arrays.items():
        path = root / uri
        if reuse_identical and path.is_file():
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if existing.dtype != value.dtype or existing.shape != value.shape or not np.array_equal(existing, value):
                raise ValueError(f"existing common dataset file differs during resume: {uri}")
        else:
            _atomic_save(path, value)
    hashes = {uri: sha256_file(root / uri) for uri in files.values()}
    return files, hashes


def write_response_shard(
    root: Path,
    *,
    shard_id: int,
    tile_start: int,
    index: np.ndarray,
    response: np.ndarray,
    resume: bool = False,
) -> ShardRecord:
    """以两个原子数组和最后一个完成标记写入一个可恢复分片。"""

    if shard_id < 0 or tile_start < 0:
        raise ValueError("shard id and tile start must be nonnegative")
    index_array = np.asarray(index)
    response_array = np.asarray(response)
    if index_array.dtype != INDEX_DTYPE or index_array.ndim != 1:
        raise ValueError("index uses an unexpected dtype or shape")
    if response_array.ndim != 1 or len(response_array) != len(index_array):
        raise ValueError("response and index must contain the same number of tiles")
    if len(index_array) < 1:
        raise ValueError("a shard cannot be empty")
    expected_ids = np.arange(tile_start, tile_start + len(index_array), dtype=np.uint64)
    if not np.array_equal(index_array["tile_id"], expected_ids):
        raise ValueError("index tile ids must match the shard range")
    if np.any(index_array["reserved"] != 0):
        raise ValueError("index reserved bytes must be zero")
    required_response_fields = {"mean", "variance", "replica_mean_a", "replica_mean_b", "sample_count"}
    if response_array.dtype.names is None or set(response_array.dtype.names) != required_response_fields:
        raise ValueError("response uses an unexpected dtype")
    for field in ("mean", "variance", "replica_mean_a", "replica_mean_b"):
        if not np.all(np.isfinite(response_array[field])):
            raise ValueError(f"response field {field} contains non-finite values")
    if any(np.any(response_array[field] < 0.0) for field in ("mean", "replica_mean_a", "replica_mean_b")):
        raise ValueError("reference response values must be nonnegative")
    if np.any(response_array["variance"] < 0.0) or np.any(response_array["sample_count"] == 0):
        raise ValueError("response variance and sample counts are invalid")

    base = f"shard-{shard_id:05d}"
    index_uri = f"shards/{base}.index.npy"
    response_uri = f"shards/{base}.response.npy"
    completion_uri = f"shards/{base}.complete.json"
    completion_path = root / completion_uri
    if resume:
        completed = resume_response_shard(root, shard_id=shard_id, tile_start=tile_start, tile_count=len(index_array))
        if completed is not None:
            return completed

    _atomic_save(root / index_uri, index_array)
    _atomic_save(root / response_uri, response_array)
    record = ShardRecord(
        shard_id,
        tile_start,
        len(index_array),
        index_uri,
        response_uri,
        completion_uri,
        sha256_file(root / index_uri),
        sha256_file(root / response_uri),
    )
    _atomic_write_bytes(completion_path, (json.dumps(record.to_dict(), sort_keys=True) + "\n").encode("utf-8"))
    return record


def resume_response_shard(
    root: Path,
    *,
    shard_id: int,
    tile_start: int,
    tile_count: int,
) -> ShardRecord | None:
    """验证并返回已完成分片；不存在完成标记时返回 ``None``。"""

    base = f"shard-{shard_id:05d}"
    completion_path = root / f"shards/{base}.complete.json"
    if not completion_path.exists():
        return None
    value = json.loads(completion_path.read_text(encoding="utf-8"))
    record = ShardRecord.from_dict(value)
    if record.shard_id != shard_id or record.tile_start != tile_start or record.tile_count != tile_count:
        raise ValueError(f"completed shard {shard_id} does not match the requested range")
    if sha256_file(root / record.index_uri) != record.index_sha256:
        raise ValueError(f"completed shard {shard_id} index hash mismatch")
    if sha256_file(root / record.response_uri) != record.response_sha256:
        raise ValueError(f"completed shard {shard_id} response hash mismatch")
    return record


def write_manifest_atomic(root: Path, manifest: ReferenceDatasetManifest) -> None:
    _atomic_write_bytes(root / "manifest.json", manifest.to_json().encode("utf-8"))


class ReferenceDataset:
    """manifest 驱动、默认校验内容哈希的只读参考数据集。"""

    def __init__(self, root: Path, manifest: ReferenceDatasetManifest, *, verify_hashes: bool) -> None:
        self.root = root
        self.manifest = manifest
        self._verify_hashes = verify_hashes
        if verify_hashes:
            self._verify_common_hashes()
            self._verify_shard_hashes()
        self.material_states = np.load(root / manifest.files["material_states"], mmap_mode="r", allow_pickle=False)
        self.family_splits = np.load(root / manifest.files["family_splits"], mmap_mode="r", allow_pickle=False)
        self.view_directions = np.load(root / manifest.files["view_directions"], mmap_mode="r", allow_pickle=False)
        self.light_directions = np.load(root / manifest.files["light_directions"], mmap_mode="r", allow_pickle=False)
        self.solid_angle_weights = np.load(root / manifest.files["solid_angle_weights"], mmap_mode="r", allow_pickle=False)
        self._programs: tuple[MaterialProgram, ...] | None = None
        self._indices: dict[int, np.ndarray] = {}
        self._responses: dict[int, np.ndarray] = {}
        self.validate_structure()

    @classmethod
    def open(cls, root: Path | str, *, verify_hashes: bool = True) -> ReferenceDataset:
        path = Path(root)
        manifest_path = path / "manifest.json"
        manifest = ReferenceDatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        return cls(path, manifest, verify_hashes=verify_hashes)

    def _verify_common_hashes(self) -> None:
        for uri, expected in self.manifest.content_hashes.items():
            path = self.root / uri
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"dataset content hash mismatch: {uri}")

    def _verify_shard_hashes(self) -> None:
        for shard in self.manifest.shards:
            completion = self.root / shard.completion_uri
            if not completion.is_file():
                raise ValueError(f"dataset shard is incomplete: {shard.shard_id}")
            completed = ShardRecord.from_dict(json.loads(completion.read_text(encoding="utf-8")))
            if completed != shard:
                raise ValueError(f"dataset shard completion record mismatch: {shard.shard_id}")
            if sha256_file(self.root / shard.index_uri) != shard.index_sha256:
                raise ValueError(f"dataset shard index hash mismatch: {shard.shard_id}")
            if sha256_file(self.root / shard.response_uri) != shard.response_sha256:
                raise ValueError(f"dataset shard response hash mismatch: {shard.shard_id}")

    @property
    def material_programs(self) -> tuple[MaterialProgram, ...]:
        if self._programs is None:
            path = self.root / self.manifest.files["material_programs"]
            self._programs = tuple(MaterialProgram.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        return self._programs

    def canonical_material_ir(self, index: int):
        path = self.root / self.manifest.files["canonical_material_ir"]
        with path.open("rb") as stream:
            stream.seek(index * BINARY_SIZE)
            payload = stream.read(BINARY_SIZE)
        return unpack_layer_stack(payload)

    def _load_shard(self, shard: ShardRecord) -> tuple[np.ndarray, np.ndarray]:
        if shard.shard_id not in self._indices:
            self._indices[shard.shard_id] = np.load(self.root / shard.index_uri, mmap_mode="r", allow_pickle=False)
            self._responses[shard.shard_id] = np.load(self.root / shard.response_uri, mmap_mode="r", allow_pickle=False)
        return self._indices[shard.shard_id], self._responses[shard.shard_id]

    def _find_shard(self, tile_id: int) -> ShardRecord:
        if not 0 <= tile_id < int(self.manifest.counts["tile_count"]):
            raise IndexError(tile_id)
        for shard in self.manifest.shards:
            if shard.tile_start <= tile_id < shard.tile_start + shard.tile_count:
                return shard
        raise AssertionError("manifest shard coverage was not validated")

    def tile_index(self, tile_id: int) -> np.void:
        shard = self._find_shard(tile_id)
        index, _ = self._load_shard(shard)
        return index[tile_id - shard.tile_start]

    def statistics(self, tile_id: int) -> ReferenceStatistics:
        shard = self._find_shard(tile_id)
        _, response = self._load_shard(shard)
        record = response[tile_id - shard.tile_start]
        mean = np.asarray(record["mean"], dtype=np.float32)
        variance = np.asarray(record["variance"], dtype=np.float32)
        count = np.asarray(record["sample_count"], dtype=np.uint32)
        denominator = np.maximum(count.astype(np.float32), 1.0)[:, None]
        uncertainty_kind = str(self.manifest.statistics_encoding["uncertainty_kind"])
        if uncertainty_kind == "sample-population-variance":
            standard_error = np.sqrt(variance / denominator)
        elif uncertainty_kind == "replica-mean-variance":
            standard_error = np.sqrt(variance)
        else:
            raise ValueError(f"unsupported uncertainty kind {uncertainty_kind!r}")
        return ReferenceStatistics(
            mean,
            variance,
            standard_error,
            count,
            np.asarray(record["replica_mean_a"], dtype=np.float32),
            np.asarray(record["replica_mean_b"], dtype=np.float32),
            uncertainty_kind,
        )

    def validate_structure(self) -> None:
        counts = self.manifest.counts
        if self.material_states.dtype != MATERIAL_STATE_DTYPE or len(self.material_states) != counts["material_state_count"]:
            raise ValueError("material_states does not match manifest")
        if self.family_splits.dtype != np.dtype("uint8") or len(self.family_splits) != counts["family_count"]:
            raise ValueError("family_splits does not match manifest")
        if self.view_directions.shape != (counts["view_count"], 4):
            raise ValueError("view_directions does not match manifest")
        if self.light_directions.shape != (counts["light_count"], 4):
            raise ValueError("light_directions does not match manifest")
        if self.solid_angle_weights.shape != (counts["light_count"],):
            raise ValueError("solid_angle_weights does not match manifest")
        if not np.isclose(float(np.sum(self.solid_angle_weights, dtype=np.float64)), 2.0 * np.pi, rtol=1e-5):
            raise ValueError("solid angle weights must integrate the hemisphere")
        if np.any(self.material_states["family_index"] >= counts["family_count"]):
            raise ValueError("material state references an invalid family")
        if not np.array_equal(
            self.material_states["split"], self.family_splits[self.material_states["family_index"]]
        ):
            raise ValueError("material states violate family-level split assignment")
        ir_size = (self.root / self.manifest.files["canonical_material_ir"]).stat().st_size
        ir_count = ir_size // BINARY_SIZE
        if ir_size % BINARY_SIZE or ir_count < len(self.material_states):
            raise ValueError("canonical material IR file has an invalid size")
        if np.any(self.material_states["canonical_ir_index"] >= ir_count):
            raise ValueError("material state references an invalid canonical IR record")
        expected_tile_id = 0
        for shard in self.manifest.shards:
            index, response = self._load_shard(shard)
            if index.dtype != INDEX_DTYPE or len(index) != shard.tile_count or len(response) != shard.tile_count:
                raise ValueError(f"shard {shard.shard_id} has an invalid dtype or shape")
            ids = np.arange(expected_tile_id, expected_tile_id + shard.tile_count, dtype=np.uint64)
            if not np.array_equal(index["tile_id"], ids):
                raise ValueError(f"shard {shard.shard_id} tile ids are not contiguous")
            if np.any(index["material_state_index"] >= counts["material_state_count"]):
                raise ValueError(f"shard {shard.shard_id} references an invalid material state")
            if np.any(index["view_index"] >= counts["view_count"]):
                raise ValueError(f"shard {shard.shard_id} references an invalid view")
            state_rows = self.material_states[index["material_state_index"]]
            if not np.array_equal(index["family_index"], state_rows["family_index"]):
                raise ValueError(f"shard {shard.shard_id} family index disagrees with material state")
            if not np.array_equal(index["split"], state_rows["split"]):
                raise ValueError(f"shard {shard.shard_id} split disagrees with material state")
            if response.dtype != make_response_dtype(int(counts["light_count"])):
                raise ValueError(f"shard {shard.shard_id} response dtype disagrees with manifest")
            expected_tile_id += shard.tile_count


def validate_reference_dataset(root: Path | str, *, verify_hashes: bool = True) -> ReferenceDataset:
    return ReferenceDataset.open(root, verify_hashes=verify_hashes)


def dataset_identity(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ncls.reference-dataset\0v2\0")
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()
