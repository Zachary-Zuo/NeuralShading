from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from ncls.core.material import (
    material_program_from_layer_stack,
    pack_layer_stack,
)
from ncls.core.material.legacy_v0 import (
    LEGACY_BINARY_SIZE,
    from_legacy_stack,
    unpack_legacy_stack,
)

from .dataset import (
    INDEX_DTYPE,
    MATERIAL_STATE_DTYPE,
    dataset_identity,
    make_response_dtype,
    write_common_files,
    write_manifest_atomic,
    write_response_shard,
)
from .manifest import ReferenceDatasetManifest, sha256_file


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legacy_states(
    source: Path,
    state_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    old_states_path = source / "states.npy"
    old_splits_path = source / "family_splits.npy"
    if old_states_path.is_file():
        old = np.load(old_states_path, allow_pickle=False)
        if len(old) != state_count or old.dtype.names is None or "family_index" not in old.dtype.names:
            raise ValueError("legacy states.npy does not match stacks.bin")
        family_index = np.asarray(old["family_index"], dtype=np.uint32)
        local_name = "local_state" if "local_state" in old.dtype.names else "local_state_index"
        local_index = np.asarray(old[local_name], dtype=np.uint32)
        family_count = int(np.max(family_index)) + 1
        if old_splits_path.is_file():
            family_splits = np.asarray(np.load(old_splits_path, allow_pickle=False), dtype=np.uint8)
        elif "split" in old.dtype.names:
            family_splits = np.zeros(family_count, dtype=np.uint8)
            for family in range(family_count):
                values = np.unique(old["split"][family_index == family])
                if len(values) != 1:
                    raise ValueError("legacy states split one family across multiple partitions")
                family_splits[family] = values[0]
        else:
            family_splits = np.zeros(family_count, dtype=np.uint8)
    else:
        family_index = np.arange(state_count, dtype=np.uint32)
        local_index = np.zeros(state_count, dtype=np.uint32)
        family_splits = np.zeros(state_count, dtype=np.uint8)
    if len(family_splits) != int(np.max(family_index)) + 1:
        raise ValueError("legacy family_splits.npy has an unexpected length")
    states = np.zeros(state_count, dtype=MATERIAL_STATE_DTYPE)
    states["family_index"] = family_index
    states["local_state_index"] = local_index
    states["program_index"] = np.arange(state_count, dtype=np.uint32)
    states["canonical_ir_index"] = np.arange(state_count, dtype=np.uint32)
    states["split"] = family_splits[family_index]
    return states, family_splits


def _legacy_shards(source: Path, metadata: dict[str, Any]) -> list[tuple[Path, Path]]:
    records = metadata.get("shards")
    if records:
        result = []
        for record in records:
            tiles = record.get("tiles") or record.get("response")
            index = record.get("index")
            if not tiles or not index:
                raise ValueError("legacy shard record is incomplete")
            result.append((source / str(tiles), source / str(index)))
        return result
    tiles = source / "tiles.npy"
    index = source / "index.npy"
    if not tiles.is_file() or not index.is_file():
        raise ValueError("legacy dataset has no recognized response shards")
    return [(tiles, index)]


def _read_legacy_materials(source: Path) -> tuple[list, list[bytes]]:
    payload = (source / "stacks.bin").read_bytes()
    if not payload or len(payload) % LEGACY_BINARY_SIZE:
        raise ValueError("legacy stacks.bin has an invalid size")
    programs = []
    canonical_payloads = []
    for offset in range(0, len(payload), LEGACY_BINARY_SIZE):
        legacy_stack = unpack_legacy_stack(payload[offset : offset + LEGACY_BINARY_SIZE])
        stack = from_legacy_stack(legacy_stack)
        programs.append(
            material_program_from_layer_stack(
                stack,
                metadata={"legacy_state_index": offset // LEGACY_BINARY_SIZE},
            )
        )
        canonical_payloads.append(pack_layer_stack(stack))
    return programs, canonical_payloads


def convert_legacy_v0_dataset(
    source: Path | str,
    output: Path | str,
    *,
    resume: bool = False,
    created_at: str | None = None,
    generator_git_commit: str = "unknown",
) -> ReferenceDatasetManifest:
    """把 `ncls-direction-tiles@1` 一次性转换为稳定的 v2 数据合同。"""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("legacy conversion requires a separate output directory")
    metadata_path = source_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "ncls-direction-tiles" or int(metadata.get("format_version", -1)) != 1:
        raise ValueError("source is not an ncls-direction-tiles@1 dataset")
    programs, canonical_payloads = _read_legacy_materials(source_path)
    state_count = len(programs)
    states, family_splits = _legacy_states(source_path, state_count)
    views = np.asarray(np.load(source_path / "views.npy", allow_pickle=False), dtype=np.float32)
    lights = np.asarray(np.load(source_path / "light_directions.npy", allow_pickle=False), dtype=np.float32)
    weights = np.asarray(np.load(source_path / "solid_angle_weights.npy", allow_pickle=False), dtype=np.float32)
    files, content_hashes = write_common_files(
        output_path,
        material_programs=programs,
        canonical_material_irs=canonical_payloads,
        material_states=states,
        family_splits=family_splits,
        view_directions=views,
        light_directions=lights,
        solid_angle_weights=weights,
        reuse_identical=resume,
    )

    light_count = len(lights)
    response_dtype = make_response_dtype(light_count)
    shards = []
    tile_start = 0
    legacy_shards = _legacy_shards(source_path, metadata)
    for shard_id, (legacy_response_path, legacy_index_path) in enumerate(legacy_shards):
        old_response = np.load(legacy_response_path, mmap_mode="r", allow_pickle=False)
        old_index = np.load(legacy_index_path, mmap_mode="r", allow_pickle=False)
        if old_response.dtype.names is None or not {"mean_a", "mean_b", "count"}.issubset(old_response.dtype.names):
            raise ValueError(f"legacy response {legacy_response_path.name} has an unsupported dtype")
        if old_index.shape != (len(old_response), 2):
            raise ValueError(f"legacy index {legacy_index_path.name} has an unexpected shape")
        index = np.zeros(len(old_response), dtype=INDEX_DTYPE)
        index["tile_id"] = np.arange(tile_start, tile_start + len(index), dtype=np.uint64)
        index["material_state_index"] = old_index[:, 0]
        index["view_index"] = old_index[:, 1]
        state_rows = states[index["material_state_index"]]
        index["family_index"] = state_rows["family_index"]
        index["split"] = state_rows["split"]

        converted = np.zeros(len(old_response), dtype=response_dtype)
        mean_a = np.asarray(old_response["mean_a"], dtype=np.float32)
        mean_b = np.asarray(old_response["mean_b"], dtype=np.float32)
        if mean_a.shape != (len(old_response), light_count, 3):
            raise ValueError(f"legacy response {legacy_response_path.name} light shape does not match")
        converted["mean"] = 0.5 * (mean_a + mean_b)
        converted["variance"] = 0.5 * np.square(mean_a - mean_b)
        converted["replica_mean_a"] = mean_a
        converted["replica_mean_b"] = mean_b
        old_count = np.asarray(old_response["count"], dtype=np.uint64)
        if old_count.ndim == 1:
            old_count = np.broadcast_to(old_count[:, None], (len(old_response), light_count))
        if old_count.shape != (len(old_response), light_count):
            raise ValueError(f"legacy response {legacy_response_path.name} count shape does not match")
        total_count = old_count * np.uint64(2)
        if np.any(total_count > np.iinfo(np.uint32).max):
            raise ValueError("legacy sample count does not fit v2 uint32")
        converted["sample_count"] = total_count.astype(np.uint32)
        shards.append(
            write_response_shard(
                output_path,
                shard_id=shard_id,
                tile_start=tile_start,
                index=index,
                response=converted,
                resume=resume,
            )
        )
        tile_start += len(index)

    expected_tiles = state_count * len(views)
    if tile_start != expected_tiles:
        raise ValueError(f"legacy tile count {tile_start} does not equal material_state_count * view_count")
    source_digest = str(metadata.get("teacher_source_sha256", ""))
    if not _SHA256.fullmatch(source_digest):
        source_digest = sha256_file(metadata_path)
    config = {
        "conversion": "ncls-direction-tiles@1-to-ncls.reference-dataset@2",
        "legacy_metadata_sha256": sha256_file(metadata_path),
        "response_encoding": "float32",
        "uncertainty_kind": "replica-mean-variance",
    }
    config_digest = _json_hash(config)
    identity_parts = [
        source_digest.encode("ascii"),
        config_digest.encode("ascii"),
        *[digest.encode("ascii") for _, digest in sorted(content_hashes.items())],
        *[shard.index_sha256.encode("ascii") for shard in shards],
        *[shard.response_sha256.encode("ascii") for shard in shards],
    ]
    manifest = ReferenceDatasetManifest(
        dataset_id=dataset_identity(identity_parts),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        reference_implementation_id=f"legacy-random-walk-v0:{source_digest[:16]}",
        reference_source_sha256=source_digest,
        generator_git_commit=generator_git_commit,
        prior_id="legacy-v0-prior",
        prior_version=str(metadata.get("prior_version", "unknown")),
        resolved_config_sha256=config_digest,
        seed=int(metadata.get("seed", 0)),
        direction_parameterization={
            "light": str(metadata.get("direction_parameterization", "unknown")),
            "view": "legacy-stratified-view-directions",
            "solid_angle_measure": "steradian",
        },
        counts={
            "family_count": len(family_splits),
            "material_state_count": state_count,
            "view_count": len(views),
            "light_count": light_count,
            "tile_count": tile_start,
        },
        shapes={
            "view_directions": list(views.shape),
            "light_directions": list(lights.shape),
            "response_per_tile": [light_count, 3],
        },
        statistics_encoding={
            "mean": "float32",
            "variance": "float32",
            "replicas": "float32",
            "sample_count": "uint32-total-across-replicas",
            "uncertainty_kind": "replica-mean-variance",
            "variance_definition": "0.5 * (replica_mean_a - replica_mean_b)^2",
        },
        quantization={
            "encoding": "lossless-from-decoded-legacy-float16",
            "new_quantization_absolute_error": 0.0,
            "legacy_source_dtype": "float16",
        },
        split_policy={
            "unit": "family",
            "names": ["train", "validation", "test"],
            "codes": {"train": 0, "validation": 1, "test": 2},
        },
        files=files,
        content_hashes=content_hashes,
        shards=tuple(shards),
        legacy_source={
            "format_name": "ncls-direction-tiles",
            "format_version": 1,
            "metadata_sha256": sha256_file(metadata_path),
            "limitations": [
                "原格式没有逐样本二阶矩，variance 仅表示两个 replica 均值之间的不确定性估计。",
                "原响应先以 float16 持久化；转换不会恢复已损失的精度。",
            ],
        },
    )
    write_manifest_atomic(output_path, manifest)
    return manifest
