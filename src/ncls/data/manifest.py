from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


FORMAT_NAME = "ncls.reference-dataset"
FORMAT_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_uri(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"dataset URI must be a safe POSIX-relative path: {value!r}")
    return value


def _sha256(value: str, field_name: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ShardRecord:
    shard_id: int
    tile_start: int
    tile_count: int
    index_uri: str
    response_uri: str
    completion_uri: str
    index_sha256: str
    response_sha256: str

    def __post_init__(self) -> None:
        if self.shard_id < 0 or self.tile_start < 0 or self.tile_count < 1:
            raise ValueError("invalid shard range")
        for name in ("index_uri", "response_uri", "completion_uri"):
            object.__setattr__(self, name, _relative_uri(getattr(self, name)))
        _sha256(self.index_sha256, "index_sha256")
        _sha256(self.response_sha256, "response_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "tile_start": self.tile_start,
            "tile_count": self.tile_count,
            "index_uri": self.index_uri,
            "response_uri": self.response_uri,
            "completion_uri": self.completion_uri,
            "index_sha256": self.index_sha256,
            "response_sha256": self.response_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShardRecord:
        return cls(
            int(value["shard_id"]),
            int(value["tile_start"]),
            int(value["tile_count"]),
            str(value["index_uri"]),
            str(value["response_uri"]),
            str(value["completion_uri"]),
            str(value["index_sha256"]),
            str(value["response_sha256"]),
        )


@dataclass(frozen=True)
class ReferenceDatasetManifest:
    dataset_id: str
    created_at: str
    reference_implementation_id: str
    reference_source_sha256: str
    generator_git_commit: str
    prior_id: str
    prior_version: str
    resolved_config_sha256: str
    seed: int
    direction_parameterization: Mapping[str, Any]
    counts: Mapping[str, int]
    shapes: Mapping[str, Any]
    statistics_encoding: Mapping[str, Any]
    quantization: Mapping[str, Any]
    split_policy: Mapping[str, Any]
    files: Mapping[str, str]
    content_hashes: Mapping[str, str]
    shards: tuple[ShardRecord, ...]
    material_program_schema_version: int = 1
    canonical_ir_id: str = "ncls.layer-stack-ir@1"
    canonical_ir_abi_version: int = 1
    scattering_contract_version: int = 1
    response_measure: str = "bsdf-times-positive-light-cosine"
    color_model: str = "linear-srgb"
    legacy_source: Mapping[str, Any] | None = None
    format_name: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format_name != FORMAT_NAME or self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported reference dataset format")
        _sha256(self.dataset_id, "dataset_id")
        _sha256(self.reference_source_sha256, "reference_source_sha256")
        _sha256(self.resolved_config_sha256, "resolved_config_sha256")
        if not self.created_at or not self.reference_implementation_id:
            raise ValueError("manifest identity fields must be nonempty")
        if self.material_program_schema_version != 1 or self.canonical_ir_abi_version != 1:
            raise ValueError("unsupported material contract version")
        if self.scattering_contract_version != 1:
            raise ValueError("unsupported scattering contract version")
        if self.response_measure != "bsdf-times-positive-light-cosine":
            raise ValueError("unsupported response measure")
        if self.color_model != "linear-srgb":
            raise ValueError("unsupported color model")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        required_counts = {"family_count", "material_state_count", "view_count", "light_count", "tile_count"}
        if set(self.counts) != required_counts or any(int(value) < 1 for value in self.counts.values()):
            raise ValueError(f"counts must contain positive values for {sorted(required_counts)}")
        required_files = {
            "material_programs",
            "canonical_material_ir",
            "material_states",
            "family_splits",
            "view_directions",
            "light_directions",
            "solid_angle_weights",
        }
        if set(self.files) != required_files:
            raise ValueError(f"files must contain {sorted(required_files)}")
        for uri in self.files.values():
            _relative_uri(uri)
        if set(self.content_hashes) != set(self.files.values()):
            raise ValueError("content_hashes must cover every common dataset file exactly once")
        for uri, digest in self.content_hashes.items():
            _relative_uri(uri)
            _sha256(digest, f"content_hashes[{uri!r}]")
        ordered = sorted(self.shards, key=lambda shard: shard.tile_start)
        expected_start = 0
        seen_ids: set[int] = set()
        for shard in ordered:
            if shard.shard_id in seen_ids or shard.tile_start != expected_start:
                raise ValueError("shards must have unique ids and contiguous tile ranges")
            seen_ids.add(shard.shard_id)
            expected_start += shard.tile_count
        if expected_start != int(self.counts["tile_count"]):
            raise ValueError("shard ranges do not cover tile_count")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "material_program_schema_version": self.material_program_schema_version,
            "canonical_ir_id": self.canonical_ir_id,
            "canonical_ir_abi_version": self.canonical_ir_abi_version,
            "scattering_contract_version": self.scattering_contract_version,
            "reference_implementation_id": self.reference_implementation_id,
            "reference_source_sha256": self.reference_source_sha256,
            "generator_git_commit": self.generator_git_commit,
            "prior_id": self.prior_id,
            "prior_version": self.prior_version,
            "resolved_config_sha256": self.resolved_config_sha256,
            "seed": self.seed,
            "direction_parameterization": dict(self.direction_parameterization),
            "response_measure": self.response_measure,
            "color_model": self.color_model,
            "counts": dict(self.counts),
            "shapes": dict(self.shapes),
            "statistics_encoding": dict(self.statistics_encoding),
            "quantization": dict(self.quantization),
            "split_policy": dict(self.split_policy),
            "files": dict(self.files),
            "content_hashes": dict(self.content_hashes),
            "shards": [shard.to_dict() for shard in self.shards],
        }
        if self.legacy_source is not None:
            result["legacy_source"] = dict(self.legacy_source)
        result.update(self.extras)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceDatasetManifest:
        known = {
            "format_name", "format_version", "dataset_id", "created_at",
            "material_program_schema_version", "canonical_ir_id", "canonical_ir_abi_version",
            "scattering_contract_version", "reference_implementation_id", "reference_source_sha256",
            "generator_git_commit", "prior_id", "prior_version", "resolved_config_sha256", "seed",
            "direction_parameterization", "response_measure", "color_model", "counts", "shapes",
            "statistics_encoding", "quantization", "split_policy", "files", "content_hashes", "shards",
            "legacy_source",
        }
        return cls(
            dataset_id=str(value["dataset_id"]),
            created_at=str(value["created_at"]),
            reference_implementation_id=str(value["reference_implementation_id"]),
            reference_source_sha256=str(value["reference_source_sha256"]),
            generator_git_commit=str(value["generator_git_commit"]),
            prior_id=str(value["prior_id"]),
            prior_version=str(value["prior_version"]),
            resolved_config_sha256=str(value["resolved_config_sha256"]),
            seed=int(value["seed"]),
            direction_parameterization=value["direction_parameterization"],
            counts={str(k): int(v) for k, v in value["counts"].items()},
            shapes=value["shapes"],
            statistics_encoding=value["statistics_encoding"],
            quantization=value["quantization"],
            split_policy=value["split_policy"],
            files={str(k): str(v) for k, v in value["files"].items()},
            content_hashes={str(k): str(v) for k, v in value["content_hashes"].items()},
            shards=tuple(ShardRecord.from_dict(item) for item in value["shards"]),
            material_program_schema_version=int(value["material_program_schema_version"]),
            canonical_ir_id=str(value["canonical_ir_id"]),
            canonical_ir_abi_version=int(value["canonical_ir_abi_version"]),
            scattering_contract_version=int(value["scattering_contract_version"]),
            response_measure=str(value["response_measure"]),
            color_model=str(value["color_model"]),
            legacy_source=value.get("legacy_source"),
            format_name=str(value["format_name"]),
            format_version=int(value["format_version"]),
            extras={str(k): v for k, v in value.items() if k not in known},
        )

    @classmethod
    def from_json(cls, text: str) -> ReferenceDatasetManifest:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("reference dataset manifest root must be an object")
        return cls.from_dict(value)
