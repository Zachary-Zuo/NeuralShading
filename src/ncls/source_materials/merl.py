from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np


MERL_MATERIAL_SCHEMA = "ncls.merl-material"
MERL_MATERIAL_VERSION = 1
MERL_DIMENSIONS = (90, 90, 180)
MERL_CHANNEL_SCALE = np.asarray((1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0), dtype=np.float64)


@dataclass(frozen=True)
class MerlMaterial:
    material_id: str
    table_uri: str
    source_record: str = "zenodo:8101681"
    license: str = "CC-BY-SA-4.0"

    def __post_init__(self) -> None:
        if not self.material_id or not self.table_uri or not self.source_record:
            raise ValueError("MERL material identity, table URI and source record must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": MERL_MATERIAL_SCHEMA,
            "schema_version": MERL_MATERIAL_VERSION,
            "material_id": self.material_id,
            "table_uri": self.table_uri,
            "source_record": self.source_record,
            "license": self.license,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MerlMaterial:
        if value.get("schema_name") != MERL_MATERIAL_SCHEMA or value.get("schema_version") != MERL_MATERIAL_VERSION:
            raise ValueError("unsupported MERL material schema")
        return cls(
            str(value["material_id"]),
            str(value["table_uri"]),
            str(value.get("source_record", "zenodo:8101681")),
            str(value.get("license", "CC-BY-SA-4.0")),
        )

    @classmethod
    def from_json(cls, text: str) -> MerlMaterial:
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("MERL material JSON root must be an object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class MerlReferenceResult:
    brdf: np.ndarray
    response_cos: np.ndarray
    valid: np.ndarray


def _normalized_directions(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape [N, 3]")
    lengths = np.linalg.norm(result, axis=1)
    if np.any(np.abs(lengths - 1.0) > 2e-5):
        raise ValueError(f"{name} must contain normalized directions")
    return result


def _rotate_z(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(angles), np.sin(angles)
    return np.stack(
        (
            cosine * vectors[:, 0] - sine * vectors[:, 1],
            sine * vectors[:, 0] + cosine * vectors[:, 1],
            vectors[:, 2],
        ),
        axis=1,
    )


def _rotate_y(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(angles), np.sin(angles)
    return np.stack(
        (
            cosine * vectors[:, 0] + sine * vectors[:, 2],
            vectors[:, 1],
            -sine * vectors[:, 0] + cosine * vectors[:, 2],
        ),
        axis=1,
    )


def merl_indices(
    view_directions: Sequence[Sequence[float]],
    light_directions: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """返回官方 Rusinkiewicz 参数化的扁平索引和有效半球 mask。"""

    views = _normalized_directions("view_directions", view_directions)
    lights = _normalized_directions("light_directions", light_directions)
    if views.shape != lights.shape:
        raise ValueError("MERL view and light direction arrays must match")
    valid = (views[:, 2] > 0.0) & (lights[:, 2] > 0.0)
    half = views + lights
    half_length = np.linalg.norm(half, axis=1)
    valid &= half_length > 1e-12
    half = half / np.maximum(half_length[:, None], 1e-12)
    theta_half = np.arccos(np.clip(half[:, 2], -1.0, 1.0))
    phi_half = np.arctan2(half[:, 1], half[:, 0])

    difference = _rotate_z(lights, -phi_half)
    difference = _rotate_y(difference, -theta_half)
    theta_difference = np.arccos(np.clip(difference[:, 2], -1.0, 1.0))
    phi_difference = np.arctan2(difference[:, 1], difference[:, 0])
    phi_difference = np.where(phi_difference < 0.0, phi_difference + math.pi, phi_difference)

    theta_half_index = np.floor(
        np.sqrt(np.maximum(theta_half, 0.0) / (0.5 * math.pi)) * MERL_DIMENSIONS[0]
    ).astype(np.int64)
    theta_difference_index = np.floor(
        theta_difference / (0.5 * math.pi) * MERL_DIMENSIONS[1]
    ).astype(np.int64)
    phi_difference_index = np.floor(
        phi_difference / math.pi * MERL_DIMENSIONS[2]
    ).astype(np.int64)
    theta_half_index = np.clip(theta_half_index, 0, MERL_DIMENSIONS[0] - 1)
    theta_difference_index = np.clip(theta_difference_index, 0, MERL_DIMENSIONS[1] - 1)
    phi_difference_index = np.clip(phi_difference_index, 0, MERL_DIMENSIONS[2] - 1)
    indices = (
        theta_half_index * MERL_DIMENSIONS[1] * MERL_DIMENSIONS[2]
        + theta_difference_index * MERL_DIMENSIONS[2]
        + phi_difference_index
    )
    return indices, valid


class MerlBrdfReference:
    def __init__(self, material: MerlMaterial, asset_root: str | Path):
        self.material = material
        self.table_path = (Path(asset_root) / material.table_uri).resolve()
        with self.table_path.open("rb") as stream:
            header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"truncated MERL header: {self.table_path}")
        dimensions = struct.unpack("<3i", header)
        if dimensions != MERL_DIMENSIONS:
            raise ValueError(f"unsupported MERL dimensions {dimensions}: {self.table_path}")
        sample_count = math.prod(MERL_DIMENSIONS)
        expected_size = 12 + 3 * sample_count * np.dtype("<f8").itemsize
        if self.table_path.stat().st_size != expected_size:
            raise ValueError(
                f"MERL table size mismatch: expected={expected_size}, actual={self.table_path.stat().st_size}"
            )
        self._data = np.memmap(
            self.table_path,
            dtype="<f8",
            mode="r",
            offset=12,
            shape=(3, sample_count),
        )

    def gpu_table(self) -> np.ndarray:
        """返回按官方 channel scale 预处理、保持负测量值的紧凑 float3 表。"""

        return np.ascontiguousarray(np.asarray(self._data.T) * MERL_CHANNEL_SCALE, dtype=np.float32)

    def evaluate(
        self,
        view_directions: Sequence[Sequence[float]],
        light_directions: Sequence[Sequence[float]],
    ) -> MerlReferenceResult:
        lights = _normalized_directions("light_directions", light_directions)
        indices, valid = merl_indices(view_directions, lights)
        brdf = np.zeros((indices.shape[0], 3), dtype=np.float64)
        if np.any(valid):
            brdf[valid] = np.asarray(self._data[:, indices[valid]].T) * MERL_CHANNEL_SCALE
        response = brdf * np.maximum(lights[:, 2:3], 0.0)
        return MerlReferenceResult(brdf.astype(np.float32), response.astype(np.float32), valid)
