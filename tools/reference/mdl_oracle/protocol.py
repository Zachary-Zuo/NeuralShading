from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUEST_SCHEMA = "ncls.mdl-oracle-request"
REQUEST_VERSION = 1
RESULT_SCHEMA = "ncls.mdl-oracle-result"
RESULT_VERSION = 1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_request(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("MDL oracle request must be a JSON object")
    if value.get("schema_name") != REQUEST_SCHEMA or value.get("schema_version") != REQUEST_VERSION:
        raise ValueError("unsupported MDL oracle request schema")
    validate_request(value)
    return value, payload


def _directions(name: str, value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 1:
        raise ValueError(f"{name} must have shape [N, 3]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    lengths = np.linalg.norm(result, axis=1)
    if np.any(np.abs(lengths - 1.0) > 2e-4) or np.any(result[:, 2] <= 0.0):
        raise ValueError(f"{name} must contain normalized upper-hemisphere directions")
    return np.ascontiguousarray(result)


def validate_request(value: Mapping[str, Any]) -> None:
    source = value.get("source")
    query = value.get("query")
    if not isinstance(source, Mapping) or not isinstance(query, Mapping):
        raise ValueError("MDL oracle request requires source and query objects")
    for name in ("asset_id", "module_root", "module", "material", "source_snapshot_id"):
        if not isinstance(source.get(name), str) or not source[name]:
            raise ValueError(f"source.{name} must be nonempty")
    if not str(source["module"]).startswith("::") or "(" in str(source["material"]):
        raise ValueError("oracle source must use a qualified module and unqualified material name")
    arguments = source.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("source.arguments must be an object")

    frame = query.get("frame")
    state = query.get("state")
    if frame != {"normal": [0.0, 0.0, 1.0], "tangent": [1.0, 0.0, 0.0]}:
        raise ValueError("MDL oracle V1 requires the canonical +Z/+X frame")
    if state != {
        "animation_time": 0.0,
        "meters_per_scene_unit": 1.0,
        "object_id": 0,
        "exterior_ior": 1.0,
        "learnable": False,
        "texture_lod": 0.0,
    }:
        raise ValueError("MDL oracle state constants differ from the V1 contract")
    if query.get("response_measure") != "rgb-bsdf-times-absolute-shading-normal-light-cosine":
        raise ValueError("unsupported MDL oracle response measure")

    _directions("query.view_directions", query.get("view_directions"))
    _directions("query.light_directions", query.get("light_directions"))
    surfaces = query.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("query.surfaces must be a nonempty array")
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            raise ValueError("every query surface must be an object")
        expected_lengths = {"position": 3, "uv": 2, "uv_dx": 2, "uv_dy": 2}
        for name, length in expected_lengths.items():
            row = surface.get(name)
            if not isinstance(row, list) or len(row) != length:
                raise ValueError(f"surface.{name} must contain {length} values")
            if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in row):
                raise ValueError(f"surface.{name} must be finite")
        if surface["uv_dx"] != [0.0, 0.0] or surface["uv_dy"] != [0.0, 0.0]:
            raise ValueError("falcor2 oracle V1 parity is explicitly limited to LOD0 queries")


def query_arrays(request: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    query = request["query"]
    views = _directions("query.view_directions", query["view_directions"])
    lights = _directions("query.light_directions", query["light_directions"])
    surfaces = query["surfaces"]
    surface_count = len(surfaces)
    view_count = len(views)
    light_count = len(lights)
    position_values = np.asarray([item["position"] for item in surfaces], dtype=np.float32)
    uv_values = np.asarray([item["uv"] for item in surfaces], dtype=np.float32)
    positions = np.repeat(position_values, view_count * light_count, axis=0)
    uv = np.repeat(uv_values, view_count * light_count, axis=0)
    view_rows = np.tile(np.repeat(views, light_count, axis=0), (surface_count, 1))
    light_rows = np.tile(lights, (surface_count * view_count, 1))
    return positions, uv, view_rows, light_rows


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
