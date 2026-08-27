from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


QUERY_MAGIC = b"NCLSMQ1\0"
RESULT_MAGIC = b"NCLSMR1\0"
QUERY_RECORD_FLOATS = 11
RESULT_RECORD_FLOATS = 4


def _rows(name: str, value: np.ndarray, count: int, channels: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (count, channels) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite float32 array with shape {(count, channels)}")
    return result


def write_native_query_packet(
    path: Path,
    wo: np.ndarray,
    wi: np.ndarray,
    position: np.ndarray,
    uv: np.ndarray,
) -> None:
    views = np.asarray(wo, dtype=np.float32)
    if views.ndim != 2 or views.shape[1] != 3 or len(views) == 0:
        raise ValueError("wo must be a nonempty Nx3 array")
    count = len(views)
    rows = np.concatenate(
        (
            _rows("wo", views, count, 3),
            _rows("wi", wi, count, 3),
            _rows("position", position, count, 3),
            _rows("uv", uv, count, 2),
        ),
        axis=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(QUERY_MAGIC)
        stream.write(struct.pack("<II", count, QUERY_RECORD_FLOATS * 4))
        stream.write(np.ascontiguousarray(rows, dtype="<f4").tobytes())


def read_native_result_packet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < 16 or payload[:8] != RESULT_MAGIC:
        raise ValueError("unsupported MDL native result schema")
    count, stride = struct.unpack_from("<II", payload, 8)
    if count == 0 or stride != RESULT_RECORD_FLOATS * 4 or len(payload) != 16 + count * stride:
        raise ValueError("invalid MDL native result size")
    rows = np.frombuffer(payload, dtype="<f4", offset=16).reshape(count, RESULT_RECORD_FLOATS).copy()
    if not np.all(np.isfinite(rows)) or np.any(rows[:, 3] < 0.0):
        raise ValueError("MDL native result contains invalid response/PDF values")
    return rows[:, :3], rows[:, 3]
