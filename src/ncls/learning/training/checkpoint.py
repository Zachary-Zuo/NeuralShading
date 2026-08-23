from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_FORMAT = "ncls.learning-checkpoint"
CHECKPOINT_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def load_checkpoint(path: Path | str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != sha256_file(checkpoint_path):
        raise ValueError("checkpoint hash sidecar is missing or does not match")
    value = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("checkpoint root must be a mapping")
    if value.get("format_name") != CHECKPOINT_FORMAT or value.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint format")
    return value
