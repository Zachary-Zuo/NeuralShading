from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping


class SlotMode(str, Enum):
    PATH_TRACING = "path-tracing"
    DEFERRED = "deferred"


class SlotStatus(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    COMPILING = "compiling"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True)
class ComparisonSlot:
    package_id: str = ""
    program_id: str = ""
    asset_id: str = ""
    instance_id: str = ""
    source_snapshot_id: str = ""
    mode: SlotMode = SlotMode.PATH_TRACING
    capabilities: int = 0
    status: SlotStatus = SlotStatus.EMPTY
    diagnostic: str = ""

    def activate(self, *, package_id: str, program_id: str, asset_id: str, instance_id: str,
                 source_snapshot_id: str, capabilities: int) -> "ComparisonSlot":
        if not all(
            len(value) == 64
            for value in (package_id, program_id, asset_id, instance_id, source_snapshot_id)
        ):
            raise ValueError("slot package identities must be SHA-256 sized")
        required = 4 | 8 if self.mode == SlotMode.PATH_TRACING else 1 | 2
        if capabilities & required != required:
            return replace(self, status=SlotStatus.UNSUPPORTED, capabilities=capabilities,
                           diagnostic=f"mode {self.mode.value} requires capabilities {required}")
        return ComparisonSlot(package_id, program_id, asset_id, instance_id, source_snapshot_id,
                              self.mode, capabilities, SlotStatus.READY)


def panel_extents(width: int, height: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    if width < 2 or height < 1:
        raise ValueError("viewer extent must be at least 2x1")
    panel_width = width // 2
    divider = width - panel_width * 2
    return (0, 0, panel_width, height), (panel_width + divider, 0, panel_width, height), divider


def validate_studio(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"format_name", "format_version", "scene", "slots", "camera", "lighting", "display"}:
        raise ValueError("viewer studio fields are not canonical")
    if value["format_name"] != "ncls.viewer-studio" or value["format_version"] != 2:
        raise ValueError("unsupported viewer studio format")
    slots = value["slots"]
    if not isinstance(slots, list) or len(slots) != 2:
        raise ValueError("viewer studio requires exactly two slots")
    for slot in slots:
        if (
            not isinstance(slot, Mapping)
            or set(slot) != {"package", "mode"}
            or not isinstance(slot["package"], str)
            or (
                slot["package"] not in {"", "source-reference"}
                and re.fullmatch(r"[0-9a-f]{64}", slot["package"]) is None
            )
            or slot["mode"] not in {mode.value for mode in SlotMode}
        ):
            raise ValueError("viewer slot fields or mode are invalid")

    scene = value["scene"]
    if not isinstance(scene, Mapping) or set(scene) != {
        "geometry", "geometry_sha256", "environment", "environment_sha256"
    }:
        raise ValueError("viewer scene fields are not canonical")
    for uri_field, hash_field in (
        ("geometry", "geometry_sha256"),
        ("environment", "environment_sha256"),
    ):
        if not isinstance(scene[uri_field], str) or not scene[uri_field]:
            raise ValueError(f"viewer scene {uri_field} URI is invalid")
        if not isinstance(scene[hash_field], str) or re.fullmatch(
            r"[0-9a-f]{64}", scene[hash_field]
        ) is None:
            raise ValueError(f"viewer scene {hash_field} is invalid")

    def finite_number(number: Any, *, name: str) -> float:
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"viewer studio {name} must be numeric")
        result = float(number)
        if not math.isfinite(result):
            raise ValueError(f"viewer studio {name} must be finite")
        return result

    def vector3(vector: Any, *, name: str) -> tuple[float, float, float]:
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"viewer studio {name} must contain three values")
        return (
            finite_number(vector[0], name=name),
            finite_number(vector[1], name=name),
            finite_number(vector[2], name=name),
        )

    camera = value["camera"]
    if not isinstance(camera, Mapping) or set(camera) != {
        "target", "yaw", "pitch", "distance", "vertical_fov_degrees"
    }:
        raise ValueError("viewer camera fields are not canonical")
    vector3(camera["target"], name="camera.target")
    for field in ("yaw", "pitch"):
        finite_number(camera[field], name=f"camera.{field}")
    if finite_number(camera["distance"], name="camera.distance") <= 0:
        raise ValueError("viewer camera distance must be positive")
    vertical_fov = finite_number(
        camera["vertical_fov_degrees"], name="camera.vertical_fov_degrees"
    )
    if not 1.0 <= vertical_fov <= 179.0:
        raise ValueError("viewer camera vertical FOV is outside supported bounds")

    lighting = value["lighting"]
    if not isinstance(lighting, Mapping) or set(lighting) != {
        "environment_rotation", "environment_intensity"
    }:
        raise ValueError("viewer lighting fields are not canonical")
    finite_number(lighting["environment_rotation"], name="lighting.environment_rotation")
    if finite_number(lighting["environment_intensity"], name="lighting.environment_intensity") < 0:
        raise ValueError("viewer environment intensity must be nonnegative")

    display = value["display"]
    if not isinstance(display, Mapping) or set(display) != {"exposure_ev", "divider_color"}:
        raise ValueError("viewer display must not contain split controls")
    finite_number(display["exposure_ev"], name="display.exposure_ev")
    divider = vector3(display["divider_color"], name="display.divider_color")
    if any(channel < 0.0 for channel in divider):
        raise ValueError("viewer divider color must be nonnegative")
    return dict(value)
