from __future__ import annotations

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
    program_runtime_id: str = ""
    material_asset_id: str = ""
    source_snapshot_id: str = ""
    mode: SlotMode = SlotMode.PATH_TRACING
    capabilities: int = 0
    status: SlotStatus = SlotStatus.EMPTY
    diagnostic: str = ""

    def activate(self, *, package_id: str, program_runtime_id: str, material_asset_id: str,
                 source_snapshot_id: str, capabilities: int) -> "ComparisonSlot":
        if not all(len(value) == 64 for value in (package_id, program_runtime_id, material_asset_id, source_snapshot_id)):
            raise ValueError("slot package identities must be SHA-256 sized")
        required = 4 | 8 if self.mode == SlotMode.PATH_TRACING else 1 | 2
        if capabilities & required != required:
            return replace(self, status=SlotStatus.UNSUPPORTED, capabilities=capabilities,
                           diagnostic=f"mode {self.mode.value} requires capabilities {required}")
        return ComparisonSlot(package_id, program_runtime_id, material_asset_id, source_snapshot_id,
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
        if set(slot) != {"package", "mode"} or slot["mode"] not in {mode.value for mode in SlotMode}:
            raise ValueError("viewer slot fields or mode are invalid")
    display = value["display"]
    if set(display) != {"exposure_ev", "divider_color"}:
        raise ValueError("viewer display must not contain split controls")
    return dict(value)
