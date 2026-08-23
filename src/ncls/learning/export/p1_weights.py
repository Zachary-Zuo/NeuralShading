from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

import numpy as np
import torch


P1_PARAMETER_ORDER = (
    "type_embedding.weight",
    "interface_encoder.0.weight",
    "interface_encoder.0.bias",
    "interface_encoder.2.weight",
    "interface_encoder.2.bias",
    "compose.weight_ih",
    "compose.weight_hh",
    "compose.bias_ih",
    "compose.bias_hh",
    "view_encoder.0.weight",
    "view_encoder.0.bias",
    "head.0.weight",
    "head.0.bias",
    "head.2.weight",
    "head.2.bias",
)


@dataclass(frozen=True)
class P1WeightLayout:
    width: int
    type_width: int
    offsets: Mapping[str, int]
    shapes: Mapping[str, tuple[int, ...]]
    total_floats: int
    format_name: str = "ncls.legacy-ltc-k2-p1-weights"
    format_version: int = 1

    def __post_init__(self) -> None:
        if not 8 <= self.width <= 64 or self.width % 2 or self.type_width != 8:
            raise ValueError("Slang P1 runtime requires even width in [8, 64] and type_width=8")
        if tuple(self.offsets) != P1_PARAMETER_ORDER or tuple(self.shapes) != P1_PARAMETER_ORDER:
            raise ValueError("P1 weight layout parameter order is not canonical")
        expected = 0
        for name in P1_PARAMETER_ORDER:
            if self.offsets[name] != expected:
                raise ValueError("P1 weight offsets must be tightly packed")
            expected += int(np.prod(self.shapes[name], dtype=np.int64))
        if expected != self.total_floats:
            raise ValueError("P1 weight total does not match shapes")

    def to_dict(self) -> dict[str, object]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "width": self.width,
            "type_width": self.type_width,
            "dtype": "float32-little-endian",
            "parameter_order": list(P1_PARAMETER_ORDER),
            "offsets": dict(self.offsets),
            "shapes": {name: list(shape) for name, shape in self.shapes.items()},
            "total_floats": self.total_floats,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def flatten_p1_weights(
    state_dict: Mapping[str, torch.Tensor],
    *,
    width: int,
    type_width: int = 8,
) -> tuple[np.ndarray, P1WeightLayout]:
    offsets: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    parts = []
    offset = 0
    for name in P1_PARAMETER_ORDER:
        if name not in state_dict:
            raise ValueError(f"P1 checkpoint is missing parameter {name!r}")
        value = state_dict[name].detach().cpu().numpy().astype("<f4", copy=False)
        offsets[name] = offset
        shapes[name] = tuple(value.shape)
        flattened = np.ascontiguousarray(value).reshape(-1)
        parts.append(flattened)
        offset += len(flattened)
    layout = P1WeightLayout(width, type_width, offsets, shapes, offset)
    expected_shapes = {
        "type_embedding.weight": (4, type_width),
        "interface_encoder.0.weight": (width, 25 + type_width),
        "interface_encoder.0.bias": (width,),
        "interface_encoder.2.weight": (width, width),
        "interface_encoder.2.bias": (width,),
        "compose.weight_ih": (3 * width, width),
        "compose.weight_hh": (3 * width, width),
        "compose.bias_ih": (3 * width,),
        "compose.bias_hh": (3 * width,),
        "view_encoder.0.weight": (width // 2, 3),
        "view_encoder.0.bias": (width // 2,),
        "head.0.weight": (width, width + width // 2),
        "head.0.bias": (width,),
        "head.2.weight": (18, width),
        "head.2.bias": (18,),
    }
    if dict(layout.shapes) != expected_shapes:
        raise ValueError("P1 checkpoint shapes disagree with the Slang inference contract")
    return np.concatenate(parts).astype("<f4", copy=False), layout
