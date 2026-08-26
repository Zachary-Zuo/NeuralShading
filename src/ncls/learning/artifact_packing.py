from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def pack_fp16_parameters(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """从模型反射生成连续 FP16 权重和唯一 offset manifest。"""

    names = tuple(parameter_names)
    if len(names) != len(set(names)):
        raise ValueError("packed parameter names must be unique")
    state = model.state_dict()
    missing = [name for name in names if name not in state]
    if missing:
        raise ValueError(f"packed parameters are absent from the model: {missing}")
    offset = 0
    parts: list[bytes] = []
    layout: dict[str, dict[str, Any]] = {}
    for name in names:
        values = state[name].detach().cpu().numpy().astype("<f2", copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"packed parameter is non-finite: {name}")
        count = int(values.size)
        layout[name] = {
            "offset_elements": offset,
            "element_count": count,
            "shape": list(values.shape),
            "dtype": "float16",
        }
        parts.append(values.tobytes())
        offset += count
    return b"".join(parts), layout


__all__ = ["pack_fp16_parameters"]
