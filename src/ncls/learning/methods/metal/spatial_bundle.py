from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from ncls.learning.methods.metal.asset_read import mip_shapes
from ncls.learning.methods.metal.native_uv import UVGroup, native_hash22
from ncls.learning.methods.metal.spatial_encoder import EncodingPlan, RawSlot, build_encoding_plan


@dataclass(frozen=True)
class EncodingPart:
    group: int
    cell: tuple[int, int] | None
    plan: EncodingPlan


@dataclass(frozen=True)
class SpatialBundlePlan:
    groups: tuple[UVGroup, ...]
    parts: tuple[EncodingPart, ...]
    anchor_uv: tuple[float, float]
    uv_bounds: tuple[float, float, float, float]


def build_spatial_bundle(
    slots: Sequence[RawSlot], groups: Sequence[UVGroup],
    uv_bounds: tuple[float, float, float, float],
    maximum_uv_dx: tuple[float, float], maximum_uv_dy: tuple[float, float],
) -> SpatialBundlePlan:
    """在采样前由 CPU cohort 列出各 UV 组的完整 lookup/RF；不同组从不共享 latent 网格。"""
    if not groups or len(groups) > 9:
        raise ValueError("spatial program supports one to nine declared UV groups")
    x0, y0, x1, y1 = uv_bounds
    if x1 <= x0 or y1 <= y0:
        raise ValueError("surface query bounds are empty")
    by_slot = {slot.slot: slot for slot in slots}
    corners = np.asarray(((x0, y0), (x0, y1), (x1, y0), (x1, y1)), dtype=np.float64)
    parts = []
    for group_index, group in enumerate(groups):
        selected = tuple(by_slot[index] for index in group.slots)
        if any(not slot.spatial for slot in selected):
            raise ValueError("non-spatial lookup cannot enter a surface UV group")
        if any(slot.affine != (1., 0., 0., 0., 1., 0.) for slot in selected):
            raise ValueError("UV group must share one mapping; per-slot reprojection is not allowed")
        address_modes = {slot.address_mode for slot in selected}
        if len(address_modes) != 1:
            raise ValueError("incompatible address modes must use separate UV groups")
        shape = (max(slot.shape[0] for slot in selected), max(slot.shape[1] for slot in selected))
        shapes = mip_shapes(*shape)
        mapping = group.mapping
        a = np.array(mapping.affine).reshape(2, 3)
        coordinates = np.sum(corners[:, None, :] * a[None, :, :2], axis=-1) + a[:, 2]
        footprint = np.stack((maximum_uv_dx, maximum_uv_dy), axis=-1)
        transformed_footprint = np.sum(a[:, :, None][:, :2] * footprint[None], axis=1)
        jacobian = np.array((shape[1], shape[0]))[:, None] * transformed_footprint * mapping.lookup_scale
        rho = max(math.hypot(*jacobian[:, 0]), math.hypot(*jacobian[:, 1]))
        maximum_level = min(len(shapes) - 1, math.ceil(math.log2(max(rho, 1.))))
        if mapping.mode == "direct":
            cells = (None,)
        else:
            tilted = coordinates * mapping.cell_scale
            tilted = np.stack((tilted[:, 0] - tilted[:, 1] / math.sqrt(3.), tilted[:, 1] * (2. / math.sqrt(3.))), axis=-1)
            lo, hi = np.floor(tilted.min(axis=0)).astype(int), np.floor(tilted.max(axis=0)).astype(int)
            cells = tuple((x + mapping.hash_offset, y + mapping.hash_offset)
                          for y in range(int(lo[1]), int(hi[1]) + 2)
                          for x in range(int(lo[0]), int(hi[0]) + 2))
        for cell in cells:
            mapped = coordinates * mapping.lookup_scale
            if cell is not None:
                # 纯 CPU plan 上的原生 hash，与 GPU 行索引没有数据回传关系。
                offset = native_hash22(torch.tensor([cell], dtype=torch.int64))[0].numpy()
                mapped = mapped - offset
            mapped[:, 1] = 1. - mapped[:, 1]
            def rectangles(level_shapes):
                result = []
                for h, w in level_shapes[:maximum_level + 1]:
                    p = mapped * np.array((w, h)) - 0.5
                    lower = np.floor(p.min(axis=0)).astype(int) - 1
                    upper = np.floor(p.max(axis=0)).astype(int) + 2
                    result.append((int(lower[1]), int(lower[0]), int(upper[1] - lower[1] + 1), int(upper[0] - lower[0] + 1)))
                return result
            context_shapes = mip_shapes(*shapes[min(2, len(shapes) - 1)])
            plan = build_encoding_plan(selected, shape, next(iter(address_modes)), rectangles(shapes), rectangles(context_shapes))
            parts.append(EncodingPart(group_index, cell, plan))
    global_slots = tuple(slot for slot in slots if not slot.spatial)
    if global_slots:
        plan = build_encoding_plan(global_slots, (1, 1), "clamp", [(0, 0, 1, 1)], [(0, 0, 1, 1)])
        parts.append(EncodingPart(-1, None, plan))
    return SpatialBundlePlan(tuple(groups), tuple(parts), ((x0 + x1) * 0.5, (y0 + y1) * 0.5), uv_bounds)
