from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from ncls.core.identity import sha256_json
from ncls.learning.methods.metal.native_uv import UVGroup
from ncls.learning.methods.metal.spatial_encoder import RawSlot


@dataclass(frozen=True)
class SpatialCohort:
    bounds: tuple[float, float, float, float]
    pair_step: tuple[float, float]
    footprint_step: float


def spatial_cohort(
    slots: Sequence[RawSlot], groups: Sequence[UVGroup], options: Mapping[str, Any],
) -> SpatialCohort:
    """由逻辑 request 在 CPU 确定 query core；资源规划不读取 GPU 行索引。"""
    by_slot = {slot.slot: slot for slot in slots}
    extent_x = extent_y = 1.0
    for group in groups:
        width = max(by_slot[index].shape[1] for index in group.slots)
        height = max(by_slot[index].shape[0] for index in group.slots)
        a, b, _, d, e, _ = group.mapping.affine
        factor = group.mapping.lookup_scale
        extent_x = max(extent_x, math.hypot(width * a, height * d) * factor)
        extent_y = max(extent_y, math.hypot(width * b, height * e) * factor)
    core = int(options.get("spatial_core_texels", 128))
    if core < 1:
        raise ValueError("spatial core must contain positive texels")
    extent = min(1.0, core / max(extent_x, extent_y))
    seed = int(sha256_json({
        "recipe": "ncls.metal-spatial-cohort@1",
        "route": options.get("logical_route_name", "diagnostic"),
        "seed": options.get("logical_route_seed", 0),
        "request": options.get("logical_request_index", 0),
        "validation": bool(options.get("validation", False)),
    })[:16], 16)
    rng = np.random.default_rng(seed)
    anchor = options.get("spatial_anchor")
    if anchor is None:
        origin = rng.random(2) * (1.0 - extent)
    else:
        origin = np.asarray(anchor, dtype=np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("spatial anchor must be a finite float2")
        origin = origin - extent * 0.5
    x, y = map(float, origin)
    return SpatialCohort((x, y, x + extent, y + extent),
                         (1.0 / extent_x, 1.0 / extent_y),
                         1.0 / max(extent_x, extent_y))


def cohort_bundle(slots, groups, cohort, maximum_footprint, paired):
    from ncls.learning.methods.metal.spatial_bundle import build_spatial_bundle
    x0, y0, x1, y1 = cohort.bounds
    px, py = cohort.pair_step if paired else (0., 0.)
    f = maximum_footprint * cohort.footprint_step
    return build_spatial_bundle(slots, groups, (x0, y0, x1+px, y1+py), (f, 0.), (0., f))


def spatial_rf_cells(bundle):
    """保守的 32 texel 占用格；不相交即可证明两个 bundle 的完整 raw RF 不相交。"""
    cells = set()
    for part in bundle.parts:
        if part.group < 0:
            continue  # 非空间的原生表是 source 条件，不属于 held-out surface tile。
        for read in part.plan.raw_reads:
            slot = part.plan.slots[read.slot].slot
            y, x, h, w = read.rect
            cells.update((slot, yy, xx) for yy in range(y//32, (y+h-1)//32+1)
                         for xx in range(x//32, (x+w-1)//32+1))
    return frozenset(cells)


def freeze_spatial_split(slots, groups, options, maximum_footprint, paired):
    """先冻结 held-out RF，再选择训练 core；全程只处理 source 声明与 CPU plan。"""
    count_train = int(options.get("spatial_train_tiles", 32))
    count_validation = int(options.get("spatial_validation_tiles", 8))
    if min(count_train, count_validation) < 1:
        raise ValueError("spatial split tile counts must be positive")
    seed = int(options.get("spatial_split_seed", 0))
    heldout_cells, train, validation = set(), [], []
    with tqdm(total=count_train+count_validation, desc="metal spatial RF split", unit="tile", leave=False) as progress:
        for index in range(4096):
            cohort = spatial_cohort(slots, groups, {**options, "spatial_anchor": None,
                "logical_route_name": "frozen-rf-split", "logical_route_seed": seed,
                "logical_request_index": index, "validation": False})
            bundle = cohort_bundle(slots, groups, cohort, maximum_footprint, paired)
            cells = spatial_rf_cells(bundle)
            if cells & heldout_cells:
                continue
            if len(validation) < count_validation:
                validation.append((cohort, bundle))
                heldout_cells.update(cells)
            else:
                train.append((cohort, bundle))
            progress.update(1)
            if len(train) == count_train:
                return tuple(train), tuple(validation)
    raise RuntimeError("cannot freeze disjoint raw receptive fields within 4096 CPU candidates")
