from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ncls.learning.methods.metal.asset_read import EncodedAssetTile, mip_shapes


SEMANTIC_GROUPS = ("color", "normal", "height", "scalar", "packed")
# 来自版本化 registry 的原生 role，不使用文件名关键词或通道数猜语义。
SEMANTICS = (
    "abblur", "ambient-occlusion", "analum-multi-smudges", "analum-scratches-dist",
    "auxiliary-alpha", "base", "base-color", "brushed-antique-copper-ref",
    "brushed-antique-copper-smudge-map", "brushing", "bsdf-multiple-scattering-lookup",
    "carbon-steel-multi", "castmetal", "color-lookup", "corrspots",
    "cracked-paint-surface-bare-metal-albedo", "cracked-paint-surface-crackedpaint-multi",
    "crust", "curvature", "dents", "dirt", "drops", "flowstains",
    "foil-crumpled-r-smudge-g-splotch", "grad", "grunge", "hammer", "height",
    "impurities", "iron-pitted-steel-no-spots-metal", "mask", "metal", "metalweave-03-metal",
    "noise", "normal-tangent", "nrm", "opacity", "pit", "plate-dirt", "polish-wipe",
    "reflectivity", "roughness", "scratch", "scratch1", "scratch2", "scratch3",
    "scratcha", "scratchb", "scratches", "scratchvar", "smudge", "splats", "spotsdirt",
    "stainless-brushed-smudges", "stainless-smudges", "streaks", "trans1", "trans2",
    "transition-noise", "wash",
)
_COLOR_ROLES = {
    "base", "base-color", "color-lookup", "cracked-paint-surface-bare-metal-albedo",
    "brushed-antique-copper-ref", "reflectivity",
}
_PACKED_ROLES = {
    "analum-multi-smudges", "carbon-steel-multi", "cracked-paint-surface-crackedpaint-multi",
    "foil-crumpled-r-smudge-g-splotch",
}


def semantic_group(channel_roles: Sequence[str]) -> str:
    roles = set(channel_roles) - {"auxiliary-alpha"}
    if not roles or not roles.issubset(SEMANTICS):
        raise ValueError(f"unsupported declared texture roles: {sorted(roles)}")
    if len(roles) > 1 or roles & _PACKED_ROLES:
        return "packed"
    role = next(iter(roles))
    if role == "normal-tangent":
        return "normal"
    if role == "height":
        return "height"
    if role in _COLOR_ROLES:
        return "color"
    return "scalar"


Rect = tuple[int, int, int, int]  # y, x, height, width


@dataclass(frozen=True)
class RawSlot:
    slot: int
    shape: tuple[int, int]
    channel_roles: tuple[str, ...]
    address_mode: str = "wrap"
    # joint UV -> slot UV；只在 learned stem 之后进行坐标对齐。
    affine: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    spatial: bool = True
    depth: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.slot < 9 or min(self.shape) < 1 or not 1 <= len(self.channel_roles) <= 4:
            raise ValueError("raw slot shape/layout is invalid")
        if self.address_mode not in {"wrap", "clamp"} or len(self.affine) != 6:
            raise ValueError("raw slot coordinate contract is invalid")
        if self.depth < 1 or (self.spatial and self.depth != 1):
            raise ValueError("only non-spatial lookup resources may have multiple native slices")
        semantic_group(self.channel_roles)


@dataclass(frozen=True)
class RawRead:
    slot: int
    rect: Rect


@dataclass(frozen=True)
class PlanOperation:
    kind: str
    inputs: tuple[int, ...]
    arguments: tuple[Any, ...]


@dataclass(frozen=True)
class EncodingPlan:
    slots: tuple[RawSlot, ...]
    shape: tuple[int, int]
    address_mode: str
    operations: tuple[PlanOperation, ...]
    raw_reads: tuple[RawRead, ...]
    detail: tuple[tuple[int, Rect, tuple[int, int]], ...]
    context: tuple[tuple[int, Rect, tuple[int, int]], ...]


@dataclass(frozen=True)
class _Field:
    kind: str
    shape: tuple[int, int]
    address: str
    parents: tuple[int, ...] = ()
    arguments: tuple[Any, ...] = ()


def _segments(begin: int, count: int, extent: int, address: str) -> tuple[tuple[int, int, int], ...]:
    """canonical 段、有效长度、重复次数；每一层都单独寻址，保持奇数 mip 的相位。"""
    result = []
    while count:
        if address == "clamp" and begin < 0:
            size = min(count, -begin)
            result.append((0, 1, size))
        elif address == "clamp" and begin >= extent:
            size = count
            result.append((extent - 1, 1, size))
        else:
            start = begin % extent if address == "wrap" else begin
            size = min(count, extent - start)
            result.append((start, size, 1))
        begin += size
        count -= size
    return tuple(result)


class _PlanBuilder:
    def __init__(self, slots: Sequence[RawSlot], shape: tuple[int, int], address: str) -> None:
        self.slots = tuple(slots)
        self.shape, self.address = shape, address
        self.fields: list[_Field] = []
        self.operations: list[PlanOperation] = []
        self.raw_reads: list[RawRead] = []
        self.cache: dict[tuple[int, Rect], int] = {}

    def field(self, field: _Field) -> int:
        self.fields.append(field)
        return len(self.fields) - 1

    def operation(self, kind: str, inputs: Sequence[int], arguments: tuple[Any, ...] = ()) -> int:
        self.operations.append(PlanOperation(kind, tuple(inputs), arguments))
        return len(self.operations) - 1

    def conv(self, parent: int, name: str, kernel: int = 3, stride: int = 1) -> int:
        source = self.fields[parent]
        shape = tuple(max(1, size // stride) for size in source.shape)
        return self.field(_Field("conv", shape, source.address, (parent,), (name, kernel, stride)))

    def read(self, field_id: int, rect: Rect) -> int:
        key = field_id, rect
        if key in self.cache:
            return self.cache[key]
        field = self.fields[field_id]
        y, x, h, w = rect
        if min(h, w) < 1:
            raise ValueError("encoding rectangle is empty")
        height, width = field.shape
        if y < 0 or x < 0 or y + h > height or x + w > width:
            rows = []
            for sy, nh, ry in _segments(y, h, height, field.address):
                columns = []
                for sx, nw, rx in _segments(x, w, width, field.address):
                    value = self.read(field_id, (sy, sx, nh, nw))
                    if rx > 1 or ry > 1:
                        value = self.operation("repeat", (value,), (ry, rx))
                    columns.append(value)
                rows.append(self.operation("concat", columns, (-1,)) if len(columns) > 1 else columns[0])
            result = self.operation("concat", rows, (-2,)) if len(rows) > 1 else rows[0]
        elif field.kind == "raw":
            read_id = len(self.raw_reads)
            self.raw_reads.append(RawRead(int(field.arguments[0]), rect))
            result = self.operation("raw", (), (read_id,))
        elif field.kind == "conv":
            name, kernel, stride = field.arguments
            radius = (kernel - 1) // 2
            source_rect = (y * stride - radius, x * stride - radius,
                           (h - 1) * stride + kernel, (w - 1) * stride + kernel)
            parent = self.read(field.parents[0], source_rect)
            result = self.operation("conv", (parent,), (name, stride))
        elif field.kind == "align":
            slot_index = int(field.arguments[0])
            slot = self.slots[slot_index]
            if not slot.spatial:
                parent = self.read(field.parents[0], (0, 0, *slot.shape))
                result = self.operation("pool", (parent,), (h, w))
            else:
                a, b, c, d, e, f = slot.affine
                corners = [((ix + 0.5) / width, (iy + 0.5) / height)
                           for ix in (x, x + w - 1) for iy in (y, y + h - 1)]
                px = [(a * u + b * v + c) * slot.shape[1] - 0.5 for u, v in corners]
                py = [(d * u + e * v + f) * slot.shape[0] - 0.5 for u, v in corners]
                # 一 texel 安全边，覆盖 float32 坐标计算的舍入，不裁掉所需 raw RF。
                sx, sy = math.floor(min(px)) - 1, math.floor(min(py)) - 1
                nw, nh = math.floor(max(px)) - sx + 3, math.floor(max(py)) - sy + 3
                parent = self.read(field.parents[0], (sy, sx, nh, nw))
                result = self.operation("align", (parent,), (slot_index, rect, (sy, sx, nh, nw)))
        elif field.kind == "fusion":
            parents = [self.read(parent, rect) for parent in field.parents]
            result = self.operation("fusion", parents)
        else:
            raise ValueError(f"unknown spatial field {field.kind}")
        self.cache[key] = result
        return result


def build_encoding_plan(
    slots: Sequence[RawSlot],
    shape: tuple[int, int],
    address_mode: str,
    detail_rects: Sequence[Rect],
    context_rects: Sequence[Rect],
) -> EncodingPlan:
    """纯 CPU RF 规划，先取得完整 raw 依赖再执行 encoder；不依赖 query GPU readback。"""
    if not slots or len({slot.slot for slot in slots}) != len(slots) or len(slots) > 9:
        raise ValueError("encoding requires one to nine distinct declared slots")
    shapes = mip_shapes(*shape)
    if address_mode not in {"wrap", "clamp"} or len(detail_rects) > len(shapes):
        raise ValueError("encoding level/address plan is invalid")
    context_shapes = mip_shapes(*shapes[min(2, len(shapes) - 1)])
    if len(context_rects) > len(context_shapes):
        raise ValueError("too many Context levels")
    builder = _PlanBuilder(slots, shape, address_mode)
    aligned = []
    for index, slot in enumerate(slots):
        field = builder.field(_Field("raw", slot.shape, slot.address_mode, arguments=(index,)))
        group = semantic_group(slot.channel_roles)
        field = builder.conv(field, f"stems.{group}.0")
        field = builder.conv(field, f"stems.{group}.1")
        aligned.append(builder.field(_Field("align", shape, address_mode, (field,), (index,))))
    field = builder.field(_Field("fusion", shape, address_mode, tuple(aligned)))
    field = builder.conv(field, "trunk.0")
    field = builder.conv(field, "trunk.1")
    hierarchy = [field]
    maximum = max(len(detail_rects) - 1, min(len(context_rects) + 1, len(shapes) - 1))
    for _ in range(maximum):
        field = builder.conv(field, "mip.0", kernel=2, stride=2)
        field = builder.conv(field, "mip.1")
        hierarchy.append(field)
    detail = tuple((builder.read(hierarchy[level], rect), rect, shapes[level])
                   for level, rect in enumerate(detail_rects))
    context = tuple((builder.read(hierarchy[min(level + 2, len(shapes) - 1)], rect), rect, context_shapes[level])
                    for level, rect in enumerate(context_rects))
    return EncodingPlan(tuple(slots), shape, address_mode, tuple(builder.operations),
                        tuple(builder.raw_reads), detail, context)


class MetalSpatialEncoder(nn.Module):
    """原始数值→分组空间 stem→共享 hierarchy；无逐资产参数、无输入归一化。"""
    def __init__(self) -> None:
        super().__init__()
        self.stems = nn.ModuleDict({group: nn.ModuleList((nn.Conv2d(8, 16, 3), nn.Conv2d(16, 16, 3)))
                                    for group in SEMANTIC_GROUPS})
        self.semantic_embedding = nn.Embedding(len(SEMANTICS), 8)
        self.fusion = nn.Conv2d(225, 32, 1)
        self.trunk = nn.ModuleList((nn.Conv2d(32, 32, 3), nn.Conv2d(32, 32, 3)))
        self.mip = nn.ModuleList((nn.Conv2d(32, 32, 2, stride=2), nn.Conv2d(32, 32, 3)))
        self.detail_head = nn.Conv2d(32, 4, 1)
        self.context_head = nn.Conv2d(32, 4, 1)

    def encode_tiles(
        self, plan: EncodingPlan, raw: Mapping[str, torch.Tensor]
    ) -> tuple[tuple[EncodedAssetTile, ...], tuple[EncodedAssetTile, ...]]:
        modules = dict(self.named_modules())
        values: list[torch.Tensor | None] = []
        # 保留到最后一个消费者；autograd 自己持有 backward 所需的 activation。
        users = [0] * len(plan.operations)
        for operation in plan.operations:
            for parent in operation.inputs:
                users[parent] += 1
        for parent, _, _ in (*plan.detail, *plan.context):
            users[parent] += 1
        for operation in plan.operations:
            inputs = [values[parent] for parent in operation.inputs]
            kind, args = operation.kind, operation.arguments
            if kind == "raw":
                read_id = int(args[0])
                read = plan.raw_reads[read_id]
                slot = plan.slots[read.slot]
                source = raw[f"raw-{read_id}"]
                channels = len(slot.channel_roles)
                if source.shape != (slot.depth, channels, read.rect[2], read.rect[3]):
                    raise ValueError("raw resource does not match its declared native rectangle")
                value_channels = F.pad(source, (0, 0, 0, 0, 0, 4 - channels))
                mask = source.new_zeros((slot.depth, 4, 1, 1))
                mask[:, :channels] = 1.0
                value = torch.cat((value_channels, mask.expand(-1, -1, *source.shape[-2:])), dim=1)
            elif kind == "conv":
                value = F.silu(modules[str(args[0])](inputs[0]))
            elif kind == "repeat":
                value = inputs[0].repeat_interleave(int(args[0]), -2).repeat_interleave(int(args[1]), -1)
            elif kind == "concat":
                value = torch.cat(inputs, dim=int(args[0]))
            elif kind == "pool":
                value = inputs[0].mean(dim=(0, 2, 3), keepdim=True).expand(1, -1, int(args[0]), int(args[1]))
            elif kind == "align":
                index, rect, source_rect = args
                slot = plan.slots[index]
                y, x, h, w = rect
                sy, sx, nh, nw = source_rect
                source = inputs[0]
                gy = (torch.arange(y, y + h, device=source.device, dtype=source.dtype) + 0.5) / plan.shape[0]
                gx = (torch.arange(x, x + w, device=source.device, dtype=source.dtype) + 0.5) / plan.shape[1]
                yy, xx = torch.meshgrid(gy, gx, indexing="ij")
                a, b, c, d, e, f = slot.affine
                px = ((a * xx + b * yy + c) * slot.shape[1] - sx) / nw
                py = ((d * xx + e * yy + f) * slot.shape[0] - sy) / nh
                grid = 2.0 * torch.stack((px, py), dim=-1)[None] - 1.0
                value = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
            elif kind == "fusion":
                first = inputs[0]
                h, w = first.shape[-2:]
                pieces = [first.new_zeros((1, 25, h, w)) for _ in range(9)]
                for slot, source in zip(plan.slots, inputs, strict=True):
                    ids = torch.tensor([SEMANTICS.index(role) for role in slot.channel_roles], device=source.device)
                    weights = source.new_tensor([1 << channel for channel in range(len(slot.channel_roles))])
                    semantic = (self.semantic_embedding(ids) * weights[:, None]).sum(dim=0) / weights.sum()
                    pieces[slot.slot] = torch.cat((source, semantic[None, :, None, None].expand(1, -1, h, w),
                                                   source.new_ones((1, 1, h, w))), dim=1)
                value = F.silu(self.fusion(torch.cat(pieces, dim=1)))
            else:
                raise ValueError(f"unknown encoding operation {kind}")
            values.append(value)
            for parent in operation.inputs:
                users[parent] -= 1
                if users[parent] == 0:
                    values[parent] = None
        def heads(outputs: Sequence[tuple[int, Rect, tuple[int, int]]], head: nn.Module) -> tuple[EncodedAssetTile, ...]:
            return tuple(EncodedAssetTile(torch.tanh(head(values[parent])), shape, rect[:2])
                         for parent, rect, shape in outputs)
        return heads(plan.detail, self.detail_head), heads(plan.context, self.context_head)

    @staticmethod
    def receptive_field(level: int) -> int:
        if level < 0:
            raise ValueError("mip level cannot be negative")
        return 9 + 5 * ((1 << level) - 1)
