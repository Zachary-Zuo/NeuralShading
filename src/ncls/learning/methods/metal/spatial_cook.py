from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ncls.core.identity import sha256_json
from ncls.learning.methods.metal.asset_cook import MetalBudgetedCompiledAsset
from ncls.learning.methods.metal.asset_read import mip_shapes
from ncls.learning.methods.metal.native_uv import UVGroup
from ncls.learning.methods.metal.spatial_encoder import MetalSpatialEncoder, RawSlot, SEMANTICS, semantic_group


@dataclass(frozen=True)
class SpatialCompiledGroup:
    group: UVGroup
    asset: MetalBudgetedCompiledAsset


@dataclass(frozen=True)
class MetalSpatialCompiledAsset:
    profile_id: str
    groups: tuple[SpatialCompiledGroup, ...]
    global_condition: np.ndarray

    def __post_init__(self):
        if not 1 <= len(self.groups) <= 9 or self.global_condition.shape != (8,):
            raise ValueError("invalid spatial asset group/global layout")
        if not np.isfinite(self.global_condition).all():
            raise ValueError("non-finite spatial asset condition")

    @property
    def identity(self):
        return sha256_json({"schema": "ncls.metal-spatial-compiled-asset@1", "profile": self.profile_id,
                           "groups": [{"mapping": value.group.mapping.identity, "slots": value.group.slots,
                                       "asset": value.asset.identity} for value in self.groups],
                           "global_condition": self.global_condition.tolist()})

    @property
    def texture_reads(self):
        return sum(2 * value.group.mapping.lookup_count for value in self.groups)

    @property
    def latent_bytes(self):
        return sum(level.nbytes for group in self.groups
                   for levels in (group.asset.detail_levels, group.asset.context_levels) for level in levels)


def _read(field: torch.Tensor, rect, address):
    y, x, h, w = rect
    yy, xx = torch.arange(y, y+h), torch.arange(x, x+w)
    if address == "wrap":
        yy, xx = yy.remainder(field.shape[-2]), xx.remainder(field.shape[-1])
    else:
        yy, xx = yy.clamp(0, field.shape[-2]-1), xx.clamp(0, field.shape[-1]-1)
    return field.index_select(-2, yy).index_select(-1, xx)


class SpatialHierarchyCooker:
    """按层暂存 FP32 learned feature；GPU 只运行 tile，不重新展开深层 raw RF。"""
    def __init__(self, encoder: MetalSpatialEncoder, *, tile_size=128, host_budget_bytes=8 * 1024**3):
        if tile_size < 1 or host_budget_bytes < 1:
            raise ValueError("spatial cook budget must be positive")
        self.encoder = encoder
        self.tile_size = tile_size
        self.host_budget_bytes = host_budget_bytes
        self.device = next(encoder.parameters()).device

    def _tiles(self, shape):
        for y in range(0, shape[0], self.tile_size):
            for x in range(0, shape[1], self.tile_size):
                yield y, x, min(self.tile_size, shape[0]-y), min(self.tile_size, shape[1]-x)

    def _conv(self, reader, shape, depth, module, progress):
        stride, kernel = module.stride[0], module.kernel_size[0]
        radius = (kernel-1)//2
        result = torch.empty((depth, module.out_channels, *shape))
        for y, x, h, w in self._tiles(shape):
            raw = reader((y*stride-radius, x*stride-radius, (h-1)*stride+kernel, (w-1)*stride+kernel))
            value = F.silu(module(raw.to(self.device)))
            result[..., y:y+h, x:x+w] = value.cpu()
            progress.update(h*w*depth)
        return result

    def _encode(self, slots, shape, address, load_raw, progress):
        # host staging 包含各 stem、融合输出和两个相邻 trunk/mip 层；预算在分配前检查。
        retained, stem_peak = 0, 0
        for slot in slots:
            size = slot.depth * 16 * np.prod(slot.shape) * 4
            stem_peak = max(stem_peak, retained + 2 * size)
            retained += size if slot.spatial else 16 * 4
        feature_bytes = 32 * np.prod(shape) * 4
        estimate = max(stem_peak, retained + feature_bytes, 2 * feature_bytes)
        if estimate > self.host_budget_bytes:
            raise RuntimeError(f"spatial cook host staging needs {estimate} bytes, budget={self.host_budget_bytes}")
        stems = []
        for slot in slots:
            def raw(rect):
                values = load_raw(slot, rect)
                padded = F.pad(values, (0, 0, 0, 0, 0, 4-values.shape[1]))
                mask = torch.zeros((slot.depth, 4, *values.shape[-2:]))
                mask[:, :len(slot.channel_roles)] = 1
                return torch.cat((padded, mask), dim=1)
            modules = self.encoder.stems[semantic_group(slot.channel_roles)]
            first = self._conv(raw, slot.shape, slot.depth, modules[0], progress)
            second = self._conv(lambda rect: _read(first, rect, slot.address_mode), slot.shape, slot.depth, modules[1], progress)
            del first
            stems.append(second if slot.spatial else second.mean(dim=(0, 2, 3), keepdim=True))
        fused = torch.empty((1, 32, *shape))
        for y, x, h, w in self._tiles(shape):
            pieces = [torch.zeros((1, 25, h, w), device=self.device) for _ in range(9)]
            for slot, stem in zip(slots, stems, strict=True):
                if slot.spatial:
                    # 只有同 UV 组进入融合；原生尺寸不同的 feature 在 stem 之后 bilinear 对齐。
                    gy = (torch.arange(y, y+h, device=self.device).float()+0.5) / shape[0]
                    gx = (torch.arange(x, x+w, device=self.device).float()+0.5) / shape[1]
                    sy = int(np.floor((y+0.5)*slot.shape[0]/shape[0]-0.5))-1
                    sx = int(np.floor((x+0.5)*slot.shape[1]/shape[1]-0.5))-1
                    nh = int(np.floor((y+h-0.5)*slot.shape[0]/shape[0]-0.5))-sy+3
                    nw = int(np.floor((x+w-0.5)*slot.shape[1]/shape[1]-0.5))-sx+3
                    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
                    grid = 2*torch.stack(((xx*slot.shape[1]-sx)/nw, (yy*slot.shape[0]-sy)/nh), dim=-1)[None]-1
                    values = F.grid_sample(_read(stem, (sy, sx, nh, nw), slot.address_mode).to(self.device),
                                           grid, mode="bilinear", padding_mode="border", align_corners=False)
                else:
                    values = stem.to(self.device).expand(1, -1, h, w)
                ids = torch.tensor([SEMANTICS.index(role) for role in slot.channel_roles], device=self.device)
                weights = values.new_tensor([1 << channel for channel in range(len(slot.channel_roles))])
                semantic = (self.encoder.semantic_embedding(ids)*weights[:, None]).sum(0)/weights.sum()
                pieces[slot.slot] = torch.cat((values, semantic[None, :, None, None].expand(1, -1, h, w),
                                               values.new_ones((1, 1, h, w))), dim=1)
            fused[..., y:y+h, x:x+w] = F.silu(self.encoder.fusion(torch.cat(pieces, dim=1))).cpu()
            progress.update(h*w)
        del stems
        for module in self.encoder.trunk:
            fused = self._conv(lambda rect: _read(fused, rect, address), shape, 1, module, progress)
        return fused

    def _head(self, field, head, *, quantized=True):
        h, w = field.shape[-2:]
        output = np.empty((h, w, 4), dtype=np.int8 if quantized else np.float32)
        for y, x, nh, nw in self._tiles((h, w)):
            values = torch.tanh(head(field[..., y:y+nh, x:x+nw].to(self.device)))
            if quantized:
                values = torch.round(values.clamp(-1, 1)*127).to(torch.int8)
            output[y:y+nh, x:x+nw] = values[0].permute(1, 2, 0).cpu().numpy()
        return output

    @torch.inference_mode()
    def encode(self, slots: Sequence[RawSlot], shape, address, load_raw: Callable, *, quantized=True):
        shapes = mip_shapes(*shape)
        total = sum(2*slot.depth*np.prod(slot.shape) for slot in slots) + 3*np.prod(shape)
        total += sum(2*np.prod(level) for level in shapes[1:])
        with tqdm(total=int(total), unit="texel", desc="metal spatial cook", leave=False) as progress:
            field = self._encode(slots, shape, address, load_raw, progress)
            detail, context = [], []
            for level, level_shape in enumerate(shapes):
                if level:
                    for module in self.encoder.mip:
                        field = self._conv(lambda rect: _read(field, rect, address), level_shape, 1, module, progress)
                detail.append(self._head(field, self.encoder.detail_head, quantized=quantized))
                if level >= min(2, len(shapes)-1):
                    context.append(self._head(field, self.encoder.context_head, quantized=quantized))
        return tuple(detail), tuple(context)


def compile_spatial_asset(model, assets, asset_index, slots, groups, *, tile_size=128):
    cooker = SpatialHierarchyCooker(model.asset.encoder, tile_size=tile_size)
    descriptor = assets.descriptors[asset_index]
    by_slot = {slot.slot: slot for slot in slots}
    load = lambda slot, rect: assets.read_raw_tile(asset_index, slot, rect)
    compiled = []
    for group in groups:
        selected = tuple(by_slot[index] for index in group.slots)
        if any(slot.affine != (1., 0., 0., 0., 1., 0.) for slot in selected):
            raise ValueError("cook cannot reproject different UV slots into one group")
        modes = {slot.address_mode for slot in selected}
        if len(modes) != 1:
            raise ValueError("one UV group must share address mode")
        shape = (max(slot.shape[0] for slot in selected), max(slot.shape[1] for slot in selected))
        mode = next(iter(modes))
        detail, context = cooker.encode(selected, shape, mode, load)
        compiled.append(SpatialCompiledGroup(group, MetalBudgetedCompiledAsset(
            model.profile.profile_id, "encoder-only@1", assets.collection_id,
            descriptor.asset_id, descriptor.schema_id, mode, detail, context)))
    nonspatial = tuple(slot for slot in slots if not slot.spatial)
    global_condition = np.zeros(8, dtype=np.float32)
    if nonspatial:
        detail, context = cooker.encode(nonspatial, (1, 1), "clamp", load, quantized=False)
        global_condition = np.concatenate((detail[0].ravel(), context[0].ravel())).astype(np.float16).astype(np.float32)
    return MetalSpatialCompiledAsset(model.profile.profile_id, tuple(compiled), global_condition)
