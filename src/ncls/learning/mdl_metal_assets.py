from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from ncls.core.identity import sha256_json
from ncls.learning.source_adaptation import (
    NativeAssetDescriptor,
    NativeAssetDomain,
    NativeAssetRole,
    NativeAssetTile,
    NativeAssetTileRequest,
    _WorkingSetCache,
    _tile_requests,
    _validate_tile_request,
)
from ncls.references.mdl import MdlCompiledArtifact, create_mdl_program_provider
from ncls.source_materials.families.mdl import MdlFamilyDefinition
from ncls.source_materials.mdl_metal import MdlMetalRegistry


_PIXEL_LAYOUTS = {
    "Sint8": (np.dtype(np.uint8), 1),
    "Rgb": (np.dtype(np.uint8), 3),
    "Rgba": (np.dtype(np.uint8), 4),
    "Rgb_16": (np.dtype(np.uint16), 3),
    "Rgba_16": (np.dtype(np.uint16), 4),
    "Float32": (np.dtype(np.float32), 1),
    "Float32<2>": (np.dtype(np.float32), 2),
    "Float32<3>": (np.dtype(np.float32), 3),
    "Float32<4>": (np.dtype(np.float32), 4),
    "Rgb_fp": (np.dtype(np.float32), 3),
    "Color": (np.dtype(np.float32), 4),
}


def _mip_shapes(height: int, width: int) -> tuple[tuple[int, int], ...]:
    result = [(height, width)]
    while result[-1] != (1, 1):
        previous_height, previous_width = result[-1]
        result.append((max(1, previous_height // 2), max(1, previous_width // 2)))
    return tuple(result)


def _channel_groups(slot: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    channels = dict(slot["channels"])
    if "RGB" in channels:
        result = [(str(channels["RGB"]), 3)]
        if "A" in channels:
            result.append((str(channels["A"]), 1))
        return tuple(result)
    ordered = [(channel, channels[channel]) for channel in "RGBA" if channel in channels]
    result: list[tuple[str, int]] = []
    for _, semantic in ordered:
        semantic = str(semantic)
        if result and result[-1][0] == semantic:
            result[-1] = (semantic, result[-1][1] + 1)
        else:
            result.append((semantic, 1))
    if not result:
        raise ValueError("Metal asset slot has no decoded channels")
    return tuple(result)


def _roles(slot: Mapping[str, Any]) -> tuple[NativeAssetRole, ...]:
    offset = 0
    result = []
    occurrences: dict[str, int] = {}
    for semantic, count in _channel_groups(slot):
        occurrence = occurrences.get(semantic, 0)
        occurrences[semantic] = occurrence + 1
        role_id = semantic if occurrence == 0 else f"{semantic}-{occurrence}"
        result.append(
            NativeAssetRole(
                role_id,
                semantic,
                offset,
                count,
                str(slot["transfer"]),
                str(slot["mip_rule"]),
            )
        )
        offset += count
    return tuple(result)


@dataclass(frozen=True)
class _AssetSource:
    export_id: str
    slots: Mapping[int, Mapping[str, Any]]


class MdlMetalNativeAssetCollection:
    """52 个 Metal texture-set 的 memmap/tile+halo canonical collection。"""

    def __init__(
        self,
        registry: MdlMetalRegistry,
        module_root: Path,
        *,
        working_set_capacity: int = 16,
    ) -> None:
        self.registry = registry
        self.module_root = module_root.resolve()
        if not self.module_root.is_dir():
            raise FileNotFoundError("vMaterials 2 module root is missing")
        by_texture_set: dict[str, str] = {}
        for record in registry.exports:
            by_texture_set.setdefault(record.texture_set_id, record.export_id)
        if set(by_texture_set) != set(registry.texture_sets):
            raise ValueError("Metal registry has an unreachable opaque texture set")
        self._sources = tuple(
            _AssetSource(
                by_texture_set[texture_set_id],
                {
                    int(slot["slot_index"]): slot
                    for slot in registry.texture_sets[texture_set_id]["slots"]
                },
            )
            for texture_set_id in sorted(registry.texture_sets)
        )
        descriptors = []
        for texture_set_id in sorted(registry.texture_sets):
            texture_set = registry.texture_sets[texture_set_id]
            domains = []
            for slot in texture_set["slots"]:
                width, height, depth = map(int, slot["dimensions"])
                shape = str(slot["shape"])
                if shape == "bsdf_data":
                    height *= depth
                    coordinate_space = "mdl-bsdf-table-3d-flattened-z-y"
                    address_mode = "clamp"
                elif shape == "2d":
                    coordinate_space = "surface-uv"
                    address_mode = str(texture_set["tile_policy"]["address_mode"])
                else:
                    raise ValueError(f"unsupported Metal native asset shape {shape!r}")
                domains.append(
                    NativeAssetDomain(
                        f"slot-{int(slot['slot_index'])}",
                        coordinate_space,
                        address_mode,
                        _mip_shapes(height, width),
                        _roles(slot),
                    )
                )
            schema_id = sha256_json(
                {
                    "schema": "ncls.mdl-metal-texture-role-schema@1",
                    "domains": [domain.to_dict() for domain in domains],
                }
            )
            descriptors.append(NativeAssetDescriptor(texture_set_id, schema_id, tuple(domains)))
        self.descriptors = tuple(descriptors)
        if len(self.descriptors) != 52:
            raise ValueError("Metal native asset collection must contain 52 texture sets")
        self._cache = _WorkingSetCache(working_set_capacity)
        self._artifact_cache: dict[int, MdlCompiledArtifact] = {}

    @property
    def collection_id(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.native-asset-collection@1",
                "source_kind": "mdl-metal-opaque-texture-sets@1",
                "registry_identity": self.registry.identity,
                "assets": [descriptor.to_dict() for descriptor in self.descriptors],
                "decode": {
                    "integer_normalization": "unorm",
                    "transfer": "registry-declared",
                    "origin": "provider-declared",
                    "mip": "role-declared-box-or-normal-renormalize",
                },
                "working_set_capacity": self._cache.capacity,
            }
        )

    def iter_tile_requests(
        self, asset_index: int, domain_id: str, max_core_texels: int, halo: int
    ) -> Iterator[NativeAssetTileRequest]:
        descriptor = self.descriptors[asset_index]
        domain = descriptor.domain(domain_id)
        yield from _tile_requests(descriptor, asset_index, domain, max_core_texels, halo)

    def _artifact(self, asset_index: int) -> MdlCompiledArtifact:
        artifact = self._artifact_cache.get(asset_index)
        if artifact is not None:
            return artifact
        record = self.registry.export(self._sources[asset_index].export_id)
        locator = {**record.exact_locator, "module_root": str(self.module_root)}
        snapshot = MdlFamilyDefinition().load_snapshot(locator)
        artifact = create_mdl_program_provider(self.module_root).compile_snapshot(snapshot)
        artifact.require_runtime_supported()
        if artifact.manifest["texture_payloads"] != "decoded":
            raise ValueError("Metal native assets require decoded MDL texture payloads")
        self._artifact_cache[asset_index] = artifact
        return artifact

    @staticmethod
    def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
        return np.where(
            values <= np.float32(0.04045),
            values / np.float32(12.92),
            ((values + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
        ).astype(np.float32, copy=False)

    def _base_array(
        self, asset_index: int, slot_index: int
    ) -> tuple[np.memmap, Mapping[str, Any], Mapping[str, Any]]:
        artifact = self._artifact(asset_index)
        manifest_slot = next(
            item for item in artifact.manifest["textures"] if int(item["index"]) == slot_index
        )
        data = manifest_slot.get("data")
        if not data:
            raise ValueError("decoded Metal texture payload is missing")
        layout = _PIXEL_LAYOUTS.get(str(manifest_slot["pixel_type"]))
        if layout is None:
            raise ValueError("decoded Metal texture pixel layout is unsupported")
        dtype, channels = layout
        width = int(manifest_slot["width"])
        height = int(manifest_slot["height"])
        depth = int(manifest_slot["depth"])
        if manifest_slot["shape"] == "bsdf_data":
            shape = (depth * height, width, channels)
        else:
            shape = (height, width, channels)
        values = np.memmap(
            artifact.root / str(data), mode="r", dtype=dtype, shape=shape
        )
        return values, manifest_slot, self._sources[asset_index].slots[slot_index]

    @staticmethod
    def _indices(begin: int, count: int, extent: int, address_mode: str) -> np.ndarray:
        values = np.arange(begin, begin + count, dtype=np.int64)
        if address_mode == "wrap":
            return np.remainder(values, extent)
        return np.clip(values, 0, extent - 1)

    def _read_block(
        self,
        values: np.memmap,
        manifest_slot: Mapping[str, Any],
        slot: Mapping[str, Any],
        y: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        if (
            manifest_slot.get("shape") == "2d"
            and manifest_slot.get("data_origin") == "lower_left"
        ):
            y = values.shape[0] - 1 - y
        elif manifest_slot.get("data_origin") not in {"top_left", "lower_left"}:
            raise ValueError("decoded Metal texture row origin is unsupported")
        source = np.asarray(values[np.ix_(y, x)]).astype(np.float32)
        dtype = values.dtype
        if np.issubdtype(dtype, np.integer):
            source /= np.float32(np.iinfo(dtype).max)
        if slot["transfer"] == "srgb-to-linear":
            source[..., : min(3, source.shape[2])] = self._srgb_to_linear(
                source[..., : min(3, source.shape[2])]
            )
        groups = _channel_groups(slot)
        channel_count = sum(count for _, count in groups)
        return source[..., :channel_count]

    def _load_tile(
        self,
        request: NativeAssetTileRequest,
        domain: NativeAssetDomain,
        device: torch.device,
    ) -> torch.Tensor:
        slot_index = int(request.domain_id.removeprefix("slot-"))
        values, manifest_slot, slot = self._base_array(request.asset_index, slot_index)
        scale = 1 << request.mip_level
        target_height = request.core_shape[0] + 2 * request.halo
        target_width = request.core_shape[1] + 2 * request.halo
        base_height, base_width = values.shape[:2]
        origin_y = (request.origin_yx[0] - request.halo) * scale
        origin_x = (request.origin_yx[1] - request.halo) * scale
        channels = domain.channel_count
        result = np.empty((target_height, target_width, channels), dtype=np.float32)
        # 每个输出 texel只保留自己的source footprint；高mip会流式读取而不展开全量host tensor。
        source_texels = target_height * target_width * scale * scale
        if source_texels <= 4_194_304:
            y = self._indices(
                origin_y, target_height * scale, base_height, domain.address_mode
            )
            x = self._indices(
                origin_x, target_width * scale, base_width, domain.address_mode
            )
            block = self._read_block(values, manifest_slot, slot, y, x)
            result[...] = block.reshape(
                target_height,
                scale,
                target_width,
                scale,
                channels,
            ).mean(axis=(1, 3))
        else:
            for target_y in range(target_height):
                y = self._indices(
                    origin_y + target_y * scale, scale, base_height, domain.address_mode
                )
                for target_x in range(target_width):
                    x = self._indices(
                        origin_x + target_x * scale, scale, base_width, domain.address_mode
                    )
                    block = self._read_block(values, manifest_slot, slot, y, x)
                    result[target_y, target_x] = block.mean(axis=(0, 1))
        cursor = 0
        for role in domain.roles:
            if role.semantic == "normal-tangent":
                normal = result[..., cursor : cursor + role.channel_count] * 2.0 - 1.0
                if role.channel_count >= 3:
                    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
                result[..., cursor : cursor + role.channel_count] = normal * 0.5 + 0.5
            cursor += role.channel_count
        if not np.isfinite(result).all():
            raise ValueError("Metal native asset tile decode produced non-finite values")
        return torch.as_tensor(result, dtype=torch.float32, device=device)

    def acquire_tile(
        self, request: NativeAssetTileRequest, device: torch.device
    ) -> NativeAssetTile:
        _, domain = _validate_tile_request(self.descriptors, request)
        key = (
            request.asset_index,
            request.domain_id,
            request.mip_level,
            request.origin_yx,
            request.core_shape,
            request.halo,
            str(device),
        )
        lease = self._cache.acquire(key, lambda: self._load_tile(request, domain, device))
        try:
            return NativeAssetTile(request, domain.roles, lease.tensor, lease)
        except BaseException:
            lease.release()
            raise


__all__ = ["MdlMetalNativeAssetCollection"]
