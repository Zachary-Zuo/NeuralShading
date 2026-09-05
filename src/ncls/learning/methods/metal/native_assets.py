from __future__ import annotations

from collections import deque
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from ncls.core.identity import sha256_json
from ncls.learning.conditioning_resources import ConditioningResource
from ncls.learning.methods.metal.spatial_encoder import EncodingPlan, RawSlot
from ncls.learning.methods.metal.spatial_bundle import SpatialBundlePlan
from ncls.data import (
    GpuResidencyManager,
    HostPipeline,
    HostRequest,
    PipelineTrace,
    ResidentAllocation,
    ResidencyKey,
)
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


def semantic_role_class(semantic: str, channel_count: int) -> int:
    value = semantic.lower()
    if "normal" in value:
        return 1
    if channel_count == 1 or any(
        token in value
        for token in (
            "rough", "mask", "ao", "height", "bump", "dirt", "weight",
            "lookup", "variation", "noise", "scrape", "scratch", "wear",
        )
    ):
        return 2
    if channel_count in {2, 4} or any(
        token in value for token in ("packed", "metallic", "transition")
    ):
        return 3
    return 0


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


def _canonicalize_decoded_channels(
    source: np.ndarray, groups: tuple[tuple[str, int], ...]
) -> np.ndarray:
    """Adapt SDK's physical channel count to the registry's semantic layout."""

    expected = sum(count for _, count in groups)
    actual = int(source.shape[-1])
    if actual == expected:
        return source
    if actual == 1 and expected == 3 and len(groups) == 1 and groups[0][1] == 3:
        return np.broadcast_to(source, (*source.shape[:-1], 3)).copy()
    if actual == expected - 1 and groups[-1][1] == 1:
        alpha = np.ones((*source.shape[:-1], 1), dtype=source.dtype)
        return np.concatenate((source, alpha), axis=-1)
    raise ValueError("decoded Metal texture cannot satisfy its semantic channel contract")


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


@dataclass(frozen=True)
class _ResidentMipPyramid:
    texels: torch.Tensor
    shapes: torch.Tensor
    offsets: torch.Tensor


@dataclass(frozen=True)
class _HostMipDecodeRequest:
    path: str
    dtype: str
    shape: tuple[int, int, int]
    manifest_slot: Mapping[str, Any]
    slot: Mapping[str, Any]
    domain: NativeAssetDomain


@dataclass(frozen=True)
class _HostRawDecodeRequest:
    payload: _HostMipDecodeRequest
    rect: tuple[int, int, int, int]
    depth: int


def _decode_host_raw(request: _HostRawDecodeRequest) -> torch.Tensor:
    payload = request.payload
    values = np.memmap(payload.path, mode="r", dtype=np.dtype(payload.dtype), shape=payload.shape)
    collection = MdlMetalNativeAssetCollection.__new__(MdlMetalNativeAssetCollection)
    return collection._read_raw_rectangle(values, payload.manifest_slot, payload.slot,
                                           request.rect, request.depth)


def _decode_host_resource(request):
    if isinstance(request, _HostRawDecodeRequest):
        return _decode_host_raw(request)
    return _decode_host_mip_pyramid(request)


class _BundleLease:
    def __init__(self, stack: ExitStack) -> None:
        self.stack = stack

    def release(self) -> None:
        self.stack.close()


def _decode_host_mip_pyramid(
    request: _HostMipDecodeRequest,
) -> tuple[torch.Tensor, ...]:
    values = np.memmap(
        request.path,
        mode="r",
        dtype=np.dtype(request.dtype),
        shape=request.shape,
    )
    collection = MdlMetalNativeAssetCollection.__new__(MdlMetalNativeAssetCollection)
    collection._base_array = lambda _asset_index, _slot_index: (
        values,
        request.manifest_slot,
        request.slot,
    )
    levels = collection._decode_mip_pyramid(0, 0, request.domain)
    return tuple(
        torch.from_numpy(np.ascontiguousarray(level.transpose(2, 0, 1)))
        for level in levels
    )


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
        if not self.descriptors:
            raise ValueError("Metal native asset collection cannot be empty")
        self._cache = _WorkingSetCache(working_set_capacity)
        self._artifact_cache: dict[int, MdlCompiledArtifact] = {}
        self._mip_pyramid_key: tuple[int, int] | None = None
        self._mip_pyramid: tuple[np.ndarray, ...] = ()
        self._cook_asset_index: int | None = None
        self._gpu_residency: GpuResidencyManager[_ResidentMipPyramid] | None = None
        self._gpu_device: torch.device | None = None
        self._gpu_trace: PipelineTrace | None = None
        self._gpu_slot_mask: torch.Tensor | None = None
        self._gpu_role_class: torch.Tensor | None = None
        self._host_pipeline: HostPipeline | None = None
        self._host_prefetch = 0
        self._host_request_id = 0
        self._host_scheduled: deque[
            tuple[int, int, NativeAssetDomain]
        ] = deque()
        self._host_scheduled_keys: set[tuple[int, str]] = set()

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

    def enable_gpu_sampling(
        self,
        device: torch.device,
        *,
        budget_bytes: int,
        trace: PipelineTrace,
        num_workers: int = 0,
        host_prefetch: int = 1,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Metal resident patch sampling requires a CUDA device")
        if self._gpu_residency is not None:
            if self._gpu_device != device:
                raise RuntimeError("Metal GPU residency cannot change device")
            return
        slot_masks = np.zeros((len(self.descriptors), 9), dtype=np.bool_)
        role_classes = np.zeros((len(self.descriptors), 9), dtype=np.int64)
        for asset_index, descriptor in enumerate(self.descriptors):
            for slot_position, domain in enumerate(descriptor.domains):
                if slot_position >= 9:
                    raise ValueError("Metal source asset exceeds the nine-slot profile")
                slot_masks[asset_index, slot_position] = True
                role_classes[asset_index, slot_position] = (
                    3
                    if len(domain.roles) > 1
                    else semantic_role_class(
                        domain.roles[0].semantic, domain.roles[0].channel_count
                    )
                )
        self._gpu_device = device
        self._gpu_trace = trace
        self._gpu_residency = GpuResidencyManager(budget_bytes, trace=trace)
        self._gpu_slot_mask = torch.as_tensor(
            slot_masks, dtype=torch.bool, device=device
        )
        self._gpu_role_class = torch.as_tensor(
            role_classes, dtype=torch.int64, device=device
        )
        if num_workers:
            self._host_pipeline = HostPipeline(
                _decode_host_resource,
                num_workers=num_workers,
                capacity=host_prefetch,
                stage="metal-mip-decode",
                trace=trace,
            )
            self._host_prefetch = host_prefetch

    @contextmanager
    def cook_session(self, asset_index: int) -> Iterator[None]:
        if not 0 <= asset_index < len(self.descriptors):
            raise ValueError("Metal native asset cook index is out of range")
        if self._cook_asset_index is not None:
            raise RuntimeError("Metal native asset cook sessions cannot be nested")
        self._cache.clear()
        self._cook_asset_index = asset_index
        try:
            yield
        finally:
            self._cache.clear()
            self._cook_asset_index = None
            self._mip_pyramid_key = None
            self._mip_pyramid = ()

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
        artifact.verify_texture_payloads()
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
        return _canonicalize_decoded_channels(source, groups)

    def raw_slots(self, asset_index: int) -> tuple[RawSlot, ...]:
        """原生声明；lookup 域不参与表面 UV 尺寸、mip 或边界策略。"""
        slots = []
        for position, domain in enumerate(self.descriptors[asset_index].domains):
            native_index = int(domain.domain_id.removeprefix("slot-"))
            declaration = self._sources[asset_index].slots[native_index]
            width, height, depth = map(int, declaration["dimensions"])
            roles = tuple(role.semantic for role in domain.roles for _ in range(role.channel_count))
            spatial = declaration["shape"] == "2d" and "color-lookup" not in roles
            slots.append(RawSlot(position, (height, width), roles,
                                 domain.address_mode if spatial else "clamp",
                                 spatial=spatial, depth=depth))
        return tuple(slots)

    def _read_raw_rectangle(
        self, values: np.ndarray, manifest: Mapping[str, Any], slot: Mapping[str, Any],
        rect: tuple[int, int, int, int], depth: int = 1,
    ) -> torch.Tensor:
        y, x, h, w = rect
        height = values.shape[0] // depth
        if min(y, x) < 0 or y + h > height or x + w > values.shape[1]:
            raise ValueError("raw rectangle must be inside its canonical native slice")
        xx = np.arange(x, x + w, dtype=np.int64)
        blocks = []
        for z in range(depth):
            yy = np.arange(y, y + h, dtype=np.int64) + z * height
            blocks.append(self._read_block(values, manifest, slot, yy, xx).transpose(2, 0, 1))
        # normal 的 decode/strength/normalize 属于 native graph，不能提前在每 texel 上单位化。
        return torch.from_numpy(np.ascontiguousarray(np.stack(blocks)))

    def acquire_encoding_resource(self, asset_index: int, plan: EncodingPlan) -> ConditioningResource:
        """只取 RF planner 列出的原生 mip0 矩形；共享 residency 按实际 bytes 保有 lease。"""
        if self._gpu_residency is None or self._gpu_device is None:
            raise RuntimeError("raw encoder requires configured GPU residency")
        descriptor = self.descriptors[asset_index]
        identity = sha256_json({"schema": "ncls.raw-metal-encoding-plan@1",
                                "asset": descriptor.asset_id, "collection": self.collection_id,
                                "plan": asdict(plan)})
        tensors = {}
        with ExitStack() as leases:
            for read_id, read in enumerate(plan.raw_reads):
                slot = plan.slots[read.slot]
                domain = descriptor.domains[slot.slot]
                native_index = int(domain.domain_id.removeprefix("slot-"))
                key = ResidencyKey(sha256_json({"collection": self.collection_id,
                    "asset": descriptor.asset_id, "domain": domain.domain_id, "rect": read.rect,
                    "depth": slot.depth}), "native-fixed-decode-mip0@1", str(self._gpu_device))
                size = slot.depth * len(slot.channel_roles) * read.rect[2] * read.rect[3] * 4

                def materialize():
                    if self._host_pipeline is not None:
                        if self._host_scheduled:
                            raise RuntimeError("raw tile and historical mip schedules cannot be mixed")
                        request = _HostRawDecodeRequest(self._host_decode_request(asset_index, native_index, domain),
                                                       read.rect, slot.depth)
                        logical_id = self._host_request_id
                        self._host_request_id += 1
                        self._host_pipeline.submit(HostRequest(logical_id, request, {"asset": descriptor.asset_id}))
                        result = self._host_pipeline.next_result()
                        if result.logical_id != logical_id:
                            raise RuntimeError("raw host decode returned a different logical request")
                        decoded = result.payload
                    else:
                        values, manifest, declaration = self._base_array(asset_index, native_index)
                        decoded = self._read_raw_rectangle(values, manifest, declaration, read.rect, slot.depth)
                    tensor = decoded.to(device=self._gpu_device)
                    return ResidentAllocation(tensor, tensor.nelement() * tensor.element_size())

                lease = self._gpu_residency.acquire(key, estimated_bytes=size, materialize=materialize)
                leases.callback(lease.release)
                tensors[f"raw-{read_id}"] = lease.value
            return ConditioningResource(identity, tensors, {"plan": plan, "asset_id": descriptor.asset_id},
                                        _BundleLease(leases.pop_all()))

    def read_raw_tile(self, asset_index: int, slot: RawSlot, rect: tuple[int, int, int, int]) -> torch.Tensor:
        """cook 的有界 host tile；与训练共享固定 decode，逐 slice 保持原生布局。"""
        domain = self.descriptors[asset_index].domains[slot.slot]
        native_index = int(domain.domain_id.removeprefix("slot-"))
        values, manifest, declaration = self._base_array(asset_index, native_index)
        y, x, h, w = rect
        yy, xx = np.arange(y, y + h), np.arange(x, x + w)
        if slot.address_mode == "wrap":
            yy, xx = yy % slot.shape[0], xx % slot.shape[1]
        else:
            yy, xx = yy.clip(0, slot.shape[0] - 1), xx.clip(0, slot.shape[1] - 1)
        return torch.from_numpy(np.ascontiguousarray(np.stack([
            self._read_block(values, manifest, declaration, yy + z * slot.shape[0], xx).transpose(2, 0, 1)
            for z in range(slot.depth)
        ])))

    def acquire_spatial_bundle(self, asset_index: int, plan: SpatialBundlePlan) -> ConditioningResource:
        tensors = {}
        with ExitStack() as leases:
            for index, part in enumerate(plan.parts):
                resource = self.acquire_encoding_resource(asset_index, part.plan)
                if resource.lease is not None:
                    leases.callback(resource.lease.release)
                tensors.update({f"part-{index}/{key}": value for key, value in resource.tensors.items()})
            key = sha256_json({"schema": "ncls.metal-uv-bundle@1", "collection": self.collection_id,
                               "asset": self.descriptors[asset_index].asset_id, "plan": asdict(plan)})
            return ConditioningResource(key, tensors, {"bundle": plan}, _BundleLease(leases.pop_all()))

    def _sample_mip_patches(
        self,
        asset_index: int,
        slot_index: int,
        mip_level: int,
        uv: np.ndarray,
        patch_size: int,
    ) -> np.ndarray:
        """从canonical box mip随机访问局部patch，不持久化派生mip。"""

        descriptor = self.descriptors[asset_index]
        domain = descriptor.domain(f"slot-{slot_index}")
        level = min(max(0, int(mip_level)), len(domain.level_shapes) - 1)
        target_height, target_width = domain.level_shapes[level]
        values, manifest_slot, slot = self._base_array(asset_index, slot_index)
        scale = 1 << level
        center_x = np.floor(uv[:, 0] * target_width).astype(np.int64)
        center_y = np.floor(uv[:, 1] * target_height).astype(np.int64)
        offsets = np.arange(patch_size, dtype=np.int64) - patch_size // 2
        target_y = center_y[:, None] + offsets[None, :]
        target_x = center_x[:, None] + offsets[None, :]
        if domain.address_mode == "wrap":
            target_y = np.remainder(target_y, target_height)
            target_x = np.remainder(target_x, target_width)
        else:
            target_y = np.clip(target_y, 0, target_height - 1)
            target_x = np.clip(target_x, 0, target_width - 1)
        footprint = np.arange(scale, dtype=np.int64)
        source_y = target_y[..., None] * scale + footprint
        source_x = target_x[..., None] * scale + footprint
        source_y = np.clip(source_y, 0, values.shape[0] - 1)
        source_x = np.clip(source_x, 0, values.shape[1] - 1)
        if (
            manifest_slot.get("shape") == "2d"
            and manifest_slot.get("data_origin") == "lower_left"
        ):
            source_y = values.shape[0] - 1 - source_y
        elif manifest_slot.get("data_origin") not in {"top_left", "lower_left"}:
            raise ValueError("decoded Metal texture row origin is unsupported")
        source_texels = int(
            uv.shape[0] * patch_size * patch_size * scale * scale
        )
        channel_count = sum(count for _, count in _channel_groups(slot))
        result = np.empty(
            (uv.shape[0], patch_size, patch_size, channel_count),
            dtype=np.float32,
        )

        def decode(block: np.ndarray) -> np.ndarray:
            decoded = np.asarray(block).astype(np.float32)
            if np.issubdtype(values.dtype, np.integer):
                decoded /= np.float32(np.iinfo(values.dtype).max)
            if slot["transfer"] == "srgb-to-linear":
                decoded[..., : min(3, decoded.shape[-1])] = self._srgb_to_linear(
                    decoded[..., : min(3, decoded.shape[-1])]
                )
            return _canonicalize_decoded_channels(decoded, _channel_groups(slot))

        if source_texels <= 4_194_304:
            block = values[
                source_y[:, :, None, :, None],
                source_x[:, None, :, None, :],
            ]
            result[...] = decode(block).mean(axis=(3, 4))
        else:
            for row in range(uv.shape[0]):
                for patch_y in range(patch_size):
                    for patch_x in range(patch_size):
                        block = values[
                            np.ix_(
                                source_y[row, patch_y],
                                source_x[row, patch_x],
                            )
                        ]
                        result[row, patch_y, patch_x] = decode(block).mean(
                            axis=(0, 1)
                        )
        cursor = 0
        for role in domain.roles:
            if role.semantic == "normal-tangent":
                normal = result[..., cursor : cursor + role.channel_count] * 2.0 - 1.0
                if role.channel_count >= 3:
                    normal /= np.maximum(
                        np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8
                    )
                result[..., cursor : cursor + role.channel_count] = normal * 0.5 + 0.5
            cursor += role.channel_count
        if not np.isfinite(result).all():
            raise ValueError("Metal random-access source patch contains non-finite values")
        return result

    def sample_local_patches(
        self,
        asset_index: torch.Tensor,
        uv: torch.Tensor,
        mip_level: torch.Tensor,
        *,
        patch_size: int,
        active_asset_indices: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回相邻两级source patches，供shared encoder端到端训练。"""

        if (
            asset_index.ndim != 1
            or uv.shape != (asset_index.shape[0], 2)
            or mip_level.shape != asset_index.shape
            or patch_size < 8
            or patch_size > 32
        ):
            raise ValueError("Metal source patch request shape/budget is invalid")
        if self._gpu_residency is not None:
            if active_asset_indices is None:
                raise ValueError(
                    "resident Metal sampling requires execution-group asset identities"
                )
            return self._sample_resident_patches(
                asset_index,
                uv,
                mip_level,
                patch_size=patch_size,
                active_asset_indices=active_asset_indices,
            )
        device = uv.device
        asset_values = asset_index.detach().to("cpu").numpy().astype(np.int64)
        uv_values = uv.detach().to("cpu").numpy().astype(np.float64)
        mip_values = np.floor(
            mip_level.detach().to("cpu").numpy()
        ).astype(np.int64)
        batch = asset_values.shape[0]
        patches = np.zeros(
            (batch, 9, 2, 4, patch_size, patch_size), dtype=np.float32
        )
        mask = np.zeros((batch, 9), dtype=np.bool_)
        role_class = np.zeros((batch, 9), dtype=np.int64)
        for current_asset in np.unique(asset_values):
            rows = np.flatnonzero(asset_values == current_asset)
            descriptor = self.descriptors[int(current_asset)]
            for slot_position, domain in enumerate(descriptor.domains):
                if slot_position >= 9:
                    raise ValueError("Metal source asset exceeds the nine-slot profile")
                slot_index = int(domain.domain_id.removeprefix("slot-"))
                mask[rows, slot_position] = True
                role_class[rows, slot_position] = (
                    3
                    if len(domain.roles) > 1
                    else semantic_role_class(
                        domain.roles[0].semantic, domain.roles[0].channel_count
                    )
                )
                for adjacent in range(2):
                    requested_levels = np.minimum(
                        mip_values[rows] + adjacent,
                        len(domain.level_shapes) - 1,
                    )
                    for level in np.unique(requested_levels):
                        selected = rows[requested_levels == level]
                        values = self._sample_mip_patches(
                            int(current_asset),
                            slot_index,
                            int(level),
                            uv_values[selected],
                            patch_size,
                        )
                        channels = min(4, values.shape[-1])
                        patches[selected, slot_position, adjacent, :channels] = (
                            values[..., :channels].transpose(0, 3, 1, 2)
                        )
        return (
            torch.as_tensor(patches, dtype=torch.float32, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
            torch.as_tensor(role_class, dtype=torch.int64, device=device),
        )

    @staticmethod
    def _sample_gpu_pyramid(
        pyramid: _ResidentMipPyramid,
        uv: torch.Tensor,
        requested_level: torch.Tensor,
        patch_size: int,
        address_mode: str,
    ) -> torch.Tensor:
        if pyramid.texels.ndim != 2:
            raise ValueError("resident Metal mip atlas must have C,N layout")
        channels = pyramid.texels.shape[0]
        shapes = pyramid.shapes.index_select(0, requested_level)
        height = shapes[:, 0]
        width = shapes[:, 1]
        base_offset = pyramid.offsets.index_select(0, requested_level)
        offsets = torch.arange(
            patch_size, dtype=torch.int64, device=uv.device
        ) - patch_size // 2
        center_x = torch.floor(uv[:, 0] * width).to(torch.int64)
        center_y = torch.floor(uv[:, 1] * height).to(torch.int64)
        x = center_x[:, None] + offsets[None, :]
        y = center_y[:, None] + offsets[None, :]
        if address_mode == "wrap":
            x = torch.remainder(x, width[:, None])
            y = torch.remainder(y, height[:, None])
        elif address_mode == "clamp":
            x = torch.minimum(torch.clamp(x, min=0), width[:, None] - 1)
            y = torch.minimum(torch.clamp(y, min=0), height[:, None] - 1)
        else:
            raise ValueError("resident Metal mip address mode is unsupported")
        flat = (
            base_offset[:, None, None]
            + y[:, :, None] * width[:, None, None]
            + x[:, None, :]
        )
        return (
            pyramid.texels.index_select(1, flat.reshape(-1))
            .reshape(channels, uv.shape[0], patch_size, patch_size)
            .permute(1, 0, 2, 3)
            .contiguous()
        )

    @staticmethod
    def _normalize_gpu_roles(
        values: torch.Tensor, domain: NativeAssetDomain
    ) -> torch.Tensor:
        cursor = 0
        result = values
        for role in domain.roles:
            following = cursor + role.channel_count
            if role.semantic == "normal-tangent":
                result = result.clone()
                normal = result[:, cursor:following] * 2.0 - 1.0
                if role.channel_count >= 3:
                    normal = normal / torch.clamp(
                        torch.linalg.vector_norm(normal, dim=1, keepdim=True),
                        min=1e-8,
                    )
                result[:, cursor:following] = normal * 0.5 + 0.5
            cursor = following
        return result

    def _decode_mip_pyramid(
        self,
        asset_index: int,
        slot_index: int,
        domain: NativeAssetDomain,
    ) -> tuple[np.ndarray, ...]:
        values, manifest_slot, slot = self._base_array(asset_index, slot_index)
        height, width = domain.level_shapes[0]
        channels = domain.channel_count
        base = np.empty((height, width, channels), dtype=np.float32)
        x = np.arange(width, dtype=np.int64)
        for begin in range(0, height, 256):
            end = min(height, begin + 256)
            y = np.arange(begin, end, dtype=np.int64)
            base[begin:end] = self._read_block(values, manifest_slot, slot, y, x)
        levels = [base]
        while len(levels) < len(domain.level_shapes):
            previous = levels[-1]
            source_height, source_width = previous.shape[:2]
            target_height, target_width = domain.level_shapes[len(levels)]
            if source_height > 1 and source_width > 1:
                height2, width2 = 2 * target_height, 2 * target_width
                value = (
                    previous[:height2:2, :width2:2]
                    + previous[1:height2:2, :width2:2]
                    + previous[:height2:2, 1:width2:2]
                    + previous[1:height2:2, 1:width2:2]
                ) * np.float32(0.25)
            elif source_height > 1:
                value = (
                    previous[: 2 * target_height : 2]
                    + previous[1 : 2 * target_height : 2]
                ) * np.float32(0.5)
            elif source_width > 1:
                value = (
                    previous[:, : 2 * target_width : 2]
                    + previous[:, 1 : 2 * target_width : 2]
                ) * np.float32(0.5)
            else:
                value = previous.copy()
            levels.append(np.ascontiguousarray(value, dtype=np.float32))
        return tuple(levels)

    def _acquire_resident_slot(
        self,
        asset_index: int,
        slot_index: int,
        domain: NativeAssetDomain,
        decoded_levels: tuple[torch.Tensor, ...] | None = None,
    ):
        if self._gpu_residency is None or self._gpu_device is None:
            raise RuntimeError("Metal GPU residency is not enabled")
        estimate = self._resident_estimate(domain)
        key = self._resident_key(asset_index, domain)

        def materialize() -> ResidentAllocation[_ResidentMipPyramid]:
            trace = self._gpu_trace
            if trace is None:
                raise RuntimeError("Metal GPU trace is not configured")
            if decoded_levels is None:
                with trace.measure("metal.host-decode-mip-pyramid"):
                    numpy_levels = self._decode_mip_pyramid(
                        asset_index, slot_index, domain
                    )
                cpu_levels = tuple(
                    torch.from_numpy(
                        np.ascontiguousarray(level.transpose(2, 0, 1))
                    )
                    for level in numpy_levels
                )
            else:
                cpu_levels = decoded_levels
            with trace.measure("metal.host-to-device-mip-pyramid"):
                level_texels = tuple(
                    int(level.shape[1] * level.shape[2]) for level in cpu_levels
                )
                offsets = np.cumsum((0, *level_texels[:-1]), dtype=np.int64)
                texels = torch.empty(
                    (domain.channel_count, sum(level_texels)),
                    dtype=torch.float32,
                    device=self._gpu_device,
                )
                for offset, count, level in zip(
                    offsets, level_texels, cpu_levels, strict=True
                ):
                    texels[:, int(offset) : int(offset) + count].copy_(
                        level.reshape(domain.channel_count, count)
                    )
                shapes = torch.tensor(
                    [level.shape[1:] for level in cpu_levels],
                    dtype=torch.int64,
                    device=self._gpu_device,
                )
                gpu_offsets = torch.as_tensor(
                    offsets, dtype=torch.int64, device=self._gpu_device
                )
                pyramid = _ResidentMipPyramid(texels, shapes, gpu_offsets)
            allocated = sum(
                value.nelement() * value.element_size()
                for value in (texels, shapes, gpu_offsets)
            )
            return ResidentAllocation(pyramid, allocated)

        return self._gpu_residency.acquire(
            key, estimated_bytes=estimate, materialize=materialize
        )

    def _resident_estimate(self, domain: NativeAssetDomain) -> int:
        return sum(
            height * width * domain.channel_count * np.dtype(np.float32).itemsize
            for height, width in domain.level_shapes
        )

    def _resident_key(
        self, asset_index: int, domain: NativeAssetDomain
    ) -> ResidencyKey:
        if self._gpu_device is None:
            raise RuntimeError("Metal GPU residency is not enabled")
        return ResidencyKey(
            sha256_json(
                {
                    "collection_id": self.collection_id,
                    "asset_id": self.descriptors[asset_index].asset_id,
                    "domain_id": domain.domain_id,
                }
            ),
            "canonical-float32-mip-pyramid@1",
            str(self._gpu_device),
        )

    def _host_decode_request(
        self,
        asset_index: int,
        slot_index: int,
        domain: NativeAssetDomain,
    ) -> _HostMipDecodeRequest:
        values, manifest_slot, slot = self._base_array(asset_index, slot_index)
        filename = getattr(values, "filename", None)
        if filename is None:
            raise ValueError("Metal host decode requires a file-backed source payload")
        return _HostMipDecodeRequest(
            str(Path(filename).resolve()),
            values.dtype.str,
            tuple(int(value) for value in values.shape),
            dict(manifest_slot),
            dict(slot),
            domain,
        )

    @staticmethod
    def _host_resource_key(
        resource: tuple[int, int, NativeAssetDomain]
    ) -> tuple[int, str]:
        return resource[0], resource[2].domain_id

    def _submit_host_resource(
        self, resource: tuple[int, int, NativeAssetDomain]
    ) -> bool:
        if self._host_pipeline is None:
            return False
        if self._host_pipeline.pending_requests >= self._host_prefetch:
            return False
        current_asset, _, domain = resource
        key = self._host_resource_key(resource)
        if key in self._host_scheduled_keys:
            return False
        if self._gpu_residency is None:
            raise RuntimeError("Metal GPU residency is not configured")
        if self._gpu_residency.is_resident(self._resident_key(current_asset, domain)):
            return False
        if self._resident_estimate(domain) > self._gpu_residency.budget_bytes:
            raise ValueError(
                "Metal resident mip exceeds the configured residency budget"
            )
        slot_index = int(domain.domain_id.removeprefix("slot-"))
        self._host_pipeline.submit(
            HostRequest(
                self._host_request_id,
                self._host_decode_request(current_asset, slot_index, domain),
                {
                    "asset_index": current_asset,
                    "domain_id": domain.domain_id,
                },
            )
        )
        self._host_request_id += 1
        self._host_scheduled.append(resource)
        self._host_scheduled_keys.add(key)
        return True

    def prefetch_gpu_sampling(
        self, active_asset_indices: tuple[int, ...]
    ) -> None:
        """Submit host-only mip decode without touching CUDA or residency state."""

        if self._host_pipeline is None:
            return
        if any(
            value < 0 or value >= len(self.descriptors)
            for value in active_asset_indices
        ):
            raise ValueError("Metal active asset index is out of range")
        resources = tuple(
            (current_asset, slot_position, domain)
            for current_asset in active_asset_indices
            for slot_position, domain in enumerate(
                self.descriptors[current_asset].domains
            )
        )
        for resource in resources:
            if self._host_pipeline.pending_requests >= self._host_prefetch:
                break
            self._submit_host_resource(resource)

    def _sample_resident_patches(
        self,
        asset_index: torch.Tensor,
        uv: torch.Tensor,
        mip_level: torch.Tensor,
        *,
        patch_size: int,
        active_asset_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._gpu_slot_mask is None
            or self._gpu_role_class is None
            or self._gpu_trace is None
        ):
            raise RuntimeError("Metal resident metadata is not initialized")
        if any(
            value < 0 or value >= len(self.descriptors)
            for value in active_asset_indices
        ):
            raise ValueError("Metal active asset index is out of range")
        batch = asset_index.shape[0]
        patches = torch.zeros(
            (batch, 9, 2, 4, patch_size, patch_size),
            dtype=torch.float32,
            device=uv.device,
        )
        integer_mip = torch.floor(mip_level).to(torch.int64)
        resources = tuple(
            (current_asset, slot_position, domain)
            for current_asset in active_asset_indices
            for slot_position, domain in enumerate(
                self.descriptors[current_asset].domains
            )
        )
        self.prefetch_gpu_sampling(active_asset_indices)
        with self._gpu_trace.measure("metal.gpu-sample-resident-patches"):
            for resource in resources:
                current_asset, slot_position, domain = resource
                asset_mask = asset_index == current_asset
                slot_index = int(domain.domain_id.removeprefix("slot-"))
                decoded_levels = None
                resident = self._gpu_residency is not None and self._gpu_residency.is_resident(
                    self._resident_key(current_asset, domain)
                )
                if self._host_scheduled and not resident:
                    assert self._host_pipeline is not None
                    if self._host_scheduled[0] != resource:
                        expected = self._host_scheduled[0]
                        raise RuntimeError(
                            "Metal host prefetch order disagrees with logical request: "
                            f"expected asset={expected[0]} domain={expected[2].domain_id}, "
                            f"got asset={current_asset} domain={domain.domain_id}"
                        )
                    result = self._host_pipeline.next_result()
                    self._host_scheduled.popleft()
                    self._host_scheduled_keys.remove(
                        self._host_resource_key(resource)
                    )
                    decoded_levels = tuple(result.payload)
                lease = self._acquire_resident_slot(
                    current_asset, slot_index, domain, decoded_levels
                )
                try:
                    pyramid = lease.value
                    level_count = int(pyramid.shapes.shape[0])
                    for adjacent in range(2):
                        requested = torch.clamp(
                            integer_mip + adjacent, max=level_count - 1
                        )
                        selected_level = self._sample_gpu_pyramid(
                            pyramid,
                            uv,
                            requested,
                            patch_size,
                            domain.address_mode,
                        )
                        selected_level = self._normalize_gpu_roles(
                            selected_level, domain
                        )
                        channels = min(4, domain.channel_count)
                        target = patches[:, slot_position, adjacent, :channels]
                        patches[:, slot_position, adjacent, :channels] = torch.where(
                            asset_mask[:, None, None, None],
                            selected_level[:, :channels],
                            target,
                        )
                finally:
                    lease.release()
                self.prefetch_gpu_sampling(active_asset_indices)
        return (
            patches,
            self._gpu_slot_mask.index_select(0, asset_index),
            self._gpu_role_class.index_select(0, asset_index),
        )

    def _raw_mip_pyramid(
        self,
        asset_index: int,
        slot_index: int,
        domain: NativeAssetDomain,
    ) -> tuple[np.ndarray, ...]:
        if self._cook_asset_index != asset_index:
            raise RuntimeError("full Metal mip pyramid is only valid inside a cook session")
        key = (asset_index, slot_index)
        if self._mip_pyramid_key == key:
            return self._mip_pyramid
        levels = list(self._decode_mip_pyramid(asset_index, slot_index, domain))
        self._mip_pyramid_key = key
        self._mip_pyramid = tuple(levels)
        return self._mip_pyramid

    def gpu_residency_snapshot(self, *, reset_trace: bool = False) -> Mapping[str, Any]:
        if self._gpu_residency is None:
            return {}
        return self._gpu_residency.snapshot(reset_trace=reset_trace)

    def close(self) -> None:
        if self._host_pipeline is not None:
            self._host_pipeline.close()
            self._host_pipeline = None
            self._host_scheduled.clear()
            self._host_scheduled_keys.clear()
        if self._gpu_residency is None:
            return
        self._gpu_residency.close()
        self._gpu_residency = None
        self._gpu_slot_mask = None
        self._gpu_role_class = None

    def _load_cook_tile(
        self,
        request: NativeAssetTileRequest,
        domain: NativeAssetDomain,
        device: torch.device,
    ) -> torch.Tensor:
        slot_index = int(request.domain_id.removeprefix("slot-"))
        level = self._raw_mip_pyramid(request.asset_index, slot_index, domain)[
            request.mip_level
        ]
        target_height = request.core_shape[0] + 2 * request.halo
        target_width = request.core_shape[1] + 2 * request.halo
        y = self._indices(
            request.origin_yx[0] - request.halo,
            target_height,
            level.shape[0],
            domain.address_mode,
        )
        x = self._indices(
            request.origin_yx[1] - request.halo,
            target_width,
            level.shape[1],
            domain.address_mode,
        )
        result = np.asarray(level[np.ix_(y, x)]).copy()
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

    def _load_lazy_tile(
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
                    normal /= np.maximum(
                        np.linalg.norm(normal, axis=2, keepdims=True), 1e-8
                    )
                result[..., cursor : cursor + role.channel_count] = normal * 0.5 + 0.5
            cursor += role.channel_count
        if not np.isfinite(result).all():
            raise ValueError("Metal native asset tile decode produced non-finite values")
        return torch.as_tensor(result, dtype=torch.float32, device=device)

    def _load_tile(
        self,
        request: NativeAssetTileRequest,
        domain: NativeAssetDomain,
        device: torch.device,
    ) -> torch.Tensor:
        return (
            self._load_cook_tile(request, domain, device)
            if self._cook_asset_index == request.asset_index
            else self._load_lazy_tile(request, domain, device)
        )

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
