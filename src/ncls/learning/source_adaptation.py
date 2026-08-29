from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np
from PIL import Image
import pyexr
import torch

from ncls.core.identity import sha256_json
from ncls.core.material import MAX_INTERFACES, MAX_MEDIA, HomogeneousMedium, LayerStackIR, pack_layer_interface
from ncls.core.material.abi_layout import INTERFACE_STRUCT


MDL_FIXED_PARAMETER_SLOTS = 64
MDL_FIXED_PARAMETER_TYPES = (
    "bool",
    "int",
    "float",
    "double",
    "enum",
    "color",
    "float2",
    "float3",
    "float4",
)
MDL_FIXED_SLOT_CHANNELS = 1 + len(MDL_FIXED_PARAMETER_TYPES) + 4


@dataclass(frozen=True)
class NativeFeatureField:
    name: str
    channels: int
    filter_rule: str

    def __post_init__(self) -> None:
        if not self.name or self.channels < 1 or not self.filter_rule:
            raise ValueError("native feature field identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channels": self.channels,
            "filter_rule": self.filter_rule,
        }


@dataclass(frozen=True)
class NativeFeatureLayout:
    family_id: str
    source_contract_version: int
    fields: tuple[NativeFeatureField, ...]
    spatial: bool

    def __post_init__(self) -> None:
        if not self.family_id or self.source_contract_version < 1 or not self.fields:
            raise ValueError("native feature layout identity is invalid")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("native feature layout fields must be unique")
        object.__setattr__(self, "fields", tuple(self.fields))

    @property
    def channel_count(self) -> int:
        return sum(field.channels for field in self.fields)

    @property
    def layout_id(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "fields": [field.to_dict() for field in self.fields],
            "spatial": self.spatial,
        }


@dataclass(frozen=True)
class NativeAssetRole:
    role_id: str
    semantic: str
    channel_offset: int
    channel_count: int
    transfer_function: str
    filter_rule: str

    def __post_init__(self) -> None:
        if (
            not self.role_id
            or not self.semantic
            or self.channel_offset < 0
            or self.channel_count < 1
            or not self.transfer_function
            or not self.filter_rule
        ):
            raise ValueError("native asset role contract is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "semantic": self.semantic,
            "channel_offset": self.channel_offset,
            "channel_count": self.channel_count,
            "transfer_function": self.transfer_function,
            "filter_rule": self.filter_rule,
        }


@dataclass(frozen=True)
class NativeAssetDomain:
    domain_id: str
    coordinate_space: str
    address_mode: str
    level_shapes: tuple[tuple[int, int], ...]
    roles: tuple[NativeAssetRole, ...]

    def __post_init__(self) -> None:
        shapes = tuple((int(height), int(width)) for height, width in self.level_shapes)
        roles = tuple(self.roles)
        if (
            not self.domain_id
            or not self.coordinate_space
            or self.address_mode not in {"clamp", "wrap"}
            or not shapes
            or any(min(shape) < 1 for shape in shapes)
            or not roles
        ):
            raise ValueError("native asset domain contract is invalid")
        if len({role.role_id for role in roles}) != len(roles):
            raise ValueError("native asset domain roles must be unique")
        for previous, current in zip(shapes, shapes[1:]):
            expected = (max(1, previous[0] // 2), max(1, previous[1] // 2))
            if current != expected:
                raise ValueError("native asset mip extents must form one canonical half chain")
        cursor = 0
        for role in roles:
            if role.channel_offset != cursor:
                raise ValueError("native asset role channel ranges must be dense and ordered")
            cursor += role.channel_count
        object.__setattr__(self, "level_shapes", shapes)
        object.__setattr__(self, "roles", roles)

    @property
    def channel_count(self) -> int:
        return sum(role.channel_count for role in self.roles)

    @property
    def role_layout_id(self) -> str:
        return sha256_json([role.to_dict() for role in self.roles])

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "coordinate_space": self.coordinate_space,
            "address_mode": self.address_mode,
            "level_shapes": [list(shape) for shape in self.level_shapes],
            "roles": [role.to_dict() for role in self.roles],
        }


@dataclass(frozen=True)
class NativeAssetDescriptor:
    asset_id: str
    schema_id: str
    domains: tuple[NativeAssetDomain, ...]

    def __post_init__(self) -> None:
        domains = tuple(self.domains)
        if (
            not self.asset_id
            or not self.schema_id
            or not domains
            or len({domain.domain_id for domain in domains}) != len(domains)
        ):
            raise ValueError("native asset descriptor is invalid")
        object.__setattr__(self, "domains", domains)

    def domain(self, domain_id: str) -> NativeAssetDomain:
        for domain in self.domains:
            if domain.domain_id == domain_id:
                return domain
        raise KeyError(f"native asset has no domain {domain_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "schema_id": self.schema_id,
            "domains": [domain.to_dict() for domain in self.domains],
        }


@dataclass(frozen=True)
class NativeAssetTileRequest:
    asset_index: int
    asset_id: str
    schema_id: str
    domain_id: str
    role_layout_id: str
    mip_level: int
    origin_yx: tuple[int, int]
    core_shape: tuple[int, int]
    halo: int

    def __post_init__(self) -> None:
        if (
            min(self.asset_index, self.mip_level, *self.origin_yx, *self.core_shape, self.halo) < 0
            or min(self.core_shape) < 1
            or not self.asset_id
            or not self.schema_id
            or not self.domain_id
            or not self.role_layout_id
        ):
            raise ValueError("native asset tile request is invalid")


@dataclass
class _WorkingSetEntry:
    tensor: torch.Tensor
    ref_count: int
    stamp: int


class NativeAssetWorkingSetLease:
    def __init__(
        self,
        tensor: torch.Tensor,
        release_callback: Callable[[], None],
    ) -> None:
        self.tensor = tensor
        self._release_callback = release_callback
        self.released = False

    def release(self) -> None:
        if not self.released:
            self._release_callback()
            self.released = True


class _WorkingSetCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("native asset working-set capacity must be positive")
        self.capacity = int(capacity)
        self._entries: dict[tuple[Any, ...], _WorkingSetEntry] = {}
        self._stamp = 0

    def acquire(
        self,
        key: tuple[Any, ...],
        loader: Callable[[], torch.Tensor],
    ) -> NativeAssetWorkingSetLease:
        entry = self._entries.get(key)
        if entry is None:
            candidates = [
                (candidate.stamp, candidate_key)
                for candidate_key, candidate in self._entries.items()
                if candidate.ref_count == 0
            ]
            if len(self._entries) >= self.capacity:
                if not candidates:
                    raise RuntimeError("native asset GPU working set is fully leased")
                _, evicted = min(candidates)
                del self._entries[evicted]
            entry = _WorkingSetEntry(loader(), 0, self._stamp)
            self._entries[key] = entry
        self._stamp += 1
        entry.stamp = self._stamp
        entry.ref_count += 1

        def release() -> None:
            current = self._entries.get(key)
            if current is not entry or current.ref_count < 1:
                raise RuntimeError("native asset working-set lease lost ownership")
            current.ref_count -= 1

        return NativeAssetWorkingSetLease(entry.tensor, release)


@dataclass(frozen=True)
class NativeAssetTile:
    request: NativeAssetTileRequest
    roles: tuple[NativeAssetRole, ...]
    values: torch.Tensor
    lease: NativeAssetWorkingSetLease = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        height, width = self.request.core_shape
        if self.values.ndim != 3 or self.values.shape[:2] != (
            height + 2 * self.request.halo,
            width + 2 * self.request.halo,
        ):
            raise ValueError("native asset tile values disagree with core/halo extent")
        roles = tuple(self.roles)
        if sum(role.channel_count for role in roles) != self.values.shape[2]:
            raise ValueError("native asset tile values disagree with role channels")
        if sha256_json([role.to_dict() for role in roles]) != self.request.role_layout_id:
            raise ValueError("native asset tile role layout identity mismatch")
        object.__setattr__(self, "roles", roles)

    @property
    def asset_index(self) -> int:
        return self.request.asset_index

    @property
    def mip_level(self) -> int:
        return self.request.mip_level

    @property
    def origin_yx(self) -> tuple[int, int]:
        return self.request.origin_yx

    @property
    def core_shape(self) -> tuple[int, int]:
        return self.request.core_shape

    @property
    def halo(self) -> int:
        return self.request.halo

    @property
    def core(self) -> torch.Tensor:
        if self.halo == 0:
            return self.values
        return self.values[
            self.halo : self.halo + self.core_shape[0],
            self.halo : self.halo + self.core_shape[1],
        ]

    def role_values(self, role_id: str, *, core: bool = False) -> torch.Tensor:
        source = self.core if core else self.values
        for role in self.roles:
            if role.role_id == role_id:
                begin = role.channel_offset
                return source[..., begin : begin + role.channel_count]
        raise KeyError(f"native asset tile has no role {role_id!r}")

    def release(self) -> None:
        self.lease.release()


class NativeAssetCollection(Protocol):
    """asset/domain/mip/role/schema-aware source tensors with bounded GPU leases."""

    descriptors: tuple[NativeAssetDescriptor, ...]
    collection_id: str

    def iter_tile_requests(
        self,
        asset_index: int,
        domain_id: str,
        max_core_texels: int,
        halo: int,
    ) -> Iterator[NativeAssetTileRequest]: ...

    def acquire_tile(
        self, request: NativeAssetTileRequest, device: torch.device
    ) -> NativeAssetTile: ...


def _tile_requests(
    descriptor: NativeAssetDescriptor,
    asset_index: int,
    domain: NativeAssetDomain,
    max_core_texels: int,
    halo: int,
) -> Iterator[NativeAssetTileRequest]:
    if max_core_texels < 1 or halo < 0:
        raise ValueError("native asset tile budget/halo is invalid")
    for mip_level, (height, width) in enumerate(domain.level_shapes):
        tile_width = min(width, max_core_texels)
        tile_height = max(1, min(height, max_core_texels // tile_width))
        for origin_y in range(0, height, tile_height):
            core_height = min(tile_height, height - origin_y)
            for origin_x in range(0, width, tile_width):
                core_width = min(tile_width, width - origin_x)
                yield NativeAssetTileRequest(
                    asset_index,
                    descriptor.asset_id,
                    descriptor.schema_id,
                    domain.domain_id,
                    domain.role_layout_id,
                    mip_level,
                    (origin_y, origin_x),
                    (core_height, core_width),
                    halo,
                )


def _gather_tile(
    source: torch.Tensor,
    request: NativeAssetTileRequest,
    domain: NativeAssetDomain,
) -> torch.Tensor:
    height, width = domain.level_shapes[request.mip_level]
    origin_y, origin_x = request.origin_yx
    core_height, core_width = request.core_shape
    y = torch.arange(origin_y - request.halo, origin_y + core_height + request.halo, device=source.device)
    x = torch.arange(origin_x - request.halo, origin_x + core_width + request.halo, device=source.device)
    if domain.address_mode == "wrap":
        y, x = torch.remainder(y, height), torch.remainder(x, width)
    else:
        y, x = torch.clamp(y, 0, height - 1), torch.clamp(x, 0, width - 1)
    return source.index_select(0, y).index_select(1, x)


def _validate_tile_request(
    descriptors: tuple[NativeAssetDescriptor, ...],
    request: NativeAssetTileRequest,
) -> tuple[NativeAssetDescriptor, NativeAssetDomain]:
    if request.asset_index >= len(descriptors):
        raise ValueError("native asset tile request asset index is out of range")
    descriptor = descriptors[request.asset_index]
    if request.asset_id != descriptor.asset_id or request.schema_id != descriptor.schema_id:
        raise ValueError("native asset tile request identity mismatch")
    try:
        domain = descriptor.domain(request.domain_id)
    except KeyError as error:
        raise ValueError("native asset tile request domain is unknown") from error
    if request.role_layout_id != domain.role_layout_id:
        raise ValueError("native asset tile request role layout mismatch")
    if request.mip_level >= len(domain.level_shapes):
        raise ValueError("native asset tile request mip is out of range")
    height, width = domain.level_shapes[request.mip_level]
    origin_y, origin_x = request.origin_yx
    core_height, core_width = request.core_shape
    if origin_y + core_height > height or origin_x + core_width > width:
        raise ValueError("native asset tile request core exceeds its mip extent")
    return descriptor, domain


class DenseNativeAssetCollection:
    def __init__(
        self,
        assets: tuple[tuple[torch.Tensor, ...], ...],
        asset_ids: tuple[str, ...],
        schema_id: str,
        domain_id: str,
        coordinate_space: str,
        address_mode: str,
        roles: tuple[NativeAssetRole, ...],
        *,
        working_set_capacity: int = 8,
    ) -> None:
        values = tuple(tuple(levels) for levels in assets)
        if not values or len(values) != len(asset_ids) or len(set(asset_ids)) != len(asset_ids):
            raise ValueError("dense native asset identities must be nonempty and unique")
        if any(not levels or any(level.ndim != 3 for level in levels) for levels in values):
            raise ValueError("dense native asset levels must have shape [height,width,feature]")
        channel_count = sum(role.channel_count for role in roles)
        if channel_count < 1 or any(level.shape[2] != channel_count for levels in values for level in levels):
            raise ValueError("dense native asset channels must agree with roles")
        if any(not bool(torch.isfinite(level).all()) for levels in values for level in levels):
            raise ValueError("dense native assets must be finite")
        self._assets = values
        self.descriptors = tuple(
            NativeAssetDescriptor(
                asset_id,
                schema_id,
                (
                    NativeAssetDomain(
                        domain_id,
                        coordinate_space,
                        address_mode,
                        tuple((int(level.shape[0]), int(level.shape[1])) for level in levels),
                        roles,
                    ),
                ),
            )
            for asset_id, levels in zip(asset_ids, values, strict=True)
        )
        self._cache = _WorkingSetCache(working_set_capacity)

    @property
    def collection_id(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.native-asset-collection@1",
                "assets": [descriptor.to_dict() for descriptor in self.descriptors],
                "working_set_capacity": self._cache.capacity,
            }
        )

    def iter_tile_requests(
        self, asset_index: int, domain_id: str, max_core_texels: int, halo: int
    ) -> Iterator[NativeAssetTileRequest]:
        descriptor = self.descriptors[asset_index]
        domain = descriptor.domain(domain_id)
        yield from _tile_requests(descriptor, asset_index, domain, max_core_texels, halo)

    def acquire_tile(
        self, request: NativeAssetTileRequest, device: torch.device
    ) -> NativeAssetTile:
        _, domain = _validate_tile_request(self.descriptors, request)
        key = (request.asset_index, request.domain_id, request.mip_level, str(device))
        lease = self._cache.acquire(
            key,
            lambda: self._assets[request.asset_index][request.mip_level].to(
                device=device, dtype=torch.float32
            ),
        )
        try:
            values = _gather_tile(lease.tensor, request, domain)
            return NativeAssetTile(request, domain.roles, values, lease)
        except BaseException:
            lease.release()
            raise


def layer_stack_native_feature_layout() -> NativeFeatureLayout:
    fields = [
        NativeFeatureField("interface-counts", 2, "constant"),
    ]
    for index in range(MAX_INTERFACES):
        fields.append(NativeFeatureField(f"interface-{index}", 19, "constant"))
    for index in range(MAX_MEDIA):
        fields.append(NativeFeatureField(f"medium-{index}", 9, "constant"))
    return NativeFeatureLayout("ncls.layer-stack@1", 1, tuple(fields), False)


def materialx_native_feature_layout() -> NativeFeatureLayout:
    return NativeFeatureLayout(
        "materialx.document@1.39.4",
        1,
        (
            NativeFeatureField("resolved-standard-surface-inputs", 24, "constant"),
            NativeFeatureField("base-color-linear", 3, "srgb-decode-then-box-mip"),
            NativeFeatureField("specular-roughness", 1, "box-mip"),
            NativeFeatureField("metalness", 1, "box-mip"),
            NativeFeatureField("normal-first-moment", 3, "lean-box-mip"),
            NativeFeatureField("normal-second-moment", 6, "lean-box-mip"),
        ),
        True,
    )


def mdl_fixed_native_feature_layout() -> NativeFeatureLayout:
    return NativeFeatureLayout(
        "mdl.program@1",
        1,
        (
            NativeFeatureField(
                "nvidia.mdl-fixed-uniform@1/parameter-slots",
                MDL_FIXED_PARAMETER_SLOTS * MDL_FIXED_SLOT_CHANNELS,
                "constant",
            ),
        ),
        False,
    )


def _mdl_scalar(
    value: object,
    descriptor: Mapping[str, Any],
    *,
    color: bool,
) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("MDL fixed-uniform parameters must be finite")
    minimum = descriptor.get("minimum")
    maximum = descriptor.get("maximum")
    if minimum is not None and maximum is not None:
        lower, upper = float(minimum), float(maximum)
        if not np.isfinite((lower, upper)).all() or not upper > lower:
            raise ValueError("MDL fixed-uniform parameter bounds are invalid")
        return float(2.0 * np.clip((result - lower) / (upper - lower), 0.0, 1.0) - 1.0)
    if color:
        return float(2.0 * np.clip(result, 0.0, 1.0) - 1.0)
    return float(np.tanh(result))


def encode_mdl_fixed_native_features(
    arguments: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, str]:
    """把单个无空间纹理 MDL snapshot 编成有界、固定宽度的常量特征。"""

    ordered = tuple(sorted((str(name), dict(value)) for name, value in arguments.items()))
    if len(ordered) > MDL_FIXED_PARAMETER_SLOTS:
        raise ValueError(
            f"MDL fixed-uniform adapter supports at most {MDL_FIXED_PARAMETER_SLOTS} parameters"
        )
    values = np.zeros(
        (MDL_FIXED_PARAMETER_SLOTS, MDL_FIXED_SLOT_CHANNELS), dtype=np.float32
    )
    schema = []
    for index, (name, descriptor) in enumerate(ordered):
        mdl_type = str(descriptor.get("mdl_type", ""))
        if mdl_type not in MDL_FIXED_PARAMETER_TYPES:
            raise ValueError(
                f"MDL fixed-uniform adapter does not support parameter {name!r} of type {mdl_type!r}"
            )
        values[index, 0] = 1.0
        values[index, 1 + MDL_FIXED_PARAMETER_TYPES.index(mdl_type)] = 1.0
        raw = descriptor.get("value")
        if mdl_type == "bool":
            if not isinstance(raw, bool):
                raise ValueError(f"MDL bool parameter {name!r} has an invalid value")
            encoded = (1.0 if raw else -1.0,)
        elif mdl_type == "enum":
            if not isinstance(raw, Mapping) or "name" not in raw:
                raise ValueError(f"MDL enum parameter {name!r} has an invalid value")
            choices = tuple(str(item["name"]) for item in descriptor.get("choices", ()))
            if str(raw["name"]) not in choices:
                raise ValueError(f"MDL enum parameter {name!r} is outside its choices")
            choice_index = choices.index(str(raw["name"]))
            encoded = (
                0.0
                if len(choices) == 1
                else 2.0 * choice_index / float(len(choices) - 1) - 1.0,
            )
        else:
            components = {
                "color": 3,
                "float2": 2,
                "float3": 3,
                "float4": 4,
            }.get(mdl_type, 1)
            raw_values = raw if isinstance(raw, (tuple, list)) else (raw,)
            if len(raw_values) != components:
                raise ValueError(f"MDL parameter {name!r} has the wrong component count")
            encoded = tuple(
                _mdl_scalar(value, descriptor, color=mdl_type == "color")
                for value in raw_values
            )
        value_offset = 1 + len(MDL_FIXED_PARAMETER_TYPES)
        values[index, value_offset : value_offset + len(encoded)] = encoded
        schema.append(
            {
                "name": name,
                "mdl_type": mdl_type,
                "choices": [
                    str(item["name"]) for item in descriptor.get("choices", ())
                ],
            }
        )
    result = values.reshape(-1)
    layout = mdl_fixed_native_feature_layout()
    if result.shape != (layout.channel_count,) or not np.isfinite(result).all():
        raise AssertionError("MDL fixed-uniform feature layout disagrees with encoder")
    return result, sha256_json(
        {"schema": "nvidia.mdl-fixed-parameter-schema@1", "parameters": schema}
    )


def _downsample_features(values: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    if height == 1 and width == 1:
        return values
    source = values.astype(np.float32, copy=False)
    if height == 1:
        source = np.concatenate((source, source), axis=0)
    elif height % 2:
        source = source[:-1]
    if width == 1:
        source = np.concatenate((source, source), axis=1)
    elif width % 2:
        source = source[:, :-1]
    result = 0.25 * (
        source[0::2, 0::2] + source[1::2, 0::2]
        + source[0::2, 1::2] + source[1::2, 1::2]
    )
    return result.astype(np.float16)


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


@dataclass(frozen=True)
class MaterialXNativeAssetCollection:
    """MaterialX spatial fields的紧凑 filtered pyramid；常量不按 texel 重复存储。"""

    constants: np.ndarray
    spatial_levels: tuple[np.ndarray, ...]
    layout_id: str
    asset_id: str
    _cache: _WorkingSetCache = field(
        default_factory=lambda: _WorkingSetCache(8), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        constants = np.asarray(self.constants, dtype=np.float32)
        if constants.shape != (24,) or not np.isfinite(constants).all():
            raise ValueError("MaterialX resolved constants must contain 24 finite values")
        levels = tuple(np.asarray(level, dtype=np.float16) for level in self.spatial_levels)
        if not levels or any(level.ndim != 3 or level.shape[2] != 14 for level in levels):
            raise ValueError("MaterialX spatial feature mips must have 14 channels")
        if any(not np.isfinite(level).all() for level in levels):
            raise ValueError("MaterialX spatial feature mips must be finite")
        layout = materialx_native_feature_layout()
        if self.layout_id != layout.layout_id:
            raise ValueError("MaterialX native feature layout identity mismatch")
        if not self.asset_id:
            raise ValueError("MaterialX native asset identity is required")
        object.__setattr__(self, "constants", constants)
        object.__setattr__(self, "spatial_levels", levels)

    @classmethod
    def from_textures(
        cls,
        constants: np.ndarray,
        *,
        base_color: Path | None,
        roughness: Path | None,
        metalness: Path | None,
        normal: Path | None,
        asset_id: str,
    ) -> "MaterialXNativeAssetCollection":
        inputs = np.asarray(constants, dtype=np.float32)
        loaded: dict[str, np.ndarray | None] = {
            "base": None if base_color is None else (
                np.asarray(Image.open(base_color).convert("RGB"), dtype=np.float32) / 255.0
            ),
            "roughness": None if roughness is None else pyexr.read(str(roughness)).astype(np.float32)[..., :1],
            "metalness": None if metalness is None else pyexr.read(str(metalness)).astype(np.float32)[..., :1],
            "normal": None if normal is None else pyexr.read(str(normal)).astype(np.float32)[..., :3],
        }
        extents = {value.shape[:2] for value in loaded.values() if value is not None}
        if len(extents) > 1:
            raise ValueError("MaterialX training textures must share one native extent")
        height, width = next(iter(extents), (1, 1))
        spatial = np.empty((height, width, 14), dtype=np.float16)
        if loaded["base"] is None:
            spatial[..., 0:3] = inputs[1:4]
        else:
            spatial[..., 0:3] = _srgb_to_linear(loaded["base"])
        spatial[..., 3] = inputs[12] if loaded["roughness"] is None else loaded["roughness"][..., 0]
        spatial[..., 4] = inputs[5] if loaded["metalness"] is None else loaded["metalness"][..., 0]
        if loaded["normal"] is None:
            normal_values = np.zeros((height, width, 3), dtype=np.float32)
            normal_values[..., 2] = 1.0
        else:
            normal_values = loaded["normal"] * 2.0 - 1.0
            normal_values[..., :2] *= float(inputs[17])
            lengths = np.linalg.norm(normal_values, axis=2, keepdims=True)
            normal_values /= np.maximum(lengths, 1e-12)
        spatial[..., 5:8] = normal_values
        spatial[..., 8] = normal_values[..., 0] * normal_values[..., 0]
        spatial[..., 9] = normal_values[..., 0] * normal_values[..., 1]
        spatial[..., 10] = normal_values[..., 0] * normal_values[..., 2]
        spatial[..., 11] = normal_values[..., 1] * normal_values[..., 1]
        spatial[..., 12] = normal_values[..., 1] * normal_values[..., 2]
        spatial[..., 13] = normal_values[..., 2] * normal_values[..., 2]
        levels = [spatial]
        while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
            levels.append(_downsample_features(levels[-1]))
        return cls(
            inputs,
            tuple(levels),
            materialx_native_feature_layout().layout_id,
            asset_id,
        )

    @property
    def descriptors(self) -> tuple[NativeAssetDescriptor, ...]:
        roles = (
            NativeAssetRole("resolved-inputs", "typed-constants", 0, 24, "linear", "constant"),
            NativeAssetRole("base-color", "base-color", 24, 3, "linear", "box-mip"),
            NativeAssetRole("roughness", "roughness", 27, 1, "linear", "box-mip"),
            NativeAssetRole("metalness", "metalness", 28, 1, "linear", "box-mip"),
            NativeAssetRole("normal-first", "normal-first-moment", 29, 3, "signed", "lean-box-mip"),
            NativeAssetRole("normal-second", "normal-second-moment", 32, 6, "linear", "lean-box-mip"),
        )
        return (
            NativeAssetDescriptor(
                self.asset_id,
                self.layout_id,
                (
                    NativeAssetDomain(
                        "surface-uv",
                        "uv0",
                        "wrap",
                        tuple(
                            (int(level.shape[0]), int(level.shape[1]))
                            for level in self.spatial_levels
                        ),
                        roles,
                    ),
                ),
            ),
        )

    @property
    def collection_id(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.native-asset-collection@1",
                "assets": [self.descriptors[0].to_dict()],
                "working_set_capacity": self._cache.capacity,
            }
        )

    def iter_tile_requests(
        self, asset_index: int, domain_id: str, max_core_texels: int, halo: int
    ) -> Iterator[NativeAssetTileRequest]:
        descriptor = self.descriptors[asset_index]
        domain = descriptor.domain(domain_id)
        yield from _tile_requests(descriptor, asset_index, domain, max_core_texels, halo)

    def _load_level(self, mip_level: int, device: torch.device) -> torch.Tensor:
        constants = torch.as_tensor(self.constants, dtype=torch.float32, device=device)
        spatial = torch.as_tensor(
            self.spatial_levels[mip_level], dtype=torch.float32, device=device
        )
        return torch.cat(
            (constants.expand(*spatial.shape[:2], len(constants)), spatial), dim=2
        )

    def acquire_tile(
        self, request: NativeAssetTileRequest, device: torch.device
    ) -> NativeAssetTile:
        _, domain = _validate_tile_request(self.descriptors, request)
        lease = self._cache.acquire(
            (request.asset_index, request.domain_id, request.mip_level, str(device)),
            lambda: self._load_level(request.mip_level, device),
        )
        try:
            return NativeAssetTile(
                request,
                domain.roles,
                _gather_tile(lease.tensor, request, domain),
                lease,
            )
        except BaseException:
            lease.release()
            raise

    @staticmethod
    def _bilinear_wrap(level: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        height, width = int(level.shape[0]), int(level.shape[1])
        wrapped = torch.remainder(uv, 1.0)
        x = wrapped[:, 0] * width - 0.5
        y = wrapped[:, 1] * height - 0.5
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        tx, ty = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
        x0w, x1w = torch.remainder(x0, width), torch.remainder(x0 + 1, width)
        y0w, y1w = torch.remainder(y0, height), torch.remainder(y0 + 1, height)
        v00 = level[y0w, x0w].to(dtype=torch.float32)
        v10 = level[y0w, x1w].to(dtype=torch.float32)
        v01 = level[y1w, x0w].to(dtype=torch.float32)
        v11 = level[y1w, x1w].to(dtype=torch.float32)
        return (
            (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty)
            + (v01 * (1.0 - tx) + v11 * tx) * ty
        )

    def sample_torch(
        self, uv: torch.Tensor, mip_level: torch.Tensor
    ) -> torch.Tensor:
        if uv.ndim != 2 or uv.shape[1] != 2 or mip_level.shape != (len(uv),):
            raise ValueError("MaterialX native feature samples require aligned uv and mip level")
        selected = torch.clamp(
            torch.round(mip_level).long(), 0, len(self.spatial_levels) - 1
        )
        result = torch.empty(
            (len(uv), 38), dtype=torch.float32, device=uv.device
        )
        # The mip selector lives on the query device. Copy the compact set of used
        # mip ids once instead of synchronizing once for every level through
        # ``bool(mask.any())``.
        used_levels = torch.unique(selected).detach().cpu().tolist()
        for index in used_levels:
            mask = selected == index
            lease = self._cache.acquire(
                (0, "surface-uv", index, str(uv.device)),
                lambda index=index: self._load_level(index, uv.device),
            )
            try:
                result[mask] = self._bilinear_wrap(lease.tensor, uv[mask])
            finally:
                lease.release()
        return result

    def sample(self, uv: np.ndarray, mip_level: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(uv, dtype=np.float32)
        selected = np.clip(
            np.rint(np.asarray(mip_level, dtype=np.float32)).astype(np.int64),
            0,
            len(self.spatial_levels) - 1,
        )
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or selected.shape != (len(coordinates),):
            raise ValueError("MaterialX native feature samples require aligned uv and mip level")
        result = np.empty((len(coordinates), 38), dtype=np.float32)
        result[:, :24] = self.constants
        for index, level in enumerate(self.spatial_levels):
            mask = selected == index
            if not np.any(mask):
                continue
            height, width = level.shape[:2]
            wrapped = np.remainder(coordinates[mask], 1.0)
            x = wrapped[:, 0] * width - 0.5
            y = wrapped[:, 1] * height - 0.5
            x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            tx, ty = (x - x0)[:, None], (y - y0)[:, None]
            x0w, x1w = np.remainder(x0, width), np.remainder(x0 + 1, width)
            y0w, y1w = np.remainder(y0, height), np.remainder(y0 + 1, height)
            values = level.astype(np.float32, copy=False)
            sampled = (
                (values[y0w, x0w] * (1.0 - tx) + values[y0w, x1w] * tx) * (1.0 - ty)
                + (values[y1w, x0w] * (1.0 - tx) + values[y1w, x1w] * tx) * ty
            )
            result[mask, 24:] = sampled
        return result


def _medium_values(medium: HomogeneousMedium) -> tuple[float, ...]:
    return (*medium.sigma_a, *medium.sigma_s, medium.g, medium.thickness)


def encode_layer_stack_native_features(stack: LayerStackIR) -> np.ndarray:
    """把 LayerStack 原生 tagged records 编成有 mask/one-hot 的 1×1 encoder 输入。"""

    values: list[float] = [
        len(stack.interfaces) / float(MAX_INTERFACES),
        len(stack.media) / float(MAX_MEDIA),
    ]
    for index in range(MAX_INTERFACES):
        if index >= len(stack.interfaces):
            values.extend([0.0] * 19)
            continue
        record = INTERFACE_STRUCT.unpack(pack_layer_interface(stack.interfaces[index]))
        kind = int(record[0])
        one_hot = [1.0 if kind == candidate else 0.0 for candidate in range(4)]
        values.extend((1.0, *one_hot, *map(float, record[2:])))
    for index in range(MAX_MEDIA):
        if index >= len(stack.media):
            values.extend([0.0] * 9)
            continue
        values.extend((1.0, *_medium_values(stack.media[index])))
    result = np.asarray(values, dtype=np.float32)
    layout = layer_stack_native_feature_layout()
    if result.shape != (layout.channel_count,) or not np.isfinite(result).all():
        raise AssertionError("LayerStack native feature layout disagrees with encoder payload")
    return result
