from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Iterable

from .layer_stack import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
    pack_layer_stack,
)


LEGACY_MAGIC = 0x4E434C53
LEGACY_VERSION = 1
LEGACY_MAX_INTERFACES = 8
LEGACY_MAX_MEDIA = LEGACY_MAX_INTERFACES - 1
_LEGACY_HEADER = struct.Struct("<4I")
_LEGACY_INTERFACE = struct.Struct("<2I14f")
_LEGACY_MEDIUM = struct.Struct("<8f")
LEGACY_BINARY_SIZE = (
    _LEGACY_HEADER.size
    + LEGACY_MAX_INTERFACES * _LEGACY_INTERFACE.size
    + LEGACY_MAX_MEDIA * _LEGACY_MEDIUM.size
)


class LegacyLayerType(IntEnum):
    ROUGH_DIELECTRIC = 0
    ROUGH_CONDUCTOR = 1
    DIFFUSE = 2
    SHEEN = 3


def _vec3(name: str, values: Iterable[float]) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class LegacyLayerInterface:
    layer_type: LegacyLayerType
    roughness_x: float
    roughness_y: float
    eta: tuple[float, float, float] = (1.5, 1.5, 1.5)
    k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    albedo: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tangent_rotation: float = 0.0
    flags: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_type", LegacyLayerType(self.layer_type))
        object.__setattr__(self, "eta", _vec3("eta", self.eta))
        object.__setattr__(self, "k", _vec3("k", self.k))
        object.__setattr__(self, "albedo", _vec3("albedo", self.albedo))
        if not 0.0 <= self.roughness_x <= 1.0 or not 0.0 <= self.roughness_y <= 1.0:
            raise ValueError("roughness must lie in [0, 1]")
        if any(value < 0.0 for value in (*self.eta, *self.k, *self.albedo)):
            raise ValueError("legacy optical parameters must be nonnegative")
        if not 0 <= self.flags <= 0xFFFFFFFF:
            raise ValueError("flags must fit uint32")


@dataclass(frozen=True)
class LegacyLayerMedium:
    sigma_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sigma_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    g: float = 0.0
    thickness: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma_a", _vec3("sigma_a", self.sigma_a))
        object.__setattr__(self, "sigma_s", _vec3("sigma_s", self.sigma_s))
        if any(value < 0.0 for value in (*self.sigma_a, *self.sigma_s)):
            raise ValueError("legacy extinction coefficients must be nonnegative")
        if not -1.0 < self.g < 1.0:
            raise ValueError("g must lie in (-1, 1)")
        if self.thickness < 0.0:
            raise ValueError("thickness must be nonnegative")


@dataclass(frozen=True)
class LegacyLayerStack:
    layers: tuple[LegacyLayerInterface, ...]
    media: tuple[LegacyLayerMedium, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "media", tuple(self.media))
        if not 1 <= len(self.layers) <= LEGACY_MAX_INTERFACES:
            raise ValueError("legacy layer count must lie in [1, 8]")
        if len(self.media) != len(self.layers) - 1:
            raise ValueError("an N-interface legacy stack must contain N-1 media")


def pack_legacy_stack(stack: LegacyLayerStack) -> bytes:
    payload = bytearray(
        _LEGACY_HEADER.pack(LEGACY_MAGIC, LEGACY_VERSION, len(stack.layers), len(stack.media))
    )
    for index in range(LEGACY_MAX_INTERFACES):
        if index >= len(stack.layers):
            payload.extend(bytes(_LEGACY_INTERFACE.size))
            continue
        layer = stack.layers[index]
        payload.extend(
            _LEGACY_INTERFACE.pack(
                int(layer.layer_type),
                layer.flags,
                layer.roughness_x,
                layer.roughness_y,
                *layer.eta,
                *layer.k,
                *layer.albedo,
                layer.tangent_rotation,
                0.0,
                0.0,
            )
        )
    for index in range(LEGACY_MAX_MEDIA):
        if index >= len(stack.media):
            payload.extend(bytes(_LEGACY_MEDIUM.size))
            continue
        medium = stack.media[index]
        payload.extend(
            _LEGACY_MEDIUM.pack(*medium.sigma_a, *medium.sigma_s, medium.g, medium.thickness)
        )
    if len(payload) != LEGACY_BINARY_SIZE:
        raise AssertionError("unexpected legacy LayerStack binary size")
    return bytes(payload)


def unpack_legacy_stack(payload: bytes) -> LegacyLayerStack:
    if len(payload) != LEGACY_BINARY_SIZE:
        raise ValueError(f"legacy LayerStack payload must be {LEGACY_BINARY_SIZE} bytes")
    magic, version, layer_count, medium_count = _LEGACY_HEADER.unpack_from(payload, 0)
    if magic != LEGACY_MAGIC or version != LEGACY_VERSION:
        raise ValueError("unsupported legacy LayerStack header")
    if not 1 <= layer_count <= LEGACY_MAX_INTERFACES or medium_count != layer_count - 1:
        raise ValueError("invalid legacy layer/media count")
    layers = []
    offset = _LEGACY_HEADER.size
    for index in range(LEGACY_MAX_INTERFACES):
        values = _LEGACY_INTERFACE.unpack_from(payload, offset)
        offset += _LEGACY_INTERFACE.size
        if index >= layer_count:
            continue
        layer_type, flags, roughness_x, roughness_y, *floats = values
        layers.append(
            LegacyLayerInterface(
                LegacyLayerType(layer_type),
                roughness_x,
                roughness_y,
                eta=tuple(floats[0:3]),
                k=tuple(floats[3:6]),
                albedo=tuple(floats[6:9]),
                tangent_rotation=floats[9],
                flags=flags,
            )
        )
    media = []
    for index in range(LEGACY_MAX_MEDIA):
        values = _LEGACY_MEDIUM.unpack_from(payload, offset)
        offset += _LEGACY_MEDIUM.size
        if index < medium_count:
            media.append(
                LegacyLayerMedium(
                    sigma_a=tuple(values[0:3]),
                    sigma_s=tuple(values[3:6]),
                    g=values[6],
                    thickness=values[7],
                )
            )
    return LegacyLayerStack(tuple(layers), tuple(media))


def from_legacy_stack(stack: LegacyLayerStack) -> LayerStackIR:
    interfaces = []
    for layer in stack.layers:
        if layer.layer_type == LegacyLayerType.ROUGH_DIELECTRIC:
            interfaces.append(
                RoughDielectricInterface(
                    layer.roughness_x,
                    layer.roughness_y,
                    layer.eta[0],
                    layer.tangent_rotation,
                )
            )
        elif layer.layer_type == LegacyLayerType.ROUGH_CONDUCTOR:
            interfaces.append(
                RoughConductorInterface(
                    layer.roughness_x,
                    layer.roughness_y,
                    layer.eta,
                    layer.k,
                    layer.tangent_rotation,
                )
            )
        elif layer.layer_type == LegacyLayerType.DIFFUSE:
            interfaces.append(DiffuseInterface(layer.albedo))
        elif layer.layer_type == LegacyLayerType.SHEEN:
            interfaces.append(SheenInterface(layer.albedo, layer.roughness_x))
        else:
            raise ValueError(f"unsupported legacy layer type {layer.layer_type}")
    media = tuple(HomogeneousMedium(item.sigma_a, item.sigma_s, item.g, item.thickness) for item in stack.media)
    return LayerStackIR(tuple(interfaces), media)


def to_legacy_stack(stack: LayerStackIR) -> LegacyLayerStack:
    layers = []
    for interface in stack.interfaces:
        if isinstance(interface, RoughDielectricInterface):
            layers.append(
                LegacyLayerInterface(
                    LegacyLayerType.ROUGH_DIELECTRIC,
                    interface.alpha_x,
                    interface.alpha_y,
                    eta=(interface.relative_ior,) * 3,
                    tangent_rotation=interface.tangent_rotation,
                )
            )
        elif isinstance(interface, RoughConductorInterface):
            layers.append(
                LegacyLayerInterface(
                    LegacyLayerType.ROUGH_CONDUCTOR,
                    interface.alpha_x,
                    interface.alpha_y,
                    eta=interface.eta,
                    k=interface.k,
                    tangent_rotation=interface.tangent_rotation,
                )
            )
        elif isinstance(interface, DiffuseInterface):
            layers.append(
                LegacyLayerInterface(LegacyLayerType.DIFFUSE, 1.0, 1.0, albedo=interface.color)
            )
        elif isinstance(interface, SheenInterface):
            layers.append(
                LegacyLayerInterface(
                    LegacyLayerType.SHEEN,
                    interface.roughness,
                    interface.roughness,
                    albedo=interface.color,
                )
            )
        else:
            raise TypeError(f"unsupported interface type {type(interface)!r}")
    media = tuple(
        LegacyLayerMedium(item.sigma_a, item.sigma_s, item.g, item.thickness)
        for item in stack.media
    )
    return LegacyLayerStack(tuple(layers), media)


def convert_legacy_payload(payload: bytes) -> bytes:
    return pack_layer_stack(from_legacy_stack(unpack_legacy_stack(payload)))


def semantic_legacy_round_trip(payload: bytes) -> bytes:
    """用于迁移测试；unused v0 字段会被规范化，不能保证逐 byte 相同。"""

    return pack_legacy_stack(to_legacy_stack(from_legacy_stack(unpack_legacy_stack(payload))))
