from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Iterable


SCHEMA_MAGIC = 0x4E434C53  # "NCLS" in little-endian bytes.
SCHEMA_VERSION = 1
MAX_LAYERS = 8
MAX_MEDIA = MAX_LAYERS - 1

_HEADER = struct.Struct("<4I")
_INTERFACE = struct.Struct("<2I14f")
_MEDIUM = struct.Struct("<8f")
BINARY_SIZE = _HEADER.size + MAX_LAYERS * _INTERFACE.size + MAX_MEDIA * _MEDIUM.size


class LayerType(IntEnum):
    ROUGH_DIELECTRIC = 0
    ROUGH_CONDUCTOR = 1
    DIFFUSE = 2
    SHEEN = 3


def _vec3(name: str, values: Iterable[float]) -> tuple[float, float, float]:
    result = tuple(float(v) for v in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return result  # type: ignore[return-value]


def _check_nonnegative(name: str, values: Iterable[float]) -> None:
    if any(v < 0.0 for v in values):
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class LayerInterface:
    layer_type: LayerType
    roughness_x: float
    roughness_y: float
    eta: tuple[float, float, float] = (1.5, 1.5, 1.5)
    k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    albedo: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tangent_rotation: float = 0.0
    flags: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_type", LayerType(self.layer_type))
        object.__setattr__(self, "eta", _vec3("eta", self.eta))
        object.__setattr__(self, "k", _vec3("k", self.k))
        object.__setattr__(self, "albedo", _vec3("albedo", self.albedo))
        if not 0.0 <= self.roughness_x <= 1.0 or not 0.0 <= self.roughness_y <= 1.0:
            raise ValueError("roughness must lie in [0, 1]")
        _check_nonnegative("eta", self.eta)
        _check_nonnegative("k", self.k)
        _check_nonnegative("albedo", self.albedo)
        if not 0 <= self.flags <= 0xFFFFFFFF:
            raise ValueError("flags must fit uint32")


@dataclass(frozen=True)
class LayerMedium:
    sigma_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sigma_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    g: float = 0.0
    thickness: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma_a", _vec3("sigma_a", self.sigma_a))
        object.__setattr__(self, "sigma_s", _vec3("sigma_s", self.sigma_s))
        _check_nonnegative("sigma_a", self.sigma_a)
        _check_nonnegative("sigma_s", self.sigma_s)
        if not -1.0 < self.g < 1.0:
            raise ValueError("g must lie in (-1, 1)")
        if self.thickness < 0.0:
            raise ValueError("thickness must be nonnegative")


@dataclass(frozen=True)
class LayerStack:
    layers: tuple[LayerInterface, ...]
    media: tuple[LayerMedium, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "media", tuple(self.media))
        if not 1 <= len(self.layers) <= MAX_LAYERS:
            raise ValueError(f"layer count must lie in [1, {MAX_LAYERS}]")
        if len(self.media) != len(self.layers) - 1:
            raise ValueError("an N-interface stack must contain N-1 media")


def _pack_interface(layer: LayerInterface) -> bytes:
    return _INTERFACE.pack(
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


def _pack_medium(medium: LayerMedium) -> bytes:
    return _MEDIUM.pack(*medium.sigma_a, *medium.sigma_s, medium.g, medium.thickness)


def pack_stack(stack: LayerStack) -> bytes:
    payload = bytearray(_HEADER.pack(SCHEMA_MAGIC, SCHEMA_VERSION, len(stack.layers), len(stack.media)))
    for index in range(MAX_LAYERS):
        payload.extend(_pack_interface(stack.layers[index]) if index < len(stack.layers) else bytes(_INTERFACE.size))
    for index in range(MAX_MEDIA):
        payload.extend(_pack_medium(stack.media[index]) if index < len(stack.media) else bytes(_MEDIUM.size))
    if len(payload) != BINARY_SIZE:
        raise AssertionError("unexpected LayerStack binary size")
    return bytes(payload)


def unpack_stack(payload: bytes) -> LayerStack:
    if len(payload) != BINARY_SIZE:
        raise ValueError(f"LayerStack payload must be {BINARY_SIZE} bytes")
    magic, version, layer_count, medium_count = _HEADER.unpack_from(payload, 0)
    if magic != SCHEMA_MAGIC:
        raise ValueError("invalid LayerStack magic")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported LayerStack version {version}")
    if not 1 <= layer_count <= MAX_LAYERS or medium_count != layer_count - 1:
        raise ValueError("invalid layer/media count in payload")

    layers: list[LayerInterface] = []
    offset = _HEADER.size
    for index in range(MAX_LAYERS):
        values = _INTERFACE.unpack_from(payload, offset)
        offset += _INTERFACE.size
        if index >= layer_count:
            continue
        layer_type, flags, roughness_x, roughness_y, *floats = values
        layers.append(
            LayerInterface(
                layer_type=LayerType(layer_type),
                flags=flags,
                roughness_x=roughness_x,
                roughness_y=roughness_y,
                eta=tuple(floats[0:3]),
                k=tuple(floats[3:6]),
                albedo=tuple(floats[6:9]),
                tangent_rotation=floats[9],
            )
        )

    media: list[LayerMedium] = []
    for index in range(MAX_MEDIA):
        values = _MEDIUM.unpack_from(payload, offset)
        offset += _MEDIUM.size
        if index >= medium_count:
            continue
        media.append(
            LayerMedium(
                sigma_a=tuple(values[0:3]),
                sigma_s=tuple(values[3:6]),
                g=values[6],
                thickness=values[7],
            )
        )
    return LayerStack(tuple(layers), tuple(media))

