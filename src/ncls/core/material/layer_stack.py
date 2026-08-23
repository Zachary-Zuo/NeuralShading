from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable, TypeAlias

from .abi_layout import (
    ABI_MAGIC,
    ABI_VERSION,
    BINARY_SIZE,
    HEADER_STRUCT,
    INTERFACE_STRUCT,
    MAX_INTERFACES,
    MAX_MEDIA,
    MEDIUM_STRUCT,
)


Rgb: TypeAlias = tuple[float, float, float]


class InterfaceKind(IntEnum):
    ROUGH_DIELECTRIC = 0
    ROUGH_CONDUCTOR = 1
    DIFFUSE = 2
    SHEEN = 3


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded(name: str, value: float, low: float, high: float) -> float:
    result = _finite(name, value)
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


def _rgb(name: str, values: Iterable[float], *, upper: float | None = None) -> Rgb:
    result = tuple(_finite(name, value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(value < 0.0 for value in result):
        raise ValueError(f"{name} must be nonnegative")
    if upper is not None and any(value > upper for value in result):
        raise ValueError(f"{name} must not exceed {upper}")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class RoughDielectricInterface:
    alpha_x: float
    alpha_y: float
    relative_ior: float
    tangent_rotation: float = 0.0

    kind = InterfaceKind.ROUGH_DIELECTRIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha_x", _bounded("alpha_x", self.alpha_x, 0.0, 1.0))
        object.__setattr__(self, "alpha_y", _bounded("alpha_y", self.alpha_y, 0.0, 1.0))
        relative_ior = _finite("relative_ior", self.relative_ior)
        if relative_ior <= 0.0:
            raise ValueError("relative_ior must be positive")
        object.__setattr__(self, "relative_ior", relative_ior)
        object.__setattr__(self, "tangent_rotation", _finite("tangent_rotation", self.tangent_rotation))


@dataclass(frozen=True)
class RoughConductorInterface:
    alpha_x: float
    alpha_y: float
    eta: Rgb
    k: Rgb
    tangent_rotation: float = 0.0

    kind = InterfaceKind.ROUGH_CONDUCTOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha_x", _bounded("alpha_x", self.alpha_x, 0.0, 1.0))
        object.__setattr__(self, "alpha_y", _bounded("alpha_y", self.alpha_y, 0.0, 1.0))
        object.__setattr__(self, "eta", _rgb("eta", self.eta))
        object.__setattr__(self, "k", _rgb("k", self.k))
        object.__setattr__(self, "tangent_rotation", _finite("tangent_rotation", self.tangent_rotation))


@dataclass(frozen=True)
class DiffuseInterface:
    color: Rgb

    kind = InterfaceKind.DIFFUSE

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", _rgb("color", self.color, upper=1.0))


@dataclass(frozen=True)
class SheenInterface:
    color: Rgb
    roughness: float

    kind = InterfaceKind.SHEEN

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", _rgb("color", self.color, upper=1.0))
        object.__setattr__(self, "roughness", _bounded("roughness", self.roughness, 0.0, 1.0))


LayerInterfaceIR: TypeAlias = RoughDielectricInterface | RoughConductorInterface | DiffuseInterface | SheenInterface


@dataclass(frozen=True)
class HomogeneousMedium:
    sigma_a: Rgb = (0.0, 0.0, 0.0)
    sigma_s: Rgb = (0.0, 0.0, 0.0)
    g: float = 0.0
    thickness: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma_a", _rgb("sigma_a", self.sigma_a))
        object.__setattr__(self, "sigma_s", _rgb("sigma_s", self.sigma_s))
        object.__setattr__(self, "g", _finite("g", self.g))
        if not -1.0 < self.g < 1.0:
            raise ValueError("g must lie in (-1, 1)")
        thickness = _finite("thickness", self.thickness)
        if thickness < 0.0:
            raise ValueError("thickness must be nonnegative")
        object.__setattr__(self, "thickness", thickness)


@dataclass(frozen=True)
class LayerStackIR:
    interfaces: tuple[LayerInterfaceIR, ...]
    media: tuple[HomogeneousMedium, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
        object.__setattr__(self, "media", tuple(self.media))
        if not 1 <= len(self.interfaces) <= MAX_INTERFACES:
            raise ValueError(f"interface count must lie in [1, {MAX_INTERFACES}]")
        if len(self.media) != len(self.interfaces) - 1:
            raise ValueError("an N-interface stack must contain N-1 media")
        if any(not isinstance(interface, RoughDielectricInterface) for interface in self.interfaces[:-1]):
            raise ValueError("only transmissive rough dielectric interfaces may appear above the opaque base")
        if isinstance(self.interfaces[-1], RoughDielectricInterface):
            raise ValueError("v1 LayerStackIR requires an opaque base interface")


def _pack_interface(interface: LayerInterfaceIR) -> bytes:
    values = [0.0] * 14
    if isinstance(interface, RoughDielectricInterface):
        values[0:3] = [interface.alpha_x, interface.alpha_y, interface.relative_ior]
        values[12] = interface.tangent_rotation
    elif isinstance(interface, RoughConductorInterface):
        values[0:2] = [interface.alpha_x, interface.alpha_y]
        values[3:6] = interface.eta
        values[6:9] = interface.k
        values[12] = interface.tangent_rotation
    elif isinstance(interface, DiffuseInterface):
        values[9:12] = interface.color
    elif isinstance(interface, SheenInterface):
        values[0:2] = [interface.roughness, interface.roughness]
        values[9:12] = interface.color
    else:
        raise TypeError(f"unsupported interface type {type(interface)!r}")
    return INTERFACE_STRUCT.pack(int(interface.kind), 0, *values)


def _unpack_interface(payload: bytes, offset: int) -> LayerInterfaceIR:
    kind_value, flags, *values = INTERFACE_STRUCT.unpack_from(payload, offset)
    if flags != 0:
        raise ValueError("LayerStackIR v1 interface flags must be zero")
    if abs(values[13]) > 0.0:
        raise ValueError("LayerStackIR v1 reserved interface field must be zero")
    kind = InterfaceKind(kind_value)
    if kind == InterfaceKind.ROUGH_DIELECTRIC:
        return RoughDielectricInterface(values[0], values[1], values[2], values[12])
    if kind == InterfaceKind.ROUGH_CONDUCTOR:
        return RoughConductorInterface(values[0], values[1], tuple(values[3:6]), tuple(values[6:9]), values[12])  # type: ignore[arg-type]
    if kind == InterfaceKind.DIFFUSE:
        return DiffuseInterface(tuple(values[9:12]))  # type: ignore[arg-type]
    if kind == InterfaceKind.SHEEN:
        if not math.isclose(values[0], values[1], rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("LayerStackIR v1 sheen roughness fields must match")
        return SheenInterface(tuple(values[9:12]), values[0])  # type: ignore[arg-type]
    raise AssertionError("unreachable interface kind")


def pack_layer_interface(interface: LayerInterfaceIR) -> bytes:
    """打包一个 64-byte LayerStackIR interface record。"""

    return _pack_interface(interface)


def unpack_layer_interface(payload: bytes) -> LayerInterfaceIR:
    """读取一个规范化的 64-byte LayerStackIR interface record。"""

    if len(payload) != INTERFACE_STRUCT.size:
        raise ValueError(f"LayerStackIR interface payload must be {INTERFACE_STRUCT.size} bytes")
    return _unpack_interface(payload, 0)


def _pack_medium(medium: HomogeneousMedium) -> bytes:
    return MEDIUM_STRUCT.pack(*medium.sigma_a, *medium.sigma_s, medium.g, medium.thickness)


def pack_layer_stack(stack: LayerStackIR) -> bytes:
    payload = bytearray(HEADER_STRUCT.pack(ABI_MAGIC, ABI_VERSION, len(stack.interfaces), len(stack.media)))
    for index in range(MAX_INTERFACES):
        payload.extend(_pack_interface(stack.interfaces[index]) if index < len(stack.interfaces) else bytes(INTERFACE_STRUCT.size))
    for index in range(MAX_MEDIA):
        payload.extend(_pack_medium(stack.media[index]) if index < len(stack.media) else bytes(MEDIUM_STRUCT.size))
    if len(payload) != BINARY_SIZE:
        raise AssertionError("unexpected LayerStackIR binary size")
    return bytes(payload)


def unpack_layer_stack(payload: bytes) -> LayerStackIR:
    if len(payload) != BINARY_SIZE:
        raise ValueError(f"LayerStackIR payload must be {BINARY_SIZE} bytes")
    magic, version, interface_count, medium_count = HEADER_STRUCT.unpack_from(payload, 0)
    if magic != ABI_MAGIC:
        raise ValueError("invalid LayerStackIR magic")
    if version != ABI_VERSION:
        raise ValueError(f"unsupported LayerStackIR ABI version {version}")
    if not 1 <= interface_count <= MAX_INTERFACES or medium_count != interface_count - 1:
        raise ValueError("invalid LayerStackIR interface/media count")

    interfaces: list[LayerInterfaceIR] = []
    offset = HEADER_STRUCT.size
    for index in range(MAX_INTERFACES):
        if index < interface_count:
            interfaces.append(_unpack_interface(payload, offset))
        elif any(payload[offset : offset + INTERFACE_STRUCT.size]):
            raise ValueError("nonzero padded interface record")
        offset += INTERFACE_STRUCT.size

    media: list[HomogeneousMedium] = []
    for index in range(MAX_MEDIA):
        values = MEDIUM_STRUCT.unpack_from(payload, offset)
        if index < medium_count:
            media.append(HomogeneousMedium(tuple(values[0:3]), tuple(values[3:6]), values[6], values[7]))  # type: ignore[arg-type]
        elif any(payload[offset : offset + MEDIUM_STRUCT.size]):
            raise ValueError("nonzero padded medium record")
        offset += MEDIUM_STRUCT.size
    return LayerStackIR(tuple(interfaces), tuple(media))
