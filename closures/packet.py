from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable

from schema import LayerInterface, LayerType


PACKET_MAGIC = 0x504C434E  # "NCLP" in little-endian bytes.
PACKET_VERSION = 1
RESIDUAL_LOBE_COUNT = 2

_HEADER = struct.Struct("<4I")
_INTERFACE = struct.Struct("<2I14f")
_LTC_LOBE = struct.Struct("<12f")
BINARY_SIZE = _HEADER.size + _INTERFACE.size + RESIDUAL_LOBE_COUNT * _LTC_LOBE.size


def _tuple(name: str, values: Iterable[float], count: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class LtcResidualLobe:
    amplitude: tuple[float, float, float]
    inverse_scale: tuple[float, float]
    shear: tuple[float, float, float]
    angle: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", _tuple("amplitude", self.amplitude, 3))
        object.__setattr__(self, "inverse_scale", _tuple("inverse_scale", self.inverse_scale, 2))
        object.__setattr__(self, "shear", _tuple("shear", self.shear, 3))
        object.__setattr__(self, "angle", float(self.angle))
        if any(value < 0.0 for value in self.amplitude):
            raise ValueError("amplitude must be nonnegative")
        scale_min = math.exp(-3.0)
        scale_max = math.exp(3.0)
        if any(value < scale_min - 1e-6 or value > scale_max + 1e-5 for value in self.inverse_scale):
            raise ValueError("inverse_scale must lie in [exp(-3), exp(3)]")
        if not math.isfinite(self.angle):
            raise ValueError("angle must be finite")


@dataclass(frozen=True)
class ClosurePacket:
    direct_top: LayerInterface
    residual_lobes: tuple[LtcResidualLobe, LtcResidualLobe]

    def __post_init__(self) -> None:
        object.__setattr__(self, "residual_lobes", tuple(self.residual_lobes))
        if len(self.residual_lobes) != RESIDUAL_LOBE_COUNT:
            raise ValueError(f"closure packet requires {RESIDUAL_LOBE_COUNT} residual lobes")


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


def _unpack_interface(payload: bytes, offset: int) -> LayerInterface:
    values = _INTERFACE.unpack_from(payload, offset)
    layer_type, flags, roughness_x, roughness_y, *floats = values
    return LayerInterface(
        layer_type=LayerType(layer_type),
        flags=flags,
        roughness_x=roughness_x,
        roughness_y=roughness_y,
        eta=tuple(floats[0:3]),
        k=tuple(floats[3:6]),
        albedo=tuple(floats[6:9]),
        tangent_rotation=floats[9],
    )


def _pack_lobe(lobe: LtcResidualLobe) -> bytes:
    return _LTC_LOBE.pack(
        *lobe.amplitude,
        0.0,
        *lobe.inverse_scale,
        lobe.shear[0],
        lobe.shear[1],
        lobe.shear[2],
        lobe.angle,
        0.0,
        0.0,
    )


def pack_packet(packet: ClosurePacket) -> bytes:
    payload = bytearray(
        _HEADER.pack(PACKET_MAGIC, PACKET_VERSION, RESIDUAL_LOBE_COUNT, 0)
    )
    payload.extend(_pack_interface(packet.direct_top))
    for lobe in packet.residual_lobes:
        payload.extend(_pack_lobe(lobe))
    if len(payload) != BINARY_SIZE:
        raise AssertionError("unexpected ClosurePacket binary size")
    return bytes(payload)


def pack_packets(packets: Iterable[ClosurePacket]) -> bytes:
    return b"".join(pack_packet(packet) for packet in packets)


def unpack_packet(payload: bytes) -> ClosurePacket:
    if len(payload) != BINARY_SIZE:
        raise ValueError(f"ClosurePacket payload must be {BINARY_SIZE} bytes")
    magic, version, lobe_count, _ = _HEADER.unpack_from(payload, 0)
    if magic != PACKET_MAGIC:
        raise ValueError("invalid ClosurePacket magic")
    if version != PACKET_VERSION or lobe_count != RESIDUAL_LOBE_COUNT:
        raise ValueError("unsupported ClosurePacket version or lobe count")
    direct_top = _unpack_interface(payload, _HEADER.size)
    lobes: list[LtcResidualLobe] = []
    offset = _HEADER.size + _INTERFACE.size
    for _ in range(RESIDUAL_LOBE_COUNT):
        values = _LTC_LOBE.unpack_from(payload, offset)
        offset += _LTC_LOBE.size
        lobes.append(
            LtcResidualLobe(
                amplitude=tuple(values[0:3]),
                inverse_scale=tuple(values[4:6]),
                shear=tuple(values[6:9]),
                angle=values[9],
            )
        )
    return ClosurePacket(direct_top, tuple(lobes))  # type: ignore[arg-type]
