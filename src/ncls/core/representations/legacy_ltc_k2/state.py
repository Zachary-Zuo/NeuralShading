from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable

from ncls.core.material import LayerInterfaceIR, pack_layer_interface, unpack_layer_interface
from ncls.core.material import BINARY_SIZE as LAYER_STACK_IR_BYTES
from ncls.core.scattering import (
    REQUIRED_REALTIME_CAPABILITIES,
    BackendCapability,
    BackendCostModel,
    BackendDescriptor,
    StateStorage,
)

from ..descriptor import RepresentationDescriptor


STATE_MAGIC = 0x324B544C  # little-endian "LTK2"
STATE_VERSION = 1
RESIDUAL_LOBE_COUNT = 2
_HEADER = struct.Struct("<4I")
_LTC_LOBE = struct.Struct("<12f")
_INTERFACE_SIZE = 64
BINARY_SIZE = _HEADER.size + _INTERFACE_SIZE + RESIDUAL_LOBE_COUNT * _LTC_LOBE.size

DESCRIPTOR = RepresentationDescriptor(
    representation_id="legacy-ltc-k2",
    representation_version=1,
    display_name="Legacy exact-top + LTC K2",
    parameter_count=18,
    state_bytes=BINARY_SIZE,
    bounded=True,
    status="research-baseline",
)


def backend_descriptor() -> BackendDescriptor:
    return BackendDescriptor(
        backend_id="legacy-ltc-k2",
        backend_version=1,
        supported_ir_ids=("ncls.layer-stack-ir@1",),
        capabilities=REQUIRED_REALTIME_CAPABILITIES | BackendCapability.REVERSE_PDF,
        state_storage=StateStorage.STRUCTURED,
        state_stride=BINARY_SIZE,
        state_alignment=16,
        deterministic_eval=True,
        bounded_execution=True,
        shader_entry_points={
            "prepare": "LegacyLtcK2Backend.prepare",
            "evaluate": "LegacyLtcK2State.evaluate",
            "sample": "LegacyLtcK2State.sample",
            "pdf": "LegacyLtcK2State.pdf",
        },
        cost_model=BackendCostModel(
            compiled_material_bytes=0,
            state_bytes_per_pixel=BINARY_SIZE,
            prepare_parameter_count=0,
            data_dependent_loops=False,
        ),
    )


def p1_backend_descriptor(*, parameter_count: int) -> BackendDescriptor:
    """描述由 P1 Slang 网络在 ``prepare()`` 中实时解码的完整后端。"""

    if parameter_count <= 0:
        raise ValueError("P1 parameter_count must be positive")
    return BackendDescriptor(
        backend_id="legacy-ltc-k2",
        backend_version=1,
        supported_ir_ids=("ncls.layer-stack-ir@1",),
        capabilities=REQUIRED_REALTIME_CAPABILITIES | BackendCapability.REVERSE_PDF,
        state_storage=StateStorage.STRUCTURED,
        state_stride=BINARY_SIZE,
        state_alignment=16,
        deterministic_eval=True,
        bounded_execution=True,
        shader_entry_points={
            "prepare": "LegacyLtcK2P1Backend.prepare",
            "evaluate": "LegacyLtcK2State.evaluate",
            "sample": "LegacyLtcK2State.sample",
            "pdf": "LegacyLtcK2State.pdf",
        },
        cost_model=BackendCostModel(
            compiled_material_bytes=LAYER_STACK_IR_BYTES,
            state_bytes_per_pixel=BINARY_SIZE,
            prepare_parameter_count=parameter_count,
            data_dependent_loops=False,
        ),
    )


def _tuple(name: str, values: Iterable[float], count: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {count} finite values")
    return result


@dataclass(frozen=True)
class LegacyLtcK2Lobe:
    amplitude: tuple[float, float, float]
    inverse_scale: tuple[float, float]
    shear: tuple[float, float, float]
    angle: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", _tuple("amplitude", self.amplitude, 3))
        object.__setattr__(self, "inverse_scale", _tuple("inverse_scale", self.inverse_scale, 2))
        object.__setattr__(self, "shear", _tuple("shear", self.shear, 3))
        angle = float(self.angle)
        if any(value < 0.0 for value in self.amplitude):
            raise ValueError("amplitude must be nonnegative")
        scale_min, scale_max = math.exp(-3.0), math.exp(3.0)
        if any(value < scale_min - 1e-6 or value > scale_max + 1e-5 for value in self.inverse_scale):
            raise ValueError("inverse_scale must lie in [exp(-3), exp(3)]")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        object.__setattr__(self, "angle", angle)


@dataclass(frozen=True)
class LegacyLtcK2State:
    direct_top: LayerInterfaceIR
    residual_lobes: tuple[LegacyLtcK2Lobe, LegacyLtcK2Lobe]

    def __post_init__(self) -> None:
        object.__setattr__(self, "residual_lobes", tuple(self.residual_lobes))
        if len(self.residual_lobes) != RESIDUAL_LOBE_COUNT:
            raise ValueError(f"legacy-ltc-k2 requires {RESIDUAL_LOBE_COUNT} residual lobes")


def _pack_lobe(lobe: LegacyLtcK2Lobe) -> bytes:
    return _LTC_LOBE.pack(
        *lobe.amplitude,
        0.0,
        *lobe.inverse_scale,
        *lobe.shear,
        lobe.angle,
        0.0,
        0.0,
    )


def pack_state(state: LegacyLtcK2State) -> bytes:
    payload = bytearray(_HEADER.pack(STATE_MAGIC, STATE_VERSION, RESIDUAL_LOBE_COUNT, 0))
    payload.extend(pack_layer_interface(state.direct_top))
    for lobe in state.residual_lobes:
        payload.extend(_pack_lobe(lobe))
    if len(payload) != BINARY_SIZE:
        raise AssertionError("unexpected legacy-ltc-k2 state size")
    return bytes(payload)


def pack_states(states: Iterable[LegacyLtcK2State]) -> bytes:
    return b"".join(pack_state(state) for state in states)


def unpack_state(payload: bytes) -> LegacyLtcK2State:
    if len(payload) != BINARY_SIZE:
        raise ValueError(f"legacy-ltc-k2 state must be {BINARY_SIZE} bytes")
    magic, version, lobe_count, flags = _HEADER.unpack_from(payload, 0)
    if magic != STATE_MAGIC or version != STATE_VERSION or lobe_count != RESIDUAL_LOBE_COUNT or flags != 0:
        raise ValueError("invalid legacy-ltc-k2 state header")
    direct_top = unpack_layer_interface(payload[_HEADER.size : _HEADER.size + _INTERFACE_SIZE])
    lobes = []
    offset = _HEADER.size + _INTERFACE_SIZE
    for _ in range(RESIDUAL_LOBE_COUNT):
        values = _LTC_LOBE.unpack_from(payload, offset)
        if values[3] != 0.0 or values[10] != 0.0 or values[11] != 0.0:
            raise ValueError("legacy-ltc-k2 reserved lobe fields must be zero")
        lobes.append(
            LegacyLtcK2Lobe(
                tuple(values[0:3]),
                tuple(values[4:6]),
                tuple(values[6:9]),
                values[9],
            )
        )
        offset += _LTC_LOBE.size
    return LegacyLtcK2State(direct_top, tuple(lobes))  # type: ignore[arg-type]
