from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias

from .abi_layout import CONTRACT_NAME, CONTRACT_VERSION, LAYOUT


Vec2: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]
Rgb: TypeAlias = Vec3


class TransportMode(IntEnum):
    RADIANCE = int(LAYOUT["transport_mode"]["Radiance"])
    IMPORTANCE = int(LAYOUT["transport_mode"]["Importance"])


class ScatteringEvent(IntFlag):
    NONE = 0
    REFLECTION = int(LAYOUT["event_flags"]["Reflection"])
    TRANSMISSION = int(LAYOUT["event_flags"]["Transmission"])
    DIFFUSE = int(LAYOUT["event_flags"]["Diffuse"])
    GLOSSY = int(LAYOUT["event_flags"]["Glossy"])
    DELTA = int(LAYOUT["event_flags"]["Delta"])
    FRONT_SIDE = int(LAYOUT["event_flags"]["FrontSide"])
    BACK_SIDE = int(LAYOUT["event_flags"]["BackSide"])
    VOLUME_BOUNDARY = int(LAYOUT["event_flags"]["VolumeBoundary"])


class BackendCapability(IntFlag):
    NONE = 0
    PREPARE = int(LAYOUT["capabilities"]["Prepare"])
    EVALUATE = int(LAYOUT["capabilities"]["Evaluate"])
    SAMPLE = int(LAYOUT["capabilities"]["Sample"])
    PDF = int(LAYOUT["capabilities"]["Pdf"])
    ANISOTROPIC_FRAME = int(LAYOUT["capabilities"]["AnisotropicFrame"])
    REVERSE_PDF = int(LAYOUT["capabilities"]["ReversePdf"])
    ANALYTIC_POLYGON_INTEGRATION = int(LAYOUT["capabilities"]["AnalyticPolygonIntegration"])
    PREFILTERED_ENVIRONMENT_INTEGRATION = int(LAYOUT["capabilities"]["PrefilteredEnvironmentIntegration"])
    NEURAL_ENVIRONMENT_INTEGRATION = int(LAYOUT["capabilities"]["NeuralEnvironmentIntegration"])
    DELTA_EVENTS = int(LAYOUT["capabilities"]["DeltaEvents"])
    TRANSMISSION = int(LAYOUT["capabilities"]["Transmission"])
    HOMOGENEOUS_VOLUME = int(LAYOUT["capabilities"]["HomogeneousVolume"])


REQUIRED_REALTIME_CAPABILITIES = (
    BackendCapability.PREPARE
    | BackendCapability.EVALUATE
    | BackendCapability.SAMPLE
    | BackendCapability.PDF
    | BackendCapability.ANISOTROPIC_FRAME
)


class StateStorage(str, Enum):
    INLINE = "inline"
    STRUCTURED = "structured"
    RAW = "raw"


def _vector(name: str, values: Iterable[float], count: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {count} finite values")
    return result


def _unit(name: str, values: Iterable[float]) -> Vec3:
    result = _vector(name, values, 3)
    length = math.sqrt(sum(value * value for value in result))
    if not math.isclose(length, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"{name} must be normalized")
    return result  # type: ignore[return-value]


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class ShadingFrame:
    normal: Vec3
    tangent: Vec3
    bitangent: Vec3

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal", _unit("normal", self.normal))
        object.__setattr__(self, "tangent", _unit("tangent", self.tangent))
        object.__setattr__(self, "bitangent", _unit("bitangent", self.bitangent))
        if max(abs(_dot(self.normal, self.tangent)), abs(_dot(self.normal, self.bitangent)), abs(_dot(self.tangent, self.bitangent))) > 1e-4:
            raise ValueError("shading frame axes must be orthogonal")


@dataclass(frozen=True)
class SurfaceInteraction:
    position: Vec3
    geometric_normal: Vec3
    shading_frame: ShadingFrame
    uv: Vec2 = (0.0, 0.0)
    uv_dx: Vec2 = (0.0, 0.0)
    uv_dy: Vec2 = (0.0, 0.0)
    material_instance_id: int = 0
    primitive_id: int = 0
    front_facing: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector("position", self.position, 3))
        object.__setattr__(self, "geometric_normal", _unit("geometric_normal", self.geometric_normal))
        object.__setattr__(self, "uv", _vector("uv", self.uv, 2))
        object.__setattr__(self, "uv_dx", _vector("uv_dx", self.uv_dx, 2))
        object.__setattr__(self, "uv_dy", _vector("uv_dy", self.uv_dy, 2))
        if self.material_instance_id < 0 or self.primitive_id < 0:
            raise ValueError("material and primitive ids must be nonnegative")


@dataclass(frozen=True)
class ScatteringContext:
    surface: SurfaceInteraction
    wo_world: Vec3
    transport_mode: TransportMode = TransportMode.RADIANCE
    component_mask: ScatteringEvent = ScatteringEvent.REFLECTION | ScatteringEvent.TRANSMISSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "wo_world", _unit("wo_world", self.wo_world))
        object.__setattr__(self, "transport_mode", TransportMode(self.transport_mode))
        object.__setattr__(self, "component_mask", ScatteringEvent(self.component_mask))


@dataclass(frozen=True)
class ScatteringPdf:
    forward: float
    reverse: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("forward", self.forward), ("reverse", self.reverse)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} pdf must be finite and nonnegative")


@dataclass(frozen=True)
class ScatteringEval:
    f: Rgb
    pdf: ScatteringPdf
    event_flags: ScatteringEvent
    valid: bool = True

    def __post_init__(self) -> None:
        f = _vector("f", self.f, 3)
        if any(value < 0.0 for value in f):
            raise ValueError("non-delta BSDF values must be nonnegative")
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "event_flags", ScatteringEvent(self.event_flags))


def positive_light_cosine(frame: ShadingFrame, wi_world: Vec3) -> float:
    """返回数据合同和 deferred lighting 唯一允许使用的入射光余弦。"""

    wi = _unit("wi_world", wi_world)
    return max(_dot(frame.normal, wi), 0.0)


def response_cosine(evaluation: ScatteringEval, frame: ShadingFrame, wi_world: Vec3) -> Rgb:
    """把公共接口的纯 BSDF ``f`` 转为磁盘/Falcor 所需的 ``f*cos``。"""

    cosine = positive_light_cosine(frame, wi_world)
    return tuple(value * cosine for value in evaluation.f)  # type: ignore[return-value]


@dataclass(frozen=True)
class ScatteringSample:
    wi_world: Vec3
    weight: Rgb
    pdf: ScatteringPdf
    eta: float
    event_flags: ScatteringEvent
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "wi_world", _unit("wi_world", self.wi_world))
        weight = _vector("weight", self.weight, 3)
        if any(value < 0.0 for value in weight):
            raise ValueError("sample weight must be nonnegative")
        object.__setattr__(self, "weight", weight)
        if not math.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("sample eta must be finite and positive")
        object.__setattr__(self, "event_flags", ScatteringEvent(self.event_flags))


@dataclass(frozen=True)
class BackendCostModel:
    compiled_material_bytes: int = 0
    state_bytes_per_pixel: int = 0
    prepare_parameter_count: int = 0
    prepare_texture_lookups: int = 0
    evaluate_texture_lookups: int = 0
    sample_texture_lookups: int = 0
    data_dependent_loops: bool = False

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "data_dependent_loops":
                continue
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BackendCostModel:
        return cls(**{name: value.get(name, default.default) for name, default in cls.__dataclass_fields__.items()})


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    backend_version: int
    supported_ir_ids: tuple[str, ...]
    capabilities: BackendCapability
    state_storage: StateStorage
    state_stride: int
    state_alignment: int
    deterministic_eval: bool
    bounded_execution: bool
    shader_entry_points: Mapping[str, str]
    cost_model: BackendCostModel = BackendCostModel()
    scattering_contract_name: str = CONTRACT_NAME
    scattering_contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.backend_id or self.backend_version < 1:
            raise ValueError("backend id and version must be valid")
        if not self.supported_ir_ids:
            raise ValueError("backend must declare at least one supported IR")
        object.__setattr__(self, "supported_ir_ids", tuple(self.supported_ir_ids))
        object.__setattr__(self, "capabilities", BackendCapability(self.capabilities))
        object.__setattr__(self, "state_storage", StateStorage(self.state_storage))
        object.__setattr__(self, "shader_entry_points", MappingProxyType(dict(self.shader_entry_points)))
        if self.scattering_contract_name != CONTRACT_NAME or self.scattering_contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported scattering contract")
        if self.state_stride < 0 or self.state_alignment < 1 or self.state_alignment & (self.state_alignment - 1):
            raise ValueError("state stride must be nonnegative and alignment must be a power of two")
        if self.state_storage == StateStorage.INLINE and self.state_stride != 0:
            raise ValueError("inline backend state_stride must be zero")
        if self.state_storage != StateStorage.INLINE and self.state_stride == 0:
            raise ValueError("stored backend state_stride must be positive")
        if self.state_stride and self.state_stride % self.state_alignment:
            raise ValueError("state_stride must be aligned")
        if self.cost_model.state_bytes_per_pixel != self.state_stride:
            raise ValueError("cost model state_bytes_per_pixel must match state_stride")

    @property
    def is_complete_realtime_backend(self) -> bool:
        return self.bounded_execution and (self.capabilities & REQUIRED_REALTIME_CAPABILITIES) == REQUIRED_REALTIME_CAPABILITIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "scattering_contract_name": self.scattering_contract_name,
            "scattering_contract_version": self.scattering_contract_version,
            "supported_ir_ids": list(self.supported_ir_ids),
            "capabilities": int(self.capabilities),
            "state_storage": self.state_storage.value,
            "state_stride": self.state_stride,
            "state_alignment": self.state_alignment,
            "deterministic_eval": self.deterministic_eval,
            "bounded_execution": self.bounded_execution,
            "shader_entry_points": dict(self.shader_entry_points),
            "cost_model": self.cost_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BackendDescriptor:
        return cls(
            str(value["backend_id"]),
            int(value["backend_version"]),
            tuple(str(item) for item in value["supported_ir_ids"]),
            BackendCapability(int(value["capabilities"])),
            StateStorage(str(value["state_storage"])),
            int(value["state_stride"]),
            int(value["state_alignment"]),
            bool(value["deterministic_eval"]),
            bool(value["bounded_execution"]),
            {str(name): str(entry) for name, entry in value["shader_entry_points"].items()},
            BackendCostModel.from_dict(value["cost_model"]),
            str(value["scattering_contract_name"]),
            int(value["scattering_contract_version"]),
        )
