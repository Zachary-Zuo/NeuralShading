from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .program import OperationId, ValueType


@dataclass(frozen=True)
class InputSpec:
    value_type: ValueType
    many: bool = False
    min_count: int = 1

    def __post_init__(self) -> None:
        if self.min_count < 0 or (not self.many and self.min_count != 1):
            raise ValueError("input min_count must be one for scalar inputs and nonnegative for arrays")


@dataclass(frozen=True)
class ParameterSpec:
    value_type: ValueType


@dataclass(frozen=True)
class OperationSpec:
    operation: OperationId
    inputs: Mapping[str, InputSpec]
    parameters: Mapping[str, ParameterSpec]
    outputs: Mapping[str, ValueType]


ROUGH_DIELECTRIC = OperationId("ncls.interface", "rough_dielectric", 1)
ROUGH_CONDUCTOR = OperationId("ncls.interface", "rough_conductor", 1)
DIFFUSE = OperationId("ncls.interface", "diffuse", 1)
SHEEN = OperationId("ncls.interface", "sheen", 1)
HOMOGENEOUS_MEDIUM = OperationId("ncls.medium", "homogeneous", 1)
LAYER_STACK = OperationId("ncls.composition", "layer_stack", 1)


OPERATION_REGISTRY: dict[OperationId, OperationSpec] = {
    ROUGH_DIELECTRIC: OperationSpec(
        ROUGH_DIELECTRIC,
        {},
        {
            "alpha_x": ParameterSpec(ValueType.FLOAT),
            "alpha_y": ParameterSpec(ValueType.FLOAT),
            "relative_ior": ParameterSpec(ValueType.FLOAT),
            "tangent_rotation": ParameterSpec(ValueType.FLOAT),
        },
        {"interface": ValueType.INTERFACE},
    ),
    ROUGH_CONDUCTOR: OperationSpec(
        ROUGH_CONDUCTOR,
        {},
        {
            "alpha_x": ParameterSpec(ValueType.FLOAT),
            "alpha_y": ParameterSpec(ValueType.FLOAT),
            "eta": ParameterSpec(ValueType.COLOR3),
            "k": ParameterSpec(ValueType.COLOR3),
            "tangent_rotation": ParameterSpec(ValueType.FLOAT),
        },
        {"interface": ValueType.INTERFACE},
    ),
    DIFFUSE: OperationSpec(
        DIFFUSE,
        {},
        {"color": ParameterSpec(ValueType.COLOR3)},
        {"interface": ValueType.INTERFACE},
    ),
    SHEEN: OperationSpec(
        SHEEN,
        {},
        {
            "color": ParameterSpec(ValueType.COLOR3),
            "roughness": ParameterSpec(ValueType.FLOAT),
        },
        {"interface": ValueType.INTERFACE},
    ),
    HOMOGENEOUS_MEDIUM: OperationSpec(
        HOMOGENEOUS_MEDIUM,
        {},
        {
            "sigma_a": ParameterSpec(ValueType.COLOR3),
            "sigma_s": ParameterSpec(ValueType.COLOR3),
            "g": ParameterSpec(ValueType.FLOAT),
            "thickness": ParameterSpec(ValueType.FLOAT),
        },
        {"medium": ValueType.MEDIUM},
    ),
    LAYER_STACK: OperationSpec(
        LAYER_STACK,
        {
            "interfaces": InputSpec(ValueType.INTERFACE, many=True),
            "media": InputSpec(ValueType.MEDIUM, many=True, min_count=0),
        },
        {},
        {"surface": ValueType.SURFACE},
    ),
}
