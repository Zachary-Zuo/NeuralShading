"""与具体拟合表示无关的散射合同。"""

from .abi_layout import CONTRACT_NAME, CONTRACT_VERSION
from .contract import (
    REQUIRED_REALTIME_CAPABILITIES,
    REQUIRED_PATH_TRACING_CAPABILITIES,
    BackendCapability,
    BackendCostModel,
    BackendDescriptor,
    ScatteringContext,
    ScatteringEval,
    ScatteringEvent,
    ScatteringPdf,
    ScatteringSample,
    ShadingFrame,
    StateStorage,
    SurfaceInteraction,
    TransportMode,
    absolute_light_cosine,
    response_cosine,
)
from .program import (
    MaterialPayload,
    ReferenceProgramDefinition,
    ReferenceProgramDescriptor,
    ReferenceProgramProviderStatus,
    RuntimePayload,
)

__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "REQUIRED_REALTIME_CAPABILITIES",
    "REQUIRED_PATH_TRACING_CAPABILITIES",
    "BackendCapability",
    "BackendCostModel",
    "BackendDescriptor",
    "ScatteringContext",
    "MaterialPayload",
    "ReferenceProgramDefinition",
    "ReferenceProgramDescriptor",
    "ReferenceProgramProviderStatus",
    "RuntimePayload",
    "ScatteringEval",
    "ScatteringEvent",
    "ScatteringPdf",
    "ScatteringSample",
    "ShadingFrame",
    "StateStorage",
    "SurfaceInteraction",
    "TransportMode",
    "absolute_light_cosine",
    "response_cosine",
]
