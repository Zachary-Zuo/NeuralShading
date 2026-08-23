"""材质族无关的 reference 查询采集与 HDF5 读写接口。"""

from .collector import CollectionConfig, collect_reference_dataset
from .contract import (
    EvaluatedBlock,
    PositionKind,
    QUERY_ROLE_NAMES,
    QueryPlan,
    QueryRole,
    ReferenceDescriptor,
    ReferenceProvider,
    SPLIT_NAMES,
    SourceState,
    SurfaceSample,
    make_state_id,
)
from .dataset import (
    COLOR_MODEL,
    FORMAT_NAME,
    FORMAT_VERSION,
    RESPONSE_MEASURE,
    ReferenceDataset,
    ReferenceDatasetManifest,
    ReferenceDatasetWriter,
    ReferenceStatistics,
    validate_reference_dataset,
)
from .directions import (
    equal_area_hemisphere,
    equal_area_sphere,
    peak_grazing_mixture_pdf,
    peak_grazing_mixture_query,
    stratified_uv,
    stratified_view_directions,
)
from .statistics import ReplicaMoments, combine_replica_moments

__all__ = [
    "COLOR_MODEL",
    "CollectionConfig",
    "EvaluatedBlock",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "PositionKind",
    "QUERY_ROLE_NAMES",
    "QueryPlan",
    "QueryRole",
    "RESPONSE_MEASURE",
    "ReferenceDescriptor",
    "ReferenceDataset",
    "ReferenceDatasetManifest",
    "ReferenceDatasetWriter",
    "ReferenceProvider",
    "ReferenceStatistics",
    "ReplicaMoments",
    "SPLIT_NAMES",
    "SourceState",
    "SurfaceSample",
    "collect_reference_dataset",
    "combine_replica_moments",
    "equal_area_hemisphere",
    "equal_area_sphere",
    "make_state_id",
    "peak_grazing_mixture_pdf",
    "peak_grazing_mixture_query",
    "stratified_uv",
    "stratified_view_directions",
    "validate_reference_dataset",
]
