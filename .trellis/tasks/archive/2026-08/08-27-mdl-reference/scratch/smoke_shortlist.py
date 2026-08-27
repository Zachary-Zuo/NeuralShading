from pathlib import Path

import numpy as np

from ncls.data.contract import PositionKind, QueryPlan, SurfaceSample
from ncls.data.providers.mdl import MdlGpuQueryRuntime
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import MDL_SDK_DIRECTORY, MdlCompiledArtifact


plan = QueryPlan(
    view_directions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    light_directions=np.asarray(
        [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]], dtype=np.float32
    ),
    solid_angle_weights=np.ones(2, dtype=np.float32),
    proposal_pdf=np.ones(2, dtype=np.float32),
    proposal_id="shortlist-smoke",
    seed=0,
)
surface = SurfaceSample(
    position_kind=PositionKind.UV,
    uv=(0.37, 0.61),
    uv_dx=(0.001, 0.0),
    uv_dy=(0.0, 0.001),
)

for slug in (
    "carpaint-shifting-flakes",
    "copper-antique-brushed-patinated",
    "aluminum-scratched",
    "ceramic-tiles-glazed-versailles",
    "velvet",
    "wood-tiles-pine-mosaic",
):
    artifact = MdlCompiledArtifact.load(
        PROJECT_ROOT / "build" / "mdl-reference" / "shortlist-audit" / slug
    )
    runtime = MdlGpuQueryRuntime(
        artifact,
        sdk_root=PROJECT_ROOT / "external" / MDL_SDK_DIRECTORY,
    )
    try:
        value, pdf = runtime.evaluate([surface], plan)
        print(slug, value.reshape(-1, 3).tolist(), pdf.reshape(-1).tolist())
    finally:
        runtime.close()
