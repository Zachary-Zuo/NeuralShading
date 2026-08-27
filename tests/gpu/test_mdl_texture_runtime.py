from __future__ import annotations

import numpy as np
import pytest

from ncls.data.collector import CollectionConfig
from ncls.data.contract import PositionKind, QueryPlan, SurfaceSample
from ncls.data.providers.mdl import MdlAssetSpec, MdlProvider, MdlProviderConfig


@pytest.mark.falcor
def test_mdl_texture_runtime_preserves_lower_left_uv_and_repeat() -> None:
    provider = MdlProvider(
        CollectionConfig(view_count=1, light_count=1, spatial_sample_count=1),
        MdlProviderConfig(
            assets=(
                MdlAssetSpec(
                    "textured-diffuse",
                    "::textured_diffuse",
                    "textured_diffuse",
                ),
            ),
        ),
    )
    try:
        state = provider.source_states()[0]
        surfaces = tuple(
            SurfaceSample(
                position_kind=PositionKind.UV,
                uv=uv,
                uv_dx=(0.01, 0.0),
                uv_dy=(0.0, 0.01),
            )
            for uv in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (1.25, 0.25))
        )
        plan = QueryPlan(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([2.0 * np.pi], dtype=np.float32),
            np.asarray([1.0 / (2.0 * np.pi)], dtype=np.float32),
            "mdl-texture-fixture",
            1,
        )
        result = provider.evaluate(state, surfaces, plan)
        expected_colors = np.asarray(
            ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result.mean[:, 0, 0], expected_colors / np.pi, rtol=3e-5, atol=3e-6
        )
        np.testing.assert_allclose(result.reference_pdf[:, 0, 0], 1.0 / np.pi, rtol=3e-6)
    finally:
        provider.close()
