from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

falcor = pytest.importorskip("falcor")

from ncls.data.falcor import create_falcor_device

KERNEL_ROOT = Path(__file__).resolve().with_name("kernels")


@pytest.mark.falcor
def test_primary_ray_cone_is_invariant_to_falcor_camera_basis_scale() -> None:
    device = create_falcor_device(falcor)
    compute = falcor.ComputePass(
        device,
        file=KERNEL_ROOT / "viewer_path_surface.cs.slang",
        cs_entry="main",
    )
    output = device.create_structured_buffer(
        struct_size=16,
        element_count=7,
        bind_flags=(
            falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
        ),
    )
    surface_output = device.create_structured_buffer(
        struct_size=16,
        element_count=2,
        bind_flags=(
            falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
        ),
    )
    compute.globals.gOutput = output
    compute.globals.gSurfaceOutput = surface_output
    compute.execute(threads_x=7)
    actual = output.to_numpy().view(np.float32).reshape(7, 4).copy()
    actual_surface = surface_output.to_numpy().view(np.float32).reshape(2, 4).copy()
    device.end_frame()

    expected_spread = 2.0 * 0.3443276 / 240.0
    ray_cone_width = expected_spread * 4.373682
    normal_projection = 1.0 / np.sqrt(1.0 + 0.2**2 + 0.1**2)
    expected_lod = -2.7890625 + np.log2(ray_cone_width / normal_projection)
    expected_footprint = np.exp2(expected_lod)
    expected = np.asarray(
        [expected_spread, expected_lod, expected_footprint, expected_footprint],
        dtype=np.float32,
    )

    np.testing.assert_allclose(actual[0], expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual[1], expected, rtol=2e-6, atol=1e-7)
    assert np.all(np.isfinite(actual))

    # More pixels across the same field of view produce a finer footprint.
    np.testing.assert_allclose(actual[2, 0], actual[0, 0] * 0.5, rtol=2e-6)
    np.testing.assert_allclose(actual[2, 1], actual[0, 1] - 1.0, rtol=2e-6)
    # Farther hits and grazing projection produce coarser footprints.
    np.testing.assert_allclose(actual[3, 1], actual[0, 1] + 1.0, rtol=2e-6)
    assert actual[4, 1] > actual[0, 1]
    # A closer hit produces a finer footprint.
    np.testing.assert_allclose(actual[5, 1], actual[0, 1] - 1.0, rtol=2e-6)
    # Invalid geometry differentials use a finite conservative full-UV fallback.
    np.testing.assert_allclose(actual[6, 1:], (0.0, 1.0, 1.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_surface[0], (0.2, 0.8, 0.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(actual_surface[1], (0.2, 0.2, 0.0, 1.0), atol=1e-7)
