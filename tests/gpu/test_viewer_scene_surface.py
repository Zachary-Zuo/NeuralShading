from pathlib import Path

import numpy as np
import pytest

falcor = pytest.importorskip("falcor")
from ncls.references.backend import create_reference_backend


@pytest.mark.falcor
def test_raster_material_sentinel_frames_and_context_match_ray_surface():
    device = create_reference_backend()._create_device(falcor)
    compute = falcor.ComputePass(device, file=Path(__file__).with_name("kernels") / "viewer_scene_surface.cs.slang", cs_entry="main")
    output = device.create_structured_buffer(struct_size=16, element_count=16,
        bind_flags=falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess)
    compute.globals.gOutput = output
    compute.execute(threads_x=4)
    values = output.to_numpy().view(np.float32).reshape(4, 4, 4).copy()
    device.end_frame()
    np.testing.assert_allclose(values[:, 0], [[0, 0, 0, 1]] * 4, atol=1e-7)
    np.testing.assert_allclose(values[:, 1], [[0.2, 0.8, 0, 1], [0.2, 0.2, 1, 1], [0.2, 0.8, 2, 0], [0.2, 0.2, 3, 0]], atol=1e-7)
    np.testing.assert_allclose(values[:, 2], [[0, 0, 1, 0]] * 4, atol=1e-6)
    np.testing.assert_allclose(values[:, 3], 0, atol=1e-7)
