from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

falcor = pytest.importorskip("falcor")

from ncls.references.falcor import create_falcor_device

KERNEL_ROOT = Path(__file__).resolve().with_name("kernels")


@pytest.mark.falcor
def test_environment_multiple_sample_mis_uses_both_sample_counts_and_delta_measure() -> None:
    device = create_falcor_device(falcor)
    compute = falcor.ComputePass(
        device,
        file=KERNEL_ROOT / "viewer_path_environment.cs.slang",
        cs_entry="main",
    )
    output = device.create_structured_buffer(
        struct_size=16,
        element_count=4,
        bind_flags=(
            falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
        ),
    )
    compute.globals.gOutput = output
    compute.execute(threads_x=4)
    actual = output.to_numpy().view(np.float32).reshape(4, 4).copy()
    device.end_frame()

    expected = np.asarray(
        [
            [1.0, 1.0, 0.5, 0.5],
            [2.0, 0.5, 4.0 / 4.25, 0.25 / 4.25],
            [2.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert np.all(np.isfinite(actual))
