from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

falcor = pytest.importorskip("falcor")

from ncls.references.backend import create_reference_backend

KERNEL_ROOT = Path(__file__).resolve().with_name("kernels")


def run_probe(device) -> np.ndarray:
    compute = falcor.ComputePass(
        device,
        file=KERNEL_ROOT / "viewer_path_sample_generator.cs.slang",
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
    return output.to_numpy().view(np.float32).reshape(4, 4).copy()


@pytest.mark.falcor
def test_path_sample_generator_is_finite_bounded_deterministic_and_per_path() -> None:
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    first = run_probe(device)
    second = run_probe(device)
    device.end_frame()

    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.all((first >= 0.0) & (first < 1.0))
    assert np.unique(first, axis=0).shape[0] == first.shape[0]
