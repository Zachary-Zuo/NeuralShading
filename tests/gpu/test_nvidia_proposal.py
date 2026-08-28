from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

falcor = pytest.importorskip("falcor")

from ncls.references.falcor import create_falcor_device

KERNEL_ROOT = Path(__file__).resolve().with_name("kernels")


@pytest.mark.falcor
def test_nvidia_two_lobe_sample_reports_its_exact_mixture_pdf() -> None:
    device = create_falcor_device(falcor)
    compute = falcor.ComputePass(
        device,
        file=KERNEL_ROOT / "nvidia_proposal_sample_pdf.cs.slang",
        cs_entry="main",
    )
    output = device.create_structured_buffer(
        struct_size=16,
        element_count=32,
        bind_flags=(
            falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
        ),
    )
    compute.globals.gOutput = output
    compute.execute(threads_x=16)
    values = output.to_numpy().view(np.float32).reshape(32, 4).copy()
    device.end_frame()

    samples, diagnostics = values[:16], values[16:]
    valid_non_null = (diagnostics[:, 1] > 0.5) & (diagnostics[:, 2] < 0.5)
    assert np.count_nonzero(valid_non_null) >= 8
    assert set(diagnostics[valid_non_null, 3].astype(np.int32)) == {0, 1}
    np.testing.assert_allclose(
        samples[valid_non_null, 3],
        diagnostics[valid_non_null, 0],
        rtol=2e-6,
        atol=1e-7,
    )
    assert np.all(np.isfinite(samples[valid_non_null]))
    assert np.all(samples[valid_non_null, 2] > 0.0)
    assert np.all(samples[valid_non_null, 3] > 0.0)
