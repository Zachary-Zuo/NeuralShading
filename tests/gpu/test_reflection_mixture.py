from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_COUNT = 512 * 1024


def _run(device, mode: int) -> tuple[np.ndarray, np.ndarray]:
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests/gpu/kernels/reflection_mixture.cs.slang",
        cs_entry="main",
    )
    flags = falcor.ResourceBindFlags.UnorderedAccess | falcor.ResourceBindFlags.ShaderResource
    direction = device.create_structured_buffer(
        struct_size=16, element_count=SAMPLE_COUNT, bind_flags=flags
    )
    check = device.create_structured_buffer(
        struct_size=16, element_count=SAMPLE_COUNT, bind_flags=flags
    )
    compute.globals.gDirectionPdf = direction
    compute.globals.gCheck = check
    compute.globals.gCount = SAMPLE_COUNT
    compute.globals.gMode = mode
    compute.execute(threads_x=SAMPLE_COUNT)
    return (
        direction.to_numpy().view(np.float32).reshape(SAMPLE_COUNT, 4).copy(),
        check.to_numpy().view(np.float32).reshape(SAMPLE_COUNT, 4).copy(),
    )


@pytest.mark.falcor
def test_reflection_mixture_sample_pdf_and_missing_mass_match_on_gpu() -> None:
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    sampled, checked = _run(device, 0)
    valid = (checked[:, 1] == 1.0) & (checked[:, 2] == 0.0)
    null = (checked[:, 1] == 1.0) & (checked[:, 2] == 1.0)
    assert np.all(valid | null)
    np.testing.assert_allclose(sampled[valid, 3], checked[valid, 0], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        np.linalg.norm(sampled[valid, :3], axis=1), 1.0, rtol=2e-6, atol=2e-7
    )
    assert np.all(sampled[valid, 2] > 0.0)
    assert np.all(sampled[valid, 3] > 0.0)

    integrated, _ = _run(device, 1)
    continuous_mass = float(2.0 * np.pi * np.mean(integrated[:, 3], dtype=np.float64))
    observed_valid = float(np.mean(valid, dtype=np.float64))
    standard_error = np.sqrt(observed_valid * (1.0 - observed_valid) / SAMPLE_COUNT)
    quadrature_error = 2.5e-4
    assert abs(continuous_mass - observed_valid) <= 6.0 * standard_error + quadrature_error
    device.end_frame()
