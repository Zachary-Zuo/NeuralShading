from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.learning.models.metal_sampler import (
    METAL_PROPOSAL_COMPONENT_COUNT,
    METAL_PROPOSAL_DISTRIBUTION_IDS,
    METAL_PROPOSAL_FRAME_INDICES,
    metal_proposal_pdf,
    metal_sample_proposal,
)
from ncls.references.backend import create_reference_backend


falcor = pytest.importorskip("falcor")
KERNEL_ROOT = Path(__file__).resolve().with_name("kernels")
SAMPLE_COUNT = 256


def _state(batch: int) -> torch.Tensor:
    probe = torch.arange(batch, dtype=torch.float32)[:, None]
    component = torch.arange(
        METAL_PROPOSAL_COMPONENT_COUNT, dtype=torch.float32
    )[None, :]
    state = torch.zeros((batch, METAL_PROPOSAL_COMPONENT_COUNT, 8))
    state[..., 0] = 0.02 + 0.002 * component + 0.0005 * torch.remainder(
        probe + 3.0 * component, 17.0
    )
    state[..., 1] = 0.05 + 0.003 * component + 0.0004 * torch.remainder(
        2.0 * probe + component, 29.0
    )
    state[..., 2] = 0.08 + 0.004 * component + 0.0003 * torch.remainder(
        probe + 2.0 * component, 31.0
    )
    state[..., 3] = -0.5 + 0.1 * component + 0.001 * torch.remainder(
        probe + 5.0 * component, 23.0
    )
    state[..., 4] = 1.0
    state[..., 5] = torch.tensor(METAL_PROPOSAL_FRAME_INDICES)
    state[..., 6] = torch.tensor(METAL_PROPOSAL_DISTRIBUTION_IDS)
    state[..., 7] = 0.5 + 0.01 * torch.remainder(probe + component, 19.0)
    return state


def _frames(batch: int) -> torch.Tensor:
    angles = torch.tensor((0.0, 0.35, -0.45, 0.7))[None, :] + 0.0007 * torch.remainder(
        torch.arange(batch, dtype=torch.float32)[:, None], 41.0
    )
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    tangent = torch.stack((cosine, torch.zeros_like(cosine), -sine), dim=2)
    bitangent = torch.zeros((batch, 4, 3), dtype=torch.float32)
    bitangent[..., 1] = 1.0
    normal = torch.stack((sine, torch.zeros_like(sine), cosine), dim=2)
    return torch.stack((tangent, bitangent, normal), dim=2)


@pytest.mark.falcor
def test_generated_slang_metal_proposal_matches_python_sample_and_pdf() -> None:
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=KERNEL_ROOT / "metal_fused_proposal.cs.slang",
        cs_entry="main",
    )
    flags = (
        falcor.ResourceBindFlags.ShaderResource
        | falcor.ResourceBindFlags.UnorderedAccess
    )
    sample_buffer = device.create_structured_buffer(
        struct_size=16, element_count=SAMPLE_COUNT, bind_flags=flags
    )
    check_buffer = device.create_structured_buffer(
        struct_size=16, element_count=SAMPLE_COUNT, bind_flags=flags
    )
    query_buffer = device.create_structured_buffer(
        struct_size=16, element_count=SAMPLE_COUNT, bind_flags=flags
    )
    compute.globals.gSample = sample_buffer
    compute.globals.gCheck = check_buffer
    compute.globals.gQuery = query_buffer
    compute.globals.gCount = SAMPLE_COUNT
    compute.execute(threads_x=SAMPLE_COUNT)
    slang_sample = (
        sample_buffer.to_numpy().view(np.float32).reshape(SAMPLE_COUNT, 4).copy()
    )
    slang_check = (
        check_buffer.to_numpy().view(np.float32).reshape(SAMPLE_COUNT, 4).copy()
    )
    slang_query = (
        query_buffer.to_numpy().view(np.float32).reshape(SAMPLE_COUNT, 4).copy()
    )
    device.end_frame()

    index = torch.arange(SAMPLE_COUNT, dtype=torch.float32)
    sample_u = torch.stack(
        (
            (index + 0.5) / SAMPLE_COUNT,
            torch.frac((index + 0.5) * 0.61803398875),
        ),
        dim=1,
    )
    probe = torch.arange(SAMPLE_COUNT, dtype=torch.float32)
    wo = torch.nn.functional.normalize(
        torch.stack(
            (
                0.15 + 0.002 * torch.remainder(probe, 53.0),
                -0.23 + 0.0015 * torch.remainder(probe, 71.0),
                torch.ones_like(probe),
            ),
            dim=1,
        ),
        dim=1,
    )
    python = metal_sample_proposal(
        _state(SAMPLE_COUNT),
        _frames(SAMPLE_COUNT),
        torch.ones(SAMPLE_COUNT, dtype=torch.bool),
        wo,
        sample_u,
    )
    assert bool(python.valid.all())
    assert np.all(slang_check[:, 2] == 1.0)
    np.testing.assert_allclose(
        slang_sample[:, 3], slang_check[:, 1], rtol=3e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        slang_sample[:, :3],
        python.wi[:, 0, :].numpy(),
        # Roughly one hundred FP32 operations plus backend-specific sin/cos/log
        # and reciprocal-square-root paths bound the accumulated probe error.
        rtol=8e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        slang_sample[:, 3], python.forward_pdf[:, 0].numpy(), rtol=3e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        slang_check[:, 0], python.reverse_pdf[:, 0].numpy(), rtol=3e-5, atol=2e-6
    )
    np.testing.assert_array_equal(
        slang_check[:, 3].astype(np.int64), python.component.numpy()
    )
    assert set(python.component.tolist()) == set(range(METAL_PROPOSAL_COMPONENT_COUNT))

    query_z = 0.01 + 0.98 * ((probe + 0.5) / SAMPLE_COUNT)
    query_phi = 2.0 * np.pi * torch.frac((probe + 0.25) * 0.41421356237)
    query_radius = torch.sqrt(torch.clamp(1.0 - query_z.square(), min=0.0))
    query_direction = torch.stack(
        (
            query_radius * torch.cos(query_phi),
            query_radius * torch.sin(query_phi),
            query_z,
        ),
        dim=1,
    )
    query_density = metal_proposal_pdf(
        _state(SAMPLE_COUNT),
        _frames(SAMPLE_COUNT),
        torch.ones(SAMPLE_COUNT, dtype=torch.bool),
        wo,
        query_direction[:, None, :],
    )
    np.testing.assert_allclose(
        slang_query[:, :3], query_direction.numpy(), rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        slang_query[:, 3], query_density.forward[:, 0].numpy(), rtol=4e-5, atol=3e-6
    )
