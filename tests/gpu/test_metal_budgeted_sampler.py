from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.learning.methods.metal.sampler import (
    metal_budgeted_proposal_pdf,
    metal_budgeted_sample_proposal,
)
from ncls.references.backend import create_reference_backend

falcor = pytest.importorskip("falcor")


@pytest.mark.falcor
def test_budgeted_sampler_matches_oracle_and_normalizes() -> None:
    rng = np.random.default_rng(9281)
    count = 4096
    states = np.zeros((5, 10, 4), dtype=np.float32)
    for c in range(5):
        states[c, :3] = [[0.55, 0.35, 0.65, c % 2], [0.35, 0.7, 0.4, 0], [0.1, 1, 1, 2]]
        for frame in range(2):
            normal = np.array([0.3 * (frame + 1), -0.2 * c, 1.0])
            normal /= np.linalg.norm(normal)
            tangent = np.cross([0.0, 1.0, 0.0], normal)
            tangent /= np.linalg.norm(tangent)
            states[c, 3+3*frame:6+3*frame, :3] = [tangent, np.cross(normal, tangent), normal]
        wo = np.array([0.8, -0.2, 0.05 if c == 2 else 0.6])
        states[c, 9, :3] = wo / np.linalg.norm(wo)
        states[c, 9, 3] = 1
    states[3, 9, 3] = 0  # invalid prepared state
    states[4, 2, 0] = 0  # missing hemisphere support
    queries = np.zeros((count, 2, 4), dtype=np.float32)
    queries[:, 0, :2] = rng.random((count, 2))
    wi = rng.normal(size=(count, 3))
    wi[:, 2] = np.abs(wi[:, 2])
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    queries[:, 1, :3] = wi
    indices = np.arange(count) % 5
    queries[:, 1, 3] = indices
    selected = torch.from_numpy(states[indices]).double()
    proposals = selected[:, :3]
    frames = selected[:, 3:9, :3].reshape(count, 2, 3, 3)
    valid = selected[:, 9, 3].bool()
    wo = selected[:, 9, :3]
    expected = metal_budgeted_sample_proposal(
        proposals, frames, valid, wo, torch.from_numpy(queries[:, 0, :2].copy()).double()
    )
    expected_query = metal_budgeted_proposal_pdf(
        proposals, frames, valid, wo, torch.from_numpy(queries[:, 1:2, :3].copy()).double()
    )
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device, file=Path(__file__).with_name("kernels") / "metal_budgeted_sampler.cs.slang", cs_entry="main"
    )
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = srv | falcor.ResourceBindFlags.UnorderedAccess

    def upload(values):
        result = device.create_structured_buffer(struct_size=16, element_count=values.size // 4, bind_flags=srv)
        result.from_numpy(values.copy())
        return result

    compute.globals.gStates = upload(states)
    compute.globals.gQueries = upload(queries)
    output = device.create_structured_buffer(struct_size=16, element_count=3*count, bind_flags=uav)
    compute.globals.gOutput = output
    compute.globals.TestCB.gCount = count
    compute.globals.TestCB.gIntegrate = 0
    compute.execute(threads_x=count)
    actual = output.to_numpy().view(np.float32).reshape(count, 3, 4).copy()
    mask = expected.valid[:, 0].numpy()
    np.testing.assert_array_equal(actual[:, 0, 3], mask.astype(np.float32))
    np.testing.assert_allclose(actual[mask, 0, :3], expected.wi[:, 0].numpy()[mask], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(actual[:, 1, 0], expected.forward_pdf[:, 0].numpy(), rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(actual[:, 1, 1], expected.reverse_pdf[:, 0].numpy(), rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(actual[:, 1, :2], actual[:, 1, 2:], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual[:, 2, 0], expected_query.forward[:, 0].numpy(), rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(actual[:, 2, 1], expected_query.reverse[:, 0].numpy(), rtol=2e-3, atol=2e-5)
    # Independent uniform-hemisphere quadrature checks mass, not just matching code.
    count = 32768
    for c in range(3):
        q = np.zeros((count, 2, 4), dtype=np.float32)
        z = rng.random(count)
        phi = 2*np.pi*rng.random(count)
        q[:, 0, :3] = np.stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z], axis=1)
        q[:, 1, 3] = c
        compute.globals.gQueries = upload(q)
        output = device.create_structured_buffer(struct_size=16, element_count=3*count, bind_flags=uav)
        compute.globals.gOutput = output
        compute.globals.TestCB.gCount = count
        compute.globals.TestCB.gIntegrate = 1
        compute.execute(threads_x=count)
        integrand = output.to_numpy().view(np.float32).reshape(count, 3, 4)[:, 0, 0].copy() * (2*np.pi)
        assert abs(integrand.mean() - 1) <= 5*integrand.std(ddof=1)/np.sqrt(count) + 0.002
    device.end_frame()
