from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from ncls.data.providers.mdl import MdlGpuQueryRuntime, MdlProvider, MdlProviderConfig
from ncls.references.mdl import MDL_SDK_DIRECTORY
from ncls.paths import PROJECT_ROOT
from ncls.data.collector import CollectionConfig
from ncls.data.batch_sources import MdlLiveReferenceBatchSource
from ncls.data.training_batch import TrainingRouteRequest


@pytest.mark.falcor
def test_mdl_shared_runtime_returns_current_cuda_tensor_without_host_readback() -> None:
    provider = MdlProvider(
        CollectionConfig(
            name="mdl-live-runtime",
            view_count=1,
            light_count=1,
            spatial_sample_count=1,
            proposal="uniform",
        ),
        MdlProviderConfig(),
    )
    state = provider.source_states()[0]
    runtime = MdlGpuQueryRuntime(
        state.runtime_state.artifact,
        sdk_root=PROJECT_ROOT / "external" / MDL_SDK_DIRECTORY,
        query_capacity=8,
        slot_count=2,
    )
    try:
        views = torch.tensor(
            [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]],
            dtype=torch.float32,
            device="cuda:0",
        )
        lights = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]],
            dtype=torch.float32,
            device="cuda:0",
        )
        uv = torch.zeros((2, 2), dtype=torch.float32, device="cuda:0")
        gradients = torch.zeros((2, 4), dtype=torch.float32, device="cuda:0")
        value, pdf = runtime.evaluate_torch(0, views, lights, uv, gradients)
        tint = torch.tensor([0.8, 0.2, 0.1], dtype=torch.float32, device="cuda:0")
        expected = tint[None, :] * lights[:, 2:3] / np.pi
        torch.testing.assert_close(value, expected, rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(pdf, lights[:, 2] / np.pi, rtol=3e-6, atol=3e-7)
        assert value.device == views.device and pdf.device == views.device
        source = inspect.getsource(MdlGpuQueryRuntime.evaluate_torch)
        assert ".cpu(" not in source and ".numpy(" not in source and "to_numpy(" not in source
        runtime._device.end_frame()
    finally:
        runtime.close()
        provider.close()


@pytest.mark.falcor
def test_mdl_live_batch_uses_shared_current_falcor_target() -> None:
    provider = MdlProvider(
        CollectionConfig(
            name="mdl-live-batch",
            view_count=1,
            light_count=1,
            spatial_sample_count=1,
            proposal="uniform",
        )
    )
    source = MdlLiveReferenceBatchSource(
        provider,
        provider.source_states()[0],
        max_batch_size=8,
        query_tile_size=8,
        seed=17,
        device="cuda:0",
    )
    request = TrainingRouteRequest(
        "train",
        8,
        1,
        0,
        0,
        23,
        {
            "direction_proposal": "uniform-hemisphere-conditioning@1",
            "mip_exponential_scale": 1.0,
            "target_estimator": "reference",
        },
    )
    try:
        batch = source.next_batch(request)
        expected = (
            torch.tensor([0.8, 0.2, 0.1], dtype=torch.float32, device="cuda:0")[None, None, :]
            * batch.tensors["wi"][:, :, 2:3]
            / np.pi
        )
        torch.testing.assert_close(batch.tensors["target"], expected, rtol=3e-6, atol=3e-7)
        assert batch.device.type == "cuda"
        assert batch.provenance["host_readback"] is False
        assert batch.provenance["producer"] == "mdl-live-reference"
        assert batch.provenance["texture_filtering"] == "explicit-lod0"
        assert batch.provenance["uv_derivatives_consumed"] is False
        assert torch.count_nonzero(batch.tensors["mip_level"]) == 0
        assert torch.count_nonzero(batch.tensors["uv_dx"]) == 0
        assert torch.count_nonzero(batch.tensors["uv_dy"]) == 0
        assert batch.tensors["native_features"].shape == (8, 3)
        batch.release()
        assert not source._active_leases
    finally:
        if source._active_leases:
            next(iter(source._active_leases.values())).release()
        source.close()
