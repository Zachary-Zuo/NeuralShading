from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.core.material import RoughDielectricInterface
from ncls.core.representations.legacy_ltc_k2 import (
    BINARY_SIZE,
    LegacyLtcK2Lobe,
    LegacyLtcK2State,
    pack_state,
)
from ncls.core.representations.legacy_ltc_k2.torch_eval import (
    evaluate_state_response_cos,
    states_to_tensors,
)


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.falcor
@pytest.mark.parametrize("entry", ["evaluateLegacyLtcK2", "evaluateLegacyLtcK2ThroughContract"])
def test_legacy_ltc_k2_python_slang_parity(entry: str) -> None:
    state = LegacyLtcK2State(
        RoughDielectricInterface(0.12, 0.08, 1.5, 0.25),
        (
            LegacyLtcK2Lobe((0.2, 0.4, 0.6), (1.2, 0.8), (0.1, -0.2, 0.3), 0.4),
            LegacyLtcK2Lobe((0.1, 0.05, 0.2), (0.7, 1.4), (-0.1, 0.2, -0.3), -0.6),
        ),
    )
    views = np.asarray([[0.2, 0.1, math.sqrt(0.95), 0.0]], dtype=np.float32)
    lights = np.asarray([[0.6, 0.0, 0.8, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    state_buffer = device.create_structured_buffer(
        struct_size=BINARY_SIZE,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    view_buffer = device.create_structured_buffer(
        struct_size=16,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    light_buffer = device.create_structured_buffer(
        struct_size=16,
        element_count=len(lights),
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    output = device.create_structured_buffer(
        struct_size=16,
        element_count=len(lights),
        bind_flags=falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess,
    )
    state_buffer.from_numpy(np.frombuffer(pack_state(state), dtype=np.uint8).copy())
    view_buffer.from_numpy(views)
    light_buffer.from_numpy(lights)
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "legacy_ltc_k2.cs.slang",
        cs_entry=entry,
    )
    compute.globals.gStates = state_buffer
    compute.globals.gViews = view_buffer
    compute.globals.gLights = light_buffer
    compute.globals.gOutput = output
    compute.globals.gLightCount = len(lights)
    compute.execute(threads_x=len(lights))
    slang = output.to_numpy().view(np.float32).reshape(len(lights), 4)[:, :3]
    python = evaluate_state_response_cos(
        states_to_tensors([state]),
        torch.from_numpy(views[:, :3]),
        torch.from_numpy(lights[:, :3]),
    )[0].detach().numpy()
    np.testing.assert_allclose(slang, python, rtol=2e-5, atol=2e-6)
