from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.core.material import (
    BINARY_SIZE,
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughDielectricInterface,
    pack_layer_stack,
)
from ncls.learning.export import flatten_p1_weights
from ncls.learning.features import encode_layer_stack
from ncls.learning.models import LegacyLtcK2P1Compiler
from ncls.learning.prediction import predict_legacy_ltc_k2_response


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _batch(stack: LayerStackIR, view: np.ndarray) -> dict[str, torch.Tensor]:
    kinds, continuous, count = encode_layer_stack(stack)
    top = stack.interfaces[0]
    assert isinstance(top, RoughDielectricInterface)
    return {
        "interface_kinds": torch.from_numpy(kinds[None]),
        "continuous": torch.from_numpy(continuous[None]),
        "interface_counts": torch.tensor([count]),
        "view": torch.from_numpy(view[None]),
        "top_kind": torch.tensor([int(top.kind)]),
        "top_alpha": torch.tensor([[top.alpha_x, top.alpha_y]], dtype=torch.float32),
        "top_relative_ior": torch.tensor([top.relative_ior], dtype=torch.float32),
        "top_eta": torch.zeros((1, 3), dtype=torch.float32),
        "top_k": torch.zeros((1, 3), dtype=torch.float32),
        "top_color": torch.zeros((1, 3), dtype=torch.float32),
        "top_rotation": torch.tensor([top.tangent_rotation], dtype=torch.float32),
    }


@pytest.mark.falcor
def test_legacy_ltc_k2_p1_python_slang_parity() -> None:
    torch.manual_seed(872341)
    width = 8
    model = LegacyLtcK2P1Compiler(width=width).eval()
    stack = LayerStackIR(
        (
            RoughDielectricInterface(0.08, 0.17, 1.5, 0.25),
            RoughDielectricInterface(0.31, 0.22, 1.27, -0.37),
            DiffuseInterface((0.63, 0.21, 0.08)),
        ),
        (
            HomogeneousMedium((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), 0.25, 0.7),
            HomogeneousMedium((0.02, 0.03, 0.04), (0.11, 0.13, 0.17), -0.18, 0.32),
        ),
    )
    view = np.asarray([0.2, -0.1, math.sqrt(0.95)], dtype=np.float32)
    lights = np.asarray(
        [[0.6, 0.0, 0.8], [0.0, 0.0, 1.0], [-0.2, 0.5, math.sqrt(0.71)]],
        dtype=np.float32,
    )
    weights, layout = flatten_p1_weights(model.state_dict(), width=width)
    assert layout.total_floats == len(weights)

    device = falcor.Device(type=falcor.DeviceType.D3D12)
    material_buffer = device.create_structured_buffer(
        struct_size=BINARY_SIZE,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    weight_buffer = device.create_structured_buffer(
        struct_size=4,
        element_count=len(weights),
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
    material_buffer.from_numpy(np.frombuffer(pack_layer_stack(stack), dtype=np.uint8).copy())
    weight_buffer.from_numpy(weights)
    view_buffer.from_numpy(np.pad(view[None], ((0, 0), (0, 1))))
    light_buffer.from_numpy(np.pad(lights, ((0, 0), (0, 1))))

    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "legacy_ltc_k2_p1.cs.slang",
        cs_entry="evaluateLegacyLtcK2P1",
    )
    compute.globals.gMaterials = material_buffer
    compute.globals.gWeights = weight_buffer
    compute.globals.gViews = view_buffer
    compute.globals.gLights = light_buffer
    compute.globals.gOutput = output
    compute.globals.gWidth = width
    compute.globals.gLightCount = len(lights)
    compute.execute(threads_x=len(lights))
    slang = output.to_numpy().view(np.float32).reshape(len(lights), 4)[:, :3]
    with torch.no_grad():
        python = predict_legacy_ltc_k2_response(
            model,
            _batch(stack, view),
            torch.from_numpy(lights),
        )[0].numpy()
    np.testing.assert_allclose(slang, python, rtol=4e-5, atol=4e-6)
