from pathlib import Path

import numpy as np
import pytest

from ncls.core.material import (
    ABI_MAGIC,
    ABI_VERSION,
    BINARY_SIZE,
    HomogeneousMedium,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
    pack_layer_stack,
)


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.falcor
def test_layer_stack_ir_cpu_slang_layout() -> None:
    stack = LayerStackIR(
        (
            RoughDielectricInterface(0.08, 0.17, 1.5, 0.25),
            RoughConductorInterface(0.3, 0.5, (0.2, 0.9, 1.1), (3.9, 2.5, 2.1), -0.4),
        ),
        (HomogeneousMedium((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), 0.25, 0.7),),
    )
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    stack_buffer = device.create_structured_buffer(
        struct_size=BINARY_SIZE,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    output_buffer = device.create_structured_buffer(
        struct_size=16,
        element_count=8,
        bind_flags=falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess,
    )
    stack_buffer.from_numpy(np.frombuffer(pack_layer_stack(stack), dtype=np.uint8).copy())
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "layer_stack_ir_layout.cs.slang",
        cs_entry="validateLayerStackIrLayout",
    )
    compute.globals.gStacks = stack_buffer
    compute.globals.gOutput = output_buffer
    compute.execute(threads_x=1)

    words = output_buffer.to_numpy().view(np.uint32).reshape(8, 4)
    floats = words.view(np.float32)
    np.testing.assert_array_equal(words[0], [ABI_MAGIC, ABI_VERSION, 2, 1])
    assert words[1, 0] == 0
    np.testing.assert_allclose(floats[1, 2:4], [0.08, 0.17], rtol=1e-6)
    np.testing.assert_allclose(floats[2, :2], [1.5, 0.25], rtol=1e-6)
    assert words[3, 0] == 1
    np.testing.assert_allclose(floats[3, 1:4], [0.3, 0.5, 0.2], rtol=1e-6)
    np.testing.assert_allclose(floats[4], [0.9, 1.1, 3.9, 2.5], rtol=1e-6)
    np.testing.assert_allclose(floats[5, :2], [2.1, -0.4], rtol=1e-6)
    np.testing.assert_allclose(floats[6], [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
    np.testing.assert_allclose(floats[7], [0.5, 0.6, 0.25, 0.7], rtol=1e-6)
