from __future__ import annotations

from pathlib import Path

import falcor
import numpy as np

from schema import BINARY_SIZE, LayerInterface, LayerMedium, LayerStack, LayerType, pack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHADER_FILE = PROJECT_ROOT / "datagen" / "kernels" / "tile_kernel.slang"


def make_probe_stack() -> LayerStack:
    return LayerStack(
        layers=(
            LayerInterface(LayerType.ROUGH_DIELECTRIC, 0.1, 0.2),
            LayerInterface(
                LayerType.ROUGH_CONDUCTOR,
                0.3,
                0.4,
                eta=(0.2, 0.9, 1.1),
                k=(3.9, 2.5, 2.1),
            ),
        ),
        media=(LayerMedium(sigma_a=(0.1, 0.2, 0.3), thickness=0.25),),
    )


def main() -> None:
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    input_buffer = device.create_structured_buffer(
        struct_size=BINARY_SIZE,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    output_buffer = device.create_structured_buffer(
        struct_size=4,
        element_count=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess,
    )

    payload = np.frombuffer(pack_stack(make_probe_stack()), dtype=np.uint8).copy()
    input_buffer.from_numpy(payload)

    compute = falcor.ComputePass(device, file=SHADER_FILE, cs_entry="validateStackLayout")
    compute.globals.gStacks = input_buffer
    compute.globals.gValidation = output_buffer
    compute.execute(threads_x=1)

    result = output_buffer.to_numpy().view(np.uint32).reshape(-1)
    if result.tolist() != [1]:
        raise RuntimeError(f"GPU LayerStack layout validation failed: {result.tolist()}")
    print(f"LayerStack CPU/GPU layout OK: {BINARY_SIZE} bytes")


if __name__ == "__main__":
    main()
