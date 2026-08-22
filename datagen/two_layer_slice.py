from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import falcor
import numpy as np

from schema import BINARY_SIZE, LayerInterface, LayerMedium, LayerStack, LayerType, pack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHADER_FILE = PROJECT_ROOT / "datagen" / "kernels" / "two_layer_reference.cs.slang"
TEACHER_SHADER_FILE = PROJECT_ROOT / "datagen" / "kernels" / "two_layer_teacher.cs.slang"
LAYER_STACK_SHADER_FILE = PROJECT_ROOT / "datagen" / "kernels" / "layer_stack_teacher.cs.slang"
SHADER_FILES = {
    "evaluateTwoLayerReference": REFERENCE_SHADER_FILE,
    "evaluateDirectCoat": REFERENCE_SHADER_FILE,
    "sampleDirectCoatTransmission": REFERENCE_SHADER_FILE,
    "sampleDirectCoatReflection": REFERENCE_SHADER_FILE,
    "evaluateTwoLayerTeacher": TEACHER_SHADER_FILE,
    "evaluateLayerStackTeacher": LAYER_STACK_SHADER_FILE,
}


@dataclass(frozen=True)
class SliceResult:
    angles_degrees: np.ndarray
    mean: np.ndarray
    variance: np.ndarray


def direction(theta_degrees: np.ndarray | float) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    result = np.zeros((np.size(theta), 4), dtype=np.float32)
    result[:, 0] = np.sin(theta).reshape(-1)
    result[:, 2] = np.cos(theta).reshape(-1)
    return result


def diffuse_stack() -> LayerStack:
    return LayerStack(
        layers=(
            LayerInterface(
                LayerType.ROUGH_DIELECTRIC,
                roughness_x=0.12,
                roughness_y=0.12,
                eta=(1.5, 1.5, 1.5),
            ),
            LayerInterface(
                LayerType.DIFFUSE,
                roughness_x=1.0,
                roughness_y=1.0,
                albedo=(0.55, 0.18, 0.06),
            ),
        ),
        media=(LayerMedium(),),
    )


def gray_diffuse_stack(albedo: float = 0.5) -> LayerStack:
    stack = diffuse_stack()
    return LayerStack(
        layers=(
            stack.layers[0],
            LayerInterface(
                LayerType.DIFFUSE,
                roughness_x=1.0,
                roughness_y=1.0,
                albedo=(albedo, albedo, albedo),
            ),
        ),
        media=stack.media,
    )


def conductor_stack() -> LayerStack:
    return LayerStack(
        layers=(
            LayerInterface(
                LayerType.ROUGH_DIELECTRIC,
                roughness_x=0.18,
                roughness_y=0.08,
                eta=(1.5, 1.5, 1.5),
            ),
            LayerInterface(
                LayerType.ROUGH_CONDUCTOR,
                roughness_x=0.28,
                roughness_y=0.12,
                eta=(0.2, 0.9, 1.1),
                k=(3.9, 2.5, 2.1),
            ),
        ),
        media=(LayerMedium(),),
    )


def _structured_buffer(device: falcor.Device, stride: int, count: int, *, writable: bool = False):
    flags = falcor.ResourceBindFlags.ShaderResource
    if writable:
        flags |= falcor.ResourceBindFlags.UnorderedAccess
    return device.create_structured_buffer(struct_size=stride, element_count=count, bind_flags=flags)


def evaluate_slice(
    stack: LayerStack,
    *,
    shader_entry: str = "evaluateTwoLayerReference",
    view_angle_degrees: float = 30.0,
    sample_count: int = 64,
    seed: int = 1,
    max_depth: int = 32,
    angles_degrees: np.ndarray | None = None,
) -> SliceResult:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if shader_entry not in SHADER_FILES:
        raise ValueError(f"unsupported shader entry: {shader_entry}")
    if shader_entry in {"evaluateTwoLayerTeacher", "evaluateLayerStackTeacher"}:
        for medium in stack.media:
            sigma_t = np.asarray(medium.sigma_a) + np.asarray(medium.sigma_s)
            if np.any(np.asarray(medium.sigma_s) > 0.0) and not np.allclose(
                sigma_t, sigma_t[0], rtol=1e-4, atol=1e-5
            ):
                raise ValueError("v0 scattering media require achromatic sigma_a + sigma_s")
    angles = np.linspace(-80.0, 80.0, 65, dtype=np.float32) if angles_degrees is None else np.asarray(angles_degrees, dtype=np.float32)
    query_count = len(angles)

    device = falcor.Device(type=falcor.DeviceType.D3D12)
    stack_buffer = _structured_buffer(device, BINARY_SIZE, query_count)
    view_buffer = _structured_buffer(device, 16, query_count)
    light_buffer = _structured_buffer(device, 16, query_count)
    mean_buffer = _structured_buffer(device, 16, query_count, writable=True)
    mean_square_buffer = _structured_buffer(device, 16, query_count, writable=True)

    packed_stack = np.frombuffer(pack_stack(stack), dtype=np.uint8)
    stack_buffer.from_numpy(np.tile(packed_stack, query_count))
    view_buffer.from_numpy(np.repeat(direction(view_angle_degrees), query_count, axis=0))
    light_buffer.from_numpy(direction(angles))

    compute = falcor.ComputePass(
        device,
        defines={"FALCOR_MATERIAL_INSTANCE_SIZE": "256"},
        file=SHADER_FILES[shader_entry],
        cs_entry=shader_entry,
    )
    compute.globals.gStacks = stack_buffer
    compute.globals.gViewDirections = view_buffer
    compute.globals.gLightDirections = light_buffer
    compute.globals.gMean = mean_buffer
    compute.globals.gMeanSquare = mean_square_buffer
    compute.globals.gQueryCount = query_count
    compute.globals.gSampleCount = sample_count
    compute.globals.gSeed = seed
    compute.globals.gMaxDepth = max_depth
    compute.execute(threads_x=query_count)

    mean = mean_buffer.to_numpy().view(np.float32).reshape(query_count, 4)[:, :3].copy()
    mean_square = mean_square_buffer.to_numpy().view(np.float32).reshape(query_count, 4)[:, :3].copy()
    variance = np.maximum(mean_square - mean * mean, 0.0)
    if not np.all(np.isfinite(mean)) or np.any(mean < 0.0):
        raise RuntimeError("teacher produced invalid directional response")
    return SliceResult(angles, mean, variance)


def main() -> None:
    report_dir = PROJECT_ROOT / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, stack in (("diffuse", diffuse_stack()), ("conductor", conductor_stack())):
        result = evaluate_slice(stack)
        output = report_dir / f"two_layer_{name}_slice.npz"
        np.savez(output, angles_degrees=result.angles_degrees, mean=result.mean, variance=result.variance)
        peak = result.mean.max(axis=0)
        print(f"{name}: {len(result.angles_degrees)} directions, RGB peak={peak}, saved={output}")


if __name__ == "__main__":
    main()
