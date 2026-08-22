from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import falcor
import numpy as np

from schema import BINARY_SIZE, LayerInterface, LayerStack, LayerType, pack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHADER_FILE = PROJECT_ROOT / "datagen" / "kernels" / "interface_validation.cs.slang"

SAMPLE_REFLECTION = 1
SAMPLE_TRANSMISSION = 2
SAMPLE_ALL = 3


@dataclass(frozen=True)
class InterfaceValidationResult:
    eval_f: np.ndarray
    eval_pdf: np.ndarray
    sample_mean: np.ndarray
    sample_variance: np.ndarray
    max_pdf_mismatch: np.ndarray
    valid_count: np.ndarray


def _buffer(device: falcor.Device, stride: int, count: int, *, writable: bool = False):
    flags = falcor.ResourceBindFlags.ShaderResource
    if writable:
        flags |= falcor.ResourceBindFlags.UnorderedAccess
    return device.create_structured_buffer(struct_size=stride, element_count=count, bind_flags=flags)


def validate_interfaces(
    layers: list[LayerInterface],
    incident_directions: np.ndarray,
    outgoing_directions: np.ndarray,
    *,
    eta_i: float = 1.0,
    eta_t: float = 1.5,
    sample_modes: int = SAMPLE_ALL,
    sample_count: int = 4096,
    seed: int = 1,
) -> InterfaceValidationResult:
    query_count = len(layers)
    incident = np.asarray(incident_directions, dtype=np.float32).reshape(query_count, 4)
    outgoing = np.asarray(outgoing_directions, dtype=np.float32).reshape(query_count, 4)
    if len(outgoing) != query_count:
        raise ValueError("layers and directions must have matching lengths")

    device = falcor.Device(type=falcor.DeviceType.D3D12)
    stack_buffer = _buffer(device, BINARY_SIZE, query_count)
    incident_buffer = _buffer(device, 16, query_count)
    outgoing_buffer = _buffer(device, 16, query_count)
    outputs = [_buffer(device, 16, query_count, writable=True) for _ in range(4)]

    stack_payloads = [np.frombuffer(pack_stack(LayerStack((layer,), ())), dtype=np.uint8) for layer in layers]
    stack_buffer.from_numpy(np.concatenate(stack_payloads))
    incident_buffer.from_numpy(incident)
    outgoing_buffer.from_numpy(outgoing)

    compute = falcor.ComputePass(device, file=SHADER_FILE, cs_entry="validateInterfaces")
    compute.globals.gStacks = stack_buffer
    compute.globals.gIncidentDirections = incident_buffer
    compute.globals.gOutgoingDirections = outgoing_buffer
    compute.globals.gEval = outputs[0]
    compute.globals.gSampleMean = outputs[1]
    compute.globals.gSampleMeanSquare = outputs[2]
    compute.globals.gSampleDiagnostics = outputs[3]
    compute.globals.gQueryCount = query_count
    compute.globals.gSampleCount = sample_count
    compute.globals.gSeed = seed
    compute.globals.gSampleModes = sample_modes
    compute.globals.gEtaI = eta_i
    compute.globals.gEtaT = eta_t
    compute.execute(threads_x=query_count)

    arrays = [output.to_numpy().view(np.float32).reshape(query_count, 4).copy() for output in outputs]
    variance = np.maximum(arrays[2][:, :3] - arrays[1][:, :3] ** 2, 0.0)
    return InterfaceValidationResult(
        eval_f=arrays[0][:, :3],
        eval_pdf=arrays[0][:, 3],
        sample_mean=arrays[1][:, :3],
        sample_variance=variance,
        max_pdf_mismatch=arrays[3][:, 0],
        valid_count=arrays[3][:, 1].astype(np.uint32),
    )


def _direction(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.array([np.sin(angle), 0.0, np.cos(angle), 0.0], dtype=np.float32)


def main() -> None:
    layers = [
        LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.5, 0.25, 0.125)),
        LayerInterface(LayerType.ROUGH_DIELECTRIC, 0.2, 0.2, eta=(1.5, 1.5, 1.5)),
        LayerInterface(LayerType.ROUGH_CONDUCTOR, 0.25, 0.1, eta=(0.2, 0.9, 1.1), k=(3.9, 2.5, 2.1)),
        LayerInterface(LayerType.SHEEN, 0.4, 0.4, albedo=(0.8, 0.2, 0.1)),
    ]
    incident = np.stack([_direction(30.0)] * len(layers))
    outgoing = np.stack([_direction(-20.0)] * len(layers))
    result = validate_interfaces(layers, incident, outgoing)
    print("eval_f=", result.eval_f)
    print("sample_mean=", result.sample_mean)
    print("max_pdf_mismatch=", result.max_pdf_mismatch)
    print("valid_count=", result.valid_count)


if __name__ == "__main__":
    main()
