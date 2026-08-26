from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import falcor  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[4]
AUDIT_KERNEL = (
    PROJECT_ROOT
    / "shaders/ncls/backends/unified_neural/unified_sampler_audit.cs.slang"
)


def _legendre_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.cos(
        np.pi * (np.arange(1, order + 1, dtype=np.float64) - 0.25) / (order + 0.5)
    )
    derivative = np.zeros_like(nodes)
    for _ in range(16):
        previous = np.ones_like(nodes)
        current = nodes.copy()
        for degree in range(2, order + 1):
            following = (
                (2.0 * degree - 1.0) * nodes * current - (degree - 1.0) * previous
            ) / degree
            previous, current = current, following
        derivative = order * (nodes * current - previous) / (nodes * nodes - 1.0)
        update = current / derivative
        nodes -= update
        if float(np.max(np.abs(update))) < 2e-15:
            break
    weights = 2.0 / ((1.0 - nodes * nodes) * derivative * derivative)
    indices = np.argsort(nodes)
    return nodes[indices], weights[indices]


def _quadrature() -> np.ndarray:
    nodes, _ = _legendre_nodes_weights(64)
    z = 0.5 * (nodes + 1.0)
    phi = (np.arange(256, dtype=np.float64) + 0.5) * (2.0 * np.pi / 256.0)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    radial = np.sqrt(np.maximum(1.0 - zz * zz, 0.0))
    return np.stack(
        (radial * np.cos(pp), radial * np.sin(pp), zz), axis=-1
    ).reshape(-1, 3).astype(np.float32)


def _buffers(device, prepared: np.ndarray, views: np.ndarray):
    srv = falcor.ResourceBindFlags.ShaderResource
    prepared_buffer = device.create_structured_buffer(
        struct_size=27 * 4, element_count=len(prepared), bind_flags=srv
    )
    view_buffer = device.create_structured_buffer(
        struct_size=16, element_count=len(views), bind_flags=srv
    )
    prepared_buffer.from_numpy(np.ascontiguousarray(prepared, dtype=np.float32))
    view_buffer.from_numpy(np.ascontiguousarray(
        np.column_stack((views, np.zeros(len(views), dtype=np.float32))),
        dtype=np.float32,
    ))
    return prepared_buffer, view_buffer


def _dispatch(
    device,
    compute,
    prepared_buffer,
    view_buffer,
    inputs: np.ndarray,
    case_index: int,
    sampler_index: int,
    method_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.column_stack((inputs, np.zeros(len(inputs), dtype=np.float32))).astype(
        np.float32, copy=False
    )
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = srv | falcor.ResourceBindFlags.UnorderedAccess
    input_buffer = device.create_structured_buffer(
        struct_size=16, element_count=len(values), bind_flags=srv
    )
    output0 = device.create_structured_buffer(
        struct_size=16, element_count=len(values), bind_flags=uav
    )
    output1 = device.create_structured_buffer(
        struct_size=16, element_count=len(values), bind_flags=uav
    )
    input_buffer.from_numpy(np.ascontiguousarray(values))
    compute.globals.gPrepared = prepared_buffer
    compute.globals.gViews = view_buffer
    compute.globals.gInput = input_buffer
    compute.globals.gOutput0 = output0
    compute.globals.gOutput1 = output1
    compute.globals.gCount = len(values)
    compute.globals.gCaseIndex = case_index
    compute.globals.gSampler = sampler_index
    compute.globals.gMethod = method_index
    compute.execute(threads_x=len(values))
    first = output0.to_numpy().view(np.float32).reshape(len(values), 4).copy()
    second = output1.to_numpy().view(np.float32).reshape(len(values), 4).copy()
    return first, second


def run(
    cases_path: Path,
    output_dir: Path,
    method: str,
    sampler: str,
    seed: int,
    *,
    case_count: int = 120,
    sample_count: int = 64 * 16_384,
) -> None:
    cases = np.load(cases_path, allow_pickle=False)
    prepared = np.asarray(cases["prepared"], dtype=np.float32)
    views = np.asarray(cases["views"], dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    prepared_buffer, view_buffer = _buffers(device, prepared, views)
    query_pass = falcor.ComputePass(
        device, file=AUDIT_KERNEL, cs_entry="queryUnifiedLearnedSampler"
    )
    sample_pass = falcor.ComputePass(
        device, file=AUDIT_KERNEL, cs_entry="sampleUnifiedLearnedSampler"
    )
    directions = _quadrature()
    sampler_index = 0 if sampler == "nvidia-diffuse-ggx9" else 1
    method_index = 0 if method == "unified" else 1
    if case_count < 1 or case_count > len(prepared) or sample_count < 1:
        raise ValueError("Falcor sampler worker limits are invalid")
    for case_index in range(case_count):
        queried, _ = _dispatch(
            device,
            query_pass,
            prepared_buffer,
            view_buffer,
            directions,
            case_index,
            sampler_index,
            method_index,
        )
        random = np.random.default_rng(
            seed + case_index * 7919 + sampler_index * 104729
        ).random((sample_count, 3), dtype=np.float32)
        sampled, metadata = _dispatch(
            device,
            sample_pass,
            prepared_buffer,
            view_buffer,
            random,
            case_index,
            sampler_index,
            method_index,
        )
        np.savez(
            output_dir / f"case-{case_index:03d}.npz",
            queried_pdf=queried[:, 0],
            sampled=sampled,
            metadata=metadata,
        )
        print(f"falcor-case={case_index + 1}/{len(prepared)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("unified", "nvidia"), required=True)
    parser.add_argument(
        "--sampler", choices=("nvidia-diffuse-ggx9", "ltc-k2"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case-count", type=int, default=120)
    parser.add_argument("--sample-count", type=int, default=64 * 16_384)
    args = parser.parse_args()
    run(
        args.cases,
        args.output,
        args.method,
        args.sampler,
        args.seed,
        case_count=args.case_count,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
