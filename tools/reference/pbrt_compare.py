from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from ncls.core.material import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
)
from ncls.references.programs import get_reference_program_for_source
from ncls.references.query import ReferenceQueryDispatcher, ScatteringQuery
from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack


LINE_PATTERN = re.compile(
    r"theta=\s*([-+0-9.]+)\s+phi=\s*([-+0-9.]+)\s+"
    r"response=([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)"
)


@dataclass(frozen=True)
class ProbeCase:
    name: str
    material: str
    view_theta: float
    view_phi: float
    optical_thickness: float
    medium_albedo: float
    g: float
    coat_alpha_x: float
    coat_alpha_y: float
    coat_ior: float
    base_alpha_x: float = 0.2
    base_alpha_y: float = 0.2
    tangent_rotation: float = 0.0
    eta: tuple[float, float, float] = (0.2, 0.9, 1.1)
    k: tuple[float, float, float] = (3.9, 2.5, 2.1)


CASES = {
    "diffuse-clear": ProbeCase(
        "diffuse-clear", "diffuse", 20.0, 0.0, 1e-6, 0.0, 0.0, 0.12, 0.12, 1.5
    ),
    "conductor-clear": ProbeCase(
        "conductor-clear", "conductor", 20.0, 15.0, 1e-6, 0.0, 0.0, 0.10, 0.10, 1.5,
        0.16, 0.42, 0.55, (0.18, 0.78, 1.20), (3.60, 2.65, 2.05),
    ),
    "conductor-absorbing": ProbeCase(
        "conductor-absorbing", "conductor", 35.0, 40.0, 0.35, 0.0, 0.0, 0.18, 0.18, 1.35,
        0.28, 0.09, -0.35, (0.35, 0.92, 1.35), (4.20, 2.85, 1.75),
    ),
    "conductor-scattering": ProbeCase(
        "conductor-scattering", "conductor", 50.0, 75.0, 0.55, 0.55, 0.3, 0.24, 0.24, 1.6,
        0.12, 0.36, 0.70, (0.22, 0.68, 1.10), (3.80, 2.40, 1.90),
    ),
}


def _direction(theta_degrees: float, phi_degrees: float) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    phi = np.deg2rad(phi_degrees)
    sin_theta = np.sin(theta)
    return np.asarray(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta), 0.0],
        dtype=np.float32,
    )


def _material(case: ProbeCase) -> LayerStackIR:
    sigma_a = (1.0 - case.medium_albedo,) * 3
    sigma_s = (case.medium_albedo,) * 3
    base = (
        DiffuseInterface((0.5, 0.5, 0.5))
        if case.material == "diffuse"
        else RoughConductorInterface(
            case.base_alpha_x,
            case.base_alpha_y,
            case.eta,
            case.k,
            case.tangent_rotation,
        )
    )
    return LayerStackIR(
        (RoughDielectricInterface(case.coat_alpha_x, case.coat_alpha_y, case.coat_ior), base),
        (HomogeneousMedium(sigma_a, sigma_s, case.g, case.optical_thickness),),
    )


def _pbrt_command(
    executable: Path,
    case: ProbeCase,
    *,
    samples: int,
    max_depth: int,
    seed: int,
) -> list[str]:
    values = [
        case.material,
        samples,
        case.view_theta,
        case.view_phi,
        max_depth,
        case.optical_thickness,
        seed,
        case.medium_albedo,
        case.g,
        case.coat_alpha_x,
        case.coat_alpha_y,
        case.coat_ior,
        case.base_alpha_x,
        case.base_alpha_y,
        case.tangent_rotation,
        *case.eta,
        *case.k,
    ]
    return [str(executable.resolve()), *(str(value) for value in values)]


def _run_case(
    executable: Path,
    case: ProbeCase,
    *,
    samples: int,
    batches: int,
    max_depth: int,
) -> tuple[float, float]:
    pbrt_batches: list[np.ndarray] = []
    directions: np.ndarray | None = None
    for batch_index in range(batches):
        completed = subprocess.run(
            _pbrt_command(
                executable,
                case,
                samples=samples,
                max_depth=max_depth,
                seed=1 + 1009 * batch_index,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        matches = [LINE_PATTERN.search(line) for line in completed.stdout.splitlines()]
        rows = [
            (
                float(item.group(1)),
                float(item.group(2)),
                (float(item.group(3)), float(item.group(4)), float(item.group(5))),
            )
            for item in matches
            if item
        ]
        if not rows:
            raise RuntimeError("pbrt probe produced no parseable direction rows")
        batch_directions = np.asarray([row[:2] for row in rows], dtype=np.float32)
        if directions is None:
            directions = batch_directions
        elif not np.array_equal(directions, batch_directions):
            raise RuntimeError("pbrt probe direction rows changed between batches")
        pbrt_batches.append(np.asarray([row[2] for row in rows], dtype=np.float32))
    assert directions is not None
    pbrt_response = np.mean(pbrt_batches, axis=0)

    light_directions = np.stack([_direction(float(theta), float(phi)) for theta, phi in directions])
    if max_depth != 64:
        raise ValueError("canonical LayerStack reference fixes maximum walk depth at 64")
    snapshot = snapshot_from_layer_stack(_material(case))
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    dispatcher = ReferenceQueryDispatcher(
        definition,
        (snapshot,),
        query_capacity=len(light_directions),
        device="cuda:0",
    )
    device = torch.device("cuda:0")
    query = ScatteringQuery(
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.as_tensor(
            _direction(case.view_theta, case.view_phi)[:3][None, :],
            device=device,
        ),
    )
    wi = torch.as_tensor(light_directions[:, :3], device=device)[None, :, :]
    chunk_means: list[np.ndarray] = []
    chunk_weights: list[int] = []
    try:
        for batch_index in range(batches):
            remaining = samples
            chunk_index = 0
            while remaining:
                sample_count = min(remaining, 256)
                seed = 53 + 1009 * batch_index + 65537 * chunk_index
                seeds = torch.arange(
                    seed,
                    seed + len(light_directions),
                    dtype=torch.int64,
                    device=device,
                )[None, :]
                result = dispatcher.evaluate(
                    query, wi, seeds, evaluation_samples=sample_count
                )
                torch._assert_async(result.valid.all())
                response = (
                    result.f[0]
                    * torch.abs(wi[0, :, 2:3])
                ).cpu().numpy().copy()
                result.lease.release()
                dispatcher.end_iteration()
                chunk_means.append(response)
                chunk_weights.append(sample_count)
                remaining -= sample_count
                chunk_index += 1
    finally:
        dispatcher.close()
    falcor_response = np.average(
        np.stack(chunk_means), axis=0, weights=np.asarray(chunk_weights)
    )
    falcor_standard_error = (
        np.std(np.stack(chunk_means), axis=0, ddof=1)
        / np.sqrt(len(chunk_means))
        if len(chunk_means) > 1
        else np.zeros_like(falcor_response)
    )
    relative_error = np.abs(falcor_response - pbrt_response) / np.maximum(
        0.5 * (falcor_response + pbrt_response), 1e-8
    )

    print(f"case={case.name}")
    for index, ((theta, phi), pbrt_value, falcor_value, error, standard_error) in enumerate(
        zip(
            directions,
            pbrt_response,
            falcor_response,
            relative_error,
            falcor_standard_error,
            strict=True,
        )
    ):
        print(
            f"  direction={index:02d} theta={theta:6.1f} phi={phi:6.1f} "
            f"pbrt={np.array2string(pbrt_value, precision=7)} "
            f"reference={np.array2string(falcor_value, precision=7)} "
            f"relative_error={np.array2string(error, precision=4)} "
            f"reference_se={np.array2string(standard_error, precision=6)}"
        )
    mean_error = float(np.mean(relative_error))
    max_error = float(np.max(relative_error))
    print(f"  mean_relative_error={mean_error:.5f} max_relative_error={max_error:.5f}")
    return mean_error, max_error


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 pbrt-v4 与 Falcor 层栈 reference 的 coated 材质切片。")
    parser.add_argument("--pbrt-exe", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES),
        help="可重复；省略时执行全部 coated diffuse/conductor 代表情形。",
    )
    args = parser.parse_args()

    selected = args.case or list(CASES)
    results = [
        _run_case(
            args.pbrt_exe,
            CASES[name],
            samples=args.samples,
            batches=args.batches,
            max_depth=args.max_depth,
        )
        for name in selected
    ]
    print(
        f"suite_mean_relative_error={np.mean([value[0] for value in results]):.5f} "
        f"suite_max_relative_error={np.max([value[1] for value in results]):.5f}"
    )


if __name__ == "__main__":
    main()
