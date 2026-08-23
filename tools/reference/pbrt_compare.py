from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

import numpy as np

from ncls.core.material import DiffuseInterface, HomogeneousMedium, LayerStackIR, RoughDielectricInterface
from ncls.data.reference import FalcorReferenceEvaluator, evaluate_reference_fixed


LINE_PATTERN = re.compile(
    r"angle=\s*([-+0-9.]+).*direct_response=([-+0-9.eE]+).*response=([-+0-9.eE]+)"
)


def _direction(theta_degrees: float) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    return np.asarray([np.sin(theta), 0.0, np.cos(theta), 0.0], dtype=np.float32)


def _material(optical_thickness: float, medium_albedo: float, g: float) -> LayerStackIR:
    sigma_a = (1.0 - medium_albedo,) * 3
    sigma_s = (medium_albedo,) * 3
    return LayerStackIR(
        (RoughDielectricInterface(0.12, 0.12, 1.5), DiffuseInterface((0.5, 0.5, 0.5))),
        (HomogeneousMedium(sigma_a, sigma_s, g, optical_thickness),),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 pbrt-v4 与 Falcor 随机游走参考解的灰色 coated diffuse 切片。")
    parser.add_argument("--pbrt-exe", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--view-angle", type=float, default=20.0)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--optical-thickness", type=float, default=1e-6)
    parser.add_argument("--medium-albedo", type=float, default=0.0)
    parser.add_argument("--g", type=float, default=0.0)
    args = parser.parse_args()

    pbrt_batches: list[np.ndarray] = []
    angles: np.ndarray | None = None
    for batch_index in range(args.batches):
        completed = subprocess.run(
            [
                str(args.pbrt_exe.resolve()),
                str(args.samples),
                str(args.view_angle),
                str(args.max_depth),
                str(args.optical_thickness),
                str(1 + 1009 * batch_index),
                str(args.medium_albedo),
                str(args.g),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        matches = [LINE_PATTERN.search(line) for line in completed.stdout.splitlines()]
        rows = [(float(item.group(1)), float(item.group(3))) for item in matches if item]
        if not rows:
            raise RuntimeError("pbrt probe produced no parseable direction rows")
        batch_angles = np.asarray([row[0] for row in rows], dtype=np.float32)
        if angles is None:
            angles = batch_angles
        elif not np.array_equal(angles, batch_angles):
            raise RuntimeError("pbrt probe direction rows changed between batches")
        pbrt_batches.append(np.asarray([row[1] for row in rows], dtype=np.float32))
    assert angles is not None
    pbrt_response = np.mean(pbrt_batches, axis=0)

    light_directions = np.stack([_direction(float(angle)) for angle in angles])
    evaluator = FalcorReferenceEvaluator(
        light_directions,
        max_depth=args.max_depth,
        max_tile_batch=1,
    )
    material = _material(args.optical_thickness, args.medium_albedo, args.g)
    falcor_batches = [
        evaluate_reference_fixed(
            evaluator,
            [material],
            _direction(args.view_angle)[None, :],
            tile_seeds=np.asarray([53 + 1009 * batch_index], dtype=np.uint32),
            samples_per_replica=args.samples,
        )
        for batch_index in range(args.batches)
    ]
    falcor_response = np.mean([batch.mean[0, :, 0] for batch in falcor_batches], axis=0)
    falcor_variance = np.mean([batch.variance[0, :, 0] for batch in falcor_batches], axis=0)
    falcor_standard_error = np.sqrt(falcor_variance / (2 * args.samples * args.batches))
    relative_error = np.abs(falcor_response - pbrt_response) / np.maximum(
        0.5 * (falcor_response + pbrt_response), 1e-8
    )
    for angle, pbrt_value, falcor_value, error, standard_error in zip(
        angles,
        pbrt_response,
        falcor_response,
        relative_error,
        falcor_standard_error,
        strict=True,
    ):
        print(
            f"angle={angle:6.1f} pbrt={pbrt_value:.8f} reference={falcor_value:.8f} "
            f"relative_error={error:.5f} reference_se={standard_error:.6f} "
            f"delta_over_reference_se={abs(falcor_value - pbrt_value) / max(standard_error, 1e-12):.3f}"
        )
    print(f"mean_relative_error={np.mean(relative_error):.5f} max_relative_error={np.max(relative_error):.5f}")


if __name__ == "__main__":
    main()
