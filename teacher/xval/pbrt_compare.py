from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

import numpy as np

from datagen.two_layer_slice import evaluate_slice, gray_diffuse_stack
from schema import LayerMedium, LayerStack


LINE_PATTERN = re.compile(
    r"angle=\s*([-+0-9.]+).*direct_response=([-+0-9.eE]+).*response=([-+0-9.eE]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pbrt-v4 and the GPU teacher on a gray coated diffuse slice.")
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
    pbrt_direct_batches: list[np.ndarray] = []
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
        parsed = [LINE_PATTERN.search(line) for line in completed.stdout.splitlines()]
        rows = [
            (float(match.group(1)), float(match.group(2)), float(match.group(3)))
            for match in parsed
            if match
        ]
        if not rows:
            raise RuntimeError("pbrt probe produced no parseable direction rows")
        batch_angles = np.asarray([row[0] for row in rows], dtype=np.float32)
        if angles is None:
            angles = batch_angles
        elif not np.array_equal(angles, batch_angles):
            raise RuntimeError("pbrt probe direction rows changed between batches")
        pbrt_direct_batches.append(np.asarray([row[1] for row in rows], dtype=np.float32))
        pbrt_batches.append(np.asarray([row[2] for row in rows], dtype=np.float32))
    assert angles is not None
    pbrt_direct = np.mean(pbrt_direct_batches, axis=0)
    pbrt_response = np.mean(pbrt_batches, axis=0)
    pbrt_standard_error = (
        np.std(pbrt_batches, axis=0, ddof=1) / np.sqrt(args.batches)
        if args.batches > 1
        else np.full_like(pbrt_response, np.nan)
    )
    stack = gray_diffuse_stack()
    if args.optical_thickness > 0.0:
        stack = LayerStack(
            stack.layers,
            (
                LayerMedium(
                    sigma_a=(1.0 - args.medium_albedo,) * 3,
                    sigma_s=(args.medium_albedo,) * 3,
                    g=args.g,
                    thickness=args.optical_thickness,
                ),
            ),
        )
    teacher_batches = [
        evaluate_slice(
            stack,
            shader_entry="evaluateTwoLayerTeacher",
            view_angle_degrees=args.view_angle,
            angles_degrees=angles,
            sample_count=args.samples,
            max_depth=args.max_depth,
            seed=53 + 1009 * batch_index,
        )
        for batch_index in range(args.batches)
    ]
    falcor_direct = evaluate_slice(
        stack,
        shader_entry="evaluateDirectCoat",
        view_angle_degrees=args.view_angle,
        angles_degrees=angles,
    ).mean[:, 0]
    teacher_response = np.mean([batch.mean[:, 0] for batch in teacher_batches], axis=0)
    teacher_second_moment = np.mean(
        [batch.variance[:, 0] + batch.mean[:, 0] ** 2 for batch in teacher_batches], axis=0
    )
    teacher_standard_error = np.sqrt(
        np.maximum(teacher_second_moment - teacher_response**2, 0.0) / (args.samples * args.batches)
    )
    relative_error = np.abs(teacher_response - pbrt_response) / np.maximum(
        0.5 * (teacher_response + pbrt_response), 1e-8
    )
    for angle, pbrt_value, teacher_value, error, standard_error, pbrt_se, pbrt_direct_value, falcor_direct_value in zip(
        angles,
        pbrt_response,
        teacher_response,
        relative_error,
        teacher_standard_error,
        pbrt_standard_error,
        pbrt_direct,
        falcor_direct,
        strict=True,
    ):
        print(
            f"angle={angle:6.1f} pbrt={pbrt_value:.8f} teacher={teacher_value:.8f} "
            f"relative_error={error:.5f} teacher_se={standard_error:.6f} "
            f"delta_over_teacher_se={abs(teacher_value - pbrt_value) / max(standard_error, 1e-12):.3f} "
            f"pbrt_se={pbrt_se:.6f} "
            f"direct_delta={falcor_direct_value - pbrt_direct_value:+.7f}"
        )
    print(f"mean_relative_error={np.mean(relative_error):.5f} max_relative_error={np.max(relative_error):.5f}")


if __name__ == "__main__":
    main()
