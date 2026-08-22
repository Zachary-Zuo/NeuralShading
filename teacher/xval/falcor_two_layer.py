from __future__ import annotations

import argparse

import numpy as np

from datagen.two_layer_slice import evaluate_slice, gray_diffuse_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate the self-contained two-layer teacher against Falcor.")
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--max-depth", type=int, default=32)
    args = parser.parse_args()

    angles = np.array([-55.0, -20.0, 0.0, 35.0, 60.0], dtype=np.float32)
    stack = gray_diffuse_stack()
    reference = evaluate_slice(
        stack,
        shader_entry="evaluateTwoLayerReference",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=args.samples,
        seed=11,
        max_depth=args.max_depth,
    )
    teacher = evaluate_slice(
        stack,
        shader_entry="evaluateTwoLayerTeacher",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=args.samples,
        seed=53,
        max_depth=args.max_depth,
    )

    standard_error = np.sqrt((reference.variance + teacher.variance) / args.samples)
    z_score = np.abs(reference.mean - teacher.mean) / np.maximum(standard_error, 1e-8)
    relative_error = np.abs(reference.mean - teacher.mean) / np.maximum(
        0.5 * (reference.mean + teacher.mean), 1e-8
    )
    for index, angle in enumerate(angles):
        print(
            f"angle={angle:6.1f} reference={reference.mean[index]} teacher={teacher.mean[index]} "
            f"relative_error={relative_error[index]} z={z_score[index]}"
        )
    print(f"max_z={np.max(z_score):.4f} p95_z={np.percentile(z_score, 95):.4f}")


if __name__ == "__main__":
    main()
