from __future__ import annotations

import argparse

import numpy as np

from datagen.two_layer_slice import diffuse_stack, evaluate_slice


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure reciprocity convergence of the Falcor layered reference.")
    parser.add_argument("--samples", nargs="+", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--entry",
        default="evaluateTwoLayerReference",
        choices=["evaluateTwoLayerReference", "evaluateTwoLayerTeacher"],
    )
    args = parser.parse_args()

    view_angle = 20.0
    light_angle = 50.0
    stack = diffuse_stack()
    for sample_count in args.samples:
        forward_result = evaluate_slice(
            stack,
            shader_entry=args.entry,
            view_angle_degrees=view_angle,
            sample_count=sample_count,
            seed=args.seed,
            angles_degrees=np.array([light_angle], dtype=np.float32),
        )
        reverse_result = evaluate_slice(
            stack,
            shader_entry=args.entry,
            view_angle_degrees=light_angle,
            sample_count=sample_count,
            seed=args.seed + 18,
            angles_degrees=np.array([view_angle], dtype=np.float32),
        )
        forward = forward_result.mean[0]
        reverse = reverse_result.mean[0]
        forward_brdf = forward / np.cos(np.deg2rad(light_angle))
        reverse_brdf = reverse / np.cos(np.deg2rad(view_angle))
        standard_error = np.sqrt(
            forward_result.variance[0] / (sample_count * np.cos(np.deg2rad(light_angle)) ** 2)
            + reverse_result.variance[0] / (sample_count * np.cos(np.deg2rad(view_angle)) ** 2)
        )
        relative_error = np.abs(forward_brdf - reverse_brdf) / np.maximum(
            0.5 * (forward_brdf + reverse_brdf), 1e-8
        )
        print(
            f"samples={sample_count:6d} "
            f"forward={forward_brdf} reverse={reverse_brdf} "
            f"relative_error={relative_error} z={np.abs(forward_brdf - reverse_brdf) / np.maximum(standard_error, 1e-8)}"
        )


if __name__ == "__main__":
    main()
