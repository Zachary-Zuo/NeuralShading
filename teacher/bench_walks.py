from __future__ import annotations

import argparse
import time

import numpy as np

from datagen.directions import equal_area_hemisphere
from datagen.gen_tiles import FalcorTileEvaluator
from schema import LayerInterface, LayerMedium, LayerStack, LayerType


def benchmark_stack() -> LayerStack:
    return LayerStack(
        (
            LayerInterface(
                LayerType.ROUGH_DIELECTRIC,
                roughness_x=0.12,
                roughness_y=0.08,
                eta=(1.45, 1.45, 1.45),
            ),
            LayerInterface(
                LayerType.ROUGH_DIELECTRIC,
                roughness_x=0.22,
                roughness_y=0.16,
                eta=(1.1, 1.1, 1.1),
                tangent_rotation=0.4,
            ),
            LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.5, 0.2, 0.08)),
        ),
        (LayerMedium(thickness=0.2), LayerMedium(thickness=0.35)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end Falcor teacher query throughput.")
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=64, help="Samples per A/B half and query")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=64)
    args = parser.parse_args()

    directions, _ = equal_area_hemisphere(args.queries)
    evaluator = FalcorTileEvaluator(directions, max_depth=args.max_depth)
    view = np.array([0.5, 0.0, np.sqrt(0.75), 0.0], dtype=np.float32)
    stack = benchmark_stack()

    evaluator.evaluate_batch(stack, view, sample_count_per_half=args.samples, seed=1)
    elapsed_samples: list[float] = []
    for repeat in range(args.repeats):
        start = time.perf_counter()
        evaluator.evaluate_batch(
            stack,
            view,
            sample_count_per_half=args.samples,
            seed=1009 + repeat * 7919,
        )
        elapsed_samples.append(time.perf_counter() - start)

    median_seconds = float(np.median(elapsed_samples))
    walks_per_repeat = 2 * args.queries * args.samples
    walks_per_second = walks_per_repeat / median_seconds
    queries_per_second = (args.queries / median_seconds)
    print(
        f"queries={args.queries} samples_per_half={args.samples} repeats={args.repeats} "
        f"median_seconds={median_seconds:.6f} walks_per_second={walks_per_second:.3e} "
        f"direction_queries_per_second={queries_per_second:.3e}"
    )


if __name__ == "__main__":
    main()
