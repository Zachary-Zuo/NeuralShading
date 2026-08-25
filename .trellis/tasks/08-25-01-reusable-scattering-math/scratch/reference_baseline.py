from __future__ import annotations

import json

import numpy as np

from ncls.core.material import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
)
from ncls.data.reference import FalcorReferenceEvaluator, evaluate_reference_fixed
from ncls.paths import ARTIFACT_ROOT


def _direction(theta_degrees: float, phi_degrees: float = 0.0) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    phi = np.deg2rad(phi_degrees)
    sine = np.sin(theta)
    return np.asarray(
        [sine * np.cos(phi), sine * np.sin(phi), np.cos(theta), 0.0],
        dtype=np.float32,
    )


def main() -> None:
    stacks = [
        LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ()),
        LayerStackIR(
            (
                RoughConductorInterface(
                    0.12,
                    0.08,
                    (0.2, 0.3, 0.5),
                    (3.0, 2.5, 2.0),
                    0.25,
                ),
            ),
            (),
        ),
        LayerStackIR(
            (
                RoughDielectricInterface(0.12, 0.08, 1.4, 0.1),
                RoughDielectricInterface(0.24, 0.16, 1.15, -0.35),
                DiffuseInterface((0.5, 0.25, 0.1)),
            ),
            (
                HomogeneousMedium((0.1, 0.05, 0.02), thickness=0.25),
                HomogeneousMedium(thickness=0.35),
            ),
        ),
    ]
    views = np.stack(
        [
            _direction(25.0, 0.0),
            _direction(55.0, 35.0),
            _direction(20.0, -20.0),
        ]
    )
    lights = np.stack(
        [
            _direction(0.0),
            _direction(50.0, 0.0),
            _direction(70.0, 75.0),
            _direction(82.0, -130.0),
        ]
    )
    evaluator = FalcorReferenceEvaluator(lights, max_depth=64, max_query_group_batch=3)
    result = evaluate_reference_fixed(
        evaluator,
        stacks,
        views,
        query_group_seeds=np.asarray([17, 101, 191], dtype=np.uint32),
        samples_per_replica=4096,
    )
    payload = {
        "direction_convention": "outward-local-z-up",
        "lights": lights[:, :3].tolist(),
        "mean": result.mean.tolist(),
        "sample_count": result.sample_count.tolist(),
        "variance": result.variance.tolist(),
        "views": views[:, :3].tolist(),
    }
    output = ARTIFACT_ROOT / "validation" / "01-scattering-math-reference-baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
