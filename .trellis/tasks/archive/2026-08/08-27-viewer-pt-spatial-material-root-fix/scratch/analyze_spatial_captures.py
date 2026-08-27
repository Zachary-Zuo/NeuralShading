from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import OpenEXR


ROOT = Path(
    "D:/01_Workspace/NeuralShading/artifacts/nvidia-faithful/"
    "materialx-recorded-200k/viewer-reference-neural"
)
LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def read_luminance(name: str) -> np.ndarray:
    rgb = OpenEXR.File(str(ROOT / name)).channels()["RGB"].pixels.astype(np.float32)
    return rgb @ LUMA


def valid_surface(luminance: np.ndarray) -> np.ndarray:
    return np.isfinite(luminance) & (luminance > 0.06)


def spatial_metrics(luminance: np.ndarray) -> dict[str, float | int]:
    mask = valid_surface(luminance)
    dx_mask = mask[:, 1:] & mask[:, :-1]
    dy_mask = mask[1:, :] & mask[:-1, :]
    dx = np.diff(luminance, axis=1)[dx_mask]
    dy = np.diff(luminance, axis=0)[dy_mask]
    gradient = np.concatenate((dx, dy))
    values = luminance[mask]
    return {
        "valid_pixels": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "gradient_rms": float(np.sqrt(np.mean(gradient * gradient))),
    }


def correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    mask = valid_surface(first) & valid_surface(second)
    dx_mask = mask[:, 1:] & mask[:, :-1]
    dy_mask = mask[1:, :] & mask[:-1, :]
    first_gradient = np.concatenate(
        (np.diff(first, axis=1)[dx_mask], np.diff(first, axis=0)[dy_mask])
    )
    second_gradient = np.concatenate(
        (np.diff(second, axis=1)[dx_mask], np.diff(second, axis=0)[dy_mask])
    )
    return {
        "common_pixels": int(mask.sum()),
        "luminance_correlation": float(np.corrcoef(first[mask], second[mask])[0, 1]),
        "gradient_correlation": float(
            np.corrcoef(first_gradient, second_gradient)[0, 1]
        ),
    }


old_denim = read_luminance("diagnostic-denim-slot-0.exr")
fixed_denim = read_luminance("denim-fixed-32spp-slot-0.exr")
walnut_source = read_luminance("walnut-direct-fixed-64spp-slot-0.exr")
walnut_neural = read_luminance("walnut-direct-fixed-64spp-slot-1.exr")
neural_pt = read_luminance("walnut-neural-pt-deferred-fixed-64spp-slot-0.exr")
neural_deferred = read_luminance(
    "walnut-neural-pt-deferred-fixed-64spp-slot-1.exr"
)

print(
    json.dumps(
        {
            "denim_before": spatial_metrics(old_denim),
            "denim_after": spatial_metrics(fixed_denim),
            "walnut_source": spatial_metrics(walnut_source),
            "walnut_neural": spatial_metrics(walnut_neural),
            "walnut_source_neural": correlation(walnut_source, walnut_neural),
            "walnut_neural_pt_deferred": correlation(neural_pt, neural_deferred),
        },
        indent=2,
        sort_keys=True,
    )
)
