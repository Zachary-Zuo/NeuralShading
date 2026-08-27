from __future__ import annotations

import argparse
from pathlib import Path

import Imath
import numpy as np
import OpenEXR


def read_rgb(path: Path) -> np.ndarray:
    source = OpenEXR.InputFile(str(path))
    bounds = source.header()["dataWindow"]
    width = bounds.max.x - bounds.min.x + 1
    height = bounds.max.y - bounds.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    return np.stack(
        [
            np.frombuffer(source.channel(channel, pixel_type), dtype=np.float32).reshape(
                height, width
            )
            for channel in "RGB"
        ],
        axis=-1,
    )


def local_median(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.median(windows, axis=(-2, -1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("beauty", type=Path)
    parser.add_argument("aov", type=Path)
    parser.add_argument("--top", type=int, default=512)
    parser.add_argument("--y-min", type=int, default=0)
    parser.add_argument("--y-max", type=int)
    args = parser.parse_args()

    beauty = read_rgb(args.beauty)
    aov = read_rgb(args.aov)
    if beauty.shape != aov.shape:
        raise ValueError(f"shape mismatch: {beauty.shape} != {aov.shape}")

    luminance = beauty @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    median = local_median(luminance)
    residual = np.maximum(luminance - median, 0.0)
    score = residual / np.maximum(median, 0.01)
    y_max = score.shape[0] if args.y_max is None else args.y_max
    eligible = np.zeros_like(score, dtype=bool)
    eligible[max(args.y_min, 0) : min(y_max, score.shape[0])] = True
    eligible_flat = np.flatnonzero(eligible.ravel())
    top_count = min(args.top, eligible_flat.size)
    eligible_score = score.ravel()[eligible_flat]
    selected = np.argpartition(eligible_score, -top_count)[-top_count:]
    top_flat = eligible_flat[selected]
    top_aov = aov.reshape(-1, 3)[top_flat]
    top_residual = residual.ravel()[top_flat]
    weighted = (top_aov * top_residual[:, None]).sum(axis=0)
    weighted /= max(float(weighted.sum()), 1e-20)
    top_channel_correlation = np.corrcoef(top_aov, rowvar=False)

    max_records = []
    for channel, label in enumerate(
        ("primary_bsdf", "primary_light", "secondary_all_direct")
    ):
        y, x = np.unravel_index(np.argmax(aov[..., channel]), aov.shape[:2])
        max_records.append(
            {
                "label": label,
                "xy": [int(x), int(y)],
                "aov": float(aov[y, x, channel]),
                "aov_rgb": [float(value) for value in aov[y, x]],
                "beauty_luminance": float(luminance[y, x]),
                "local_median": float(median[y, x]),
                "local_residual": float(residual[y, x]),
            }
        )

    surface_mask = np.sum(aov, axis=-1) > 0.0
    closure_error = np.abs(np.sum(aov, axis=-1) - luminance)
    print(
        {
            "shape": list(beauty.shape),
            "finite": bool(np.isfinite(beauty).all() and np.isfinite(aov).all()),
            "aov_max_records": max_records,
            "surface_aov_luminance_max_abs_error": float(
                closure_error[surface_mask].max(initial=0.0)
            ),
            "top_local_residual_count": top_count,
            "top_local_residual_y_range": [args.y_min, y_max],
            "top_local_residual_weighted_share": {
                "primary_bsdf": float(weighted[0]),
                "primary_light": float(weighted[1]),
                "secondary_all_direct": float(weighted[2]),
            },
            "top_channel_correlation": top_channel_correlation.tolist(),
        }
    )


if __name__ == "__main__":
    main()
