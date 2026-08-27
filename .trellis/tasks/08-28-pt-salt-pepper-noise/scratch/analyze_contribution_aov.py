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
    args = parser.parse_args()

    beauty = read_rgb(args.beauty)
    aov = read_rgb(args.aov)
    if beauty.shape != aov.shape:
        raise ValueError(f"shape mismatch: {beauty.shape} != {aov.shape}")

    luminance = beauty @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    median = local_median(luminance)
    residual = np.maximum(luminance - median, 0.0)
    score = residual / np.maximum(median, 0.01)
    top_count = min(args.top, score.size)
    top_flat = np.argpartition(score.ravel(), -top_count)[-top_count:]
    top_aov = aov.reshape(-1, 3)[top_flat]
    top_residual = residual.ravel()[top_flat]
    weighted = (top_aov * top_residual[:, None]).sum(axis=0)
    weighted /= max(float(weighted.sum()), 1e-20)

    print(
        {
            "shape": list(beauty.shape),
            "finite": bool(np.isfinite(beauty).all() and np.isfinite(aov).all()),
            "aov_max_invalid_geometry": float(aov[..., 0].max()),
            "aov_max_environment_miss": float(aov[..., 1].max()),
            "aov_max_secondary_nee_or_deeper": float(aov[..., 2].max()),
            "aov_sum_invalid_geometry": float(aov[..., 0].sum()),
            "aov_sum_environment_miss": float(aov[..., 1].sum()),
            "aov_sum_secondary_nee_or_deeper": float(aov[..., 2].sum()),
            "top_local_residual_count": top_count,
            "top_local_residual_weighted_share": {
                "invalid_geometry": float(weighted[0]),
                "environment_miss": float(weighted[1]),
                "secondary_nee_or_deeper": float(weighted[2]),
            },
        }
    )


if __name__ == "__main__":
    main()
