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


def summarize(
    path: Path, y_min: int = 0, y_max: int | None = None
) -> dict[str, float | int | bool]:
    rgb = read_rgb(path)
    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    padded = np.pad(luminance, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    median = np.median(windows, axis=(-2, -1))
    residual = np.maximum(luminance - median, 0.0)
    score = residual / np.maximum(median, 0.01)
    y_max = score.shape[0] if y_max is None else y_max
    selection = np.s_[max(y_min, 0) : min(y_max, score.shape[0]), :]
    selected_residual = residual[selection]
    selected_score = score[selection]
    return {
        "finite": bool(np.isfinite(rgb).all()),
        "residual_p99": float(np.quantile(selected_residual, 0.99)),
        "residual_p99_9": float(np.quantile(selected_residual, 0.999)),
        "residual_p99_99": float(np.quantile(selected_residual, 0.9999)),
        "score_p99": float(np.quantile(selected_score, 0.99)),
        "score_p99_9": float(np.quantile(selected_score, 0.999)),
        "score_p99_99": float(np.quantile(selected_score, 0.9999)),
        "count_score_gt_5_residual_gt_0_1": int(
            np.count_nonzero((selected_score > 5.0) & (selected_residual > 0.1))
        ),
        "top_512_residual_sum": float(
            np.partition(selected_residual.ravel(), -512)[-512:].sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--y-min", type=int, default=0)
    parser.add_argument("--y-max", type=int)
    args = parser.parse_args()
    for path in args.paths:
        print(path.parent.name, summarize(path, args.y_min, args.y_max))


if __name__ == "__main__":
    main()
