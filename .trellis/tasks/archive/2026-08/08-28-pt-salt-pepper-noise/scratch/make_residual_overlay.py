from __future__ import annotations

import argparse
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("linear", type=Path)
    parser.add_argument("display", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--y-min", type=int, default=275)
    parser.add_argument("--y-max", type=int, default=475)
    args = parser.parse_args()

    rgb = read_rgb(args.linear)
    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    padded = np.pad(luminance, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    median = np.median(windows, axis=(-2, -1))
    residual = np.maximum(luminance - median, 0.0)
    score = residual / np.maximum(median, 0.01)
    selected = (score > 5.0) & (residual > 0.1)
    selected[: args.y_min] = False
    selected[args.y_max :] = False

    display = Image.open(args.display).convert("RGB").crop(
        (0, 0, rgb.shape[1], rgb.shape[0])
    )
    draw = ImageDraw.Draw(display)
    for y, x in np.argwhere(selected):
        draw.rectangle((int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2), outline=(255, 32, 32))

    crop = display.crop((0, args.y_min, rgb.shape[1], args.y_max))
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(args.output)
    print({"output": str(args.output), "marked_pixels": int(selected.sum())})


if __name__ == "__main__":
    main()
