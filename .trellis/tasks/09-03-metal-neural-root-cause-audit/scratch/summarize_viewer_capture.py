from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import OpenEXR


def _read(path: Path) -> np.ndarray:
    pixels = OpenEXR.File(str(path)).channels()["RGB"].pixels
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"capture is not an RGB image: {path}")
    return pixels


def _stats(pixels: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = pixels[mask]
    luminance = selected.mean(axis=1)
    return {
        "mean_rgb": selected.mean(axis=0).tolist(),
        "std_rgb": selected.std(axis=0).tolist(),
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "rgb_chroma_mean": float(
            np.mean(np.max(selected, axis=1) - np.min(selected, axis=1))
        ),
        "minimum": float(selected.min()),
        "maximum": float(selected.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总固定 shaderball 球面区域的线性 EXR 输出"
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("neural", type=Path)
    parser.add_argument("--center-x", type=int, default=80)
    parser.add_argument("--center-y", type=int, default=90)
    parser.add_argument("--radius", type=int, default=55)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = _read(args.reference)
    neural = _read(args.neural)
    if reference.shape != neural.shape:
        raise ValueError("reference and neural capture shapes differ")
    height, width, _ = reference.shape
    y, x = np.ogrid[:height, :width]
    mask = (
        (x - args.center_x) ** 2 + (y - args.center_y) ** 2
        < args.radius**2
    )
    absolute_error = np.abs(reference[mask] - neural[mask])
    document = {
        "schema": "ncls.metal-viewer-learning-probe@1",
        "reference": str(args.reference),
        "neural": str(args.neural),
        "shape": list(reference.shape),
        "region": {
            "kind": "fixed-shaderball-circle",
            "center": [args.center_x, args.center_y],
            "radius": args.radius,
            "pixel_count": int(mask.sum()),
        },
        "reference_stats": _stats(reference, mask),
        "neural_stats": _stats(neural, mask),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(absolute_error)))),
    }
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
