from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyexr


CAPTURES = (
    ("carpaint-before", Path("artifacts/captures/mdl-viewer/carpaint")),
    ("carpaint-fixed", Path("artifacts/captures/mdl-viewer-fixed/carpaint")),
    ("ceramic-fixed", Path("artifacts/captures/mdl-viewer-fixed/ceramic")),
)


def main() -> None:
    for label, stem in CAPTURES:
        image = pyexr.read(str(stem.with_name(stem.name + "-slot-0.exr")))
        manifest = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        luminance = image @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        padded = np.pad(luminance, 1, mode="edge")
        neighborhoods = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        local_median = np.median(neighborhoods, axis=(-2, -1))
        isolated = (luminance > 5.0) & (luminance > 20.0 * (local_median + 1e-3))
        print(
            json.dumps(
                {
                    "label": label,
                    "shape": image.shape,
                    "finite": bool(np.isfinite(image).all()),
                    "spp": manifest["slots"][0]["spp"],
                    "mean": float(image.mean()),
                    "max_channel": float(image.max()),
                    "luminance_p999": float(np.quantile(luminance, 0.999)),
                    "luminance_p9999": float(np.quantile(luminance, 0.9999)),
                    "pixels_channel_over_10": int(np.count_nonzero(np.max(image, axis=2) > 10.0)),
                    "pixels_channel_over_100": int(np.count_nonzero(np.max(image, axis=2) > 100.0)),
                    "isolated_firefly_pixels": int(np.count_nonzero(isolated)),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
