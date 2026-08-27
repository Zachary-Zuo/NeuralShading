from __future__ import annotations

import sys
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


def connected_component_sizes(mask: np.ndarray) -> list[int]:
    pending = {tuple(index) for index in np.argwhere(mask)}
    sizes: list[int] = []
    while pending:
        seed = pending.pop()
        frontier = [seed]
        size = 0
        while frontier:
            y, x = frontier.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor = (y + dy, x + dx)
                    if neighbor in pending:
                        pending.remove(neighbor)
                        frontier.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


for argument in sys.argv[1:]:
    path = Path(argument)
    rgb = read_rgb(path)
    maximum = rgb.max(axis=-1)
    extreme = np.argwhere(maximum > 100.0)
    component_sizes = connected_component_sizes(maximum > 100.0)
    bounding_box = None
    if len(extreme):
        bounding_box = [
            int(extreme[:, 1].min()),
            int(extreme[:, 0].min()),
            int(extreme[:, 1].max()),
            int(extreme[:, 0].max()),
        ]
    print(
        path.parent.name,
        {
            "shape": list(rgb.shape),
            "finite": bool(np.isfinite(rgb).all()),
            "max": float(rgb.max()),
            "p99.9": float(np.quantile(rgb, 0.999)),
            "p99.99": float(np.quantile(rgb, 0.9999)),
            "pixels_over_100_bbox": bounding_box,
            "pixels_over_100": int(len(extreme)),
            "components_over_100": len(component_sizes),
            "isolated_pixels_over_100": component_sizes.count(1),
            "largest_component_over_100": component_sizes[0] if component_sizes else 0,
        },
    )
