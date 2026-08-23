from __future__ import annotations

import argparse
import math
from pathlib import Path


def generate_uv_sphere(path: Path, longitude_segments: int, latitude_segments: int) -> None:
    if longitude_segments < 8 or latitude_segments < 4:
        raise ValueError("UV sphere 分段数过低")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# NCLS common-camera MaterialX parity sphere", "g sphere"]
    for latitude in range(latitude_segments + 1):
        theta = -0.5 * math.pi + math.pi * latitude / latitude_segments
        radius = math.cos(theta)
        y = math.sin(theta)
        for longitude in range(longitude_segments + 1):
            phi = 2.0 * math.pi * longitude / longitude_segments
            x = -radius * math.sin(phi)
            z = -radius * math.cos(phi)
            lines.append(f"v {x:.9f} {y:.9f} {z:.9f}")
    for latitude in range(latitude_segments + 1):
        v = latitude / latitude_segments
        for longitude in range(longitude_segments + 1):
            u = longitude / longitude_segments
            lines.append(f"vt {u:.9f} {v:.9f}")
    for latitude in range(latitude_segments + 1):
        theta = -0.5 * math.pi + math.pi * latitude / latitude_segments
        radius = math.cos(theta)
        y = math.sin(theta)
        for longitude in range(longitude_segments + 1):
            phi = 2.0 * math.pi * longitude / longitude_segments
            x = -radius * math.sin(phi)
            z = -radius * math.cos(phi)
            lines.append(f"vn {x:.9f} {y:.9f} {z:.9f}")
    stride = longitude_segments + 1
    for latitude in range(latitude_segments):
        for longitude in range(longitude_segments):
            a = latitude * stride + longitude + 1
            b = a + 1
            d = a + stride
            c = d + 1
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
            lines.append(f"f {a}/{a}/{a} {c}/{c}/{c} {d}/{d}/{d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 MaterialX/Falcor parity 使用的高细分 UV sphere")
    parser.add_argument("output", type=Path)
    parser.add_argument("--longitude-segments", type=int, default=256)
    parser.add_argument("--latitude-segments", type=int, default=128)
    arguments = parser.parse_args()
    generate_uv_sphere(arguments.output.resolve(), arguments.longitude_segments, arguments.latitude_segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
