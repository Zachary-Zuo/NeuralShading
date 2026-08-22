from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyexr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sample_equirectangular(image: np.ndarray, directions: np.ndarray, azimuth: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)[..., :3]
    height, width = image.shape[:2]
    phi = np.arctan2(directions[:, 1], directions[:, 0]) + azimuth
    u = np.mod(phi / (2.0 * np.pi) + 0.5, 1.0) * width - 0.5
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    v = np.clip(theta / np.pi * height - 0.5, 0.0, height - 1.0)
    x_floor = np.floor(u)
    x0 = x_floor.astype(np.int64) % width
    x1 = (x0 + 1) % width
    y0 = np.floor(v).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (u - x_floor)[:, None]
    ty = (v - np.floor(v))[:, None]
    top = image[y0, x0] * (1.0 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1.0 - tx) + image[y1, x1] * tx
    return np.maximum(top * (1.0 - ty) + bottom * ty, 0.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen HDRIs on the v0 direction grid.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "docs" / "manifests" / "polyhaven_hdri_v0.json",
    )
    parser.add_argument(
        "--hdris",
        type=Path,
        default=PROJECT_ROOT / "data" / "hdris" / "polyhaven_1k",
    )
    parser.add_argument(
        "--directions",
        type=Path,
        default=PROJECT_ROOT / "data" / "v0_oracle_512" / "light_directions.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "hdris" / "polyhaven_probes_v0.npz",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    directions = np.load(args.directions)[:, :3]
    names: list[str] = []
    probes: list[np.ndarray] = []
    for asset in manifest["assets"]:
        image = pyexr.read(str(args.hdris / f"{asset['id']}_1k.exr"))
        for rotation_index, azimuth in enumerate(np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)):
            names.append(f"{asset['id']}-r{rotation_index}")
            probes.append(sample_equirectangular(image, directions, float(azimuth)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, names=np.asarray(names), probes=np.stack(probes))
    print(f"cached {len(probes)} probes at {args.output}")


if __name__ == "__main__":
    main()
