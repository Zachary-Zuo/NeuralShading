from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.closure_families import evaluate_exported_parameters
from baselines.oracle_fit import load_oracle_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analytic_probe_bank(lights: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Create deterministic HDR sky/sun/softbox probes on the tile direction grid."""
    directions = np.asarray(lights, dtype=np.float32)
    names: list[str] = []
    probes: list[np.ndarray] = []

    names.append("uniform-white")
    probes.append(np.ones((len(directions), 3), dtype=np.float32))
    names.append("overcast")
    overcast = (0.15 + 0.85 * directions[:, 2:3]) * np.asarray([0.85, 0.92, 1.0])
    probes.append(overcast.astype(np.float32))

    sun_thetas = np.deg2rad([15.0, 35.0, 60.0, 78.0])
    sun_phis = np.deg2rad([0.0, 90.0, 210.0])
    for theta in sun_thetas:
        for phi in sun_phis:
            axis = np.asarray(
                [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
                dtype=np.float32,
            )
            sun = np.exp(96.0 * (directions @ axis - 1.0))[:, None]
            sky = (0.03 + 0.12 * directions[:, 2:3]) * np.asarray([0.75, 0.85, 1.0])
            radiance = sky + 24.0 * sun * np.asarray([1.0, 0.88, 0.7])
            names.append(f"sun-{np.degrees(theta):.0f}-{np.degrees(phi):.0f}")
            probes.append(radiance.astype(np.float32))

    rng = np.random.default_rng(20260822)
    for probe_index in range(12):
        radiance = np.full((len(directions), 3), 0.02, dtype=np.float32)
        for _ in range(3):
            z = rng.uniform(0.15, 0.95)
            phi = rng.uniform(-np.pi, np.pi)
            axis = np.asarray(
                [np.sqrt(1.0 - z * z) * np.cos(phi), np.sqrt(1.0 - z * z) * np.sin(phi), z],
                dtype=np.float32,
            )
            sharpness = rng.uniform(4.0, 24.0)
            color = rng.uniform(0.4, 1.0, size=3)
            intensity = rng.uniform(2.0, 10.0)
            radiance += (
                intensity
                * np.exp(sharpness * (directions @ axis - 1.0))[:, None]
                * color[None, :]
            ).astype(np.float32)
        names.append(f"softbox-{probe_index:02d}")
        probes.append(radiance)
    return names, np.stack(probes)


def sample_equirectangular(image: np.ndarray, directions: np.ndarray, azimuth: float = 0.0) -> np.ndarray:
    """Bilinearly sample a lat-long HDRI using +Z as the material normal."""
    image = np.asarray(image, dtype=np.float32)[..., :3]
    directions = np.asarray(directions, dtype=np.float32)
    height, width = image.shape[:2]
    phi = np.arctan2(directions[:, 1], directions[:, 0]) + azimuth
    u = np.mod(phi / (2.0 * np.pi) + 0.5, 1.0) * width - 0.5
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    v = np.clip(theta / np.pi * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(u).astype(np.int64) % width
    x1 = (x0 + 1) % width
    y0 = np.floor(v).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (u - np.floor(u))[:, None]
    ty = (v - np.floor(v))[:, None]
    top = image[y0, x0] * (1.0 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1.0 - tx) + image[y1, x1] * tx
    return np.maximum(top * (1.0 - ty) + bottom * ty, 0.0).astype(np.float32)


def real_hdri_probe_bank(
    lights: np.ndarray,
    manifest_path: Path,
    hdri_dir: Path,
) -> tuple[list[str], np.ndarray]:
    import pyexr

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names: list[str] = []
    probes: list[np.ndarray] = []
    for asset in manifest["assets"]:
        image = pyexr.read(str(hdri_dir / f"{asset['id']}_1k.exr"))
        if not np.all(np.isfinite(image)):
            raise RuntimeError(f"HDRI contains non-finite pixels: {asset['id']}")
        for rotation_index, azimuth in enumerate(np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)):
            names.append(f"{asset['id']}-r{rotation_index}")
            probes.append(sample_equirectangular(image, lights, float(azimuth)))
    return names, np.stack(probes)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _integrate(response: np.ndarray, probes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("tbc,pbc,b->tpc", response, probes, weights, optimize=True)


def evaluate_ibl(
    dataset_dir: Path,
    oracle_dir: Path,
    output_path: Path,
    *,
    configs: list[str],
    device: str | None,
    hdri_manifest: Path | None = None,
    hdri_dir: Path | None = None,
    probe_file: Path | None = None,
) -> dict[str, object]:
    dataset = load_oracle_dataset(dataset_dir)
    if probe_file is not None:
        cached = np.load(probe_file)
        names = [str(value) for value in cached["names"]]
        probes = cached["probes"]
        probe_kind = "real-hdri"
    elif hdri_manifest is not None:
        if hdri_dir is None:
            raise ValueError("hdri_dir is required with hdri_manifest")
        names, probes = real_hdri_probe_bank(dataset.lights, hdri_manifest, hdri_dir)
        probe_kind = "real-hdri"
    else:
        names, probes = analytic_probe_bank(dataset.lights)
        probe_kind = "analytic"
    weights = np.load(dataset_dir / "solid_angle_weights.npy")
    target_ibl = _integrate(dataset.target, probes, weights)
    mean_a_ibl = _integrate(dataset.mean_a, probes, weights)
    mean_b_ibl = _integrate(dataset.mean_b, probes, weights)
    noise_relative_l1 = np.sum(np.abs(mean_a_ibl - mean_b_ibl), axis=2) / np.maximum(
        0.5 * np.sum(np.abs(mean_a_ibl) + np.abs(mean_b_ibl), axis=2), 1e-6
    )
    result: dict[str, object] = {
        "dataset": str(dataset_dir),
        "tile_count": len(dataset.target),
        "probe_count": len(probes),
        "probe_names": names,
        "probe_kind": probe_kind,
        "noise_relative_l1": _summary(noise_relative_l1),
        "families": {},
    }

    for config in configs:
        archive = np.load(oracle_dir / f"{config}.npz")
        if config.startswith("dictionary-m"):
            cosine = archive["basis_axis"] @ dataset.lights.T
            basis = np.exp(archive["basis_sharpness"][:, None] * (cosine - 1.0))
            prediction = np.einsum("mb,tmc->tbc", basis, archive["amplitude"], optimize=True)
        else:
            family = config.split("-k", maxsplit=1)[0]
            parameter_names = {
                "ggx": ("amplitude", "axis", "alpha"),
                "ltc": ("amplitude", "inverse_scale", "shear", "angle"),
                "sg": ("amplitude", "axis", "sharpness"),
            }[family]
            parameters = {name: archive[name] for name in parameter_names}
            prediction = evaluate_exported_parameters(
                family,
                parameters,
                dataset.views[:, :3],
                dataset.lights,
                device=device,
            )
        prediction_ibl = _integrate(prediction, probes, weights)
        relative_l1 = np.sum(np.abs(prediction_ibl - target_ibl), axis=2) / np.maximum(
            np.sum(np.abs(target_ibl), axis=2), 1e-6
        )
        log_rgb = np.mean(
            np.abs(np.log1p(np.maximum(prediction_ibl, 0.0)) - np.log1p(np.maximum(target_ibl, 0.0))),
            axis=2,
        )
        result["families"][config] = {
            "relative_l1": _summary(relative_l1),
            "log_rgb_l1": _summary(log_rgb),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 真实 HDRI 环境光表示上界" if probe_kind == "real-hdri" else "# 解析环境光探针表示上界",
        "",
        f"数据集：`{dataset_dir}`；{len(probes)} 个光照探针；{len(dataset.target)} 个 tiles。",
        "",
        f"A/B 环境光 relative-L1 噪声：median {100 * result['noise_relative_l1']['median']:.2f}%，"
        f"p90 {100 * result['noise_relative_l1']['p90']:.2f}%。",
        "",
        "| closure | 环境光 median relative-L1 | p90 | median log-RGB L1 |",
        "|---|---:|---:|---:|",
    ]
    for config in configs:
        metrics = result["families"][config]
        lines.append(
            f"| {config} | {100 * metrics['relative_l1']['median']:.2f}% | "
            f"{100 * metrics['relative_l1']['p90']:.2f}% | {metrics['log_rgb_l1']['median']:.4f} |"
        )
    lines.extend(
        [
            "",
            "解析探针包含均匀光、阴天、太阳与天空以及多软箱光。真实探针使用固定的 Poly Haven CC0 清单，"
            "每个 HDRI 取四个方位旋转。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fitted closure packets under analytic HDR probes.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512")
    parser.add_argument("--oracle", type=Path, default=PROJECT_ROOT / "reports" / "oracle_v0_512")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "oracle_v0_512" / "ibl.md")
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--device", type=str)
    parser.add_argument("--hdri-manifest", type=Path)
    parser.add_argument("--hdri-dir", type=Path)
    parser.add_argument("--probe-file", type=Path)
    args = parser.parse_args()
    evaluate_ibl(
        args.dataset,
        args.oracle,
        args.output,
        configs=args.config or ["ggx-k3", "ltc-k3", "sg-k8"],
        device=args.device,
        hdri_manifest=args.hdri_manifest,
        hdri_dir=args.hdri_dir,
        probe_file=args.probe_file,
    )


if __name__ == "__main__":
    main()
