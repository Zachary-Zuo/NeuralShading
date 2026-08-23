from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyexr
from PIL import Image, ImageDraw

from ncls.references.acceptance import linear_hdr_image_metrics, load_reference_acceptance
from generate_uv_sphere import generate_uv_sphere


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_linear_rgb(path: Path) -> np.ndarray:
    try:
        pixels = np.asarray(pyexr.read(str(path)), dtype=np.float32)
    except Exception as error:
        raise RuntimeError(f"无法读取线性 EXR: {path}: {error}") from error
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise RuntimeError(f"线性图像必须至少有三个通道: {path}")
    return pixels[..., :3]


def _sphere_mask(width: int, height: int, *, erosion_pixels: int) -> np.ndarray:
    x = 2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0
    y = 1.0 - 2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height
    screen_x, screen_y = np.meshgrid(x, y)
    tangent = math.tan(math.radians(45.0) * 0.5)
    directions = np.stack((screen_x * tangent, screen_y * tangent, -np.ones_like(screen_x)), axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    camera = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    b = np.sum(directions * camera, axis=-1)
    mask = b * b - (np.dot(camera, camera) - 1.0) >= 0.0
    for _ in range(erosion_pixels):
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(mask)
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            eroded &= padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        mask = eroded
    return mask


def _metrics(
    native: np.ndarray,
    falcor: np.ndarray,
    erosion_pixels: int,
    *,
    profile: str = "linear_hdr_image",
) -> dict[str, object]:
    if native.shape != falcor.shape:
        raise ValueError(f"图像尺寸不一致: native={native.shape}, Falcor={falcor.shape}")
    acceptance = load_reference_acceptance(PROJECT_ROOT / "references" / "acceptance.json")
    mask = _sphere_mask(native.shape[1], native.shape[0], erosion_pixels=erosion_pixels)
    gate = getattr(acceptance, profile)
    metrics = linear_hdr_image_metrics(native, falcor, gate, mask=mask)
    return {
        "pixel_count": metrics.pixel_count,
        "p95_relative_l1": metrics.p95_relative_l1,
        "linear_psnr_db": metrics.linear_psnr_db,
        "absolute_mae": metrics.absolute_mae,
        "peak_value": metrics.peak_value,
        "passed": metrics.passed,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear_to_srgb8(pixels: np.ndarray) -> np.ndarray:
    pixels = np.clip(pixels, 0.0, 1.0)
    encoded = np.where(
        pixels <= 0.0031308,
        12.92 * pixels,
        1.055 * np.power(pixels, 1.0 / 2.4) - 0.055,
    )
    return np.asarray(np.round(encoded * 255.0), dtype=np.uint8)


def _write_comparison_png(native: np.ndarray, falcor: np.ndarray, path: Path) -> None:
    height, width = native.shape[:2]
    geometry_mask = _sphere_mask(width, height, erosion_pixels=0)[..., None]
    native_display = np.where(geometry_mask, native, 0.0)
    falcor_display = np.where(geometry_mask, falcor, 0.0)
    difference = np.where(geometry_mask, np.abs(native - falcor) * 32.0, 0.0)
    panels = [
        _linear_to_srgb8(native_display),
        _linear_to_srgb8(falcor_display),
        _linear_to_srgb8(difference),
    ]
    header = 24
    image = Image.new("RGB", (width * 3, height + header), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    for index, (label, panel) in enumerate(zip(("MaterialX native", "Falcor same OBJ", "masked |difference| x32"), panels)):
        image.paste(Image.fromarray(panel, mode="RGB"), (index * width, header))
        draw.text((index * width + 6, 5), label, fill=(235, 235, 235))
    image.save(path)


def _replay(material: Path, sphere: Path, size: int) -> dict[str, object]:
    return {
        "format_name": "ncls.viewer-capture",
        "format_version": 2,
        "method_id": "",
        "bundle_root": str(PROJECT_ROOT / "artifacts" / "exports"),
        "source_material": str(material),
        "reference_geometry": str(sphere),
        "reference_geometry_sha256": _sha256(sphere),
        "environment": "",
        "resolution": [size * 2, size],
        "object_mode": 0,
        "reference_spp": 1,
        "reference_samples_per_frame": 1,
        "reference_max_depth": 24,
        "camera": {
            "target": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "pitch": 0.0,
            "distance": 3.0,
            "vertical_fov_degrees": 45.0,
        },
        "display": {"comparison_mode": 0, "split": 0.5, "exposure_ev": 0.0, "difference_scale": 8.0},
        "lighting": {
            "use_environment": False,
            "environment_rotation": 0.0,
            "environment_intensity": 1.0,
            "use_sun": True,
            "sun_direction": [0.36514837, 0.54772256, 0.73029673],
            "sun_intensity": 1.0,
            "sun_color": [1.0, 1.0, 1.0],
            "use_point": False,
            "point_position": [0.0, 0.0, 0.0],
            "point_intensity": 0.0,
            "point_color": [1.0, 1.0, 1.0],
            "use_rectangle": False,
            "rectangle_center": [0.0, 0.0, 0.0],
            "rectangle_axis_u": [1.0, 0.0, 0.0],
            "rectangle_axis_v": [0.0, 0.0, 1.0],
            "rectangle_intensity": 0.0,
            "rectangle_color": [1.0, 1.0, 1.0],
        },
    }


def _run_suite(arguments: argparse.Namespace) -> int:
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    requested = set(arguments.asset_id or [])
    if requested:
        assets = [asset for asset in assets if asset["asset_id"] in requested]
        missing = requested - {asset["asset_id"] for asset in assets}
        if missing:
            raise ValueError(f"未知 MaterialX asset_id: {', '.join(sorted(missing))}")
    if not assets:
        raise ValueError("MaterialX parity suite 没有待测资产")
    for path in (arguments.probe, arguments.viewer, arguments.materialx_root, arguments.asset_root):
        if not path.exists():
            raise FileNotFoundError(path)

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sphere = output / "common-sphere.obj"
    generate_uv_sphere(sphere, arguments.longitude_segments, arguments.latitude_segments)
    erosion = max(4, round(arguments.size / 60))
    process_environment = os.environ.copy()
    process_environment["PATH"] = os.pathsep.join((
        str(Path(sys.prefix) / "Library" / "bin"),
        process_environment.get("PATH", ""),
    ))

    core_material = (PROJECT_ROOT / "tests/fixtures/reference/materialx_constant.mtlx").resolve()
    core_native_path = output / "core-constant-native.exr"
    core_replay_path = output / "core-constant-replay.json"
    core_capture_path = output / "core-constant-falcor.json"
    core_replay_path.write_text(
        json.dumps(_replay(core_material, sphere, arguments.size), indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        str(arguments.probe), str(core_material), str(arguments.materialx_root), str(sphere), str(core_native_path),
        "--size", str(arguments.size),
        "--light-direction", "0.36514837", "0.54772256", "0.73029673",
    ], cwd=PROJECT_ROOT, env=process_environment, check=True, timeout=180)
    subprocess.run([
        str(arguments.viewer), "--replay", str(core_replay_path), "--headless", "--capture", str(core_capture_path),
    ], cwd=PROJECT_ROOT, env=process_environment, check=True, timeout=180)
    core_falcor_path = output / "core-constant-falcor-reference.exr"
    core_native = _read_linear_rgb(core_native_path)
    core_falcor = _read_linear_rgb(core_falcor_path)
    core_metrics = _metrics(core_native, core_falcor, erosion, profile="linear_hdr_image")
    _write_comparison_png(core_native, core_falcor, output / "core-constant-comparison.png")
    print(f"core-constant: {json.dumps(core_metrics, ensure_ascii=False)}")

    records: list[dict[str, object]] = []
    for asset in assets:
        asset_id = str(asset["asset_id"])
        material = (arguments.asset_root / str(asset["materialx_file"])).resolve()
        native_path = output / f"{asset_id}-native.exr"
        replay_path = output / f"{asset_id}-replay.json"
        capture_path = output / f"{asset_id}-falcor.json"
        replay_path.write_text(json.dumps(_replay(material, sphere, arguments.size), indent=2) + "\n", encoding="utf-8")
        subprocess.run([
            str(arguments.probe), str(material), str(arguments.materialx_root), str(sphere), str(native_path),
            "--size", str(arguments.size),
            "--light-direction", "0.36514837", "0.54772256", "0.73029673",
        ], cwd=PROJECT_ROOT, env=process_environment, check=True, timeout=180)
        subprocess.run([
            str(arguments.viewer), "--replay", str(replay_path), "--headless", "--capture", str(capture_path),
        ], cwd=PROJECT_ROOT, env=process_environment, check=True, timeout=180)
        falcor_path = output / f"{asset_id}-falcor-reference.exr"
        native_pixels = _read_linear_rgb(native_path)
        falcor_pixels = _read_linear_rgb(falcor_path)
        metrics = _metrics(native_pixels, falcor_pixels, erosion, profile="linear_hdr_textured_image")
        comparison_path = output / f"{asset_id}-comparison.png"
        _write_comparison_png(native_pixels, falcor_pixels, comparison_path)
        records.append({
            "asset_id": asset_id,
            "source_material": material.relative_to(PROJECT_ROOT).as_posix(),
            "source_material_sha256": _sha256(material),
            "native_image": native_path.name,
            "falcor_image": falcor_path.name,
            "comparison_png": comparison_path.name,
            "metrics": metrics,
        })
        print(f"{asset_id}: {json.dumps(metrics, ensure_ascii=False)}")

    acceptance = load_reference_acceptance(PROJECT_ROOT / "references" / "acceptance.json")
    core_gate = acceptance.linear_hdr_image
    textured_gate = acceptance.linear_hdr_textured_image
    report = {
        "schema_name": "ncls.materialx-falcor-image-parity",
        "schema_version": 1,
        "reference_id": "ncls.materialx-polyhaven@1",
        "native_implementation": "aswf.materialx@1.39.4-genglsl-float",
        "candidate_implementation": "ncls.falcor-materialx-standard-surface@1",
        "scene": {
            "geometry": f"generated UV sphere {arguments.longitude_segments}x{arguments.latitude_segments}",
            "geometry_file": sphere.name,
            "geometry_sha256": _sha256(sphere),
            "geometry_contract": "MaterialX native 与 Falcor 加载同一个 OBJ 文件",
            "camera": "eye=(0,0,3), target=(0,0,0), vertical_fov=45deg",
            "lighting": "directional wi=(0.36514837,0.54772256,0.73029673), linear RGB=(1,1,1)",
            "resolution": [arguments.size, arguments.size],
            "mask_erosion_pixels": erosion,
        },
        "core_probe": {
            "source_material": core_material.relative_to(PROJECT_ROOT).as_posix(),
            "profile": "linear_hdr_image",
            "metrics": core_metrics,
        },
        "gates": {
            "linear_hdr_image": {
                "luminance_floor": core_gate.luminance_floor,
                "p95_relative_l1_max": core_gate.p95_relative_l1_max,
                "linear_psnr_min_db": core_gate.linear_psnr_min_db,
                "max_absolute_mae": core_gate.max_absolute_mae,
            },
            "linear_hdr_textured_image": {
                "luminance_floor": textured_gate.luminance_floor,
                "p95_relative_l1_max": textured_gate.p95_relative_l1_max,
                "linear_psnr_min_db": textured_gate.linear_psnr_min_db,
                "max_absolute_mae": textured_gate.max_absolute_mae,
            },
        },
        "materials": records,
        "passed": bool(core_metrics["passed"]) and all(bool(record["metrics"]["passed"]) for record in records),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report: {report_path}")
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="比较上游 MaterialX float reference 与 Falcor 线性图像")
    parser.add_argument("native", type=Path, nargs="?")
    parser.add_argument("falcor", type=Path, nargs="?")
    parser.add_argument("--erosion-pixels", type=int, default=4)
    parser.add_argument("--diagnose-flips", action="store_true")
    parser.add_argument("--suite", action="store_true", help="运行全部锁定 Poly Haven MaterialX 资产")
    parser.add_argument("--asset-id", action="append")
    parser.add_argument("--size", type=int, default=240)
    parser.add_argument("--longitude-segments", type=int, default=256)
    parser.add_argument("--latitude-segments", type=int, default=128)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "references/materialx-polyhaven-v1/assets.json")
    parser.add_argument(
        "--asset-root", type=Path, default=PROJECT_ROOT / "data/source-materials/materialx-polyhaven/v1")
    parser.add_argument(
        "--materialx-root", type=Path, default=PROJECT_ROOT / "build/materialx-reference-install")
    parser.add_argument(
        "--probe", type=Path, default=PROJECT_ROOT / "build/materialx-probe/Release/ncls_materialx_probe.exe")
    parser.add_argument(
        "--viewer", type=Path,
        default=PROJECT_ROOT / "external/Falcor/build/windows-vs2022/bin/Release/NclsViewer.exe")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/validation/materialx-parity/suite")
    arguments = parser.parse_args()

    if arguments.suite:
        return _run_suite(arguments)
    if arguments.native is None or arguments.falcor is None:
        parser.error("compare 模式需要 native 和 falcor 两个 EXR 路径，或使用 --suite")

    native = _read_linear_rgb(arguments.native.resolve())
    falcor = _read_linear_rgb(arguments.falcor.resolve())
    results: dict[str, object] = {"same": _metrics(native, falcor, arguments.erosion_pixels)}
    if arguments.diagnose_flips:
        results.update({
            "flip_y": _metrics(native, falcor[::-1], arguments.erosion_pixels),
            "flip_x": _metrics(native, falcor[:, ::-1], arguments.erosion_pixels),
            "flip_xy": _metrics(native, falcor[::-1, ::-1], arguments.erosion_pixels),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if bool(results["same"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
