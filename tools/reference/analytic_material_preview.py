from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ncls.source_materials import MerlBrdfReference, MerlMaterial, OpenPBRMaterial, OpenPBRReference


OPENPBR_DEFAULTS = (
    "open_pbr_carpaint",
    "open_pbr_aluminum_brushed",
    "open_pbr_velvet",
    "open_pbr_pearl",
)
MERL_DEFAULTS = ("alum-bronze", "blue-metallic-paint", "beige-fabric", "red-plastic")
ACESCG_TO_LINEAR_SRGB = np.asarray(
    (
        (1.70505099, -0.62179212, -0.08325887),
        (-0.13025642, 1.14080474, -0.01054832),
        (-0.02400336, -0.12896898, 1.15297233),
    ),
    dtype=np.float64,
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _sphere(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    y = 1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0
    grid_x, grid_y = np.meshgrid(x, y)
    mask = grid_x * grid_x + grid_y * grid_y <= 1.0
    normals = np.stack((grid_x[mask], grid_y[mask], np.sqrt(1.0 - grid_x[mask] ** 2 - grid_y[mask] ** 2)), axis=1)
    seed = np.broadcast_to(np.asarray((1.0, 0.0, 0.0)), normals.shape).copy()
    near_x = np.abs(normals[:, 0]) > 0.95
    seed[near_x] = (0.0, 1.0, 0.0)
    tangents = seed - np.sum(seed * normals, axis=1, keepdims=True) * normals
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    bitangents = np.cross(normals, tangents)
    return mask, normals, tangents, bitangents


def _local(directions: np.ndarray, tangents: np.ndarray, bitangents: np.ndarray, normals: np.ndarray) -> np.ndarray:
    world = np.broadcast_to(directions, normals.shape)
    return np.stack(
        (np.sum(world * tangents, axis=1), np.sum(world * bitangents, axis=1), np.sum(world * normals, axis=1)),
        axis=1,
    )


def _write_preview(path: Path, mask: np.ndarray, response: np.ndarray, exposure: float) -> None:
    linear = np.zeros((*mask.shape, 3), dtype=np.float64)
    linear[:] = (0.055, 0.055, 0.065)
    color = np.maximum(response * exposure, 0.0)
    color = np.clip((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0)
    color = np.where(color <= 0.0031308, color * 12.92, 1.055 * np.power(color, 1.0 / 2.4) - 0.055)
    linear[mask] = color
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(linear * 255.0).astype(np.uint8), "RGB").save(path)


def _reflection_lighting() -> tuple[tuple[np.ndarray, float], ...]:
    return (
        (_normalize(np.asarray((-0.35, 0.45, 0.82), dtype=np.float64)), 3.0),
        (_normalize(np.asarray((0.65, -0.2, 0.74), dtype=np.float64)), 0.65),
    )


def _openpbr_lighting() -> tuple[tuple[np.ndarray, float], ...]:
    return _reflection_lighting() + (
        (_normalize(np.asarray((0.15, 0.1, -0.98), dtype=np.float64)), 1.0),
    )


def _render_openpbr(root: Path, material_id: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    source = root / "external" / "OpenPBR" / "examples" / f"{material_id}.mtlx"
    if not source.is_file():
        raise ValueError(f"未知 OpenPBR 官方材质：{material_id}")
    material = OpenPBRMaterial.from_materialx(source)
    reference = OpenPBRReference(root / "build" / "openpbr-probe" / "Release" / "ncls_openpbr_probe.exe")
    mask, normals, tangents, _ = _sphere(width, height)
    view = np.broadcast_to(np.asarray((0.0, 0.0, 1.0), dtype=np.float32), normals.shape)
    geometries = tuple(
        {"N": normal, "T": tangent, "coat_N": normal, "coat_T": tangent}
        for normal, tangent in zip(normals, tangents, strict=True)
    )
    response = np.zeros_like(normals)
    for light, intensity in _openpbr_lighting():
        lights = np.broadcast_to(light.astype(np.float32), normals.shape)
        value = reference.evaluate(material, view, lights, geometries=geometries).response_cos
        response += value * intensity
    if material.color_space.lower() == "acescg":
        response = response @ ACESCG_TO_LINEAR_SRGB.T
    return mask, response


def _merl_index(root: Path) -> dict[str, str]:
    path = root / "references" / "merl-brdf-v1" / "materials.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["material_id"]): str(item["table_uri"]) for item in value["materials"]}


def _render_merl(root: Path, material_id: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    index = _merl_index(root)
    if material_id not in index:
        raise ValueError(f"未知 MERL 材质：{material_id}")
    reference = MerlBrdfReference(
        MerlMaterial(material_id, index[material_id]),
        root / "data" / "source-materials" / "merl-brdf" / "v1",
    )
    mask, normals, tangents, bitangents = _sphere(width, height)
    world_view = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    local_view = _local(world_view, tangents, bitangents, normals)
    response = np.zeros_like(normals)
    for light, intensity in _reflection_lighting():
        local_light = _local(light, tangents, bitangents, normals)
        value = reference.evaluate(local_view, local_light).response_cos
        response += np.maximum(value, 0.0) * intensity
    return mask, response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染 OpenPBR 或 MERL 原始源材质的离线球体预览")
    parser.add_argument("family", choices=("openpbr", "merl"))
    parser.add_argument("material_ids", nargs="*")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.exposure <= 0:
        raise SystemExit("尺寸和 exposure 必须为正数")
    root = Path(__file__).resolve().parents[2]
    defaults: tuple[str, ...]
    renderer: Callable[[Path, str, int, int], tuple[np.ndarray, np.ndarray]]
    if args.family == "openpbr":
        defaults, renderer = OPENPBR_DEFAULTS, _render_openpbr
    else:
        defaults, renderer = MERL_DEFAULTS, _render_merl
    material_ids = tuple(args.material_ids) or defaults
    output_dir = args.output_dir or Path("artifacts") / "reference-previews" / args.family
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    for material_id in material_ids:
        mask, response = renderer(root, material_id, args.width, args.height)
        output = output_dir / f"{material_id}.png"
        _write_preview(output, mask, response, args.exposure)
        print(f"{material_id}: {output}")


if __name__ == "__main__":
    main()
