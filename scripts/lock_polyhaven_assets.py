from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import urllib.request
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "references" / "materialx-polyhaven-v1" / "assets.json"
API_ROOT = "https://api.polyhaven.com"
USER_AGENT = "NeuralShading research importer/1.0 (Powered by Poly Haven)"
RESOLUTION = "4k"
SELECTED = (
    ("american_walnut_veneer", "wood-clean"),
    ("bark_brown_02", "wood-natural-rough"),
    ("denim_fabric", "fabric-woven"),
    ("curly_teddy_natural", "fabric-pile-texture"),
    ("rusty_metal_02", "metal-weathered"),
    ("metal_plate", "metal-industrial"),
    ("lichen_rock", "stone-organic-mixed"),
    ("monastery_stone_floor", "stone-man-made"),
)


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _file_records(value: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if {"url", "size", "md5"}.issubset(item):
                basename = Path(urlparse(str(item["url"])).path).name
                previous = result.get(basename)
                if previous is not None and previous != item:
                    raise RuntimeError(f"Poly Haven API has ambiguous file identity: {basename}")
                result[basename] = item
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return result


def _native_references(document: bytes) -> tuple[str, ...]:
    root = ET.fromstring(document)
    references = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "input" and element.attrib.get("type") == "filename":
            value = element.attrib.get("value")
            if value and value not in references:
                references.append(value)
    return tuple(references)


def _semantic(path: str) -> str:
    name = Path(path).name.lower()
    for marker, semantic in (
        ("_diff_", "base-color-srgb"),
        ("_rough_", "roughness-raw"),
        ("_nor_gl_", "normal-opengl-tangent-raw"),
        ("_disp_", "normalized-displacement-raw"),
        ("_ao_", "ambient-occlusion-raw"),
        ("_metal_", "metalness-raw"),
    ):
        if marker in name:
            return semantic
    return "native-materialx-resource"


def main() -> None:
    metadata = _get_json(f"{API_ROOT}/assets?type=textures")
    assets = []
    total_size = 0
    for asset_id, coverage_role in SELECTED:
        asset_metadata = metadata[asset_id]
        files = _get_json(f"{API_ROOT}/files/{asset_id}")
        package = files.get("mtlx", {}).get(RESOLUTION, {}).get("mtlx")
        if not package:
            raise RuntimeError(f"Poly Haven asset has no {RESOLUTION} MaterialX package: {asset_id}")
        document = _get_bytes(package["url"])
        if len(document) != int(package["size"]) or hashlib.md5(document).hexdigest() != package["md5"]:
            raise RuntimeError(f"Poly Haven MaterialX document identity mismatch: {asset_id}")
        records = [
            {
                "path": f"{asset_id}/{Path(package['url']).name}",
                "source_url": package["url"],
                "size": package["size"],
                "md5": package["md5"],
                "semantic": "materialx-document",
            }
        ]
        available_files = _file_records(files)
        for include_path in _native_references(document):
            include = available_files.get(Path(include_path).name)
            if include is None:
                raise RuntimeError(f"MaterialX document dependency is absent from Poly Haven API: {include_path}")
            records.append(
                {
                    "path": f"{asset_id}/{include_path}",
                    "source_url": include["url"],
                    "size": include["size"],
                    "md5": include["md5"],
                    "semantic": _semantic(include_path),
                }
            )
        total_size += sum(int(record["size"]) for record in records)
        assets.append(
            {
                "asset_id": asset_id,
                "name": asset_metadata["name"],
                "coverage_role": coverage_role,
                "category": asset_metadata.get("category"),
                "category_id": asset_metadata.get("category_id"),
                "categories": asset_metadata.get("categories", []),
                "authors": asset_metadata.get("authors", {}),
                "files_hash": asset_metadata["files_hash"],
                "max_resolution_px": asset_metadata["max_resolution"],
                "physical_size_mm": asset_metadata["dimensions"],
                "materialx_file": records[0]["path"],
                "files": records,
            }
        )
    manifest = {
        "schema_name": "ncls.polyhaven-materialx-assets",
        "schema_version": 1,
        "provider": "Poly Haven",
        "provider_url": "https://polyhaven.com",
        "api_url": API_ROOT,
        "api_credit": "Powered by Poly Haven",
        "license": "CC0-1.0",
        "resolution": RESOLUTION,
        "materialx_reference": "aswf.materialx@1.39.4",
        "native_semantics": {
            "closure": "the standard_surface and displacement graphs in each native MaterialX document",
            "texture_coordinates": "MaterialX texcoord index 0 with document-authored connections",
            "normal": "OpenGL tangent-space normal through the native normalmap node",
            "displacement": "normalized source height interpreted by each native MaterialX displacement node and scale",
            "physical_size": "Poly Haven API dimensions in millimeters",
        },
        "total_size": total_size,
        "assets": assets,
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"locked {len(assets)} Poly Haven MaterialX assets ({total_size} bytes) -> {OUTPUT}")


if __name__ == "__main__":
    main()
