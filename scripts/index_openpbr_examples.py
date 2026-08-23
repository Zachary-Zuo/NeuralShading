from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "external" / "OpenPBR"
OUTPUT = PROJECT_ROOT / "references" / "openpbr-1.1.1-v1" / "materials.json"
EXPECTED_REVISION = "f8d6d947dfae4c9b599965a86c22826ea7a8dbfb"
COVERAGE = {
    "open_pbr_carpaint": "coat",
    "open_pbr_aluminum_brushed": "anisotropic-metal",
    "open_pbr_velvet": "fuzz",
    "open_pbr_brass": "metal",
    "open_pbr_pearl": "coat-thin-film-subsurface",
    "open_pbr_glass": "transmission-volume",
    "open_pbr_soapbubble": "thin-film-delta-transmission",
}


def main() -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"OpenPBR revision mismatch: expected={EXPECTED_REVISION}, actual={revision}")
    if subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "status", "--short"], text=True, encoding="utf-8"
    ).strip():
        raise RuntimeError("external/OpenPBR must be a clean worktree")

    materials = []
    for path in sorted((SOURCE_ROOT / "examples").glob("*.mtlx")):
        content = path.read_bytes()
        root = ET.fromstring(content)
        surface = next((element for element in root if element.tag == "open_pbr_surface"), None)
        if surface is None:
            raise RuntimeError(f"official example has no open_pbr_surface: {path}")
        material = next((element for element in root if element.tag == "surfacematerial"), None)
        authored = tuple(item.get("name", "") for item in surface.findall("input"))
        materials.append(
            {
                "material_id": path.stem,
                "display_name": material.get("name", path.stem) if material is not None else path.stem,
                "document": f"examples/{path.name}",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "color_space": root.get("colorspace", "acescg"),
                "authored_parameters": authored,
                "coverage_role": COVERAGE.get(path.stem),
            }
        )
    value = {
        "schema_name": "ncls.openpbr-material-index",
        "schema_version": 1,
        "upstream": "aswf.openpbr@1.1.1",
        "revision": revision,
        "native_representation": "official MaterialX open_pbr_surface examples",
        "materials": materials,
    }
    OUTPUT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"indexed {len(materials)} official OpenPBR materials -> {OUTPUT}")


if __name__ == "__main__":
    main()
