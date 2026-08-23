from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ncls.source_materials import ConstantBinding, GeometryBinding, OpenPBRMaterial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把原生 OpenPBR MaterialX 文档解析为 Falcor resolved-input source adapter"
    )
    parser.add_argument("input", type=Path, help="包含 open_pbr_surface 的原始 .mtlx")
    parser.add_argument("output", type=Path, help="输出 ncls.openpbr-material JSON")
    arguments = parser.parse_args()
    material = OpenPBRMaterial.from_materialx(arguments.input)
    unsupported = [
        name
        for name, binding in material.parameters.items()
        if not isinstance(binding, (ConstantBinding, GeometryBinding))
    ]
    if unsupported:
        raise RuntimeError(
            "Falcor resolved-input v1 不能静默烘焙 graph/texture binding：" + ", ".join(unsupported)
        )
    material = replace(material, source_document=str(arguments.input.resolve()))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(material.to_json(), encoding="utf-8")
    print(f"OpenPBR viewer source adapter: {arguments.output}")


if __name__ == "__main__":
    main()
