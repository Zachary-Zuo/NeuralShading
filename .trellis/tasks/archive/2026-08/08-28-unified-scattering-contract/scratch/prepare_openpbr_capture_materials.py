from __future__ import annotations

import sys
from pathlib import Path

from ncls.source_materials import OpenPBRMaterial


if len(sys.argv) < 3 or len(sys.argv[1:]) % 2:
    raise SystemExit("usage: prepare_openpbr_capture_materials.py SOURCE OUTPUT [SOURCE OUTPUT ...]")

for source_argument, output_argument in zip(sys.argv[1::2], sys.argv[2::2]):
    source = Path(source_argument).resolve()
    output = Path(output_argument).resolve()
    material = OpenPBRMaterial.from_materialx(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(material.to_json(), encoding="utf-8")
