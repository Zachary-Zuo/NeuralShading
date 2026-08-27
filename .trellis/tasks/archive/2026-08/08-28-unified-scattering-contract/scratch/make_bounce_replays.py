from __future__ import annotations

import json
import sys
from pathlib import Path


source = Path(sys.argv[1])
output_root = Path(sys.argv[2])
document = json.loads(source.read_text(encoding="utf-8"))
viewer_scene = document.get("viewer_scene")
if viewer_scene and not Path(viewer_scene).is_absolute():
    document["viewer_scene"] = str((source.parent / viewer_scene).resolve())
for value in map(int, sys.argv[3:]):
    output = output_root / f"bounce-{value}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    replay = dict(document)
    replay["reference_scene_max_bounces"] = value
    output.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
