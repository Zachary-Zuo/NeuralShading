import json
from pathlib import Path

import numpy as np
import pyexr


root = Path("outputs/metal-spatial-probe-bronze-scratched/260905-220316-6185cc/eval/step-00000002-2d716c")
capture = json.loads((root / "capture.json").read_text(encoding="utf-8"))
assert all(slot["status"] == "ready" for slot in capture["slots"])
assert capture["slots"][0]["spp"] == capture["slots"][0]["target_spp"] == 128
result = {"capture": str(root / "capture.json"), "slots": capture["slots"], "linear": {}}
for key in ("slot_0_linear", "slot_1_linear", "comparison_linear", "difference_linear"):
    path = root / capture["files"][key]
    value = pyexr.read(path)
    finite = np.isfinite(value)
    assert finite.all(), path
    assert value.dtype == np.float32, path
    result["linear"][key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                              "nonfinite": int((~finite).sum()), "minimum": float(value.min()),
                              "maximum": float(value.max())}
target = Path(__file__).parents[1] / "research" / "spatial-capture-check.json"
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result["linear"], ensure_ascii=False))
