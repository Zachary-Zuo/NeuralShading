"""只读检查 P1 corpus 中的 LayerStack 原生 state 元数据。"""

from __future__ import annotations

import glob
import json

import h5py


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


seen: dict[str, dict[str, object]] = {}
for path in glob.glob("data/reference-responses/layer-stack-p1-v1/*.h5"):
    with h5py.File(path, "r") as stream:
        for index, raw_id in enumerate(stream["states/state_id"][:]):
            state_id = _text(raw_id)
            if state_id in seen:
                continue
            begin = int(stream["states/payload_offsets"][index])
            end = int(stream["states/payload_offsets"][index + 1])
            payload = json.loads(bytes(stream["states/payload_blob"][begin:end]))
            seen[state_id] = {
                "family": _text(stream["states/structure_family_id"][index]),
                "difficulty": _text(stream["states/difficulty_class"][index]),
                "tags": json.loads(_text(stream["states/difficulty_tags_json"][index])),
                "split": int(stream["states/split"][index]),
                "payload": payload,
            }

for state_id, row in sorted(seen.items()):
    nodes = row["payload"]["nodes"]
    interface_rows = []
    for node in nodes:
        operation = node.get("operation", {}).get("name")
        if not node.get("id", "").startswith("interface-"):
            continue
        parameters = node.get("parameters", {})
        compact = {name: value.get("value") for name, value in parameters.items()}
        interface_rows.append({"operation": operation, "parameters": compact})
    print(
        state_id,
        row["family"],
        row["difficulty"],
        row["tags"],
        "split=",
        row["split"],
        "interfaces=",
        json.dumps(interface_rows, ensure_ascii=False, sort_keys=True),
    )
