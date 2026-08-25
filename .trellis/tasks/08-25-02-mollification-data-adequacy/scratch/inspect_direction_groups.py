"""只读检查 adequacy 代表 state 的 v5 query group 方向布局。"""

from __future__ import annotations

import glob
import json

import h5py
import numpy as np


STATE_IDS = {
    "diffuse": "1b6dc2ae36e7fb076e1942242317b9af5b8bc723d71e17e5bccf33402403c49f",
    "narrow-conductor": "3d925881a55b0bbee135592d539b71dec694be51852793d74071d68d161ce5b5",
    "tail-conductor": "bd6de2e9d0cf5b32e6259d90af99b718983783cbfff1a0a9c3025a54ff3672db",
    "tail-diffuse-a": "6fff05aa8f142b26631bd354c21c871e900a75bbd0f33e8426d3e05d73b9a3e9",
    "tail-sheen": "4ebd9258461716ed23d523a0b221d015feb190270471e2a55b63c3a219fcb1e7",
    "tail-diffuse-b": "1796065779d0932fe7ded3cc2c40b84a8a19190dd7e68732f06bc518ae7fe54a",
}


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


result: dict[str, dict[str, object]] = {label: {} for label in STATE_IDS}
for path in glob.glob("data/reference-responses/layer-stack-p1-v1/*.h5"):
    with h5py.File(path, "r") as stream:
        ids = [_text(item) for item in stream["states/state_id"][:]]
        for label, state_id in STATE_IDS.items():
            if state_id not in ids:
                continue
            state_index = ids.index(state_id)
            selected = np.flatnonzero(stream["queries/state_index"][:] == state_index)
            if not len(selected):
                continue
            role = path.rsplit("-", 2)[-2] if "dense_slice" not in path else "dense_slice"
            views = np.asarray(stream["queries/wo"][selected], dtype=np.float64)
            result[label][role] = {
                "path": path.replace("\\", "/"),
                "view_count": len(views),
                "direction_count": int(stream.attrs["direction_count"]),
                "wo": np.round(views, 8).tolist() if role == "dense_slice" else None,
                "minimum_wo_z": float(np.min(views[:, 2])),
                "maximum_wo_z": float(np.max(views[:, 2])),
            }

print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
