"""诊断单个 frozen supplement target 在高 sample cap 下的双 replica moments。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ncls.data.contract import QueryPlan, QueryRole, SurfaceSample
from ncls.data.mollification import (
    _make_layer_stack_provider,
    _reference_relative_standard_error,
    _seed32,
    load_mollification_supplement_anchor_lock,
    mollification_cone_directions,
)


protocol, anchor_lock, manifest, supplement_lock = load_mollification_supplement_anchor_lock(
    "configs/corpus/layer-stack-p1-mollification-adequacy-v1.json",
    "artifacts/mollification/layer-stack-p1-v1-anchor-lock.json",
    "artifacts/mollification/layer-stack-p1-v1-audit-c.json",
    "artifacts/mollification/layer-stack-p1-mollification-v1-reference-se-anchor-lock.json",
)
state_id = "1796065779d0932fe7ded3cc2c40b84a8a19190dd7e68732f06bc518ae7fe54a"
state = next(item for item in supplement_lock["states"] if item["state_id"] == state_id)
view_index = 2
level_index = 0
view = state["views"][view_index]
jitter = mollification_cone_directions(
    view["wo"],
    protocol.document["curriculum"]["radius_degrees"][level_index],
    256,
    _seed32(protocol.document["seed"], state_id, view_index, level_index, "supplement-cone"),
)
lights = np.asarray(view["wi"], dtype=np.float32)
plan = QueryPlan(
    jitter,
    np.broadcast_to(lights[None], (256, 64, 3)).copy(),
    np.ones((256, 64), dtype=np.float32),
    np.ones((256, 64), dtype=np.float32),
    "mollification-supplement-upper-cap-v1",
    _seed32(protocol.document["seed"], state_id, view_index, level_index, "supplement-reference"),
    np.full(256, int(QueryRole.TRAIN), dtype=np.uint8),
)
provider, states = _make_layer_stack_provider(
    protocol, manifest, (state_id,), fixed_samples_per_replica=65536
)
rows = []
targets = {}
try:
    for paths in (32768, 65536):
        evaluated = provider.evaluate_fixed(
            states[state_id], (SurfaceSample(),), plan, samples_per_replica=paths
        )
        replica_a = np.mean(evaluated.replica_mean_a[0], axis=0, dtype=np.float64)
        replica_b = np.mean(evaluated.replica_mean_b[0], axis=0, dtype=np.float64)
        mean = 0.5 * (replica_a + replica_b)
        se = 0.5 * np.abs(replica_a - replica_b)
        relative = _reference_relative_standard_error(
            mean, se, group_axes=(0, 1), absolute_floor=1e-6
        )
        maximum = np.unravel_index(int(np.argmax(relative)), relative.shape)
        targets[paths] = (replica_a, replica_b, mean)
        rows.append(
            {
                "paths": paths,
                "p95": float(np.quantile(relative, 0.95)),
                "maximum": float(np.max(relative)),
                "maximum_index": list(maximum),
                "mean_at_maximum": float(mean[maximum]),
                "replica_a_at_maximum": float(replica_a[maximum]),
                "replica_b_at_maximum": float(replica_b[maximum]),
                "group_peak": float(np.max(np.abs(mean))),
            }
        )
finally:
    provider.close()
for name, index in (("replica_a", 0), ("replica_b", 1), ("mean", 2)):
    rows.append(
        {
            "comparison": name,
            "maximum_absolute_difference": float(
                np.max(np.abs(targets[65536][index] - targets[32768][index]))
            ),
        }
    )
output = Path("artifacts/mollification/reference-cap-diagnostic.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(json.dumps(rows, indent=2))
