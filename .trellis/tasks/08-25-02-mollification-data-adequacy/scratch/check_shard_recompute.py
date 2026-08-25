from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


manifest = json.loads(
    Path("artifacts/corpus/layer-stack-p1-mollification-v1.json").read_text(encoding="utf-8")
)
rows = []
for shard in manifest["shards"]:
    with h5py.File(Path(shard["uri"]), "r") as stream:
        mean = np.asarray(stream["responses/mean"], dtype=np.float64)
        variance = np.asarray(stream["responses/variance"], dtype=np.float64)
        replica_a = np.asarray(stream["responses/replica_mean_a"], dtype=np.float64)
        replica_b = np.asarray(stream["responses/replica_mean_b"], dtype=np.float64)
        reconstructed_mean = 0.5 * (replica_a + replica_b)
        reconstructed_variance = 0.5 * (
            (replica_a - reconstructed_mean) ** 2
            + (replica_b - reconstructed_mean) ** 2
        )
        target_se = 0.5 * np.abs(replica_a - replica_b)
        peak = np.max(np.abs(reconstructed_mean), axis=(2, 3), keepdims=True)
        relative = target_se / np.maximum(
            np.abs(reconstructed_mean), np.maximum(0.005 * peak, 1e-6)
        )
        expected_p95 = np.quantile(relative, 0.95, axis=(2, 3))
        expected_max = np.max(relative, axis=(2, 3))
        stored_p95 = np.asarray(stream["responses/relative_se_p95"], dtype=np.float64)
        stored_max = np.asarray(stream["responses/relative_se_max"], dtype=np.float64)
        rows.append(
            {
                "state_id": shard["state_id"],
                "mean_abs": float(np.max(np.abs(mean - reconstructed_mean))),
                "variance_abs": float(np.max(np.abs(variance - reconstructed_variance))),
                "variance_scale": float(np.max(np.abs(variance))),
                "p95_abs": float(np.max(np.abs(stored_p95 - expected_p95))),
                "max_abs": float(np.max(np.abs(stored_max - expected_max))),
            }
        )

for field in ("mean_abs", "variance_abs", "p95_abs", "max_abs"):
    print(field, max(rows, key=lambda row: row[field]))
