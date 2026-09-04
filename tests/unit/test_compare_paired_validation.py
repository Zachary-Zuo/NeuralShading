from __future__ import annotations

import json
from pathlib import Path

from tools.learning.compare_paired_validation import _paired_bootstrap, compare


def test_paired_bootstrap_reports_direction_and_is_deterministic() -> None:
    first = _paired_bootstrap(
        (-0.5, -0.4, -0.3, -0.2), seed=17, replicates=2_000
    )
    second = _paired_bootstrap(
        (-0.5, -0.4, -0.3, -0.2), seed=17, replicates=2_000
    )

    assert first == second
    assert first["candidate_better"] is True
    assert first["baseline_better"] is False
    assert first["bootstrap_mean_delta_ci95"][1] < 0.0


def test_compare_supports_paired_rows_at_different_milestones(
    tmp_path: Path,
) -> None:
    metrics = (
        "validation/loss/appearance",
        "validation/appearance/log_rgb",
        "validation/appearance/linear_rgb",
        "validation/appearance/chroma",
        "validation/appearance/peak_rgb",
        "validation/appearance/spatial_gradient",
    )
    path = tmp_path / "metrics.jsonl"
    rows = []
    for step, offset in ((128, 0.5), (256, 0.1)):
        for index in range(4):
            rows.append(
                {
                    "record_kind": "validation",
                    "step": step,
                    "training_config_sha256": "config",
                    **{name: offset + index * 0.01 for name in metrics},
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = compare(
        path,
        path,
        baseline_step=128,
        candidate_step=256,
        seed=29,
        replicates=2_000,
    )

    assert result["baseline_step"] == 128
    assert result["candidate_step"] == 256
    assert all(
        metric["candidate_better"] for metric in result["metrics"].values()
    )
