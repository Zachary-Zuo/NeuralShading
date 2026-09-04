from __future__ import annotations

from tools.learning.compare_paired_validation import _paired_bootstrap


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
