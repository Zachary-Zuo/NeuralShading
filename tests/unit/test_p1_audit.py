from __future__ import annotations

import numpy as np

from ncls.learning.evaluation.p1_audit import (
    _deadzone_row_metrics,
    _noise_row_metrics,
    _state_signed_metrics,
    _tail_stability,
)


def test_deadzone_metrics_distinguish_rgb_any_and_all_direction_counts() -> None:
    preclamp = np.asarray(
        ((-1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)), dtype=np.float64
    )
    target = np.ones((2, 3), dtype=np.float64)
    result = _deadzone_row_metrics(
        preclamp,
        target,
        np.asarray((1.0, 3.0)),
        np.ones(3),
    )
    assert result["negative_rgb_count"] == 4
    assert result["rgb_count"] == 6
    assert result["any_negative_direction_count"] == 2
    assert result["all_negative_direction_count"] == 1
    assert result["solid_angle_negative_rgb_numerator"] == 10.0
    assert result["solid_angle_rgb_denominator"] == 12.0
    assert result["solid_angle_any_direction_numerator"] == 4.0
    assert result["solid_angle_all_direction_numerator"] == 3.0
    assert np.all(result["normalized_negative_depth"] == 1.0)


def test_noise_metrics_match_generation_relative_se_definition() -> None:
    mean = np.ones((2, 4, 3), dtype=np.float64)
    standard_error = np.full_like(mean, 0.1)
    group_p95, weighted = _noise_row_metrics(
        mean,
        standard_error,
        np.ones((2, 4), dtype=np.float64),
    )
    assert np.allclose(group_p95, 0.1)
    assert np.allclose(weighted, 0.1)


def test_signed_energy_uses_ratio_of_sums_and_excludes_near_zero_wo() -> None:
    result = _state_signed_metrics(
        np.asarray((8.0, 4.0, 2.0)),
        np.asarray((10.0, 5.0, 2.5)),
        np.asarray((14.0, 1e-10)),
        np.asarray((17.5, 1e-20)),
    )
    assert np.isclose(result["energy_ratio"], 0.8)
    assert np.isclose(result["signed_relative_bias"], -0.2)
    assert all(np.isclose(value, 0.8) for value in result["channel_energy_ratio"].values())
    assert result["wo_included_count"] == 1
    assert result["wo_excluded_near_zero_count"] == 1
    assert np.isclose(result["signed_relative_bias_by_wo"]["median"], -0.2)


def test_tail_stability_records_numpy_p95_support_states() -> None:
    state_ids = [f"state-{index:02d}" for index in range(30)]
    values = np.arange(30, dtype=np.float64)
    result = _tail_stability(state_ids, values, iterations=1000, seed=7)
    assert np.isclose(result["p95"], 27.55)
    support = result["numpy_linear_quantile_support"]
    assert np.isclose(support["fractional_index"], 27.55)
    assert support["lower_state_id"] == "state-27"
    assert support["upper_state_id"] == "state-28"
    assert result["worst_states"][0] == {
        "state_id": "state-29",
        "directional_l1": 29.0,
    }
