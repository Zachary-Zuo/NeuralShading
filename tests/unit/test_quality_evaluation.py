from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ncls.learning.evaluation import (
    QUALITY_SUITE,
    build_quality_report,
    compare_quality_reports,
    finalize_quality_report,
    quality_metric_rows,
    write_quality_report,
)
from ncls.learning.evaluation.benchmark import _measure


def _batch() -> dict[str, np.ndarray]:
    wi = np.asarray(
        [
            [[0.0, 0.0, 1.0], [0.8, 0.0, 0.6], [0.0, 0.8, 0.6], [-0.8, 0.0, 0.6]],
            [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.0, 0.6, 0.8], [-0.6, 0.0, 0.8]],
        ],
        dtype=np.float32,
    )
    f = np.asarray(
        [
            [[1.0, 0.5, 0.25], [2.0, 1.0, 0.5], [0.5, 0.25, 0.125], [0.2, 0.1, 0.05]],
            [[0.8, 0.4, 0.2], [0.4, 0.2, 0.1], [1.6, 0.8, 0.4], [0.1, 0.05, 0.025]],
        ],
        dtype=np.float32,
    )
    return {
        "state_index": np.asarray([0, 1], dtype=np.int64),
        "wi": wi,
        "mean": f * np.abs(wi[..., 2:3]),
        "standard_error": np.full_like(f, 1e-3),
        "solid_angle_weight": np.full((2, 4), 2.0 * np.pi / 4, dtype=np.float32),
        "wo": np.asarray(((0.0, 0.0, 1.0), (0.6, 0.0, 0.8)), dtype=np.float32),
        "reciprocal_mean": f * np.asarray((1.0, 0.8), dtype=np.float32)[:, None, None],
        "f": f,
    }


def _report(scale: float = 1.0):
    batch = _batch()
    f = batch.pop("f")
    rows = quality_metric_rows(f * scale, batch, f * scale)
    return build_quality_report(
        rows,
        state_ids=np.asarray(["state-a", "state-b"], dtype=object),
        family_ids=np.asarray(["layer-stack", "layer-stack"], dtype=object),
        structure_family_ids=np.asarray(["layers-1", "layers-2"], dtype=object),
        difficulty_classes=np.asarray(["W", "S"], dtype=object),
        difficulty_tags=((), ("M",)),
        evaluation_cohorts=np.asarray(["g2", "g2s"], dtype=object),
        data_id="a" * 64,
        evaluation_role="test",
        provenance_checks={
            "dataset_hash_verified": True,
            "checkpoint_recovered": True,
            "fitted_state_train_only": True,
        },
    )


def test_quality_v1_applies_cosine_once_and_has_four_layers() -> None:
    report = _report()
    assert report["valid"]
    assert report["suite"] == QUALITY_SUITE
    assert report["sanity"]["passed"]
    assert report["primary"]["directional_l1_by_state"]["maximum"] < 1e-7
    assert report["primary"]["energy_relative_error_by_state_wo"]["maximum"] < 1e-7
    assert set(report["scorecard"]) >= {
        "log_l1",
        "peak_support_angle_degrees",
        "peak_ratio_log_error",
        "top_5_percent_energy_recall",
        "breakdowns",
    }
    assert set(report["diagnostics"]) == {
        "absolute_error_sum",
        "reference_se_sum",
        "model_error_over_reference_se",
    }
    assert report["scorecard"]["breakdowns"]["difficulty_tags"]["M"]
    assert report["scorecard"]["breakdowns"]["difficulty"]["W"][
        "directional_l1_by_state"
    ]["maximum"] < 1e-7
    assert len(report["states"]["state-a"]["energy_relative_error_by_wo"]) == 1
    assert report["scorecard"]["source_aware_reciprocity_deviation"]["maximum"] < 1e-7


def test_sanity_is_the_only_invalidating_layer() -> None:
    batch = _batch()
    prediction = batch.pop("f")
    prediction[0, 0, 0] = -0.1
    rows = quality_metric_rows(prediction, batch, prediction)
    report = build_quality_report(
        rows,
        state_ids=np.asarray(["state-a", "state-b"], dtype=object),
        family_ids=np.asarray(["layer-stack", "layer-stack"], dtype=object),
        structure_family_ids=np.asarray(["layers-1", "layers-2"], dtype=object),
        difficulty_classes=np.asarray(["W", "S"], dtype=object),
        difficulty_tags=((), ("M",)),
        evaluation_cohorts=np.asarray(["g2", "g2s"], dtype=object),
        data_id="a" * 64,
        evaluation_role="test",
    )
    assert not report["valid"]
    assert not report["sanity"]["checks"]["color_contract"]
    assert "primary" in report and "scorecard" in report and "diagnostics" in report


def test_state_block_paired_bootstrap_uses_matched_test_states(tmp_path) -> None:
    baseline = _report(1.25)
    candidate = _report(1.05)
    for report in (baseline, candidate):
        originals = list(report["states"].values())
        report["states"] = {
            f"state-{index:03d}": dict(originals[index % len(originals)])
            for index in range(30)
        }
        report["training"] = {
            "stage": "P1",
            "capacity": "S",
            "steps": 25000,
            "seed": 17,
            "dataset_selection": {},
        }
    baseline = finalize_quality_report(baseline)
    candidate = finalize_quality_report(candidate)
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    write_quality_report(baseline_path, baseline)
    write_quality_report(candidate_path, candidate)
    comparison = compare_quality_reports(
        baseline_path,
        candidate_path,
        iterations=1000,
        seed=17,
    )
    assert comparison["state_count"] == 30
    assert comparison["iterations"] == 1000
    assert all(
        value["difference"] < 0.0 for value in comparison["statistics"].values()
    )
    assert json.loads(baseline_path.read_text(encoding="utf-8"))["suite"] == QUALITY_SUITE
    assert comparison["suite"] == QUALITY_SUITE
    assert "energy_relative_error.state_wo_p95" in comparison["statistics"]

    candidate_with_capacity_change = dict(candidate)
    candidate_with_capacity_change["training"] = {
        **candidate["training"],
        "capacity": "M",
    }
    candidate_with_capacity_change = finalize_quality_report(
        candidate_with_capacity_change
    )
    write_quality_report(candidate_path, candidate_with_capacity_change)
    with pytest.raises(ValueError, match="declared varied fields"):
        compare_quality_reports(baseline_path, candidate_path, iterations=1000, seed=17)
    capacity_comparison = compare_quality_reports(
        baseline_path,
        candidate_path,
        iterations=1000,
        seed=17,
        varied_fields=("capacity",),
    )
    assert capacity_comparison["matched"]["varied_training_fields"] == {
        "capacity": {"baseline": "S", "candidate": "M"}
    }


def test_cpu_query_benchmark_reports_finite_latency() -> None:
    value = torch.ones((1, 3), dtype=torch.float32)
    timing, prediction = _measure(
        lambda: value * 2.0,
        torch.device("cpu"),
        warmup=1,
        iterations=3,
    )
    assert timing["synchronized_wall_ms"]["median"] >= 0.0
    assert torch.equal(prediction, value * 2.0)
