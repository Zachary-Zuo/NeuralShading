from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .quality import QUALITY_SUITE, QUALITY_SUITE_DOCUMENT


def _read_report(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("format_name") != "quality-report" or value.get("suite") != QUALITY_SUITE:
        raise ValueError("comparison requires quality-v1 reports")
    if not value.get("valid") or value.get("evaluation_role") != "test":
        raise ValueError("comparison requires valid test reports")
    claimed = value.get("report_sha256")
    payload = dict(value)
    payload.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if claimed != actual:
        raise ValueError("quality report hash mismatch")
    return value


def _bootstrap_difference(
    baseline: tuple[np.ndarray, ...],
    candidate: tuple[np.ndarray, ...],
    statistic: Callable[[np.ndarray], float],
    *,
    iterations: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires matching nonempty state blocks")
    if any(left.shape != right.shape for left, right in zip(baseline, candidate, strict=True)):
        raise ValueError("paired bootstrap state blocks have different shapes")
    observed = float(
        statistic(np.concatenate(candidate)) - statistic(np.concatenate(baseline))
    )
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(baseline), size=len(baseline))
        candidate_sample = np.concatenate([candidate[item] for item in selected])
        baseline_sample = np.concatenate([baseline[item] for item in selected])
        samples[index] = statistic(candidate_sample) - statistic(baseline_sample)
    tail = 0.5 * (1.0 - confidence)
    low, high = np.quantile(samples, (tail, 1.0 - tail))
    conclusion = (
        "candidate-better" if high < 0.0
        else "baseline-better" if low > 0.0
        else "no-significant-difference"
    )
    return {
        "difference": observed,
        "confidence_interval": [float(low), float(high)],
        "confidence": confidence,
        "conclusion": conclusion,
    }


def compare_quality_reports(
    baseline_path: Path | str,
    candidate_path: Path | str,
    *,
    iterations: int | None = None,
    seed: int = 20260824,
) -> dict[str, Any]:
    comparison_config = QUALITY_SUITE_DOCUMENT["comparison"]
    minimum_iterations = int(comparison_config["minimum_bootstrap_iterations"])
    resolved_iterations = minimum_iterations if iterations is None else iterations
    confidence = float(comparison_config["confidence"])
    if resolved_iterations < minimum_iterations:
        raise ValueError(
            f"comparison requires at least {minimum_iterations} bootstrap iterations"
        )
    baseline = _read_report(baseline_path)
    candidate = _read_report(candidate_path)
    if baseline["data_id"] != candidate["data_id"]:
        raise ValueError("paired comparison requires the same data_id")
    baseline_states = baseline["states"]
    candidate_states = candidate["states"]
    if set(baseline_states) != set(candidate_states):
        raise ValueError("paired comparison requires exactly the same test states")
    state_ids = sorted(baseline_states)
    if len(state_ids) < 50:
        raise ValueError("formal comparison requires at least 50 matched test states")
    matched_fields = ("capacity", "steps", "seed", "dataset_selection")
    baseline_training = baseline.get("training")
    candidate_training = candidate.get("training")
    if not isinstance(baseline_training, dict) or not isinstance(candidate_training, dict):
        raise ValueError("comparison reports must include training provenance")
    mismatched = [
        field
        for field in matched_fields
        if baseline_training.get(field) != candidate_training.get(field)
    ]
    if mismatched:
        raise ValueError(f"comparison is not matched on training fields: {mismatched}")
    direction_a = tuple(np.asarray([baseline_states[state]["directional_l1"]]) for state in state_ids)
    direction_b = tuple(np.asarray([candidate_states[state]["directional_l1"]]) for state in state_ids)
    energy_a = tuple(
        np.asarray(baseline_states[state]["energy_relative_error_by_wo"], dtype=np.float64)
        for state in state_ids
    )
    energy_b = tuple(
        np.asarray(candidate_states[state]["energy_relative_error_by_wo"], dtype=np.float64)
        for state in state_ids
    )
    rng = np.random.default_rng(seed)
    statistics = {
        "directional_l1.state_median": (direction_a, direction_b, np.median),
        "directional_l1.state_p95": (
            direction_a,
            direction_b,
            lambda values: float(np.quantile(values, 0.95)),
        ),
        "energy_relative_error.state_wo_median": (energy_a, energy_b, np.median),
        "energy_relative_error.state_wo_p95": (
            energy_a,
            energy_b,
            lambda values: float(np.quantile(values, 0.95)),
        ),
    }
    result: dict[str, Any] = {
        "format_name": "comparison-report",
        "format_version": 1,
        "suite": dict(QUALITY_SUITE),
        "data_id": baseline["data_id"],
        "baseline_report": str(Path(baseline_path)),
        "candidate_report": str(Path(candidate_path)),
        "state_count": len(state_ids),
        "iterations": resolved_iterations,
        "seed": seed,
        "matched": {
            "data_id": True,
            "test_states": True,
            **{field: baseline_training.get(field) for field in matched_fields},
        },
        "statistics": {
            name: _bootstrap_difference(
                left,
                right,
                statistic,
                iterations=resolved_iterations,
                confidence=confidence,
                rng=rng,
            )
            for name, (left, right, statistic) in statistics.items()
        },
    }
    payload = json.dumps(
        result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    result["report_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return result


def write_comparison_report(path: Path | str, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
