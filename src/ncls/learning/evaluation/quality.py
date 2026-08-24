from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ncls.paths import PROJECT_ROOT


QUALITY_SUITE_NAME = "quality-v1"


def _load_quality_suite() -> tuple[dict[str, Any], str]:
    path = PROJECT_ROOT / "configs" / "evaluation" / "quality-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {
        "schema", "name", "sanity", "primary", "checkpoint_selection", "comparison"
    }:
        raise ValueError("quality-v1 fields do not match the fixed suite")
    if value.get("schema") != {"name": "quality-suite", "version": 1}:
        raise ValueError("unsupported quality suite schema")
    if value.get("name") != QUALITY_SUITE_NAME:
        raise ValueError("the fixed quality suite must be named quality-v1")
    if value.get("sanity") != {
        "invalidate_on_failure": True,
        "checks": [
            "dataset-hash", "split-leak", "checkpoint-recovery",
            "train-only-fitted-state", "finite-output", "family-color-contract",
        ],
    }:
        raise ValueError("quality-v1 sanity contract does not match the implementation")
    primary = value.get("primary", {})
    if (
        set(primary) != {"directional", "energy", "report", "reference_lines"}
        or primary.get("directional") != "solid-angle-normalized-l1-by-state"
        or primary.get("energy") != "integrated-relative-error-by-state-wo"
        or primary.get("report") != ["median", "p95"]
        or primary.get("reference_lines") != {
            "directional_state_median": 0.05,
            "directional_state_p95": 0.15,
            "energy_median": 0.03,
        }
    ):
        raise ValueError("quality-v1 primary metrics do not match the implementation")
    if value.get("checkpoint_selection") != {
        "metric": "directional_state_median",
        "tie_break": "directional_state_p95",
    }:
        raise ValueError("quality-v1 checkpoint selection is unsupported")
    comparison = value.get("comparison", {})
    if comparison != {
        "resampling_unit": "state",
        "minimum_bootstrap_iterations": 1000,
        "confidence": 0.95,
    }:
        raise ValueError("quality-v1 comparison contract is unsupported")
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return value, hashlib.sha256(payload.encode("utf-8")).hexdigest()


QUALITY_SUITE_DOCUMENT, QUALITY_SUITE_SHA256 = _load_quality_suite()
QUALITY_SUITE = {
    "name": QUALITY_SUITE_NAME,
    "sha256": QUALITY_SUITE_SHA256,
}


def finalize_quality_report(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value.pop("report_sha256", None)
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    value["report_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return value


def summarize(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data):
        raise ValueError("metric summary requires a nonempty vector")
    return {
        "minimum": float(np.min(data)),
        "p5": float(np.quantile(data, 0.05)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "maximum": float(np.max(data)),
    }


def quality_metric_rows(
    prediction_f: np.ndarray,
    batch: Mapping[str, np.ndarray],
    reciprocal_prediction_f: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """把 evaluator 的线性 `f` 变成 HDF5 measure，并返回逐 `(state, wo)` 行。"""

    prediction = np.asarray(prediction_f, dtype=np.float64)
    target = np.asarray(batch["mean"], dtype=np.float64)
    standard_error = np.asarray(batch["standard_error"], dtype=np.float64)
    wi = np.asarray(batch["wi"], dtype=np.float64)
    weights = np.asarray(batch["solid_angle_weight"], dtype=np.float64)[..., None]
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("quality suite expects prediction_f and target [group,direction,RGB]")
    if standard_error.shape != target.shape or wi.shape != target.shape:
        raise ValueError("quality suite batch shapes disagree")
    finite = np.isfinite(prediction)
    safe_f = np.where(finite, prediction, 0.0)
    prediction_y = safe_f * np.abs(wi[..., 2:3])
    absolute = np.abs(prediction_y - target)
    direction_numerator = np.sum(absolute * weights, axis=(1, 2))
    direction_denominator = np.maximum(
        np.sum(np.abs(target) * weights, axis=(1, 2)), 1e-12
    )
    target_energy = np.sum(target * weights, axis=1)
    predicted_energy = np.sum(prediction_y * weights, axis=1)
    energy_error = np.sum(np.abs(predicted_energy - target_energy), axis=1) / np.maximum(
        np.sum(np.abs(target_energy), axis=1), 1e-12
    )

    target_magnitude = np.sum(np.abs(target), axis=-1)
    predicted_magnitude = np.sum(np.abs(prediction_y), axis=-1)
    target_peak = np.max(target_magnitude, axis=1)
    predicted_peak = np.max(predicted_magnitude, axis=1)
    peak_ratio = predicted_peak / np.maximum(target_peak, 1e-12)
    predicted_peak_index = np.argmax(predicted_magnitude, axis=1)
    peak_support_angle = np.empty(len(target), dtype=np.float64)
    for row in range(len(target)):
        support = target_magnitude[row] >= 0.95 * target_peak[row]
        directions = wi[row, support]
        predicted_direction = wi[row, predicted_peak_index[row]]
        cosine = np.clip(directions @ predicted_direction, -1.0, 1.0)
        peak_support_angle[row] = np.degrees(np.arccos(np.max(cosine)))

    top_count = max(1, int(np.ceil(target.shape[1] * 0.05)))
    target_contribution = target_magnitude * weights[..., 0]
    predicted_contribution = predicted_magnitude * weights[..., 0]
    top_recall = np.empty(len(target), dtype=np.float64)
    for row in range(len(target)):
        target_top = np.argpartition(target_contribution[row], -top_count)[-top_count:]
        predicted_top = np.argpartition(predicted_contribution[row], -top_count)[-top_count:]
        overlap = np.intersect1d(target_top, predicted_top, assume_unique=False)
        top_recall[row] = np.sum(target_contribution[row, overlap]) / max(
            np.sum(target_contribution[row, target_top]), 1e-12
        )

    scale = 0.01 * np.max(np.abs(target), axis=(1, 2), keepdims=True) + 1e-6
    log_error = np.mean(np.abs(
        np.log1p(np.maximum(prediction_y, 0.0) / scale)
        - np.log1p(np.maximum(target, 0.0) / scale)
    ), axis=(1, 2))
    absolute_sum = np.sum(absolute, axis=(1, 2))
    se_sum = np.sum(np.abs(standard_error), axis=(1, 2))
    result = {
        "state_index": np.asarray(batch["state_index"], dtype=np.int64),
        "direction_numerator": direction_numerator,
        "direction_denominator": direction_denominator,
        "energy_relative_error": energy_error,
        "log_l1": log_error,
        "peak_support_angle_degrees": peak_support_angle,
        "peak_ratio_log_error": np.abs(np.log(np.maximum(peak_ratio, 1e-12))),
        "top_5_percent_energy_recall": top_recall,
        "absolute_error_sum": absolute_sum,
        "reference_se_sum": se_sum,
        "model_error_over_reference_se": absolute_sum / np.maximum(se_sum, 1e-12),
        "finite_rate": np.mean(finite, axis=(1, 2)),
        "minimum_f": np.min(safe_f, axis=(1, 2)),
    }
    if reciprocal_prediction_f is not None:
        reciprocal_f = np.asarray(reciprocal_prediction_f, dtype=np.float64)
        reciprocal_target = np.asarray(batch["reciprocal_mean"], dtype=np.float64)
        if reciprocal_f.shape != target.shape or reciprocal_target.shape != target.shape:
            raise ValueError("reciprocal predictions and reference must match target shape")
        wo_cosine = np.abs(np.asarray(batch["wo"], dtype=np.float64)[:, 2])[:, None, None]
        wi_cosine = np.abs(wi[..., 2:3])
        reciprocal_y = reciprocal_f * wo_cosine
        predicted_cross = prediction_y * wo_cosine - reciprocal_y * wi_cosine
        target_cross = target * wo_cosine - reciprocal_target * wi_cosine
        numerator = np.sum(np.abs(predicted_cross - target_cross) * weights, axis=(1, 2))
        denominator = np.maximum(
            np.sum(
                (np.abs(target) * wo_cosine + np.abs(reciprocal_target) * wi_cosine)
                * weights,
                axis=(1, 2),
            ),
            1e-12,
        )
        result["source_aware_reciprocity_deviation"] = numerator / denominator
    return result


def _group_scorecard(
    rows: Mapping[str, np.ndarray],
    selected: np.ndarray,
) -> dict[str, dict[str, float]]:
    names = [
        "log_l1",
        "peak_support_angle_degrees",
        "peak_ratio_log_error",
        "top_5_percent_energy_recall",
    ]
    if "source_aware_reciprocity_deviation" in rows:
        names.append("source_aware_reciprocity_deviation")
    return {name: summarize(np.asarray(rows[name])[selected]) for name in names}


def _directional_by_state(
    rows: Mapping[str, np.ndarray],
    selected: np.ndarray,
) -> np.ndarray:
    state_index = np.asarray(rows["state_index"], dtype=np.int64)
    numerator = np.asarray(rows["direction_numerator"], dtype=np.float64)
    denominator = np.asarray(rows["direction_denominator"], dtype=np.float64)
    return np.asarray([
        np.sum(numerator[selected & (state_index == index)])
        / max(np.sum(denominator[selected & (state_index == index)]), 1e-12)
        for index in np.unique(state_index[selected])
    ], dtype=np.float64)


def build_quality_report(
    rows: Mapping[str, np.ndarray],
    *,
    state_ids: np.ndarray,
    family_ids: np.ndarray,
    structure_family_ids: np.ndarray,
    difficulty_classes: np.ndarray,
    difficulty_tags: tuple[tuple[str, ...], ...],
    evaluation_cohorts: np.ndarray,
    data_id: str,
    evaluation_role: str,
    color_contract: str = "nonnegative",
    provenance_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    state_index = np.asarray(rows["state_index"], dtype=np.int64)
    selected_states = np.unique(state_index)
    state_records: dict[str, Any] = {}
    state_direction = []
    for index in selected_states:
        selected = state_index == index
        direction = float(np.sum(np.asarray(rows["direction_numerator"])[selected]) / max(
            np.sum(np.asarray(rows["direction_denominator"])[selected]), 1e-12
        ))
        energy = np.asarray(rows["energy_relative_error"])[selected]
        state_direction.append(direction)
        state_records[str(state_ids[index])] = {
            "family_id": str(family_ids[index]),
            "structure_family_id": str(structure_family_ids[index]),
            "difficulty_class": str(difficulty_classes[index]),
            "difficulty_tags": list(difficulty_tags[index]),
            "evaluation_cohort": str(evaluation_cohorts[index]),
            "directional_l1": direction,
            "energy_relative_error_by_wo": energy.tolist(),
            "energy_relative_error": summarize(energy),
        }
    state_direction_values = np.asarray(state_direction, dtype=np.float64)

    checks = {
        "prediction_finite": bool(np.all(np.asarray(rows["finite_rate"]) == 1.0)),
        "color_contract": bool(
            color_contract == "signed"
            or np.all(np.asarray(rows["minimum_f"]) >= -1e-7)
        ),
        **dict(provenance_checks or {}),
    }
    report: dict[str, Any] = {
        "format_name": "quality-report",
        "format_version": 1,
        "suite": dict(QUALITY_SUITE),
        "data_id": data_id,
        "evaluation_role": evaluation_role,
        "valid": all(checks.values()),
        "sanity": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "primary": {
            "directional_l1_by_state": summarize(state_direction_values),
            "energy_relative_error_by_state_wo": summarize(
                np.asarray(rows["energy_relative_error"], dtype=np.float64)
            ),
        },
        "scorecard": _group_scorecard(rows, np.ones(len(state_index), dtype=bool)),
        "diagnostics": {
            "absolute_error_sum": summarize(np.asarray(rows["absolute_error_sum"])),
            "reference_se_sum": summarize(np.asarray(rows["reference_se_sum"])),
            "model_error_over_reference_se": summarize(
                np.asarray(rows["model_error_over_reference_se"])
            ),
        },
        "states": state_records,
    }
    if "source_aware_reciprocity_deviation" not in rows:
        report["scorecard"]["source_aware_reciprocity_deviation"] = {
            "status": "not-available",
            "reason": "输入没有 reciprocal paired response",
        }
    breakdowns: dict[str, Any] = {}
    axes = {
        "difficulty": difficulty_classes[state_index],
        "difficulty_tags": np.asarray([
            "+".join(difficulty_tags[index]) or "none" for index in state_index
        ]),
        "family": family_ids[state_index],
        "structure_family": structure_family_ids[state_index],
        "cohort": evaluation_cohorts[state_index],
    }
    for axis_name, values in axes.items():
        breakdowns[axis_name] = {
            str(value): {
                "directional_l1_by_state": summarize(
                    _directional_by_state(rows, values == value)
                ),
                "energy_relative_error": summarize(
                    np.asarray(rows["energy_relative_error"])[values == value]
                ),
                **_group_scorecard(rows, values == value),
            }
            for value in sorted(set(map(str, values.tolist())))
        }
    report["scorecard"]["breakdowns"] = breakdowns
    return finalize_quality_report(report)


def write_quality_report(path: Path | str, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(finalize_quality_report(report), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
