from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ncls.learning.data import open_reference_store
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig

from .quality import QUALITY_SUITE, summarize


P1_AUDIT_FORMAT_NAME = "p1-audit-report"
P1_AUDIT_FORMAT_VERSION = 1
P1_AUDIT_ROLES = ("train", "validation", "test", "adversarial_probe", "dense_slice")

_MODEL_FIELDS = ("state_index", "wo", "wi", "mean", "solid_angle_weight")
_NOISE_FIELDS = (
    "state_index",
    "mean",
    "variance",
    "sample_count",
    "solid_angle_weight",
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finalize_report(report: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value.pop("report_sha256", None)
    value["report_sha256"] = _sha256_json(value)
    return value


def _write_report(path: Path | str, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _finalize_report(report),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _rankdata(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        stop = start + 1
        while stop < len(data) and data[order[stop]] == data[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    left = _rankdata(np.asarray(x, dtype=np.float64))
    right = _rankdata(np.asarray(y, dtype=np.float64))
    if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    # 这里只需要两个很短的 state rank 向量，不需要 BLAS。Windows 上在
    # PyTorch 已加载 LLVM OpenMP 后调用 NumPy BLAS 会再次加载 Intel OpenMP。
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    numerator = float(np.sum(left_centered * right_centered))
    denominator = math.sqrt(
        float(np.sum(left_centered * left_centered))
        * float(np.sum(right_centered * right_centered))
    )
    return numerator / denominator if denominator > 0.0 else None


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    point = _spearman(left, right)
    result: dict[str, Any] = {
        "state_count": int(len(left)),
        "spearman_rho": point,
        "bootstrap_iterations": int(iterations),
        "confidence": 0.95,
    }
    if point is None:
        result["confidence_interval"] = None
        return result
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        selected = rng.integers(0, len(left), size=len(left))
        value = _spearman(left[selected], right[selected])
        if value is not None:
            samples.append(value)
    if not samples:
        result["confidence_interval"] = None
    else:
        result["confidence_interval"] = [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ]
        result["valid_bootstrap_samples"] = len(samples)
    return result


def _bootstrap_statistic(
    values: np.ndarray,
    *,
    statistic: str,
    iterations: int,
    seed: int,
) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data):
        raise ValueError("bootstrap requires a nonempty vector")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(data), size=(iterations, len(data)))
    samples = data[selected]
    if statistic == "median":
        estimates = np.median(samples, axis=1)
    elif statistic == "p95":
        estimates = np.quantile(samples, 0.95, axis=1)
    else:
        raise ValueError(f"unsupported bootstrap statistic: {statistic}")
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _tail_stability(
    state_ids: Sequence[str],
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    ids = np.asarray(state_ids, dtype=object)
    if data.ndim != 1 or len(data) != len(ids) or len(data) < 3:
        raise ValueError("tail stability requires at least three matched states")
    order = np.argsort(data)
    quantile_index = 0.95 * (len(data) - 1)
    lower = int(math.floor(quantile_index))
    upper = int(math.ceil(quantile_index))
    leave_one_out = np.asarray(
        [np.quantile(np.delete(data, index), 0.95) for index in range(len(data))],
        dtype=np.float64,
    )
    top = order[::-1][: min(10, len(order))]
    return {
        "state_count": int(len(data)),
        "p95": float(np.quantile(data, 0.95)),
        "bootstrap_95_ci": _bootstrap_statistic(
            data,
            statistic="p95",
            iterations=iterations,
            seed=seed,
        ),
        "leave_one_state_out_p95": {
            "minimum": float(np.min(leave_one_out)),
            "median": float(np.median(leave_one_out)),
            "maximum": float(np.max(leave_one_out)),
        },
        "numpy_linear_quantile_support": {
            "fractional_index": float(quantile_index),
            "lower_sorted_index": lower,
            "upper_sorted_index": upper,
            "lower_state_id": str(ids[order[lower]]),
            "upper_state_id": str(ids[order[upper]]),
            "lower_value": float(data[order[lower]]),
            "upper_value": float(data[order[upper]]),
        },
        "worst_states": [
            {"state_id": str(ids[index]), "directional_l1": float(data[index])}
            for index in top
        ],
    }


def _noise_row_metrics(
    mean: np.ndarray,
    standard_error: np.ndarray,
    solid_angle_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(mean, dtype=np.float64)
    error = np.asarray(standard_error, dtype=np.float64)
    weights = np.asarray(solid_angle_weight, dtype=np.float64)[..., None]
    peak = np.max(np.abs(target), axis=(1, 2), keepdims=True)
    denominator = np.maximum(
        np.abs(target),
        np.maximum(0.005 * peak, 1e-8),
    )
    group_p95 = np.quantile(error / denominator, 0.95, axis=(1, 2))
    weighted = np.sum(error * weights, axis=(1, 2)) / np.maximum(
        np.sum(np.abs(target) * weights, axis=(1, 2)),
        1e-12,
    )
    return group_p95, weighted


def _deadzone_row_metrics(
    preclamp_f: np.ndarray,
    target_y: np.ndarray,
    solid_angle_weight: np.ndarray,
    output_scale: np.ndarray,
) -> dict[str, Any]:
    preclamp = np.asarray(preclamp_f, dtype=np.float64)
    target = np.asarray(target_y, dtype=np.float64)
    weights = np.asarray(solid_angle_weight, dtype=np.float64)[:, None]
    scale = np.asarray(output_scale, dtype=np.float64)[None, :]
    negative = preclamp < 0.0
    any_negative = np.any(negative, axis=1)
    all_negative = np.all(negative, axis=1)
    normalized_depth = np.maximum(-preclamp / scale, 0.0)[negative]
    return {
        "negative_rgb_count": int(np.sum(negative)),
        "rgb_count": int(negative.size),
        "any_negative_direction_count": int(np.sum(any_negative)),
        "all_negative_direction_count": int(np.sum(all_negative)),
        "direction_count": int(len(preclamp)),
        "solid_angle_negative_rgb_numerator": float(np.sum(negative * weights)),
        "solid_angle_rgb_denominator": float(3.0 * np.sum(weights)),
        "solid_angle_any_direction_numerator": float(np.sum(any_negative * weights[:, 0])),
        "solid_angle_all_direction_numerator": float(np.sum(all_negative * weights[:, 0])),
        "solid_angle_direction_denominator": float(np.sum(weights)),
        "target_energy_negative_numerator": float(
            np.sum(negative * np.abs(target) * weights)
        ),
        "target_energy_denominator": float(np.sum(np.abs(target) * weights)),
        "normalized_negative_depth": normalized_depth,
    }


def _configured_noise_budget(
    corpus_document: Mapping[str, Any],
    role: str,
    state_id: str,
) -> dict[str, Any]:
    budget = corpus_document["plan"]["reference_budget"]
    if role == "train":
        target = float(budget["training_target_relative_se_p95"])
        maximum = float(budget["training_maximum_query_group_relative_se_p95"])
        enforced = True
    elif role in {"validation", "test"}:
        target = float(budget["target_relative_se_p95"])
        maximum = float(budget["maximum_query_group_relative_se_p95"])
        enforced = True
    else:
        target = float(budget["diagnostic_target_relative_se_p95"])
        maximum = float(budget["diagnostic_maximum_query_group_relative_se_p95"])
        enforced = False
    base_maximum = maximum
    promoted = False
    for item in budget.get("state_sample_promotions", []):
        if state_id == str(item["state_id"]) and role in item["query_roles"]:
            maximum = max(maximum, float(item["maximum_query_group_relative_se_p95"]))
            promoted = maximum > base_maximum
    return {
        "target_relative_se_p95": target,
        "maximum_query_group_relative_se_p95": maximum,
        "base_maximum_query_group_relative_se_p95": base_maximum,
        "maximum_enforced": enforced,
        "promoted": promoted,
    }


def _role_indices(store: Any, role: str) -> np.ndarray:
    if role in {"adversarial_probe", "dense_slice"}:
        return store.indices_for_query_role(role)
    return store.partition_indices("target-visible-v1", role)


def _audit_reference_noise_role(
    store: Any,
    corpus_document: Mapping[str, Any],
    role: str,
    *,
    batch_size: int = 16,
) -> dict[str, Any]:
    state_ids = list(map(str, store.state_strings("state_id").tolist()))
    group_p95_by_state: list[list[np.ndarray]] = [[] for _ in state_ids]
    weighted_by_state: list[list[np.ndarray]] = [[] for _ in state_ids]
    integrated_se_numerator_by_state = np.zeros(len(state_ids), dtype=np.float64)
    integrated_target_denominator_by_state = np.zeros(len(state_ids), dtype=np.float64)
    configured_maximum_by_state = [
        _configured_noise_budget(corpus_document, role, state_id) for state_id in state_ids
    ]
    group_caps: list[np.ndarray] = []
    all_group_p95: list[np.ndarray] = []
    all_weighted: list[np.ndarray] = []
    indices = _role_indices(store, role)
    for raw in store.iter_batches(indices, batch_size, fields=_NOISE_FIELDS):
        sample_count = np.maximum(np.asarray(raw["sample_count"], dtype=np.float64), 1.0)
        standard_error = np.sqrt(
            np.maximum(np.asarray(raw["variance"], dtype=np.float64), 0.0)
            / sample_count[..., None]
        )
        group_p95, weighted = _noise_row_metrics(
            raw["mean"], standard_error, raw["solid_angle_weight"]
        )
        solid_angle_weight = np.asarray(
            raw["solid_angle_weight"], dtype=np.float64
        )[..., None]
        integrated_se_numerator = np.sum(
            standard_error * solid_angle_weight, axis=(1, 2)
        )
        integrated_target_denominator = np.sum(
            np.abs(np.asarray(raw["mean"], dtype=np.float64)) * solid_angle_weight,
            axis=(1, 2),
        )
        states = np.asarray(raw["state_index"], dtype=np.int64)
        caps = np.asarray(
            [configured_maximum_by_state[int(index)]["maximum_query_group_relative_se_p95"] for index in states],
            dtype=np.float64,
        )
        all_group_p95.append(group_p95)
        all_weighted.append(weighted)
        group_caps.append(caps)
        for state_index in np.unique(states):
            selected = states == state_index
            group_p95_by_state[int(state_index)].append(group_p95[selected])
            weighted_by_state[int(state_index)].append(weighted[selected])
            integrated_se_numerator_by_state[int(state_index)] += float(
                np.sum(integrated_se_numerator[selected])
            )
            integrated_target_denominator_by_state[int(state_index)] += float(
                np.sum(integrated_target_denominator[selected])
            )
    combined_p95 = np.concatenate(all_group_p95)
    combined_weighted = np.concatenate(all_weighted)
    combined_caps = np.concatenate(group_caps)
    states: dict[str, Any] = {}
    promoted_state_ids = []
    for state_index, state_id in enumerate(state_ids):
        values = np.concatenate(group_p95_by_state[state_index])
        weighted = np.concatenate(weighted_by_state[state_index])
        configured = configured_maximum_by_state[state_index]
        if configured["promoted"]:
            promoted_state_ids.append(state_id)
        states[state_id] = {
            "query_group_count": int(len(values)),
            "configured": configured,
            "achieved_group_relative_se_p95": summarize(values),
            "achieved_energy_weighted_relative_se": summarize(weighted),
            "integrated_reference_se_ratio": float(
                integrated_se_numerator_by_state[state_index]
                / max(integrated_target_denominator_by_state[state_index], 1e-15)
            ),
            "groups_above_target": int(
                np.sum(values > configured["target_relative_se_p95"])
            ),
            "groups_above_configured_maximum": int(
                np.sum(values > configured["maximum_query_group_relative_se_p95"])
            ),
        }
    return {
        "query_group_count": int(len(combined_p95)),
        "promoted_state_ids": promoted_state_ids,
        "achieved_group_relative_se_p95": summarize(combined_p95),
        "achieved_energy_weighted_relative_se": summarize(combined_weighted),
        "integrated_reference_se_ratio": float(
            np.sum(integrated_se_numerator_by_state)
            / max(np.sum(integrated_target_denominator_by_state), 1e-15)
        ),
        "groups_above_configured_maximum": int(np.sum(combined_p95 > combined_caps)),
        "states": states,
    }


def _state_signed_metrics(
    predicted_energy: np.ndarray,
    target_energy: np.ndarray,
    predicted_group_energy: np.ndarray,
    target_group_energy: np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_energy, dtype=np.float64)
    target = np.asarray(target_energy, dtype=np.float64)
    channel_floor = max(float(np.max(target)) * 1e-12, 1e-15)
    channel_ratio = [
        float(predicted[index] / target[index]) if target[index] > channel_floor else None
        for index in range(3)
    ]
    ratio = float(np.sum(predicted) / max(np.sum(target), 1e-15))
    group_predicted = np.asarray(predicted_group_energy, dtype=np.float64)
    group_target = np.asarray(target_group_energy, dtype=np.float64)
    group_floor = max(float(np.max(group_target)) * 1e-6, 1e-12)
    selected = group_target >= group_floor
    group_ratios = group_predicted[selected] / group_target[selected]
    return {
        "energy_ratio": ratio,
        "signed_relative_bias": ratio - 1.0,
        "channel_energy_ratio": {
            name: channel_ratio[index] for index, name in enumerate(("R", "G", "B"))
        },
        "wo_energy_floor": group_floor,
        "wo_included_count": int(np.sum(selected)),
        "wo_excluded_near_zero_count": int(np.sum(~selected)),
        "signed_relative_bias_by_wo": summarize(group_ratios - 1.0),
        "log_energy_ratio_by_wo": summarize(np.log(np.maximum(group_ratios, 1e-12))),
    }


def _empty_state_accumulator() -> dict[str, Any]:
    return {
        "direction_numerator": 0.0,
        "direction_denominator": 0.0,
        "predicted_energy": np.zeros(3, dtype=np.float64),
        "target_energy": np.zeros(3, dtype=np.float64),
        "predicted_group_energy": [],
        "target_group_energy": [],
        "core_energy": np.zeros(3, dtype=np.float64),
        "deadzone": {
            "negative_rgb_count": 0,
            "rgb_count": 0,
            "any_negative_direction_count": 0,
            "all_negative_direction_count": 0,
            "direction_count": 0,
            "solid_angle_negative_rgb_numerator": 0.0,
            "solid_angle_rgb_denominator": 0.0,
            "solid_angle_any_direction_numerator": 0.0,
            "solid_angle_all_direction_numerator": 0.0,
            "solid_angle_direction_denominator": 0.0,
            "target_energy_negative_numerator": 0.0,
            "target_energy_denominator": 0.0,
            "normalized_negative_depth": [],
        },
    }


def _merge_deadzone(target: dict[str, Any], row: Mapping[str, Any]) -> None:
    for name in (
        "negative_rgb_count",
        "rgb_count",
        "any_negative_direction_count",
        "all_negative_direction_count",
        "direction_count",
        "solid_angle_negative_rgb_numerator",
        "solid_angle_rgb_denominator",
        "solid_angle_any_direction_numerator",
        "solid_angle_all_direction_numerator",
        "solid_angle_direction_denominator",
        "target_energy_negative_numerator",
        "target_energy_denominator",
    ):
        target[name] += row[name]
    depth = np.asarray(row["normalized_negative_depth"], dtype=np.float64)
    if len(depth):
        target["normalized_negative_depth"].append(depth)


def _finalize_deadzone(value: Mapping[str, Any]) -> dict[str, Any]:
    depth_parts = value["normalized_negative_depth"]
    depth = np.concatenate(depth_parts) if depth_parts else np.empty(0, dtype=np.float64)
    return {
        "negative_rgb_fraction": float(value["negative_rgb_count"] / max(value["rgb_count"], 1)),
        "any_negative_direction_fraction": float(
            value["any_negative_direction_count"] / max(value["direction_count"], 1)
        ),
        "all_negative_direction_fraction": float(
            value["all_negative_direction_count"] / max(value["direction_count"], 1)
        ),
        "solid_angle_weighted_negative_rgb_fraction": float(
            value["solid_angle_negative_rgb_numerator"]
            / max(value["solid_angle_rgb_denominator"], 1e-15)
        ),
        "solid_angle_weighted_any_negative_direction_fraction": float(
            value["solid_angle_any_direction_numerator"]
            / max(value["solid_angle_direction_denominator"], 1e-15)
        ),
        "solid_angle_weighted_all_negative_direction_fraction": float(
            value["solid_angle_all_direction_numerator"]
            / max(value["solid_angle_direction_denominator"], 1e-15)
        ),
        "target_energy_weighted_negative_rgb_fraction": float(
            value["target_energy_negative_numerator"]
            / max(value["target_energy_denominator"], 1e-15)
        ),
        "normalized_negative_depth": summarize(depth) if len(depth) else None,
        "negative_rgb_count": int(value["negative_rgb_count"]),
        "rgb_count": int(value["rgb_count"]),
        "direction_count": int(value["direction_count"]),
    }


def _pipeline_core_probe(pipeline: Any) -> tuple[Callable[..., torch.Tensor] | None, bool]:
    """探测 pipeline 的可选成员：`core_f(model, batch, store, device)` 给出解析 core（报告
    `E_core/E_ref`）；`has_signed_residual` 为真时才做 `clamp(core + Δ, 0)` 死区诊断。"""

    core_f = getattr(pipeline, "core_f", None)
    return (core_f if callable(core_f) else None), bool(getattr(pipeline, "has_signed_residual", False))


@torch.no_grad()
def _audit_checkpoint_role(
    model: torch.nn.Module,
    pipeline: Any,
    fitted_state: Mapping[str, Any],
    store: Any,
    config: TrainingConfig,
    role: str,
    noise_report: Mapping[str, Any],
    device: torch.device,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    indices = store.select_indices(
        pipeline.evaluation_indices(store, role),
        config.dataset_selection,
    )
    state_ids = list(map(str, store.state_strings("state_id").tolist()))
    family_ids = list(map(str, store.state_strings("family_id").tolist()))
    structure_ids = list(map(str, store.state_strings("structure_family_id").tolist()))
    difficulties = list(map(str, store.state_strings("difficulty_class").tolist()))
    tags = [json.loads(value) for value in store.state_strings("difficulty_tags_json")]
    accumulators = [_empty_state_accumulator() for _ in state_ids]
    core_f, has_signed_residual = _pipeline_core_probe(pipeline)
    output_scale = np.asarray(fitted_state["output_scale"], dtype=np.float64)
    model.eval()
    for raw in store.iter_batches(indices, config.batch_size, fields=_MODEL_FIELDS):
        batch = {
            name: torch.as_tensor(raw[name], device=device) for name in _MODEL_FIELDS
        }
        states_t = batch["state_index"].long()
        prediction_f = pipeline.predict_f(model, batch, store, device)
        core = None if core_f is None else core_f(model, batch, store, device)
        preclamp = None
        if core is not None and has_signed_residual:
            preclamp = core + model(states_t, batch["wo"].float(), batch["wi"].float())
        if not torch.all(torch.isfinite(prediction_f)):
            raise ValueError("P1 audit encountered a non-finite prediction")
        prediction = prediction_f.cpu().numpy().astype(np.float64, copy=False)
        wi = np.asarray(raw["wi"], dtype=np.float64)
        target = np.asarray(raw["mean"], dtype=np.float64)
        weights = np.asarray(raw["solid_angle_weight"], dtype=np.float64)
        prediction_y = prediction * np.abs(wi[..., 2:3])
        states = np.asarray(raw["state_index"], dtype=np.int64)
        core_y = None if core is None else (
            core.cpu().numpy().astype(np.float64, copy=False) * np.abs(wi[..., 2:3])
        )
        preclamp_numpy = None if preclamp is None else preclamp.cpu().numpy()
        for row, state_index_value in enumerate(states):
            state_index = int(state_index_value)
            accumulator = accumulators[state_index]
            row_weights = weights[row, :, None]
            absolute = np.abs(prediction_y[row] - target[row])
            accumulator["direction_numerator"] += float(np.sum(absolute * row_weights))
            accumulator["direction_denominator"] += float(
                np.sum(np.abs(target[row]) * row_weights)
            )
            predicted_energy = np.sum(prediction_y[row] * row_weights, axis=0)
            target_energy = np.sum(target[row] * row_weights, axis=0)
            accumulator["predicted_energy"] += predicted_energy
            accumulator["target_energy"] += target_energy
            accumulator["predicted_group_energy"].append(float(np.sum(predicted_energy)))
            accumulator["target_group_energy"].append(float(np.sum(target_energy)))
            if core_y is not None:
                accumulator["core_energy"] += np.sum(core_y[row] * row_weights, axis=0)
            if preclamp_numpy is not None:
                deadzone = _deadzone_row_metrics(
                    preclamp_numpy[row],
                    target[row],
                    weights[row],
                    output_scale[state_index],
                )
                _merge_deadzone(accumulator["deadzone"], deadzone)

    states_report: dict[str, Any] = {}
    directional_values = []
    energy_ratios = []
    promoted_state_ids = set(map(str, noise_report["promoted_state_ids"]))
    for state_index, state_id in enumerate(state_ids):
        accumulator = accumulators[state_index]
        directional = float(
            accumulator["direction_numerator"]
            / max(accumulator["direction_denominator"], 1e-15)
        )
        signed = _state_signed_metrics(
            accumulator["predicted_energy"],
            accumulator["target_energy"],
            np.asarray(accumulator["predicted_group_energy"]),
            np.asarray(accumulator["target_group_energy"]),
        )
        record: dict[str, Any] = {
            "family_id": family_ids[state_index],
            "structure_family_id": structure_ids[state_index],
            "difficulty_class": difficulties[state_index],
            "difficulty_tags": tags[state_index],
            "directional_l1": directional,
            "signed_energy": signed,
            "reference_noise": noise_report["states"][state_id],
        }
        if core_f is not None:
            record["analytic_core_energy_ratio"] = float(
                np.sum(accumulator["core_energy"])
                / max(np.sum(accumulator["target_energy"]), 1e-15)
            )
        if has_signed_residual:
            record["deadzone"] = _finalize_deadzone(accumulator["deadzone"])
        states_report[state_id] = record
        directional_values.append(directional)
        energy_ratios.append(signed["energy_ratio"])

    directional_array = np.asarray(directional_values, dtype=np.float64)
    ratio_array = np.asarray(energy_ratios, dtype=np.float64)
    log_ratios = np.log(np.maximum(ratio_array, 1e-12))
    result: dict[str, Any] = {
        "query_group_count": int(len(indices)),
        "directional_l1_by_state": summarize(directional_array),
        "tail_stability": _tail_stability(
            state_ids,
            directional_array,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "signed_energy_by_state": {
            "energy_ratio": summarize(ratio_array),
            "signed_relative_bias": summarize(ratio_array - 1.0),
            "fraction_below_one": float(np.mean(ratio_array < 1.0)),
            "median_log_energy_ratio": float(np.median(log_ratios)),
            "median_log_energy_ratio_bootstrap_95_ci": _bootstrap_statistic(
                log_ratios,
                statistic="median",
                iterations=bootstrap_iterations,
                seed=seed + 1,
            ),
            "global_ratio_of_sums": float(
                sum(np.sum(item["predicted_energy"]) for item in accumulators)
                / max(sum(np.sum(item["target_energy"]) for item in accumulators), 1e-15)
            ),
        },
        "states": states_report,
    }
    noise_values = np.asarray(
        [
            noise_report["states"][state_id]["achieved_group_relative_se_p95"]["p95"]
            for state_id in state_ids
        ],
        dtype=np.float64,
    )
    result["correlations"] = {
        "reference_se_p95_vs_directional_l1": _bootstrap_spearman(
            noise_values,
            directional_array,
            iterations=bootstrap_iterations,
            seed=seed + 2,
        ),
        "reference_se_p95_vs_signed_energy_ratio": _bootstrap_spearman(
            noise_values,
            ratio_array,
            iterations=bootstrap_iterations,
            seed=seed + 3,
        ),
    }
    integrated_noise_values = np.asarray(
        [
            noise_report["states"][state_id]["integrated_reference_se_ratio"]
            for state_id in state_ids
        ],
        dtype=np.float64,
    )
    result["correlations"]["integrated_reference_se_ratio_vs_directional_l1"] = (
        _bootstrap_spearman(
            integrated_noise_values,
            directional_array,
            iterations=bootstrap_iterations,
            seed=seed + 4,
        )
    )
    result["correlations"]["integrated_reference_se_ratio_vs_signed_energy_ratio"] = (
        _bootstrap_spearman(
            integrated_noise_values,
            ratio_array,
            iterations=bootstrap_iterations,
            seed=seed + 5,
        )
    )
    correlations: dict[str, Any] = {}
    if has_signed_residual:
        deadzone_names = (
            "negative_rgb_fraction",
            "all_negative_direction_fraction",
            "solid_angle_weighted_negative_rgb_fraction",
            "target_energy_weighted_negative_rgb_fraction",
        )
        keep = np.asarray([state_id not in promoted_state_ids for state_id in state_ids])
        for offset, name in enumerate(deadzone_names):
            values = np.asarray(
                [states_report[state_id]["deadzone"][name] for state_id in state_ids],
                dtype=np.float64,
            )
            correlations[f"{name}_vs_directional_l1"] = _bootstrap_spearman(
                values,
                directional_array,
                iterations=bootstrap_iterations,
                seed=seed + 10 + offset,
            )
            correlations[f"{name}_vs_directional_l1_excluding_promoted_states"] = (
                _bootstrap_spearman(
                    values[keep],
                    directional_array[keep],
                    iterations=bootstrap_iterations,
                    seed=seed + 20 + offset,
                )
            )
    if core_f is not None:
        core_ratios = np.asarray(
            [states_report[state_id]["analytic_core_energy_ratio"] for state_id in state_ids],
            dtype=np.float64,
        )
        correlations["analytic_core_energy_ratio_vs_directional_l1"] = _bootstrap_spearman(
            core_ratios,
            directional_array,
            iterations=bootstrap_iterations,
            seed=seed + 30,
        )
    if correlations:
        result["deadzone_correlations"] = correlations
    return result


def _load_audit_checkpoint(
    path: Path,
    store: Any,
    device: torch.device,
) -> tuple[dict[str, Any], TrainingConfig, Any, torch.nn.Module, Mapping[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=device)
    config = TrainingConfig.from_dict(checkpoint["training_config"])
    if checkpoint.get("training_config_sha256") != config.resolved_sha256:
        raise ValueError("checkpoint training config hash is unsupported")
    pipeline = create_pipeline(str(checkpoint.get("pipeline", "")))
    if pipeline.descriptor.stage != "P1":
        raise ValueError("P1 audit only accepts P1 checkpoints")
    if checkpoint.get("pipeline_sha256") != pipeline.descriptor.sha256:
        raise ValueError("checkpoint learning pipeline identity is unsupported")
    if checkpoint.get("data_id") != store.data_id:
        raise ValueError("checkpoint data_id does not match the audit corpus")
    fitted_state = checkpoint.get("fitted_training_state")
    if not isinstance(fitted_state, dict):
        raise ValueError("checkpoint fitted training state is missing")
    if checkpoint.get("fitted_training_state_sha256") != _sha256_json(fitted_state):
        raise ValueError("checkpoint fitted training state hash is invalid")
    if list(map(str, fitted_state.get("state_ids", []))) != list(
        map(str, store.state_strings("state_id").tolist())
    ):
        raise ValueError("checkpoint fitted state order disagrees with the corpus")
    pipeline.load_training_state(fitted_state)
    model = pipeline.create_model(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, config, pipeline, model, fitted_state


def run_p1_audit(
    data_path: Path | str,
    checkpoints: Mapping[str, Path | str],
    output_path: Path | str,
    *,
    roles: Sequence[str] = P1_AUDIT_ROLES,
    device_name: str | None = None,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260825,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    selected_roles = tuple(roles)
    if not selected_roles or len(set(selected_roles)) != len(selected_roles):
        raise ValueError("P1 audit roles must be nonempty and unique")
    if any(role not in P1_AUDIT_ROLES for role in selected_roles):
        raise ValueError(f"unsupported P1 audit role: {selected_roles}")
    if bootstrap_iterations < 1000:
        raise ValueError("P1 audit requires at least 1,000 bootstrap iterations")
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("P1 audit checkpoint labels must be nonempty and unique")
    data_file = Path(data_path).resolve()
    corpus_document = json.loads(data_file.read_text(encoding="utf-8"))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    module_path = Path(__file__).resolve()
    with open_reference_store(data_file) as store:
        reference_noise: dict[str, Any] = {}
        for role in selected_roles:
            if progress:
                progress(f"reference-noise role={role}")
            reference_noise[role] = _audit_reference_noise_role(
                store, corpus_document, role
            )
        checkpoint_reports: dict[str, Any] = {}
        for checkpoint_index, (label, checkpoint_path) in enumerate(checkpoints.items()):
            resolved_checkpoint = Path(checkpoint_path).resolve()
            if progress:
                progress(f"checkpoint={label} load={resolved_checkpoint}")
            checkpoint, config, pipeline, model, fitted_state = _load_audit_checkpoint(
                resolved_checkpoint, store, device
            )
            roles_report: dict[str, Any] = {}
            for role_index, role in enumerate(selected_roles):
                if progress:
                    progress(f"checkpoint={label} role={role}")
                roles_report[role] = _audit_checkpoint_role(
                    model,
                    pipeline,
                    fitted_state,
                    store,
                    config,
                    role,
                    reference_noise[role],
                    device,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=bootstrap_seed + checkpoint_index * 100 + role_index * 10,
                )
            checkpoint_reports[label] = {
                "checkpoint": {
                    "uri": str(resolved_checkpoint),
                    "sha256": sha256_file(resolved_checkpoint),
                    "step": int(checkpoint["step"]),
                    "pipeline": pipeline.descriptor.name,
                    "pipeline_sha256": pipeline.descriptor.sha256,
                    "training_config_sha256": config.resolved_sha256,
                    "seed": int(config.seed),
                    "capacity": config.capacity,
                },
                "cost": dict(pipeline.parameter_costs(model)),
                "roles": roles_report,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report = {
            "format_name": P1_AUDIT_FORMAT_NAME,
            "format_version": P1_AUDIT_FORMAT_VERSION,
            "scope": {
                "kind": "retrospective-p1-v1-diagnostic",
                "causal_claims": False,
                "quality_v1_unchanged": True,
            },
            "implementation": {
                "module": str(module_path),
                "module_sha256": sha256_file(module_path),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
                ),
            },
            "data": {
                "uri": str(data_file),
                "manifest_sha256": sha256_file(data_file),
                "data_id": store.data_id,
            },
            "frozen_quality_suite": QUALITY_SUITE,
            "roles": list(selected_roles),
            "bootstrap": {
                "iterations": int(bootstrap_iterations),
                "seed": int(bootstrap_seed),
                "resampling_unit": "state",
                "confidence": 0.95,
            },
            "reference_noise": reference_noise,
            "checkpoints": checkpoint_reports,
        }
    finalized = _finalize_report(report)
    _write_report(output_path, finalized)
    return finalized


def parse_checkpoint_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
        else:
            raw_path = value
            path = Path(raw_path)
            label = f"{path.parent.parent.name}-{path.stem}"
        if not label or label in result:
            raise ValueError(f"duplicate or empty checkpoint label: {label!r}")
        result[label] = Path(raw_path)
    return result


__all__ = [
    "P1_AUDIT_FORMAT_NAME",
    "P1_AUDIT_FORMAT_VERSION",
    "P1_AUDIT_ROLES",
    "parse_checkpoint_specs",
    "run_p1_audit",
]
