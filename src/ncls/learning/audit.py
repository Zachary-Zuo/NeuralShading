from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ncls.data import PositionKind, QUERY_ROLE_NAMES, ReferenceDataset, SPLIT_NAMES
from ncls.learning.gates import evaluate_supervision_gate, load_supervision_gate


AUDIT_FORMAT = "ncls.supervision-audit"
AUDIT_VERSION = 6
TRANSFORM_STATISTICS_FORMAT = "ncls.target-transform-statistics"
TRANSFORM_STATISTICS_VERSION = 2
_SPLIT_CODES = {name: index for index, name in enumerate(SPLIT_NAMES)}
_QUERY_ROLE_CODES = {name: index for index, name in enumerate(QUERY_ROLE_NAMES)}
_LUMINANCE = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
_PEAK_RELEVANT_TOP_1_PERCENT_ENERGY_FRACTION = 0.1


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    )


def _select_evenly(indices: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if len(values) <= limit:
        return values
    positions = np.linspace(0, len(values) - 1, limit, dtype=np.int64)
    return values[positions]


def _percentiles(values: np.ndarray, quantiles: Iterable[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {}
    result: dict[str, float] = {}
    for quantile in quantiles:
        key = f"p{100.0 * quantile:g}".replace(".", "_")
        result[key] = float(np.quantile(array, quantile))
    return result


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"count": 0}
    absolute = np.abs(array)
    nonzero = absolute[absolute > 0.0]
    result: dict[str, Any] = {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "zero_fraction": float(np.mean(array == 0.0)),
        "negative_fraction": float(np.mean(array < 0.0)),
        "finite_rate": float(np.mean(np.isfinite(array))),
        "absolute": _percentiles(absolute, (0.5, 0.9, 0.95, 0.99, 0.999)),
        "nonzero_absolute": _percentiles(nonzero, (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)),
    }
    if len(nonzero):
        denominator = max(float(np.quantile(nonzero, 0.01)), np.finfo(np.float64).tiny)
        result["dynamic_range_max_over_nonzero_p1"] = float(np.max(nonzero) / denominator)
    return result


def _peak_relevant_summary(top1: np.ndarray, spacing: np.ndarray) -> dict[str, Any]:
    concentration = np.asarray(top1, dtype=np.float64)
    angles = np.asarray(spacing, dtype=np.float64)
    selected = concentration >= _PEAK_RELEVANT_TOP_1_PERCENT_ENERGY_FRACTION
    return {
        "top_1_percent_energy_fraction_minimum": _PEAK_RELEVANT_TOP_1_PERCENT_ENERGY_FRACTION,
        "query_group_count": int(np.sum(selected)),
        "peak_nearest_neighbor_angle_degrees": _percentiles(
            angles[selected], (0.05, 0.5, 0.9, 0.95)
        ),
    }


def _split_leaks(keys: np.ndarray, splits: np.ndarray) -> dict[str, Any]:
    memberships: dict[str, set[int]] = {}
    for key, split in zip(keys.tolist(), splits.tolist(), strict=True):
        text = str(key)
        if not text:
            continue
        memberships.setdefault(text, set()).add(int(split))
    leaked = sorted(key for key, values in memberships.items() if len(values) > 1)
    return {"leak_count": len(leaked), "examples": leaked[:16]}


def _transform_family_statistics(values: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    response = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    channels: list[dict[str, Any]] = []
    for channel in range(3):
        raw = response[..., channel][mask]
        absolute_nonzero = np.abs(raw[np.abs(raw) > 0.0])
        scale = float(np.quantile(absolute_nonzero, 0.5)) if len(absolute_nonzero) else 1.0
        scale = max(scale, 1e-8)
        transformed = np.arcsinh(raw / scale)
        channel_result: dict[str, Any] = {
            "channel": channel,
            "signed_asinh_scale": scale,
            "signed_asinh_mean": float(np.mean(transformed)) if len(transformed) else 0.0,
            "signed_asinh_std": float(np.std(transformed)) if len(transformed) else 1.0,
        }
        if len(raw) and np.min(raw) >= 0.0:
            log_values = np.log1p(raw / scale)
            channel_result.update({
                "nonnegative_log1p_scale": scale,
                "nonnegative_log1p_mean": float(np.mean(log_values)),
                "nonnegative_log1p_std": float(np.std(log_values)),
            })
        channels.append(channel_result)
    return {"channels": channels}


def _decode_json_attribute(value: Any) -> Any:
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return json.loads(text)


def _row_hashes(values: Any, *, chunk_size: int = 256) -> np.ndarray:
    hashes: list[str] = []
    for start in range(0, len(values), chunk_size):
        rows = np.asarray(values[start : start + chunk_size])
        for row in rows:
            hashes.append(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest())
    return np.asarray(hashes, dtype=object)


def _cross_partition_hash_audit(
    hashes: np.ndarray,
    partition_codes: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any]:
    sets = {
        name: set(map(str, hashes[partition_codes == code].tolist()))
        for code, name in enumerate(names)
    }
    return {
        "unique_count_by_partition": {name: len(values) for name, values in sets.items()},
        "overlap_counts": {
            f"{first}_{second}": len(sets[first] & sets[second])
            for first_index, first in enumerate(names)
            for second in names[first_index + 1 :]
        },
    }


def _footprint_scale_keys(length_x: np.ndarray, length_y: np.ndarray) -> set[tuple[float, float]]:
    """以 log2 尺度量化 footprint，避免旋转后的 float32 舍入被误计为新尺度。"""

    result: set[tuple[float, float]] = set()
    for x_value, y_value in zip(length_x, length_y, strict=True):
        if x_value <= 0.0 and y_value <= 0.0:
            continue
        result.add((
            round(float(np.log2(max(x_value, np.finfo(np.float64).tiny))), 4),
            round(float(np.log2(max(y_value, np.finfo(np.float64).tiny))), 4),
        ))
    return result


def _uv_seam_coverage(uv: np.ndarray, *, edge_threshold: float = 0.01) -> dict[str, Any]:
    """检查 U/V 两轴是否各有横向坐标匹配的 seam 两侧 query。"""

    values = np.asarray(uv, dtype=np.float64)
    transverse_tolerance = 1e-5

    def axis_coverage(axis: int) -> dict[str, Any]:
        transverse_axis = 1 - axis
        low = values[values[:, axis] <= edge_threshold]
        high = values[values[:, axis] >= 1.0 - edge_threshold]
        matched = bool(
            len(low)
            and len(high)
            and np.min(np.abs(low[:, None, transverse_axis] - high[None, :, transverse_axis]))
            <= transverse_tolerance
        )
        return {
            "low_edge_count": int(len(low)),
            "high_edge_count": int(len(high)),
            "matched_opposite_edge_pair": matched,
        }

    by_axis = {"u": axis_coverage(0), "v": axis_coverage(1)}
    return {
        "edge_threshold": edge_threshold,
        "transverse_tolerance": transverse_tolerance,
        "by_axis": by_axis,
        "opposite_edge_pair_axis_count": sum(
            int(value["matched_opposite_edge_pair"]) for value in by_axis.values()
        ),
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    split = audit["split_audit"]
    coverage = audit["coverage"]
    uncertainty = audit["reference_uncertainty"]
    proposal_names = ", ".join(item["proposal_id"] for item in coverage["proposals"])
    high_resolution = coverage["adversarial_profile_presence"]
    return "\n".join((
        "# Supervision audit",
        "",
        "## 它是什么",
        "",
        f"这是数据集 `{audit['dataset']['dataset_id']}` 的只读 E0 监督审计。它不训练模型，也不使用 validation/test 估计 target transform。",
        "",
        "## 当前结论",
        "",
        f"- HDF5 合同与内容哈希：通过；抽样了 {audit['sampling']['query_group_count_used']} / {audit['sampling']['query_group_count_available']} 个 query group 计算分布统计。",
        f"- split_group 泄漏：{split['split_group_id']['leak_count']}；source asset 泄漏：{split['source_sha256']['leak_count']}；父子状态跨 source split：{split['parent_child_cross_split_count']}。",
        f"- query role 计数：{json.dumps(split['query_role_counts'], ensure_ascii=False)}。",
        f"- 当前 proposal：{proposal_names or '无'}。",
        f"- peak/grazing/transmission 高分辨率 profile：{json.dumps(high_resolution, ensure_ascii=False)}。",
        f"- replica normalized L1 p95：{uncertainty['replica_normalized_l1'].get('p95', 0.0):.6g}；相对 standard error p95：{uncertainty['relative_standard_error'].get('p95', 0.0):.6g}。",
        f"- 最坏 query group 的 relative SE p95：{uncertainty['worst_query_group']['relative_standard_error_p95']:.6g}；replica normalized L1：{uncertainty['worst_query_group']['replica_normalized_l1']:.6g}。",
        "",
        "## 下一步",
        "",
        "把本报告与独立高分辨率 probe 一起对照版本化 E0 gate；若 proposal 或 peak/掠射/透射覆盖不足，先修正 query/state distribution，再生成正式训练 H5。",
        "",
    ))


def audit_supervision(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    verify_hashes: bool = True,
    max_distribution_query_groups: int = 8192,
    gate_path: Path | str | None = None,
) -> dict[str, Any]:
    """只依赖公共 ReferenceDataset 合同生成 E0 监督审计与 train-only transform 统计。"""

    if max_distribution_query_groups < 1:
        raise ValueError("max_distribution_query_groups must be positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("supervision audit output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    with ReferenceDataset.open(dataset_path, verify_hashes=verify_hashes) as dataset:
        state_splits = dataset.state_splits.astype(np.int64)
        state_ids = dataset.state_strings("state_id")
        family_ids = dataset.state_strings("family_id")
        asset_ids = dataset.state_strings("asset_id")
        split_group_ids = dataset.state_strings("split_group_id")
        source_hashes = dataset.state_strings("source_sha256")
        parent_ids = dataset.state_strings("parent_state_id")
        state_index = np.asarray(dataset.stream["queries/state_index"], dtype=np.int64)
        query_source_splits = state_splits[state_index]
        query_roles = dataset.query_roles.astype(np.int64)
        query_families = family_ids[state_index]
        all_indices = np.arange(dataset.query_group_count, dtype=np.int64)
        selected = _select_evenly(all_indices, max_distribution_query_groups)
        batch = dataset.group_batch(selected)
        selected_families = query_families[selected]
        selected_source_splits = query_source_splits[selected]
        selected_roles = query_roles[selected]

        parent_lookup = {str(state_id): index for index, state_id in enumerate(state_ids.tolist())}
        parent_cross_split: list[dict[str, Any]] = []
        for child_index, parent_id in enumerate(parent_ids.tolist()):
            parent_index = parent_lookup.get(str(parent_id))
            if parent_index is not None and state_splits[parent_index] != state_splits[child_index]:
                parent_cross_split.append({
                    "child_state_id": str(state_ids[child_index]),
                    "parent_state_id": str(parent_id),
                })

        proposal_ids = _decode_json_attribute(dataset.stream.attrs["proposal_ids_json"])
        proposal_codes = np.asarray(dataset.stream["queries/proposal_code"], dtype=np.int64)
        proposals: list[dict[str, Any]] = []
        for code in np.unique(proposal_codes):
            mask = proposal_codes == code
            item: dict[str, Any] = {
                "code": int(code),
                "proposal_id": str(proposal_ids[str(int(code))]),
                "query_group_count": int(np.sum(mask)),
                "by_source_split": {
                    name: int(np.sum(mask & (query_source_splits == split_code)))
                    for name, split_code in _SPLIT_CODES.items()
                },
                "by_query_role": {
                    name: int(np.sum(mask & (query_roles == role_code)))
                    for name, role_code in _QUERY_ROLE_CODES.items()
                },
            }
            proposals.append(item)

        wi_grid_hashes = _row_hashes(dataset.stream["queries/wi"])
        wo_hashes = _row_hashes(dataset.stream["queries/wo"])

        response_by_family: dict[str, Any] = {}
        group_energy_parts: list[np.ndarray] = []
        top1_parts: list[np.ndarray] = []
        top5_parts: list[np.ndarray] = []
        peak_spacing_parts: list[np.ndarray] = []
        group_top1 = np.empty(len(selected), dtype=np.float64)
        group_top5 = np.empty(len(selected), dtype=np.float64)
        group_peak_spacing = np.empty(len(selected), dtype=np.float64)
        group_peak_to_median = np.empty(len(selected), dtype=np.float64)
        worst_groups: list[dict[str, Any]] = []
        for family_id in sorted(set(map(str, selected_families.tolist()))):
            family_mask = selected_families == family_id
            response = np.asarray(batch["mean"][family_mask], dtype=np.float64)
            valid = np.asarray(batch["valid"][family_mask], dtype=bool)
            weights = np.asarray(batch["solid_angle_weight"][family_mask], dtype=np.float64)
            wi = np.asarray(batch["wi"][family_mask], dtype=np.float64)
            channels = [
                _distribution(response[..., channel][valid])
                for channel in range(3)
            ]
            energy_rgb = np.sum(response * weights[..., None] * valid[..., None], axis=1)
            group_energy_parts.append(np.sum(np.abs(energy_rgb), axis=1))
            magnitude = np.sum(np.abs(response), axis=-1) * weights * valid
            order = np.argsort(magnitude, axis=1)[:, ::-1]
            sorted_energy = np.take_along_axis(magnitude, order, axis=1)
            total_energy = np.maximum(np.sum(sorted_energy, axis=1), 1e-20)
            top1_count = max(1, int(np.ceil(dataset.direction_count * 0.01)))
            top5_count = max(1, int(np.ceil(dataset.direction_count * 0.05)))
            top1 = np.sum(sorted_energy[:, :top1_count], axis=1) / total_energy
            top5 = np.sum(sorted_energy[:, :top5_count], axis=1) / total_energy
            top1_parts.append(top1)
            top5_parts.append(top5)

            response_magnitude = np.sum(np.abs(response), axis=-1) * valid
            peak_indices = np.argmax(response_magnitude, axis=1)
            median_magnitude = np.asarray([
                float(np.median(row[row > 0.0])) if np.any(row > 0.0) else 0.0
                for row in response_magnitude
            ])
            peak_to_median = np.max(response_magnitude, axis=1) / np.maximum(median_magnitude, 1e-20)
            spacing = np.empty(len(peak_indices), dtype=np.float64)
            for local_index, peak_index in enumerate(peak_indices.tolist()):
                dots = np.clip(wi[local_index] @ wi[local_index, peak_index], -1.0, 1.0)
                dots[peak_index] = -1.0
                spacing[local_index] = np.degrees(np.arccos(np.max(dots)))
            peak_spacing_parts.append(spacing)
            response_by_family[family_id] = {
                "query_group_count": int(np.sum(family_mask)),
                "channels": channels,
                "integrated_absolute_energy": _distribution(np.sum(np.abs(energy_rgb), axis=1)),
                "top_1_percent_energy_fraction": _percentiles(top1, (0.5, 0.9, 0.95)),
                "top_5_percent_energy_fraction": _percentiles(top5, (0.5, 0.9, 0.95)),
                "peak_nearest_neighbor_angle_degrees": _percentiles(spacing, (0.05, 0.5, 0.9, 0.95)),
                "peak_to_median_response_ratio": _percentiles(peak_to_median, (0.5, 0.9, 0.95, 0.99)),
                "peak_relevant": _peak_relevant_summary(top1, spacing),
            }
            global_family_indices = np.flatnonzero(family_mask)
            group_top1[global_family_indices] = top1
            group_top5[global_family_indices] = top5
            group_peak_spacing[global_family_indices] = spacing
            group_peak_to_median[global_family_indices] = peak_to_median
            for local_index in np.argsort(top1)[-4:][::-1]:
                selected_position = int(global_family_indices[local_index])
                group_index = int(selected[selected_position])
                state = int(batch["state_index"][selected_position])
                peak_index = int(peak_indices[local_index])
                worst_groups.append({
                    "query_group_id": group_index,
                    "state_id": str(state_ids[state]),
                    "asset_id": str(asset_ids[state]),
                    "family_id": family_id,
                    "source_split": SPLIT_NAMES[int(selected_source_splits[selected_position])],
                    "query_role": QUERY_ROLE_NAMES[int(selected_roles[selected_position])],
                    "wo": np.asarray(batch["wo"][selected_position]).astype(float).tolist(),
                    "peak_wi": np.asarray(batch["wi"][selected_position, peak_index]).astype(float).tolist(),
                    "peak_response_rgb": np.asarray(batch["mean"][selected_position, peak_index]).astype(float).tolist(),
                    "top_1_percent_energy_fraction": float(top1[local_index]),
                    "top_5_percent_energy_fraction": float(top5[local_index]),
                    "peak_nearest_neighbor_angle_degrees": float(spacing[local_index]),
                })

        target = np.asarray(batch["mean"], dtype=np.float64)
        standard_error = np.asarray(batch["standard_error"], dtype=np.float64)
        valid = np.asarray(batch["valid"], dtype=bool)
        group_peak = np.max(np.abs(target), axis=(1, 2), keepdims=True)
        denominator = np.maximum(np.abs(target), np.maximum(0.005 * group_peak, 1e-8))
        relative_standard_error_rows = standard_error / denominator
        relative_standard_error = relative_standard_error_rows[valid]
        replica_delta = np.sum(
            np.abs(np.asarray(batch["replica_mean_a"], dtype=np.float64) - np.asarray(batch["replica_mean_b"], dtype=np.float64)),
            axis=(1, 2),
        ) / np.maximum(np.sum(np.abs(target), axis=(1, 2)), 1e-8)
        group_weights = np.asarray(batch["solid_angle_weight"], dtype=np.float64)
        group_energy = np.sum(np.abs(target) * group_weights[..., None] * valid[..., None], axis=(1, 2))
        group_sample_count = np.max(np.asarray(batch["sample_count"], dtype=np.float64), axis=1)
        group_relative_p95 = np.asarray([
            float(np.quantile(relative_standard_error_rows[index][valid[index]], 0.95))
            if np.any(valid[index]) else 0.0
            for index in range(len(selected))
        ])

        def grouped_uncertainty(mask: np.ndarray) -> dict[str, Any]:
            selected_mask = np.asarray(mask, dtype=bool)
            relative = relative_standard_error_rows[selected_mask]
            selected_valid = valid[selected_mask]
            return {
                "query_group_count": int(np.sum(selected_mask)),
                "relative_standard_error": _percentiles(
                    relative[selected_valid], (0.5, 0.9, 0.95, 0.99)
                ),
                "query_group_relative_standard_error_p95": _percentiles(
                    group_relative_p95[selected_mask], (0.5, 0.9, 0.95, 0.99)
                ),
                "replica_normalized_l1": _percentiles(
                    replica_delta[selected_mask], (0.5, 0.9, 0.95, 0.99)
                ),
                "integrated_absolute_energy": _distribution(group_energy[selected_mask]),
                "maximum_sample_count": _distribution(group_sample_count[selected_mask]),
            }

        selected_state_indices = np.asarray(batch["state_index"], dtype=np.int64)
        uncertainty_by_state: list[dict[str, Any]] = []
        for state in np.unique(selected_state_indices):
            state_mask = selected_state_indices == state
            uncertainty_by_state.append({
                "state_id": str(state_ids[state]),
                "asset_id": str(asset_ids[state]),
                "family_id": str(family_ids[state]),
                "source_split": SPLIT_NAMES[int(state_splits[state])],
                **grouped_uncertainty(state_mask),
            })
        uncertainty_by_state.sort(
            key=lambda item: item["replica_normalized_l1"].get("p95", 0.0),
            reverse=True,
        )
        energy_p50, energy_p90 = np.quantile(group_energy, (0.5, 0.9))
        uncertainty_by_energy = {
            "low_lt_p50": {
                "energy_range": {"maximum_exclusive": float(energy_p50)},
                **grouped_uncertainty(group_energy < energy_p50),
            },
            "middle_p50_to_lt_p90": {
                "energy_range": {"minimum": float(energy_p50), "maximum_exclusive": float(energy_p90)},
                **grouped_uncertainty((group_energy >= energy_p50) & (group_energy < energy_p90)),
            },
            "high_ge_p90": {
                "energy_range": {"minimum": float(energy_p90)},
                **grouped_uncertainty(group_energy >= energy_p90),
            },
        }
        worst_uncertainty_groups = []
        for selected_position in np.argsort(group_relative_p95)[-16:][::-1]:
            state = int(selected_state_indices[selected_position])
            worst_uncertainty_groups.append({
                "query_group_id": int(selected[selected_position]),
                "state_id": str(state_ids[state]),
                "asset_id": str(asset_ids[state]),
                "family_id": str(family_ids[state]),
                "source_split": SPLIT_NAMES[int(selected_source_splits[selected_position])],
                "query_role": QUERY_ROLE_NAMES[int(selected_roles[selected_position])],
                "wo": np.asarray(batch["wo"][selected_position]).astype(float).tolist(),
                "integrated_absolute_energy": float(group_energy[selected_position]),
                "relative_standard_error_p95": float(group_relative_p95[selected_position]),
                "replica_normalized_l1": float(replica_delta[selected_position]),
                "maximum_sample_count": int(group_sample_count[selected_position]),
            })

        response_by_query_role = {
            name: {
                "query_group_count": int(np.sum(selected_roles == code)),
                "integrated_absolute_energy": _distribution(group_energy[selected_roles == code]),
                "top_1_percent_energy_fraction": _percentiles(
                    group_top1[selected_roles == code], (0.5, 0.9, 0.95)
                ),
                "top_5_percent_energy_fraction": _percentiles(
                    group_top5[selected_roles == code], (0.5, 0.9, 0.95)
                ),
                "peak_nearest_neighbor_angle_degrees": _percentiles(
                    group_peak_spacing[selected_roles == code], (0.05, 0.5, 0.9, 0.95)
                ),
                "peak_to_median_response_ratio": _percentiles(
                    group_peak_to_median[selected_roles == code], (0.5, 0.9, 0.95, 0.99)
                ),
                "peak_relevant": _peak_relevant_summary(
                    group_top1[selected_roles == code], group_peak_spacing[selected_roles == code]
                ),
            }
            for name, code in _QUERY_ROLE_CODES.items()
        }

        train_available = np.flatnonzero(
            (query_source_splits == _SPLIT_CODES["train"])
            & (query_roles == _QUERY_ROLE_CODES["train"])
        )
        train_selected = _select_evenly(train_available, max_distribution_query_groups)
        train_batch = dataset.group_batch(train_selected)
        train_families = query_families[train_selected]
        transform_families = {
            family_id: _transform_family_statistics(
                train_batch["mean"][train_families == family_id],
                train_batch["valid"][train_families == family_id],
            )
            for family_id in sorted(set(map(str, train_families.tolist())))
        }
        transform_statistics: dict[str, Any] = {
            "format_name": TRANSFORM_STATISTICS_FORMAT,
            "format_version": TRANSFORM_STATISTICS_VERSION,
            "dataset_id": dataset.manifest.dataset_id,
            "fit_source_split": "train",
            "fit_query_role": "train",
            "query_group_count_available": int(len(train_available)),
            "query_group_count_used": int(len(train_selected)),
            "selection": "all-or-evenly-spaced@1",
            "families": transform_families,
        }
        transform_statistics["statistics_sha256"] = _sha256_json(transform_statistics)

        adversarial_proposal_text = " ".join(
            str(item["proposal_id"])
            for item in proposals
            if int(item["by_query_role"]["adversarial_probe"]) > 0
        ).lower()
        wi = np.asarray(batch["wi"], dtype=np.float64)
        wo = np.asarray(batch["wo"], dtype=np.float64)
        all_position_kinds = np.asarray(dataset.stream["queries/position_kind"], dtype=np.uint8)
        all_uv = np.asarray(dataset.stream["queries/uv"], dtype=np.float64)
        all_uv_dx = np.asarray(dataset.stream["queries/uv_dx"], dtype=np.float64)
        all_uv_dy = np.asarray(dataset.stream["queries/uv_dy"], dtype=np.float64)
        uv_mask = all_position_kinds == int(PositionKind.UV)
        uv = all_uv[uv_mask]
        uv_dx = all_uv_dx[uv_mask]
        uv_dy = all_uv_dy[uv_mask]
        footprint_length_x = np.linalg.norm(uv_dx, axis=1)
        footprint_length_y = np.linalg.norm(uv_dy, axis=1)
        footprint_dot = np.sum(uv_dx * uv_dy, axis=1)
        footprint_scale_keys = _footprint_scale_keys(footprint_length_x, footprint_length_y)
        footprint_rotations = np.mod(np.arctan2(uv_dx[:, 1], uv_dx[:, 0]), np.pi)
        footprint_rotation_keys = {
            round(float(value), 6)
            for value, length in zip(footprint_rotations, footprint_length_x, strict=True)
            if length > 0.0
        }
        adversarial_uv_mask = uv_mask & (query_roles == _QUERY_ROLE_CODES["adversarial_probe"])
        adversarial_uv_dx = all_uv_dx[adversarial_uv_mask]
        adversarial_uv_dy = all_uv_dy[adversarial_uv_mask]
        adversarial_length_x = np.linalg.norm(adversarial_uv_dx, axis=1)
        adversarial_length_y = np.linalg.norm(adversarial_uv_dy, axis=1)
        adversarial_rotations = {
            round(float(value), 6)
            for value, length in zip(
                np.mod(np.arctan2(adversarial_uv_dx[:, 1], adversarial_uv_dx[:, 0]), np.pi),
                adversarial_length_x,
                strict=True,
            )
            if length > 0.0
        }
        adversarial_scales = _footprint_scale_keys(adversarial_length_x, adversarial_length_y)
        seam_coverage = _uv_seam_coverage(uv) if len(uv) else {
            "edge_threshold": 0.01,
            "transverse_tolerance": 1e-5,
            "by_axis": {
                "u": {"low_edge_count": 0, "high_edge_count": 0, "matched_opposite_edge_pair": False},
                "v": {"low_edge_count": 0, "high_edge_count": 0, "matched_opposite_edge_pair": False},
            },
            "opposite_edge_pair_axis_count": 0,
        }
        adversarial_presence = {
            "peak": "peak" in adversarial_proposal_text or "microfacet" in adversarial_proposal_text,
            "grazing": "graz" in adversarial_proposal_text,
            "transmission_critical": "transmission" in adversarial_proposal_text or "critical" in adversarial_proposal_text,
            "spatial_footprint_rotation": len(adversarial_scales) >= 2 and len(adversarial_rotations) >= 2,
        }

        audit: dict[str, Any] = {
            "format_name": AUDIT_FORMAT,
            "format_version": AUDIT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "uri": str(Path(dataset_path).resolve()),
                "dataset_id": dataset.manifest.dataset_id,
                "created_at": dataset.manifest.created_at,
                "generator_git_commit": dataset.manifest.generator_git_commit,
                "query_profile_ids": list(dataset.manifest.query_profile_ids),
                "generation_config": dict(dataset.manifest.generation_config),
                "provider_metadata": [dict(item) for item in dataset.manifest.provider_metadata],
                "counts": dict(dataset.manifest.counts),
                "content_hash_verified": verify_hashes,
            },
            "roles": {
                "source_split": "源状态/资产泛化轴；train、validation、test 的 split_group 不得交叉",
                "query_role.train": "训练 query；只有 source train × query train 可拟合 target transform",
                "query_role.validation": "同一合同中的固定 validation query；不参与 transform 统计",
                "query_role.test": "独立 held-out query；不参与模型选择或 transform 统计",
                "query_role.adversarial_probe": "peak/掠射/透射/footprint 覆盖诊断；不参与模型选择",
            },
            "sampling": {
                "selection": "all-or-evenly-spaced@1",
                "query_group_count_available": dataset.query_group_count,
                "query_group_count_used": int(len(selected)),
                "limit": max_distribution_query_groups,
            },
            "state_distribution": {
                family_id: {
                    name: int(np.sum((family_ids == family_id) & (state_splits == code)))
                    for name, code in _SPLIT_CODES.items()
                }
                for family_id in sorted(set(map(str, family_ids.tolist())))
            },
            "split_audit": {
                "split_group_id": _split_leaks(split_group_ids, state_splits),
                "asset_id": _split_leaks(asset_ids, state_splits),
                "source_sha256": _split_leaks(source_hashes, state_splits),
                "parent_child_cross_split_count": len(parent_cross_split),
                "parent_child_cross_split_examples": parent_cross_split[:16],
                "state_counts": {
                    name: int(np.sum(state_splits == code)) for name, code in _SPLIT_CODES.items()
                },
                "query_group_counts": {
                    name: int(np.sum(query_source_splits == code)) for name, code in _SPLIT_CODES.items()
                },
                "query_role_counts": {
                    name: int(np.sum(query_roles == code)) for name, code in _QUERY_ROLE_CODES.items()
                },
            },
            "response": {
                "measure": dataset.manifest.response_measure,
                "color_model": dataset.manifest.color_model,
                "by_family": response_by_family,
                "by_query_role": response_by_query_role,
                "integrated_absolute_energy": _distribution(np.concatenate(group_energy_parts)),
                "top_1_percent_energy_fraction": _percentiles(np.concatenate(top1_parts), (0.5, 0.9, 0.95)),
                "top_5_percent_energy_fraction": _percentiles(np.concatenate(top5_parts), (0.5, 0.9, 0.95)),
                "peak_nearest_neighbor_angle_degrees": _percentiles(np.concatenate(peak_spacing_parts), (0.05, 0.5, 0.9, 0.95)),
                "high_concentration_groups": sorted(
                    worst_groups,
                    key=lambda item: item["top_1_percent_energy_fraction"],
                    reverse=True,
                )[:16],
            },
            "reference_uncertainty": {
                "relative_standard_error": _percentiles(relative_standard_error, (0.5, 0.9, 0.95, 0.99)),
                "replica_normalized_l1": _percentiles(replica_delta, (0.5, 0.9, 0.95, 0.99)),
                "deterministic_query_fraction": float(np.mean(np.asarray(batch["variance"]) == 0.0)),
                "sample_count": _distribution(np.asarray(batch["sample_count"], dtype=np.float64)),
                "by_source_split": {
                    name: grouped_uncertainty(selected_source_splits == code)
                    for name, code in _SPLIT_CODES.items()
                },
                "by_query_role": {
                    name: grouped_uncertainty(selected_roles == code)
                    for name, code in _QUERY_ROLE_CODES.items()
                },
                "by_state": uncertainty_by_state,
                "by_integrated_energy": uncertainty_by_energy,
                "worst_query_group": {
                    "relative_standard_error_p95": float(np.max(group_relative_p95)),
                    "replica_normalized_l1": float(np.max(replica_delta)),
                },
                "highest_relative_standard_error_groups": worst_uncertainty_groups,
            },
            "coverage": {
                "proposals": proposals,
                "adversarial_profile_presence": adversarial_presence,
                "direction_grid_independence": {
                    "source_split": {
                        "wi_grid": _cross_partition_hash_audit(wi_grid_hashes, query_source_splits, SPLIT_NAMES),
                        "wo": _cross_partition_hash_audit(wo_hashes, query_source_splits, SPLIT_NAMES),
                    },
                    "query_role": {
                        "wi_grid": _cross_partition_hash_audit(wi_grid_hashes, query_roles, QUERY_ROLE_NAMES),
                        "wo": _cross_partition_hash_audit(wo_hashes, query_roles, QUERY_ROLE_NAMES),
                    },
                    "interpretation": "exact float32 hash overlap；正式 gate 使用 query role 轴，source split 轴另行报告",
                },
                "absolute_wo_cosine": _distribution(np.abs(wo[:, 2])),
                "absolute_wi_cosine": _distribution(np.abs(wi[..., 2])),
                "wi_transmission_fraction": float(np.mean(wi[..., 2] < 0.0)),
                "wi_grazing_fraction_abs_cos_below_sin_5deg": float(np.mean(np.abs(wi[..., 2]) < np.sin(np.deg2rad(5.0)))),
                "wo_grazing_fraction_abs_cos_below_sin_5deg": float(np.mean(np.abs(wo[:, 2]) < np.sin(np.deg2rad(5.0)))),
                "by_query_role": {
                    name: {
                        "query_group_count": int(np.sum(selected_roles == code)),
                        "wi_transmission_fraction": float(np.mean(wi[selected_roles == code, ..., 2] < 0.0))
                        if np.any(selected_roles == code) else 0.0,
                        "wi_grazing_fraction_abs_cos_below_sin_5deg": float(np.mean(
                            np.abs(wi[selected_roles == code, ..., 2]) < np.sin(np.deg2rad(5.0))
                        )) if np.any(selected_roles == code) else 0.0,
                        "wo_grazing_fraction_abs_cos_below_sin_5deg": float(np.mean(
                            np.abs(wo[selected_roles == code, 2]) < np.sin(np.deg2rad(5.0))
                        )) if np.any(selected_roles == code) else 0.0,
                    }
                    for name, code in _QUERY_ROLE_CODES.items()
                },
                "position_kinds": {
                    str(int(code)): int(np.sum(np.asarray(batch["position_kind"]) == code))
                    for code in np.unique(np.asarray(batch["position_kind"]))
                },
                "footprint_length_x": _distribution(footprint_length_x),
                "footprint_length_y": _distribution(footprint_length_y),
                "footprint_axis_dot": _distribution(footprint_dot),
                "unique_footprint_scale_count": len(footprint_scale_keys),
                "unique_footprint_rotation_count": len(footprint_rotation_keys),
                "uv_seam": seam_coverage,
            },
            "target_transform_statistics": {
                "uri": "target_transform_statistics.json",
                "statistics_sha256": transform_statistics["statistics_sha256"],
                "fit_source_split": "train",
                "fit_query_role": "train",
            },
        }
        audit["audit_sha256"] = _sha256_json(audit)

    _write_json_atomic(output / "audit.json", audit)
    _write_json_atomic(output / "target_transform_statistics.json", transform_statistics)
    _write_text_atomic(output / "report.md", _audit_markdown(audit))
    if gate_path is not None:
        gate_result = evaluate_supervision_gate(audit, load_supervision_gate(gate_path))
        _write_json_atomic(output / "gate_result.json", gate_result)
        audit["gate_result"] = gate_result
    return audit
