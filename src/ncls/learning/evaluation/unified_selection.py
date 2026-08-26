from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SELECTION_PROTOCOL_PATH = PROJECT_ROOT / "configs/evaluation/unified-method-selection-v1.json"
_SELECTION_AXES = (
    "directional_l1_by_state",
    "signed_energy_absolute_error_by_state",
    "cosine_relative_variance_by_state",
    "single_query_time_microseconds_by_state",
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_report(
    path: Path | str,
    *,
    identity_field: str = "report_sha256",
) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    claimed = str(document.get(identity_field, ""))
    unsigned = dict(document)
    unsigned.pop(identity_field, None)
    if len(claimed) != 64 or claimed != _sha256_json(unsigned):
        raise ValueError(f"artifact identity is invalid: {source}")
    return source, document


def load_unified_selection_protocol(
    path: Path | str = SELECTION_PROTOCOL_PATH,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {
        "schema", "name", "seed", "bootstrap_iterations", "cells", "rules"
    }:
        raise ValueError("unified selection protocol fields are not frozen v1")
    if value["schema"] != {"name": "unified-method-selection", "version": 1}:
        raise ValueError("unsupported unified selection protocol")
    if value["name"] != "unified-method-selection-v1":
        raise ValueError("unified selection protocol name is unsupported")
    if value["seed"] != 20260824 or value["bootstrap_iterations"] < 1000:
        raise ValueError("unified selection seed/bootstrap contract drifted")
    expected_cells = {
        "A": {"evaluator": "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1", "sampler": "nvidia-diffuse-ggx9", "baseline": True},
        "B": {"evaluator": "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1", "sampler": "ltc-k2", "baseline": False},
        "C": {"evaluator": "core-frame-neural-v1", "sampler": "nvidia-diffuse-ggx9", "baseline": False},
        "D": {"evaluator": "core-frame-neural-v1", "sampler": "ltc-k2", "baseline": False},
    }
    if value["cells"] != expected_cells:
        raise ValueError("unified 2x2 cell identities drifted")
    if value["rules"] != {
        "require_implementation_correctness": True,
        "require_evaluator_convergence": True,
        "require_sampler_convergence": True,
        "require_sampler_correctness": True,
        "require_checkpoint_parity": True,
        "credible_interval": 0.95,
        "credible_improvement_requires_upper_below_zero": True,
        "credible_regression_requires_lower_above_zero": True,
        "fallback_cell": "A",
        "candidate_tiebreak": [
            "credible_improvement_axis_count_descending",
            "single_query_time_median_ascending",
            "total_static_bytes_ascending",
            "cell_id_ascending",
        ],
        "baseline_ineligible_local_pair": ["C", "D"],
        "baseline_ineligible_priority": ["B", "C", "D"],
    }:
        raise ValueError("unified mechanical selection rules drifted")
    return value


def unified_selection_protocol_sha256(
    path: Path | str = SELECTION_PROTOCOL_PATH,
) -> str:
    return _sha256_json(load_unified_selection_protocol(path))


def paired_state_difference(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    iterations: int = 1000,
    seed: int = 20260824,
) -> dict[str, Any]:
    """state为重采样单位，返回candidate-baseline均值差的冻结CI。"""

    if iterations < 1000 or set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired state bootstrap requires >=1000 matched nonempty states")
    state_ids = sorted(baseline)
    difference = np.asarray(
        [float(candidate[state]) - float(baseline[state]) for state in state_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(difference), size=(iterations, len(difference)))
    samples = np.mean(difference[selected], axis=1)
    interval = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
    return {
        "state_count": len(state_ids),
        "iterations": iterations,
        "seed": seed,
        "candidate_minus_baseline": float(np.mean(difference)),
        "bootstrap_95_ci": interval,
        "credible_improvement": interval[1] < 0.0,
        "credible_regression": interval[0] > 0.0,
    }


def _finite_state_metric(cell: Mapping[str, Any], axis: str) -> dict[str, float]:
    metrics = cell.get("metrics")
    values = None if not isinstance(metrics, Mapping) else metrics.get(axis)
    if not isinstance(values, Mapping) or len(values) != 30:
        raise ValueError(f"unified selection axis {axis} requires exactly 30 states")
    result = {str(name): float(value) for name, value in values.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError(f"unified selection axis {axis} must be finite and nonnegative")
    return result


def _total_static_bytes(cell: Mapping[str, Any]) -> int:
    cost = cell.get("cost")
    if not isinstance(cost, Mapping):
        raise ValueError("unified selection cell cost is missing")
    asset = int(cost.get("B_asset", -1))
    shared = int(cost.get("B_shared", -1))
    if asset < 0 or shared < 0:
        raise ValueError("unified selection static byte costs must be nonnegative")
    return asset + shared


def compare_unified_cells(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    iterations: int = 1000,
    seed: int = 20260824,
) -> dict[str, Any]:
    """按state block为四个主轴生成candidate-baseline CI，并精确比较静态bytes。"""

    axes = {
        axis: paired_state_difference(
            _finite_state_metric(baseline, axis),
            _finite_state_metric(candidate, axis),
            iterations=iterations,
            seed=seed + index * 104729,
        )
        for index, axis in enumerate(_SELECTION_AXES)
    }
    baseline_bytes = _total_static_bytes(baseline)
    candidate_bytes = _total_static_bytes(candidate)
    byte_delta = candidate_bytes - baseline_bytes
    memory = {
        "baseline_total_static_bytes": baseline_bytes,
        "candidate_total_static_bytes": candidate_bytes,
        "candidate_minus_baseline": byte_delta,
        "credible_improvement": byte_delta < 0,
        "credible_regression": byte_delta > 0,
    }
    improvement_axes = [
        name for name, evidence in {**axes, "total_static_bytes": memory}.items()
        if evidence["credible_improvement"]
    ]
    regression_axes = [
        name for name, evidence in {**axes, "total_static_bytes": memory}.items()
        if evidence["credible_regression"]
    ]
    return {
        "axes": axes,
        "total_static_bytes": memory,
        "credible_improvement_axes": improvement_axes,
        "credible_regression_axes": regression_axes,
        "pareto_eligible": bool(improvement_axes and not regression_axes),
    }


def _validate_cell_identity(
    cell_id: str,
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> bool:
    expected = protocol["cells"][cell_id]
    if cell.get("evaluator") != expected["evaluator"] or cell.get("sampler") != expected["sampler"]:
        raise ValueError(f"unified selection cell {cell_id} identity drifted")
    implementation = cell.get("implementation_correctness")
    evaluator_convergence = cell.get("evaluator_convergence")
    sampler_convergence = cell.get("sampler_convergence")
    correctness = cell.get("sampler_correctness")
    parity = cell.get("checkpoint_parity")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(evaluator_convergence, Mapping)
        or not isinstance(sampler_convergence, Mapping)
        or not isinstance(correctness, Mapping)
        or not isinstance(parity, Mapping)
    ):
        raise ValueError(f"unified selection cell {cell_id} gate evidence is missing")
    for artifact_name in ("evaluator_checkpoint", "sampler_checkpoint", "compiled_set"):
        artifact = cell.get(artifact_name)
        if (
            not isinstance(artifact, Mapping)
            or not str(artifact.get("uri", ""))
            or len(str(artifact.get("sha256", ""))) != 64
        ):
            raise ValueError(f"unified selection cell {cell_id} artifact identity is invalid")
    for axis in _SELECTION_AXES:
        _finite_state_metric(cell, axis)
    _total_static_bytes(cell)
    for name in ("slang_implementation_sha256", "layout_sha256"):
        value = str(cell.get(name, ""))
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"unified selection cell {cell_id} {name} is invalid")
    return bool(
        implementation.get("passed") is True
        and evaluator_convergence.get("passed") is True
        and sampler_convergence.get("passed") is True
        and correctness.get("passed") is True
        and parity.get("passed") is True
    )


def _selection_cell_from_artifacts(
    cell_id: str,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "audit", "checkpoint_label", "implementation_correctness",
        "evaluator_convergence", "sampler_convergence", "sampler_correctness",
        "benchmark", "compiled", "parity",
    }
    if set(spec) != required:
        raise ValueError(f"selection cell {cell_id} artifact inputs drifted")
    audit_path, audit = _load_hashed_report(spec["audit"])
    implementation_path, implementation = _load_hashed_report(
        spec["implementation_correctness"]
    )
    evaluator_convergence_path, evaluator_convergence = _load_hashed_report(
        spec["evaluator_convergence"]
    )
    sampler_convergence_path, sampler_convergence = _load_hashed_report(
        spec["sampler_convergence"]
    )
    correctness_path, correctness = _load_hashed_report(spec["sampler_correctness"])
    benchmark_path, benchmark = _load_hashed_report(spec["benchmark"])
    parity_path, parity = _load_hashed_report(spec["parity"])
    compiled_path, compiled = _load_hashed_report(
        spec["compiled"], identity_field="compiled_set_id"
    )
    expected = protocol["cells"][cell_id]
    label = str(spec["checkpoint_label"])
    checkpoints = audit.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or label not in checkpoints:
        raise ValueError(f"selection audit checkpoint label is missing: {label}")
    audited = checkpoints[label]
    roles = audited.get("roles") if isinstance(audited, Mapping) else None
    test = roles.get("test") if isinstance(roles, Mapping) else None
    evaluator_checkpoint = audited.get("checkpoint") if isinstance(audited, Mapping) else None
    if not isinstance(test, Mapping) or not isinstance(evaluator_checkpoint, Mapping):
        raise ValueError("selection requires an audited evaluator test role")
    if evaluator_checkpoint.get("pipeline") != expected["evaluator"]:
        raise ValueError(f"selection cell {cell_id} evaluator identity mismatch")
    evaluator_sha256 = str(evaluator_checkpoint.get("sha256", ""))
    if len(evaluator_sha256) != 64:
        raise ValueError("selection evaluator checkpoint hash is invalid")
    evaluator_implementation = evaluator_checkpoint.get("implementation_identity")
    if not isinstance(evaluator_implementation, Mapping):
        raise ValueError("selection evaluator implementation identity is missing")
    data_id = str(audit.get("data", {}).get("data_id", ""))
    if len(data_id) != 64:
        raise ValueError("selection audit data identity is invalid")

    if (
        correctness.get("format_name") != "unified-sampler-correctness-report"
        or correctness.get("status") != "complete"
        or correctness.get("pipeline") != expected["evaluator"]
        or correctness.get("sampler") != expected["sampler"]
        or correctness.get("data_id") != data_id
        or correctness.get("evaluator_checkpoint", {}).get("sha256") != evaluator_sha256
    ):
        raise ValueError(f"selection cell {cell_id} sampler correctness identity mismatch")
    sampler_checkpoint = correctness.get("sampler_checkpoint")
    if not isinstance(sampler_checkpoint, Mapping) or len(
        str(sampler_checkpoint.get("sha256", ""))
    ) != 64:
        raise ValueError("selection sampler checkpoint identity is invalid")
    sampler_sha256 = str(sampler_checkpoint["sha256"])

    if (
        implementation.get("format_name") != "unified-method-correctness-report"
        or implementation.get("format_version") != 1
        or implementation.get("status") != "complete"
        or implementation.get("pipeline") != expected["evaluator"]
        or implementation.get("data_id") != data_id
    ):
        raise ValueError(f"selection cell {cell_id} implementation evidence mismatch")
    if (
        evaluator_convergence.get("format_name")
        != "unified-training-convergence-report"
        or evaluator_convergence.get("format_version") != 1
        or evaluator_convergence.get("status") != "complete"
        or evaluator_convergence.get("stage") != "evaluator"
        or evaluator_convergence.get("pipeline") != expected["evaluator"]
        or evaluator_convergence.get("data_id") != data_id
        or evaluator_convergence.get("checkpoint", {}).get("sha256")
        != evaluator_sha256
    ):
        raise ValueError(f"selection cell {cell_id} evaluator convergence mismatch")
    if (
        sampler_convergence.get("format_name")
        != "unified-training-convergence-report"
        or sampler_convergence.get("format_version") != 1
        or sampler_convergence.get("status") != "complete"
        or sampler_convergence.get("stage") != "sampler"
        or sampler_convergence.get("pipeline") != expected["evaluator"]
        or sampler_convergence.get("sampler") != expected["sampler"]
        or sampler_convergence.get("data_id") != data_id
        or sampler_convergence.get("evaluator_checkpoint", {}).get("sha256")
        != evaluator_sha256
        or sampler_convergence.get("checkpoint", {}).get("sha256")
        != sampler_sha256
    ):
        raise ValueError(f"selection cell {cell_id} sampler convergence mismatch")
    sampler_implementation = sampler_checkpoint.get("implementation_identity")
    if not isinstance(sampler_implementation, Mapping):
        raise ValueError("selection sampler implementation identity is missing")

    if (
        compiled.get("format_name") != "unified-compiled-material-set"
        or compiled.get("pipeline") != expected["evaluator"]
        or compiled.get("sampler") != expected["sampler"]
        or compiled.get("data_id") != data_id
        or compiled.get("evaluator_checkpoint_sha256") != evaluator_sha256
        or compiled.get("sampler_checkpoint_sha256") != sampler_sha256
        or compiled.get("evaluator_implementation_identity") != evaluator_implementation
        or compiled.get("sampler_implementation_identity") != sampler_implementation
    ):
        raise ValueError(f"selection cell {cell_id} compiled set identity mismatch")
    if (
        parity.get("format_name") != "unified-checkpoint-parity-report"
        or parity.get("pipeline") != expected["evaluator"]
        or parity.get("sampler") != expected["sampler"]
        or parity.get("data_id") != data_id
        or parity.get("compiled_set_id") != compiled["compiled_set_id"]
        or parity.get("evaluator_checkpoint_sha256") != evaluator_sha256
        or parity.get("sampler_checkpoint_sha256") != sampler_sha256
    ):
        raise ValueError(f"selection cell {cell_id} checkpoint parity identity mismatch")
    if (
        benchmark.get("schema") != {"name": "p1-query-benchmark", "version": 1}
        or benchmark.get("data_id") != data_id
        or benchmark.get("pipeline") != expected["evaluator"]
        or benchmark.get("checkpoint_sha256") != evaluator_sha256
    ):
        raise ValueError(f"selection cell {cell_id} benchmark identity mismatch")
    shared_identities = (
        str(implementation.get("slang_implementation_sha256", "")),
        str(implementation.get("layout_sha256", "")),
    )
    if (
        evaluator_convergence.get("slang_implementation_sha256")
        != shared_identities[0]
        or evaluator_convergence.get("layout_sha256") != shared_identities[1]
        or sampler_convergence.get("slang_implementation_sha256")
        != shared_identities[0]
        or sampler_convergence.get("layout_sha256") != shared_identities[1]
        or correctness.get("slang_implementation_sha256") != shared_identities[0]
        or correctness.get("layout_sha256") != shared_identities[1]
        or compiled.get("slang_implementation_sha256") != shared_identities[0]
        or compiled.get("layout_sha256") != shared_identities[1]
        or parity.get("slang_implementation_sha256") != shared_identities[0]
        or parity.get("layout_sha256") != shared_identities[1]
        or evaluator_implementation.get("slang_implementation_sha256")
        != shared_identities[0]
        or evaluator_implementation.get("layout_sha256") != shared_identities[1]
        or sampler_implementation.get("slang_implementation_sha256")
        != shared_identities[0]
        or sampler_implementation.get("layout_sha256") != shared_identities[1]
    ):
        raise ValueError(f"selection cell {cell_id} Slang/layout identity mismatch")

    states = test.get("states")
    if not isinstance(states, Mapping) or len(states) != 30:
        raise ValueError("selection test metrics require exactly 30 states")
    directional = {
        str(state_id): float(record["directional_l1"])
        for state_id, record in states.items()
    }
    signed_energy = {
        str(state_id): abs(float(record["signed_energy"]["signed_relative_bias"]))
        for state_id, record in states.items()
    }
    variance_parts: dict[str, list[float]] = {str(state_id): [] for state_id in states}
    cases = correctness.get("cases")
    if not isinstance(cases, list) or len(cases) != 120:
        raise ValueError("selection sampler variance requires the frozen 30x4 cases")
    for case in cases:
        state_id = str(case.get("state_id", ""))
        if state_id not in variance_parts:
            raise ValueError("selection sampler state does not match evaluator test states")
        variance_parts[state_id].append(
            float(case["mc_unbiasedness"]["cosine_relative_variance"])
        )
    if any(len(values) != 4 for values in variance_parts.values()):
        raise ValueError("selection sampler variance requires four views per state")
    variance = {
        state_id: float(np.mean(values)) for state_id, values in variance_parts.items()
    }
    timing = benchmark.get("single_query_time_microseconds_by_state")
    if not isinstance(timing, Mapping) or set(map(str, timing)) != set(states):
        raise ValueError("selection benchmark requires all 30 matched states")

    return {
        "evaluator": expected["evaluator"],
        "sampler": expected["sampler"],
        "evaluator_checkpoint": dict(evaluator_checkpoint),
        "sampler_checkpoint": dict(sampler_checkpoint),
        "compiled_set": {
            "uri": str(compiled_path),
            "sha256": _sha256_file(compiled_path),
            "compiled_set_id": compiled["compiled_set_id"],
        },
        "implementation_correctness": {
            "uri": str(implementation_path),
            "sha256": _sha256_file(implementation_path),
            "report_sha256": implementation["report_sha256"],
            "passed": implementation.get("passed") is True,
        },
        "evaluator_convergence": {
            "uri": str(evaluator_convergence_path),
            "sha256": _sha256_file(evaluator_convergence_path),
            "report_sha256": evaluator_convergence["report_sha256"],
            "passed": evaluator_convergence.get("passed") is True,
        },
        "sampler_convergence": {
            "uri": str(sampler_convergence_path),
            "sha256": _sha256_file(sampler_convergence_path),
            "report_sha256": sampler_convergence["report_sha256"],
            "passed": sampler_convergence.get("passed") is True,
        },
        "sampler_correctness": {
            "uri": str(correctness_path),
            "sha256": _sha256_file(correctness_path),
            "report_sha256": correctness["report_sha256"],
            "passed": correctness.get("passed") is True,
        },
        "checkpoint_parity": {
            "uri": str(parity_path),
            "sha256": _sha256_file(parity_path),
            "report_sha256": parity["report_sha256"],
            "passed": parity.get("passed") is True,
        },
        "metrics": {
            "directional_l1_by_state": directional,
            "signed_energy_absolute_error_by_state": signed_energy,
            "cosine_relative_variance_by_state": variance,
            "single_query_time_microseconds_by_state": {
                str(state_id): float(value) for state_id, value in timing.items()
            },
        },
        "cost": dict(compiled["cost"]),
        "evidence": {
            "audit": {"uri": str(audit_path), "sha256": _sha256_file(audit_path)},
            "benchmark": {
                "uri": str(benchmark_path), "sha256": _sha256_file(benchmark_path)
            },
        },
        "slang_implementation_sha256": shared_identities[0],
        "layout_sha256": shared_identities[1],
        "data_id": data_id,
    }


def build_unified_selection_manifest(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    data_id: str,
    source_git_commit: str,
    protocol_path: Path | str = SELECTION_PROTOCOL_PATH,
) -> dict[str, Any]:
    """执行冻结2x2规则并生成唯一选择身份；不允许调用方手工覆盖结果。"""

    protocol = load_unified_selection_protocol(protocol_path)
    if set(cells) != {"A", "B", "C", "D"}:
        raise ValueError("unified selection requires exactly cells A-D")
    for name, value in {"data_id": data_id}.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"unified selection {name} must be a SHA-256 identity")
    if len(source_git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_git_commit
    ):
        raise ValueError("unified selection source_git_commit must be a full Git SHA-1")

    normalized = {cell_id: dict(cells[cell_id]) for cell_id in "ABCD"}
    eligible = {
        cell_id: _validate_cell_identity(cell_id, normalized[cell_id], protocol)
        for cell_id in "ABCD"
    }
    for cell_id in "ABCD":
        normalized[cell_id]["eligible"] = eligible[cell_id]

    iterations = int(protocol["bootstrap_iterations"])
    seed = int(protocol["seed"])
    comparisons_to_a = {
        cell_id: compare_unified_cells(
            normalized["A"], normalized[cell_id], iterations=iterations, seed=seed
        )
        for cell_id in "BCD"
    }
    for cell_id, comparison in comparisons_to_a.items():
        normalized[cell_id]["comparison_to_A"] = comparison

    if eligible["A"]:
        viable = [
            cell_id for cell_id in "BCD"
            if eligible[cell_id] and comparisons_to_a[cell_id]["pareto_eligible"]
        ]
        if viable:
            viable.sort(key=lambda cell_id: (
                -len(comparisons_to_a[cell_id]["credible_improvement_axes"]),
                float(np.median(list(_finite_state_metric(
                    normalized[cell_id], "single_query_time_microseconds_by_state"
                ).values()))),
                _total_static_bytes(normalized[cell_id]),
                cell_id,
            ))
            selected = viable[0]
            reason = (
                f"cell {selected} has valid implementation/convergence/correctness evidence and credible Pareto improvement "
                f"on {comparisons_to_a[selected]['credible_improvement_axes']} without a main-axis regression"
            )
        else:
            selected = "A"
            reason = "no eligible candidate has a credible Pareto improvement without regression; selected frozen baseline A"
    else:
        direct_alternative = "B" if eligible["B"] else None
        core_selected: str | None = None
        if eligible["C"]:
            core_selected = "C"
            if eligible["D"]:
                d_to_c = compare_unified_cells(
                    normalized["C"], normalized["D"], iterations=iterations, seed=seed
                )
                normalized["D"]["comparison_to_C"] = d_to_c
                if d_to_c["pareto_eligible"]:
                    core_selected = "D"
        elif eligible["D"]:
            core_selected = "D"
        available = [value for value in (direct_alternative, core_selected) if value]
        if not available:
            raise ValueError(
                "unified selection has no implementation/convergence/correctness-eligible cell"
            )
        priority = protocol["rules"]["baseline_ineligible_priority"]
        selected = min(available, key=priority.index)
        reason = f"baseline A is ineligible; selected first legal cell {selected} under the frozen fallback priority"

    manifest: dict[str, Any] = {
        "format_name": "unified-method-selection",
        "format_version": 1,
        "protocol_sha256": unified_selection_protocol_sha256(protocol_path),
        "data_id": data_id,
        "source_git_commit": source_git_commit,
        "cells": normalized,
        "selected_cell": selected,
        "reason": reason,
    }
    manifest["selection_id"] = _sha256_json(manifest)
    return manifest


def build_unified_selection_from_artifacts(
    inputs_path: Path | str,
    output_path: Path | str,
    *,
    source_git_commit: str,
    protocol_path: Path | str = SELECTION_PROTOCOL_PATH,
) -> dict[str, Any]:
    """从正式audit/benchmark/compiled/parity产物组装四格并执行冻结选择。"""

    inputs_source = Path(inputs_path).resolve()
    inputs = json.loads(inputs_source.read_text(encoding="utf-8"))
    if set(inputs) != {"format_name", "format_version", "cells"} or inputs.get(
        "format_name"
    ) != "unified-selection-inputs" or inputs.get("format_version") != 1:
        raise ValueError("unified selection inputs document is unsupported")
    specs = inputs.get("cells")
    if not isinstance(specs, Mapping) or set(specs) != {"A", "B", "C", "D"}:
        raise ValueError("unified selection inputs require exactly cells A-D")
    protocol = load_unified_selection_protocol(protocol_path)
    cells = {
        cell_id: _selection_cell_from_artifacts(cell_id, specs[cell_id], protocol)
        for cell_id in "ABCD"
    }
    data_ids = {str(cell.pop("data_id")) for cell in cells.values()}
    if len(data_ids) != 1:
        raise ValueError("unified selection cells do not share one data identity")
    for cell in cells.values():
        cell["evidence"]["selection_inputs"] = {
            "uri": str(inputs_source),
            "sha256": _sha256_file(inputs_source),
        }
    manifest = build_unified_selection_manifest(
        cells,
        data_id=data_ids.pop(),
        source_git_commit=source_git_commit,
        protocol_path=protocol_path,
    )
    write_unified_selection_manifest(output_path, manifest)
    return manifest


def write_unified_selection_manifest(
    path: Path | str,
    manifest: Mapping[str, Any],
) -> None:
    expected = dict(manifest)
    selection_id = expected.pop("selection_id", None)
    if selection_id != _sha256_json(expected):
        raise ValueError("unified selection manifest identity is invalid")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
