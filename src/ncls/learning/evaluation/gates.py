from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ncls.learning.training.checkpoint import sha256_file


GATE_FORMAT = "ncls.evaluator-acceptance-gate"
GATE_VERSION = 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_evaluator_gate(path: Path | str) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("format_name") != GATE_FORMAT or value.get("format_version") != GATE_VERSION:
        raise ValueError("unsupported evaluator acceptance gate format")
    if "@" not in str(value.get("gate_id", "")) or value.get("status") != "frozen":
        raise ValueError("evaluator acceptance gate must be versioned and frozen")
    if not isinstance(value.get("metric_requirements"), Mapping) or not isinstance(
        value.get("cost_requirements"), Mapping
    ):
        raise ValueError("evaluator acceptance gate requirements must be objects")
    return value


def evaluate_evaluator_gate(
    run_manifest_path: Path | str,
    test_metrics_path: Path | str,
    adversarial_metrics_path: Path | str,
    gate_path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    run_path = Path(run_manifest_path)
    run = _read_json(run_path)
    evaluations = {
        "test": _read_json(test_metrics_path),
        "adversarial_probe": _read_json(adversarial_metrics_path),
    }
    gate = load_evaluator_gate(gate_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, actual: Any, expected: Any, passed: bool) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "passed": bool(passed)})

    check("run.status", run.get("status"), "complete", run.get("status") == "complete")
    check(
        "run.research_role",
        run.get("research_role"),
        gate.get("required_research_role"),
        run.get("research_role") == gate.get("required_research_role"),
    )
    check(
        "run.held_out_test_accessed",
        run.get("held_out_test_accessed"),
        False,
        run.get("held_out_test_accessed") is False,
    )
    best = run.get("checkpoints", {}).get("best", {})
    checkpoint_path = (run_path.parent / str(best.get("uri", ""))).resolve()
    expected_checkpoint_hash = str(best.get("sha256", ""))
    actual_checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    check(
        "checkpoint.best.sha256",
        actual_checkpoint_hash,
        expected_checkpoint_hash,
        actual_checkpoint_hash == expected_checkpoint_hash,
    )

    for role, evaluation in evaluations.items():
        check(
            f"evaluation.{role}.role",
            evaluation.get("evaluation_role"),
            role,
            evaluation.get("evaluation_role") == role,
        )
        check(
            f"evaluation.{role}.dataset_id",
            evaluation.get("dataset_id"),
            run.get("dataset_id"),
            evaluation.get("dataset_id") == run.get("dataset_id"),
        )
        for metric_path, requirement in gate["metric_requirements"].get(role, {}).items():
            metric_name, summary_name = metric_path.split(".", 1)
            actual = float(evaluation["metrics"][metric_name][summary_name])
            minimum = requirement.get("minimum")
            maximum = requirement.get("maximum")
            passed = (minimum is None or actual >= float(minimum)) and (
                maximum is None or actual <= float(maximum)
            )
            check(f"metric.{role}.{metric_path}", actual, dict(requirement), passed)

    costs = run.get("model_costs", {})
    for name, requirement in gate["cost_requirements"].items():
        actual_value = costs.get(name)
        actual = float(actual_value) if isinstance(actual_value, (int, float)) else None
        minimum = requirement.get("minimum")
        maximum = requirement.get("maximum")
        passed = actual is not None and (minimum is None or actual >= float(minimum)) and (
            maximum is None or actual <= float(maximum)
        )
        check(f"cost.{name}", actual, dict(requirement), passed)

    result: dict[str, Any] = {
        "format_name": "ncls.evaluator-acceptance-gate-result",
        "format_version": 1,
        "gate_id": gate["gate_id"],
        "gate_sha256": _canonical_sha256(gate),
        "run_id": run.get("run_id"),
        "pipeline_id": run.get("pipeline_id"),
        "candidate_id": run.get("candidate_id"),
        "dataset_id": run.get("dataset_id"),
        "checkpoint_sha256": actual_checkpoint_hash,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    result["result_sha256"] = _canonical_sha256(result)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    return result
