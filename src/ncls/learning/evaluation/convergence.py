from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ncls.learning.models import (
    NvidiaNeuralAppearanceModel,
    adapt_nvidia_model_for_sampler,
)
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig
from ncls.learning.training.sampler_config import SamplerTrainingConfig
from ncls.learning.training.sampler_runner import evaluate_sampler_validation

from .evaluator import evaluate_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ConvergenceProtocol:
    """只冻结相对收敛的统计协议，不包含任何绝对材质质量线。"""

    name: str
    bootstrap_iterations: int
    bootstrap_seed: int
    confidence: float
    late_window_fraction: float
    minimum_late_validation_records: int
    required_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.name != "validation-relative-convergence-v1":
            raise ValueError("unsupported convergence protocol name")
        if self.bootstrap_iterations < 1000 or self.bootstrap_seed < 0:
            raise ValueError("convergence bootstrap settings are invalid")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("convergence confidence must lie in (0.5, 1.0)")
        if not 0.0 < self.late_window_fraction <= 1.0:
            raise ValueError("convergence late-window fraction must lie in (0, 1]")
        if self.minimum_late_validation_records < 3:
            raise ValueError("convergence late window requires at least three records")
        if len(self.required_seeds) < 1 or len(set(self.required_seeds)) != len(
            self.required_seeds
        ):
            raise ValueError("convergence protocol requires unique seed identities")
        if min(self.required_seeds) < 0:
            raise ValueError("convergence seeds must be nonnegative")

    @classmethod
    def load(cls, path: Path | str) -> "ConvergenceProtocol":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {
            "schema",
            "name",
            "bootstrap_iterations",
            "bootstrap_seed",
            "confidence",
            "late_window_fraction",
            "minimum_late_validation_records",
            "required_seeds",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("convergence protocol fields are not frozen v1")
        if value["schema"] != {"name": "convergence-protocol", "version": 1}:
            raise ValueError("unsupported convergence protocol schema")
        return cls(
            name=str(value["name"]),
            bootstrap_iterations=int(value["bootstrap_iterations"]),
            bootstrap_seed=int(value["bootstrap_seed"]),
            confidence=float(value["confidence"]),
            late_window_fraction=float(value["late_window_fraction"]),
            minimum_late_validation_records=int(
                value["minimum_late_validation_records"]
            ),
            required_seeds=tuple(map(int, value["required_seeds"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": {"name": "convergence-protocol", "version": 1},
            "name": self.name,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence": self.confidence,
            "late_window_fraction": self.late_window_fraction,
            "minimum_late_validation_records": self.minimum_late_validation_records,
            "required_seeds": list(self.required_seeds),
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _metric_by_state(
    report: Mapping[str, Any],
    state_metric: str,
) -> dict[str, float]:
    if report.get("evaluation_role") != "validation" or report.get("valid") is not True:
        raise ValueError("convergence evidence must be a valid validation report")
    states = report.get("states")
    if not isinstance(states, Mapping) or not states:
        raise ValueError("convergence validation report has no state metrics")
    result = {
        str(state_id): float(record[state_metric])
        for state_id, record in states.items()
    }
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("convergence state metrics must be finite")
    return result


def _bootstrap_mean_interval(
    values: np.ndarray,
    protocol: ConvergenceProtocol,
    *,
    seed_offset: int,
) -> tuple[float, list[float]]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("convergence bootstrap requires finite one-dimensional values")
    rng = np.random.default_rng(protocol.bootstrap_seed + seed_offset)
    selected = rng.integers(
        0,
        len(values),
        size=(protocol.bootstrap_iterations, len(values)),
    )
    samples = np.mean(values[selected], axis=1)
    tail = (1.0 - protocol.confidence) / 2.0
    return float(np.mean(values)), [
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    ]


def analyze_convergence_records(
    initialization: Mapping[str, Any],
    validation_history: Sequence[Mapping[str, Any]],
    optimization_trace: Sequence[Mapping[str, Any]],
    recovered: Mapping[str, Any],
    protocol: ConvergenceProtocol,
    *,
    best_step: int,
    seed_offset: int = 0,
    state_metric: str = "directional_l1",
) -> dict[str, Any]:
    """分析相对初始化的改善、后期发散与 checkpoint 复算，不判断最终质量。"""

    if not validation_history:
        raise ValueError("convergence analysis requires validation history")
    history = sorted(validation_history, key=lambda record: int(record["step"]))
    best_matches = [record for record in history if int(record["step"]) == best_step]
    if len(best_matches) != 1:
        raise ValueError("best checkpoint step is absent or duplicated in validation history")
    best = best_matches[0]
    initial_state = _metric_by_state(initialization, state_metric)
    best_state = _metric_by_state(best, state_metric)
    recovered_state = _metric_by_state(recovered, state_metric)
    if set(initial_state) != set(best_state) or set(best_state) != set(recovered_state):
        raise ValueError("convergence validation state identities disagree")
    state_ids = sorted(initial_state)
    difference = np.asarray(
        [best_state[state_id] - initial_state[state_id] for state_id in state_ids],
        dtype=np.float64,
    )
    mean_difference, improvement_ci = _bootstrap_mean_interval(
        difference,
        protocol,
        seed_offset=seed_offset,
    )
    improvement = {
        "axis": f"best_minus_initial_{state_metric}",
        "state_count": len(state_ids),
        "mean_difference": mean_difference,
        "bootstrap_confidence_interval": improvement_ci,
        "statistically_supported": improvement_ci[1] < 0.0,
        "absolute_quality_threshold_used": False,
    }

    late_count = max(
        protocol.minimum_late_validation_records,
        int(math.ceil(len(history) * protocol.late_window_fraction)),
    )
    late_sufficient = len(history) >= late_count
    slope_mean: float | None = None
    slope_ci: list[float] | None = None
    credible_divergence = False
    if late_sufficient:
        late = history[-late_count:]
        steps = np.asarray([int(record["step"]) for record in late], dtype=np.float64)
        steps = (steps - steps[0]) / (steps[-1] - steps[0])
        centered_steps = steps - np.mean(steps)
        slope_denominator = float(np.sum(centered_steps * centered_steps))
        if slope_denominator <= 0.0:
            raise ValueError("convergence late-window steps must be distinct")
        slopes = []
        for state_id in state_ids:
            values = np.asarray(
                [_metric_by_state(record, state_metric)[state_id] for record in late],
                dtype=np.float64,
            )
            centered_values = values - np.mean(values)
            slopes.append(
                float(np.sum(centered_steps * centered_values) / slope_denominator)
            )
        slope_mean, slope_ci = _bootstrap_mean_interval(
            np.asarray(slopes),
            protocol,
            seed_offset=seed_offset + 104729,
        )
        credible_divergence = slope_ci[0] > 0.0
    late_window = {
        "available_validation_records": len(history),
        "required_late_records": late_count,
        "used_late_records": late_count if late_sufficient else 0,
        "sufficient": late_sufficient,
        "mean_normalized_slope": slope_mean,
        "bootstrap_confidence_interval": slope_ci,
        "credible_divergence": credible_divergence,
        "passed": late_sufficient and not credible_divergence,
    }

    historical_values = np.asarray([best_state[state] for state in state_ids])
    recovered_values = np.asarray([recovered_state[state] for state in state_ids])
    recovery = {
        "state_count": len(state_ids),
        "maximum_absolute_difference": float(
            np.max(np.abs(recovered_values - historical_values))
        ),
        "rtol": 1e-6,
        "atol": 1e-8,
        "passed": bool(
            np.allclose(recovered_values, historical_values, rtol=1e-6, atol=1e-8)
        ),
    }

    trace_finite = bool(optimization_trace) and all(
        _all_numeric_finite(record)
        and record.get("gradient", {}).get("all_finite") is True
        and record.get("parameters", {}).get("all_finite") is True
        and record.get("optimizer_step_skipped") is False
        for record in optimization_trace
    )
    finite = {
        "initialization": _all_numeric_finite(initialization),
        "validation_history": _all_numeric_finite(history),
        "optimization_trace": trace_finite,
        "recovered_validation": _all_numeric_finite(recovered),
    }
    finite["passed"] = all(finite.values())
    passed = bool(
        finite["passed"]
        and improvement["statistically_supported"]
        and late_window["passed"]
        and recovery["passed"]
    )
    return {
        "finite_execution": finite,
        "validation_improvement": improvement,
        "late_window": late_window,
        "checkpoint_recovery": recovery,
        "passed": passed,
        "quality_threshold_used": False,
    }


def _training_protocol_sha256(config: TrainingConfig) -> str:
    value = config.to_dict()
    value.pop("seed", None)
    return _sha256_json(value)


def run_convergence_audit(
    data_path: Path | str,
    run_directories: Sequence[Path | str],
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """恢复每个 best checkpoint，并汇总预冻结 seed 的 validation-only 收敛证据。"""

    protocol = ConvergenceProtocol.load(protocol_path)
    if not run_directories:
        raise ValueError("convergence audit requires run directories")
    runs: list[dict[str, Any]] = []
    shared_identities: set[tuple[str, str, str]] = set()
    seeds: list[int] = []
    for run_index, directory_value in enumerate(run_directories):
        directory = Path(directory_value).resolve()
        manifest_path = directory / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("convergence audit requires a complete training run")
        if manifest.get("held_out_test_accessed") is not False:
            raise ValueError("convergence audit rejects runs that accessed held-out test")
        initialization_path = directory / manifest["initialization_validation"]["uri"]
        history_path = directory / manifest["validation_history"]["uri"]
        trace_path = directory / manifest["optimization_trace"]["uri"]
        for path, record in (
            (initialization_path, manifest["initialization_validation"]),
            (history_path, manifest["validation_history"]),
            (trace_path, manifest["optimization_trace"]),
        ):
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"convergence evidence hash mismatch: {path.name}")
        checkpoint_path = directory / manifest["checkpoints"]["best"]["uri"]
        if sha256_file(checkpoint_path) != manifest["checkpoints"]["best"]["sha256"]:
            raise ValueError("convergence best checkpoint hash mismatch")
        checkpoint = load_checkpoint(checkpoint_path)
        config = TrainingConfig.from_dict(checkpoint["training_config"])
        seed = config.seed
        seeds.append(seed)
        shared_identities.add(
            (
                str(manifest["data_id"]),
                str(manifest["pipeline_sha256"]),
                _training_protocol_sha256(config),
            )
        )
        initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        recovered = evaluate_checkpoint(
            data_path,
            checkpoint_path,
            split="validation",
            device_name=device_name,
        )
        analysis = analyze_convergence_records(
            initialization,
            history,
            trace,
            recovered,
            protocol,
            best_step=int(manifest["best_validation"]["step"]),
            seed_offset=run_index * 1000003,
        )
        runs.append(
            {
                "run_id": manifest["run_id"],
                "run_directory": str(directory),
                "seed": seed,
                "data_id": manifest["data_id"],
                "pipeline": manifest["pipeline"],
                "pipeline_sha256": manifest["pipeline_sha256"],
                "training_protocol_sha256": _training_protocol_sha256(config),
                "best_checkpoint_sha256": sha256_file(checkpoint_path),
                "held_out_test_accessed": False,
                **analysis,
            }
        )
    seed_identity_passed = sorted(seeds) == sorted(protocol.required_seeds)
    matched_protocol = len(shared_identities) == 1
    report: dict[str, Any] = {
        "format_name": "validation-relative-convergence-report",
        "format_version": 1,
        "protocol": protocol.to_dict(),
        "protocol_sha256": protocol.sha256,
        "evidence_role": "validation",
        "held_out_test_accessed": False,
        "runs": runs,
        "run_summary": {
            "required_seeds": list(protocol.required_seeds),
            "observed_seeds": seeds,
            "seed_identity_passed": seed_identity_passed,
            "matched_data_pipeline_and_protocol": matched_protocol,
            "all_runs_passed": all(run["passed"] for run in runs),
        },
        "quality_threshold_used": False,
    }
    report["passed"] = bool(
        seed_identity_passed
        and matched_protocol
        and report["run_summary"]["all_runs_passed"]
    )
    report["report_sha256"] = _sha256_json(report)
    _write_json_atomic(Path(output_path).resolve(), report)
    return report


def _sampler_training_protocol_sha256(config: SamplerTrainingConfig) -> str:
    value = config.to_dict()
    value.pop("seed", None)
    return _sha256_json(value)


def run_sampler_convergence_audit(
    data_path: Path | str,
    run_directories: Sequence[Path | str],
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """恢复 frozen-evaluator sampler checkpoint，并生成独立的多 seed 收敛证据。"""

    protocol = ConvergenceProtocol.load(protocol_path)
    if not run_directories:
        raise ValueError("sampler convergence audit requires run directories")
    device = torch.device(device_name)
    runs: list[dict[str, Any]] = []
    shared_identities: set[tuple[str, str, str, str, str]] = set()
    seeds: list[int] = []
    for run_index, directory_value in enumerate(run_directories):
        directory = Path(directory_value).resolve()
        manifest = json.loads(
            (directory / "run_manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("format_name") != "unified-sampler-training-run"
            or manifest.get("status") != "complete"
        ):
            raise ValueError("sampler convergence requires a complete sampler run")
        if (
            manifest.get("held_out_test_accessed") is not False
            or manifest.get("shared_evaluator_detached") is not True
            or manifest.get("target_head_reinitialized") is not True
        ):
            raise ValueError(
                "sampler convergence rejects leaked, non-detached, or inherited-head runs"
            )
        initialization_path = directory / manifest["initialization_validation"]["uri"]
        history_path = directory / manifest["validation_history"]["uri"]
        trace_path = directory / manifest["optimization_trace"]["uri"]
        for path, record in (
            (initialization_path, manifest["initialization_validation"]),
            (history_path, manifest["validation_history"]),
            (trace_path, manifest["optimization_trace"]),
        ):
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"sampler convergence hash mismatch: {path.name}")
        sampler_checkpoint_path = directory / manifest["checkpoints"]["best"]["uri"]
        if (
            sha256_file(sampler_checkpoint_path)
            != manifest["checkpoints"]["best"]["sha256"]
        ):
            raise ValueError("sampler convergence checkpoint hash mismatch")
        sampler_checkpoint = load_checkpoint(
            sampler_checkpoint_path, map_location=device
        )
        config = SamplerTrainingConfig.from_dict(
            sampler_checkpoint["sampler_training_config"]
        )
        source_path = Path(config.evaluator_checkpoint)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        source_path = source_path.resolve()
        if sha256_file(source_path) != sampler_checkpoint[
            "source_evaluator_checkpoint_sha256"
        ]:
            raise ValueError("sampler convergence source evaluator hash mismatch")
        source = load_checkpoint(source_path, map_location=device)
        pipeline = create_pipeline(config.evaluator_pipeline)
        store = pipeline.open_store(str(data_path))
        try:
            if source["data_id"] != store.data_id or manifest["data_id"] != store.data_id:
                raise ValueError("sampler convergence data identity mismatch")
            pipeline.load_training_state(source["fitted_training_state"])
            source_config = TrainingConfig.from_dict(source["training_config"])
            source_model = pipeline.create_model(source_config.model).to(device)
            source_model.load_state_dict(source["model_state"])
            model = (
                adapt_nvidia_model_for_sampler(source_model, config.sampler).to(device)
                if isinstance(source_model, NvidiaNeuralAppearanceModel)
                else source_model
            )
            model.load_state_dict(sampler_checkpoint["model_state"])
            validation_indices = store.partition_indices(
                pipeline.descriptor.partition_policy_id, "validation"
            )
            recovered = {
                "step": int(sampler_checkpoint["step"]),
                **evaluate_sampler_validation(
                    model,
                    pipeline,
                    store,
                    validation_indices,
                    device,
                    config,
                ),
            }
        finally:
            store.close()
        initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        analysis = analyze_convergence_records(
            initialization,
            history,
            trace,
            recovered,
            protocol,
            best_step=int(sampler_checkpoint["step"]),
            seed_offset=run_index * 1000003,
            state_metric="evaluator_relative_cross_entropy",
        )
        seeds.append(config.seed)
        protocol_sha256 = _sampler_training_protocol_sha256(config)
        shared_identities.add(
            (
                str(manifest["data_id"]),
                str(manifest["evaluator_pipeline"]),
                str(manifest["sampler"]),
                str(manifest["evaluator_checkpoint"]["sha256"]),
                protocol_sha256,
            )
        )
        runs.append(
            {
                "run_id": manifest["run_id"],
                "run_directory": str(directory),
                "seed": config.seed,
                "data_id": manifest["data_id"],
                "evaluator_pipeline": manifest["evaluator_pipeline"],
                "sampler": manifest["sampler"],
                "evaluator_checkpoint_sha256": manifest["evaluator_checkpoint"][
                    "sha256"
                ],
                "training_protocol_sha256": protocol_sha256,
                "sampler_checkpoint_sha256": sha256_file(sampler_checkpoint_path),
                "implementation_identity": manifest["implementation_identity"],
                "shared_evaluator_detached": True,
                "held_out_test_accessed": False,
                **analysis,
            }
        )
    seed_identity_passed = sorted(seeds) == sorted(protocol.required_seeds)
    matched_protocol = len(shared_identities) == 1
    report: dict[str, Any] = {
        "format_name": "sampler-validation-relative-convergence-report",
        "format_version": 1,
        "protocol": protocol.to_dict(),
        "protocol_sha256": protocol.sha256,
        "evidence_role": "validation",
        "held_out_test_accessed": False,
        "runs": runs,
        "run_summary": {
            "required_seeds": list(protocol.required_seeds),
            "observed_seeds": seeds,
            "seed_identity_passed": seed_identity_passed,
            "matched_data_evaluator_sampler_and_protocol": matched_protocol,
            "all_runs_passed": all(run["passed"] for run in runs),
        },
        "quality_threshold_used": False,
    }
    report["passed"] = bool(
        seed_identity_passed
        and matched_protocol
        and report["run_summary"]["all_runs_passed"]
    )
    report["report_sha256"] = _sha256_json(report)
    _write_json_atomic(Path(output_path).resolve(), report)
    return report


__all__ = [
    "ConvergenceProtocol",
    "analyze_convergence_records",
    "run_convergence_audit",
    "run_sampler_convergence_audit",
]
