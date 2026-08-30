from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from ncls.core.identity import sha256_json, write_json_atomic
from ncls.learning.method import MethodDescriptor

from .checkpoint import TrainingCheckpoint
from .config import TrainingConfig


_REVIEW_WINDOW_CAP = 32
_BOOTSTRAP_REPLICATES = 2000


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("a review quantile requires observations")
    index = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    weight = index - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def _phase_loss_summary(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> Mapping[str, Any]:
    losses = [float(row["loss"]) for row in rows]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("training review phase losses must be finite and nonempty")
    window = min(_REVIEW_WINDOW_CAP, max(1, len(losses) // 4))
    initial = losses[:window]
    final = losses[-window:]
    generator = random.Random(seed)
    deltas = []
    for _ in range(_BOOTSTRAP_REPLICATES):
        initial_mean = sum(generator.choice(initial) for _ in initial) / len(initial)
        final_mean = sum(generator.choice(final) for _ in final) / len(final)
        deltas.append(final_mean - initial_mean)
    deltas.sort()
    return {
        "record_count": len(losses),
        "window_records": window,
        "initial_mean": sum(initial) / len(initial),
        "final_mean": sum(final) / len(final),
        "initial_median": median(initial),
        "final_median": median(final),
        "minimum": min(losses),
        "observed_final_minus_initial": sum(final) / len(final) - sum(initial) / len(initial),
        "bootstrap_mean_delta_ci95": [
            _quantile(deltas, 0.025),
            _quantile(deltas, 0.975),
        ],
        "interpretation": "report-only; not a quality or completion gate",
    }


def load_metric_rows(path: Path, *, config_sha256: str) -> list[Mapping[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("training_config_sha256") != config_sha256:
            raise ValueError(f"metric row {line_number} belongs to another training config")
        numeric = [
            float(item)
            for name, item in value.items()
            if name not in {"record_kind", "training_config_sha256"}
            and isinstance(item, (int, float))
        ]
        if not all(math.isfinite(item) for item in numeric):
            raise ValueError(f"metric row {line_number} contains NaN or Inf")
        rows.append(value)
    if not rows:
        raise ValueError("training review requires metric rows")
    return rows


def _segmented_training_elapsed(rows: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    segment_max = 0.0
    previous = -math.inf
    for row in rows:
        elapsed = float(row.get("elapsed_seconds", 0.0))
        if elapsed < previous:
            total += segment_max
            segment_max = 0.0
        segment_max = max(segment_max, elapsed)
        previous = elapsed
    return total + segment_max


def build_training_review(
    config: TrainingConfig,
    descriptor: MethodDescriptor,
    checkpoint: TrainingCheckpoint,
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    metric_rows: Sequence[Mapping[str, Any]],
    metrics_bytes: int,
    elapsed_seconds: float,
    checkpoint_write_seconds: Sequence[float] = (),
) -> Mapping[str, Any]:
    if checkpoint.training_config_sha256 != config.sha256:
        raise ValueError("training review checkpoint belongs to another config")
    checkpoint.validate_method(descriptor)
    training_rows = [row for row in metric_rows if row.get("record_kind") == "training"]
    validation_rows = [row for row in metric_rows if row.get("record_kind") == "validation"]
    phases = []
    for phase_index, phase in enumerate(config.phases):
        rows = [
            row
            for row in training_rows
            if int(float(row["phase_index"])) == phase_index
        ]
        entry: dict[str, Any] = {
            "name": phase.name,
            "planned_steps": phase.steps,
            "completed_steps": max(
                (int(float(row["phase_step"])) for row in rows), default=0
            ),
        }
        if rows:
            entry["loss"] = _phase_loss_summary(
                rows, seed=config.seed + 104729 * phase_index
            )
            phase_profile_keys = sorted(
                {name for row in rows for name in row if name.startswith("profile/")}
            )
            entry["profile_medians"] = {
                name.removeprefix("profile/"): median(
                    float(row[name]) for row in rows if name in row
                )
                for name in phase_profile_keys
            }
        phases.append(entry)
    finite_coverage = all(
        all(
            bool(value[field])
            for field in (
                "finite_observed",
                "nonzero_gradient_observed",
                "parameter_update_observed",
            )
        )
        for value in checkpoint.gradient_coverage.values()
    )
    peak_memory = max(
        (int(float(row.get("peak_memory_bytes", 0))) for row in training_rows),
        default=0,
    )
    rates = [float(row["steps_per_second"]) for row in training_rows]
    profile_keys = sorted(
        {
            name
            for row in metric_rows
            for name in row
            if name.startswith("profile/")
        }
    )
    profile_medians = {
        name.removeprefix("profile/"): median(
            float(row[name]) for row in metric_rows if name in row
        )
        for name in profile_keys
    }
    body = {
        "schema": "ncls.training-review@1",
        "method_key": config.method_key,
        "method_descriptor_sha256": descriptor.descriptor_sha256,
        "training_config_sha256": config.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "global_step": checkpoint.global_step,
        "planned_steps": config.total_steps,
        "complete": checkpoint.global_step == config.total_steps,
        "source_count": len(config.source["materials"]),
        "phase_summaries": phases,
        "health": {
            "all_metric_values_finite": True,
            "gradient_update_coverage_complete": finite_coverage,
            "validation_records": len(validation_rows),
        },
        "observed_cost": {
            "training_elapsed_seconds": _segmented_training_elapsed(training_rows),
            "latest_process_elapsed_seconds": float(elapsed_seconds),
            "median_steps_per_second": median(rates) if rates else 0.0,
            "peak_memory_bytes": peak_memory,
            "checkpoint_bytes": int(checkpoint_bytes),
            "metrics_bytes": int(metrics_bytes),
            "checkpoint_write_seconds_total": sum(checkpoint_write_seconds),
            "checkpoint_write_seconds_median": (
                median(checkpoint_write_seconds) if checkpoint_write_seconds else 0.0
            ),
            "profile_medians": profile_medians,
        },
        "bounded_runtime": dict(descriptor.bounded_execution),
        "automatic_followups": [],
        "next_action": "user-review-required",
    }
    return {**body, "identity": sha256_json(body)}


def write_training_review(path: Path, review: Mapping[str, Any]) -> None:
    write_json_atomic(path, review)


__all__ = [
    "build_training_review",
    "load_metric_rows",
    "write_training_review",
]
