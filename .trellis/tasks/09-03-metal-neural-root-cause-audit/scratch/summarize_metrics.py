from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metrics",
        type=Path,
        nargs="?",
        default=Path("artifacts/metal-linux-training/long/checkpoint.metrics.jsonl"),
    )
    metrics_path = parser.parse_args().metrics
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_kinds = sorted({row.get("record_kind") for row in rows})
    print(
        "record_kinds",
        {kind: sum(row.get("record_kind") == kind for row in rows) for kind in record_kinds},
    )

    training_rows = [row for row in rows if row.get("record_kind") == "training"]
    print(
        "training_records",
        len(training_rows),
        "step_range",
        (training_rows[0]["step"], training_rows[-1]["step"]),
    )

    phase_indices = sorted({int(row["phase_index"]) for row in training_rows})
    for phase_index in phase_indices:
        phase_rows = [
            row for row in training_rows if int(row["phase_index"]) == phase_index
        ]
        seconds_per_step = []
        for previous, current in zip(phase_rows, phase_rows[1:], strict=False):
            step_delta = current["step"] - previous["step"]
            if step_delta > 0:
                seconds_per_step.append(
                    (current["elapsed_seconds"] - previous["elapsed_seconds"])
                    / step_delta
                )
        batch_prepare = [
            row.get("profile/batch_prepare_wall_seconds_mean", math.nan)
            for row in phase_rows
        ]
        batch_prepare = [value for value in batch_prepare if math.isfinite(value)]
        print(
            "phase",
            phase_index,
            "records",
            len(phase_rows),
            "steps",
            (phase_rows[0]["step"], phase_rows[-1]["step"]),
            "sec_per_step_med_p90_max",
            tuple(
                round(value, 4)
                for value in (
                    statistics.median(seconds_per_step),
                    percentile(seconds_per_step, 0.9),
                    max(seconds_per_step),
                )
            )
            if seconds_per_step
            else None,
            "prepare_med_p90_max",
            tuple(
                round(value, 4)
                for value in (
                    statistics.median(batch_prepare),
                    percentile(batch_prepare, 0.9),
                    max(batch_prepare),
                )
            )
            if batch_prepare
            else None,
            "peak_memory_first_last",
            (
                int(phase_rows[0]["peak_memory_bytes"]),
                int(phase_rows[-1]["peak_memory_bytes"]),
            ),
        )

        slow_intervals = sorted(
            (
                (
                    (current["elapsed_seconds"] - previous["elapsed_seconds"])
                    / (current["step"] - previous["step"]),
                    int(previous["step"]),
                    int(current["step"]),
                    current.get("profile/batch_prepare_wall_seconds"),
                )
                for previous, current in zip(phase_rows, phase_rows[1:], strict=False)
                if current["step"] > previous["step"]
            ),
            reverse=True,
        )[:10]
        print("phase_slowest_intervals", phase_index, slow_intervals)
        if phase_index == 1:
            metric_names = (
                "loss",
                "response_robust_loss",
                "linear_energy_loss",
                "peak_support_loss",
                "compiler_distillation_loss",
            )
            window = max(1, min(20, len(phase_rows) // 2))
            for metric_name in metric_names:
                first = [float(row[metric_name]) for row in phase_rows[:window]]
                last = [float(row[metric_name]) for row in phase_rows[-window:]]
                print(
                    "phase1_window_mean",
                    metric_name,
                    "first",
                    statistics.fmean(first),
                    "last",
                    statistics.fmean(last),
                    "delta",
                    statistics.fmean(last) - statistics.fmean(first),
                )

    print("phase_boundary")
    selected_keys = (
        "step",
        "phase_index",
        "phase_step",
        "loss",
        "elapsed_seconds",
        "steps_per_second",
        "peak_memory_bytes",
        "profile/batch_prepare_wall_seconds_mean",
        "profile/forward_gpu_seconds",
        "profile/backward_gpu_seconds",
        "profile/optimizer_gpu_seconds",
    )
    for row in training_rows:
        if 19_950 <= row["step"] <= 20_100:
            print({key: row.get(key) for key in selected_keys})

    print("validation_rows")
    for row in rows:
        if row.get("record_kind") == "validation":
            print(
                {
                    key: row.get(key)
                    for key in (
                        "step",
                        "phase_index",
                        "validation/loss",
                        "profile/validation_wall_seconds",
                    )
                }
            )


if __name__ == "__main__":
    main()
