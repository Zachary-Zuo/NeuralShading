from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ncls.core.identity import sha256_file, write_json_atomic


DEFAULT_METRICS = (
    "validation/loss/appearance",
    "validation/appearance/log_rgb",
    "validation/appearance/linear_rgb",
    "validation/appearance/chroma",
    "validation/appearance/peak_rgb",
    "validation/appearance/spatial_gradient",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对同序 validation row 计算 candidate-baseline paired bootstrap。"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--baseline-step", type=int)
    parser.add_argument("--candidate-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090403)
    parser.add_argument("--replicates", type=int, default=20_000)
    return parser.parse_args()


def _rows(path: Path, step: int) -> list[Mapping[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_kind") == "validation" and int(row["step"]) == step:
            result.append(row)
    if not result:
        raise ValueError(f"{path} 在 step {step} 没有 validation row")
    return result


def _paired_bootstrap(
    differences: Sequence[float],
    *,
    seed: int,
    replicates: int,
) -> Mapping[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap 需要至少两个有限差值")
    if replicates < 100:
        raise ValueError("paired bootstrap 至少需要100次重采样")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    observed = float(values.mean())
    return {
        "row_count": len(values),
        "observed_mean_delta": observed,
        "bootstrap_mean_delta_ci95": [float(low), float(high)],
        "candidate_better": bool(high < 0.0),
        "baseline_better": bool(low > 0.0),
    }


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    baseline_step: int,
    candidate_step: int,
    seed: int,
    replicates: int,
) -> Mapping[str, Any]:
    baseline = _rows(baseline_path, baseline_step)
    candidate = _rows(candidate_path, candidate_step)
    if len(baseline) != len(candidate):
        raise ValueError("paired validation 两侧 row 数量不同")
    metrics = {}
    for metric_index, name in enumerate(DEFAULT_METRICS):
        if any(name not in row for row in (*baseline, *candidate)):
            raise ValueError(f"paired validation 缺少 metric {name!r}")
        metrics[name] = _paired_bootstrap(
            [
                float(candidate_row[name]) - float(baseline_row[name])
                for baseline_row, candidate_row in zip(baseline, candidate)
            ],
            seed=seed + metric_index,
            replicates=replicates,
        )
    return {
        "schema": "ncls.paired-validation-comparison@1",
        "delta": "candidate-minus-baseline",
        "baseline_step": baseline_step,
        "candidate_step": candidate_step,
        "seed": seed,
        "replicates": replicates,
        "baseline": {
            "path": baseline_path.as_posix(),
            "sha256": sha256_file(baseline_path),
            "training_config_sha256": baseline[0]["training_config_sha256"],
        },
        "candidate": {
            "path": candidate_path.as_posix(),
            "sha256": sha256_file(candidate_path),
            "training_config_sha256": candidate[0]["training_config_sha256"],
        },
        "metrics": metrics,
    }


def main() -> None:
    args = _parse_args()
    if args.step is not None:
        if args.baseline_step is not None or args.candidate_step is not None:
            raise ValueError("--step 不能与 --baseline-step/--candidate-step 混用")
        baseline_step = candidate_step = args.step
    else:
        if args.baseline_step is None or args.candidate_step is None:
            raise ValueError("必须提供 --step，或同时提供两个里程碑 step")
        baseline_step = args.baseline_step
        candidate_step = args.candidate_step
    result = compare(
        args.baseline.resolve(),
        args.candidate.resolve(),
        baseline_step=baseline_step,
        candidate_step=candidate_step,
        seed=args.seed,
        replicates=args.replicates,
    )
    write_json_atomic(args.output.resolve(), result)
    print(f"paired validation comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()
