from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from baselines.closure_families import fit_oracle_batch
from baselines.oracle_fit import _summary, _smape_numpy, load_oracle_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fit_residual(
    dataset_dir: Path,
    direct_path: Path,
    output_dir: Path,
    *,
    family: str,
    lobe_count: int,
    fit_batch: int,
    steps: int,
    restarts: int,
    device: str | None,
    seed: int,
) -> dict[str, object]:
    dataset = load_oracle_dataset(dataset_dir)
    direct = np.load(direct_path, mmap_mode="r")
    residual = np.maximum(dataset.target - direct, 0.0)
    prediction_parts: list[np.ndarray] = []
    parameter_parts: dict[str, list[np.ndarray]] = {}
    start = time.perf_counter()
    for batch_start in range(0, len(residual), fit_batch):
        batch_end = min(batch_start + fit_batch, len(residual))
        fitted = fit_oracle_batch(
            residual[batch_start:batch_end],
            dataset.views[batch_start:batch_end, :3],
            dataset.lights,
            family=family,
            lobe_count=lobe_count,
            steps=steps,
            restarts=restarts,
            device=device,
            seed=seed + batch_start * 17,
        )
        prediction_parts.append(fitted.prediction)
        for name, values in fitted.parameters.items():
            parameter_parts.setdefault(name, []).append(values)
    residual_prediction = np.concatenate(prediction_parts)
    prediction = np.asarray(direct) + residual_prediction
    smape = _smape_numpy(prediction, dataset.target)
    relative_l1 = np.sum(np.abs(prediction - dataset.target), axis=(1, 2)) / np.maximum(
        np.sum(np.abs(dataset.target), axis=(1, 2)), 1e-8
    )
    name = f"direct-{family}-k{lobe_count}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{name}.npz",
        smape=smape,
        relative_l1=relative_l1,
        prediction=prediction.astype(np.float16),
        state_indices=dataset.state_indices,
        view_indices=dataset.view_indices,
        **{key: np.concatenate(parts) for key, parts in parameter_parts.items()},
    )
    summary = {
        "dataset": str(dataset_dir),
        "direct": str(direct_path),
        "family": family,
        "lobe_count": lobe_count,
        "seconds": time.perf_counter() - start,
        "clamped_negative_residual_fraction": float(np.mean(dataset.target < direct)),
        "smape": _summary(smape),
        "relative_l1": _summary(relative_l1),
    }
    (output_dir / f"{name}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit analytic closures to the residual after direct top reflection.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512")
    parser.add_argument(
        "--direct", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512" / "direct_top.npy"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "oracle_v0_512")
    parser.add_argument("--family", choices=("ggx", "ltc", "sg"), default="ltc")
    parser.add_argument("--lobes", type=int, default=2)
    parser.add_argument("--fit-batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    fit_residual(
        args.dataset,
        args.direct,
        args.output,
        family=args.family,
        lobe_count=args.lobes,
        fit_batch=args.fit_batch,
        steps=args.steps,
        restarts=args.restarts,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
