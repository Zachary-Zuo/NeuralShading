from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
    }


def validate_v0(
    dataset_dir: Path,
    output_path: Path,
    *,
    noise_sample_tiles: int,
    seed: int,
) -> dict[str, object]:
    start_time = time.perf_counter()
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    states = np.load(dataset_dir / "states.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    total_tiles = 0
    total_bytes = 0
    sampled_a: list[np.ndarray] = []
    sampled_b: list[np.ndarray] = []
    sample_budget = min(noise_sample_tiles, int(metadata["tile_count"]))
    for shard in metadata["shards"]:
        tile_path = dataset_dir / shard["tiles"]
        index_path = dataset_dir / shard["index"]
        tiles = np.load(tile_path, mmap_mode="r")
        index = np.load(index_path, mmap_mode="r")
        expected_count = int(shard["tile_count"])
        if len(tiles) != expected_count or index.shape != (expected_count, 2):
            raise ValueError(f"shape mismatch in {shard['tiles']}")
        if not np.all(np.isfinite(tiles["mean_a"])) or not np.all(np.isfinite(tiles["mean_b"])):
            raise ValueError(f"non-finite response in {shard['tiles']}")
        if np.any(tiles["count"] == 0):
            raise ValueError(f"zero sample count in {shard['tiles']}")
        if np.max(index[:, 0]) >= metadata["state_count"] or np.max(index[:, 1]) >= metadata["view_count"]:
            raise ValueError(f"out-of-range index in {shard['index']}")
        take = int(round(sample_budget * expected_count / metadata["tile_count"]))
        take = min(max(take, 1), expected_count)
        selected = np.sort(rng.choice(expected_count, size=take, replace=False))
        sampled_a.append(np.asarray(tiles["mean_a"][selected], dtype=np.float32))
        sampled_b.append(np.asarray(tiles["mean_b"][selected], dtype=np.float32))
        total_tiles += expected_count
        total_bytes += tile_path.stat().st_size + index_path.stat().st_size
    if total_tiles != metadata["tile_count"]:
        raise ValueError("shard tile total does not match metadata")

    mean_a = np.concatenate(sampled_a)[:sample_budget]
    mean_b = np.concatenate(sampled_b)[:sample_budget]
    target = 0.5 * (mean_a + mean_b)
    peak = np.max(target, axis=(1, 2), keepdims=True)
    floor = 1e-3 * peak + 1e-5
    smape = np.mean(
        2.0 * np.abs(mean_a - mean_b) / (np.abs(mean_a) + np.abs(mean_b) + floor),
        axis=(1, 2),
    )
    relative_l1 = np.sum(np.abs(mean_a - mean_b), axis=(1, 2)) / np.maximum(
        np.sum(np.abs(target), axis=(1, 2)), 1e-8
    )
    family_splits = np.load(dataset_dir / "family_splits.npy")
    result: dict[str, object] = {
        "dataset": str(dataset_dir),
        "format": metadata["format"],
        "teacher_source_sha256": metadata["teacher_source_sha256"],
        "family_count": int(metadata["family_count"]),
        "state_count": int(metadata["state_count"]),
        "tile_count": total_tiles,
        "shard_count": len(metadata["shards"]),
        "response_and_index_bytes": total_bytes,
        "all_responses_finite": True,
        "split_family_counts": {
            name: int(np.count_nonzero(family_splits == split_index))
            for split_index, name in enumerate(metadata["split_names"])
        },
        "noise_sample_tiles": len(mean_a),
        "ab_smape": _summary(smape),
        "ab_relative_l1": _summary(relative_l1),
        "seconds": time.perf_counter() - start_time,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-validate a sharded v0 dataset.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_train")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "v0_train_validation.json",
    )
    parser.add_argument("--noise-sample-tiles", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    validate_v0(
        args.dataset,
        args.output,
        noise_sample_tiles=args.noise_sample_tiles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
