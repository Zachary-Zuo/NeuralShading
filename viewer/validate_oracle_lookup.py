from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from closures.oracle import load_oracle_packets
from viewer.oracle_lookup import FalcorOracleLookup


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_oracle_lookup(
    dataset_dir: Path,
    archive_path: Path,
    output_path: Path,
    *,
    packet_batch: int,
) -> dict[str, float | int | str]:
    with np.load(archive_path) as archive:
        expected = np.asarray(archive["prediction"], dtype=np.float32)
        view_indices = np.asarray(archive["view_indices"], dtype=np.uint32)
    all_views = np.load(dataset_dir / "views.npy")
    lights = np.load(dataset_dir / "light_directions.npy")
    views = all_views[view_indices]
    packets = load_oracle_packets(dataset_dir, archive_path)
    evaluator = FalcorOracleLookup(lights, max_packet_batch=packet_batch)
    actual_parts: list[np.ndarray] = []
    for start in range(0, len(packets), packet_batch):
        end = min(start + packet_batch, len(packets))
        actual_parts.append(evaluator.evaluate(packets[start:end], views[start:end]))
    actual = np.concatenate(actual_parts)
    absolute = np.abs(actual - expected)
    relative_l1 = np.sum(absolute, axis=(1, 2)) / np.maximum(
        np.sum(np.abs(expected), axis=(1, 2)), 1e-8
    )
    worst_flat = int(np.argmax(absolute))
    worst_tile, worst_bin, worst_channel = np.unravel_index(worst_flat, absolute.shape)
    global_peak = float(np.max(np.abs(expected)))
    result: dict[str, float | int | str] = {
        "dataset": str(dataset_dir),
        "archive": str(archive_path),
        "packet_count": len(packets),
        "packet_bytes": 176,
        "median_relative_l1_vs_fp16_archive": float(np.median(relative_l1)),
        "p99_relative_l1_vs_fp16_archive": float(np.quantile(relative_l1, 0.99)),
        "max_relative_l1_vs_fp16_archive": float(np.max(relative_l1)),
        "max_absolute_vs_fp16_archive": float(np.max(absolute)),
        "max_absolute_fraction_of_global_peak": float(np.max(absolute) / max(global_peak, 1e-8)),
        "worst_tile": int(worst_tile),
        "worst_bin": int(worst_bin),
        "worst_channel": int(worst_channel),
        "worst_expected": float(expected[worst_tile, worst_bin, worst_channel]),
        "worst_actual": float(actual[worst_tile, worst_bin, worst_channel]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate oracle closure packets in Falcor.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512")
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "reports" / "oracle_v0_512" / "direct-ltc-k2.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "oracle_v0_512" / "falcor_lookup_validation.json",
    )
    parser.add_argument("--packet-batch", type=int, default=128)
    args = parser.parse_args()
    validate_oracle_lookup(args.dataset, args.archive, args.output, packet_batch=args.packet_batch)


if __name__ == "__main__":
    main()
