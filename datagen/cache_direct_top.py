from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datagen.gen_tiles import FalcorTileEvaluator
from schema import BINARY_SIZE, unpack_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def cache_direct_top(dataset_dir: Path, output: Path, *, tile_batch: int = 128) -> None:
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    payload = (dataset_dir / "stacks.bin").read_bytes()
    stacks = [
        unpack_stack(payload[offset : offset + BINARY_SIZE])
        for offset in range(0, len(payload), BINARY_SIZE)
    ]
    views = np.load(dataset_dir / "views.npy")
    lights = np.load(dataset_dir / "light_directions.npy")
    evaluator = FalcorTileEvaluator(lights, max_tile_batch=tile_batch)
    result = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(metadata["tile_count"], metadata["bin_count"], 3),
    )
    for shard in metadata["shards"]:
        index = np.load(dataset_dir / shard["index"], mmap_mode="r")
        global_start = shard["tile_start"]
        for local_start in range(0, len(index), tile_batch):
            local_end = min(local_start + tile_batch, len(index))
            batch_index = index[local_start:local_end]
            direct = evaluator.evaluate_direct_tiles(
                [stacks[int(state_index)] for state_index in batch_index[:, 0]],
                views[batch_index[:, 1]],
            )
            result[global_start + local_start : global_start + local_end] = direct
    result.flush()
    print(f"cached direct-top response at {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache the analytic top-interface response per tile.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "v0_oracle_512")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "v0_oracle_512" / "direct_top.npy",
    )
    parser.add_argument("--tile-batch", type=int, default=128)
    args = parser.parse_args()
    cache_direct_top(args.dataset, args.output, tile_batch=args.tile_batch)


if __name__ == "__main__":
    main()
