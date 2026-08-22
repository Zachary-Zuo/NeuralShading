from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model.features import StackFeatureTable, load_stack_feature_table


class DirectionTileStore:
    """Random batch access over the sharded 14-byte/bin v0 tile format."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.metadata = json.loads(
            (self.dataset_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.features: StackFeatureTable = load_stack_feature_table(self.dataset_dir)
        self.views = np.load(self.dataset_dir / "views.npy")[:, :3].astype(np.float32)
        self.lights = np.load(self.dataset_dir / "light_directions.npy")[:, :3].astype(np.float32)
        self.states = np.load(self.dataset_dir / "states.npy", mmap_mode="r")
        self.shards = [
            (
                int(record["tile_start"]),
                int(record["tile_count"]),
                np.load(self.dataset_dir / record["tiles"], mmap_mode="r"),
                np.load(self.dataset_dir / record["index"], mmap_mode="r"),
            )
            for record in self.metadata["shards"]
        ]
        self.tile_count = int(self.metadata["tile_count"])
        split_by_tile = np.repeat(
            np.asarray(self.states["split"], dtype=np.uint8), int(self.metadata["view_count"])
        )
        if len(split_by_tile) != self.tile_count:
            raise ValueError("dataset tile ordering does not match state_count * view_count")
        self.split_indices = {
            name: np.flatnonzero(split_by_tile == split_index).astype(np.int64)
            for split_index, name in enumerate(self.metadata["split_names"])
        }
        self.shard_split_indices = {
            name: [
                indices[
                    np.searchsorted(indices, start) : np.searchsorted(indices, start + count)
                ]
                for start, count, _, _ in self.shards
            ]
            for name, indices in self.split_indices.items()
        }

    def sample_batch_indices(
        self,
        split: str,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one shard-local batch to keep memmap reads contiguous in file scope."""
        candidates = self.shard_split_indices[split]
        counts = np.asarray([len(indices) for indices in candidates], dtype=np.float64)
        shard_index = int(rng.choice(len(candidates), p=counts / np.sum(counts)))
        return rng.choice(candidates[shard_index], size=batch_size, replace=True)

    def batch(self, tile_indices: np.ndarray) -> dict[str, np.ndarray]:
        requested = np.asarray(tile_indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("tile_indices must be one-dimensional")
        bin_count = int(self.metadata["bin_count"])
        mean_a = np.empty((len(requested), bin_count, 3), dtype=np.float32)
        mean_b = np.empty_like(mean_a)
        state_indices = np.empty(len(requested), dtype=np.int64)
        view_indices = np.empty(len(requested), dtype=np.int64)
        found = np.zeros(len(requested), dtype=bool)
        for shard_start, shard_count, tiles, index in self.shards:
            positions = np.flatnonzero(
                (requested >= shard_start) & (requested < shard_start + shard_count)
            )
            if not len(positions):
                continue
            local = requested[positions] - shard_start
            mean_a[positions] = np.asarray(tiles["mean_a"][local], dtype=np.float32)
            mean_b[positions] = np.asarray(tiles["mean_b"][local], dtype=np.float32)
            state_indices[positions] = index[local, 0]
            view_indices[positions] = index[local, 1]
            found[positions] = True
        if not np.all(found):
            raise IndexError("one or more tile indices are outside the dataset")
        state = state_indices
        table = self.features
        return {
            "layer_types": table.layer_types[state],
            "continuous": table.continuous[state],
            "layer_counts": table.layer_counts[state],
            "view": self.views[view_indices],
            "mean_a": mean_a,
            "mean_b": mean_b,
            "top_type": table.top_type[state],
            "top_roughness": table.top_roughness[state],
            "top_eta": table.top_eta[state],
            "top_k": table.top_k[state],
            "top_albedo": table.top_albedo[state],
            "top_rotation": table.top_rotation[state],
        }
