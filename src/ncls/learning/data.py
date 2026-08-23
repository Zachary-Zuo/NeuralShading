from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.data import ReferenceDataset

from .features import StackFeatureTable, load_feature_table


SPLIT_NAMES = ("train", "validation", "test")


class ReferenceTileStore:
    """对 v2 reference shards 做 shard-local 随机访问，不猜测磁盘格式。"""

    def __init__(self, dataset_dir: Path | str, *, verify_hashes: bool = True) -> None:
        self.dataset = ReferenceDataset.open(dataset_dir, verify_hashes=verify_hashes)
        self.features: StackFeatureTable = load_feature_table(self.dataset)
        self.views = np.array(self.dataset.view_directions[:, :3], dtype=np.float32, copy=True)
        self.lights = np.array(self.dataset.light_directions[:, :3], dtype=np.float32, copy=True)
        self.tile_count = int(self.dataset.manifest.counts["tile_count"])
        self.shards = []
        split_parts = {name: [] for name in SPLIT_NAMES}
        for shard in self.dataset.manifest.shards:
            index = np.load(self.dataset.root / shard.index_uri, mmap_mode="r", allow_pickle=False)
            response = np.load(self.dataset.root / shard.response_uri, mmap_mode="r", allow_pickle=False)
            self.shards.append((shard.tile_start, shard.tile_count, index, response))
            for code, name in enumerate(SPLIT_NAMES):
                split_parts[name].append(
                    np.asarray(index["tile_id"][index["split"] == code], dtype=np.int64)
                )
        self.split_indices = {
            name: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            for name, parts in split_parts.items()
        }
        self.shard_split_indices = {
            name: [
                indices[np.searchsorted(indices, start) : np.searchsorted(indices, start + count)]
                for start, count, _, _ in self.shards
            ]
            for name, indices in self.split_indices.items()
        }

    def sample_batch_indices(self, split: str, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if split not in SPLIT_NAMES or batch_size < 1:
            raise ValueError("invalid split or batch size")
        candidates = self.shard_split_indices[split]
        counts = np.asarray([len(indices) for indices in candidates], dtype=np.float64)
        if not np.any(counts):
            raise ValueError(f"dataset split {split!r} is empty")
        shard_index = int(rng.choice(len(candidates), p=counts / np.sum(counts)))
        return rng.choice(candidates[shard_index], size=batch_size, replace=True)

    def batch(self, tile_indices: np.ndarray) -> dict[str, np.ndarray]:
        requested = np.asarray(tile_indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("tile_indices must be one-dimensional")
        light_count = int(self.dataset.manifest.counts["light_count"])
        shape = (len(requested), light_count, 3)
        mean = np.empty(shape, dtype=np.float32)
        variance = np.empty(shape, dtype=np.float32)
        replica_a = np.empty(shape, dtype=np.float32)
        replica_b = np.empty(shape, dtype=np.float32)
        sample_count = np.empty((len(requested), light_count), dtype=np.uint32)
        state_indices = np.empty(len(requested), dtype=np.int64)
        view_indices = np.empty(len(requested), dtype=np.int64)
        splits = np.empty(len(requested), dtype=np.uint8)
        found = np.zeros(len(requested), dtype=bool)
        for start, count, index, response in self.shards:
            positions = np.flatnonzero((requested >= start) & (requested < start + count))
            if not len(positions):
                continue
            local = requested[positions] - start
            mean[positions] = response["mean"][local]
            variance[positions] = response["variance"][local]
            replica_a[positions] = response["replica_mean_a"][local]
            replica_b[positions] = response["replica_mean_b"][local]
            sample_count[positions] = response["sample_count"][local]
            state_indices[positions] = index["material_state_index"][local]
            view_indices[positions] = index["view_index"][local]
            splits[positions] = index["split"][local]
            found[positions] = True
        if not np.all(found):
            raise IndexError("one or more tile indices are outside the dataset")
        table = self.features
        states = state_indices
        uncertainty_kind = str(self.dataset.manifest.statistics_encoding["uncertainty_kind"])
        if uncertainty_kind == "sample-population-variance":
            standard_error = np.sqrt(
                variance / np.maximum(sample_count.astype(np.float32), 1.0)[..., None]
            )
        elif uncertainty_kind == "replica-mean-variance":
            standard_error = np.sqrt(variance)
        else:
            raise ValueError(f"unsupported uncertainty kind {uncertainty_kind!r}")
        return {
            "tile_id": requested,
            "split": splits,
            "interface_kinds": table.interface_kinds[states],
            "continuous": table.continuous[states],
            "interface_counts": table.interface_counts[states],
            "view": self.views[view_indices],
            "mean": mean,
            "variance": variance,
            "sample_count": sample_count,
            "standard_error": standard_error.astype(np.float32),
            "replica_mean_a": replica_a,
            "replica_mean_b": replica_b,
            "top_kind": table.top_kind[states],
            "top_alpha": table.top_alpha[states],
            "top_relative_ior": table.top_relative_ior[states],
            "top_eta": table.top_eta[states],
            "top_k": table.top_k[states],
            "top_color": table.top_color[states],
            "top_rotation": table.top_rotation[states],
        }
