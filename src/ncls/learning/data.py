from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.data import ReferenceDataset, SPLIT_NAMES

from .features import StackFeatureTable, load_feature_table


class ReferenceQueryStore:
    """只解释公共 query/response 合同，不依赖任何源材质表示。"""

    def __init__(self, dataset_path: Path | str, *, verify_hashes: bool = True) -> None:
        self.dataset = ReferenceDataset.open(dataset_path, verify_hashes=verify_hashes)
        self.query_group_count = self.dataset.query_group_count
        self.split_indices = {
            name: self.dataset.group_indices(name)
            for name in SPLIT_NAMES
        }

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> "ReferenceQueryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample_batch_indices(self, split: str, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if split not in SPLIT_NAMES or batch_size < 1:
            raise ValueError("invalid split or batch size")
        candidates = self.split_indices[split]
        if not len(candidates):
            raise ValueError(f"dataset split {split!r} is empty")
        return np.asarray(rng.choice(candidates, size=batch_size, replace=True), dtype=np.int64)

    def batch(self, query_group_indices: np.ndarray) -> dict[str, np.ndarray]:
        requested = np.asarray(query_group_indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("query_group_indices must be one-dimensional")
        result = self.dataset.group_batch(requested)
        result["view"] = result["wo"]
        result["lights"] = result["wi"]
        return result


class LayerStackReferenceStore(ReferenceQueryStore):
    """当前 LayerStack baseline 的显式源材质解码层。"""

    def __init__(self, dataset_path: Path | str, *, verify_hashes: bool = True) -> None:
        super().__init__(dataset_path, verify_hashes=verify_hashes)
        self.features: StackFeatureTable = load_feature_table(self.dataset)
        first = self.dataset.group_batch((0,))["wi"][0]
        for start in range(0, self.query_group_count, 4096):
            stop = min(start + 4096, self.query_group_count)
            batch_lights = np.asarray(self.dataset.stream["queries/wi"][start:stop], dtype=np.float32)
            if not np.all(batch_lights == first[None, ...]):
                raise ValueError("current LayerStack baseline requires one shared incident-direction grid")
        self.lights = np.array(first, dtype=np.float32, copy=True)

    def batch(self, query_group_indices: np.ndarray) -> dict[str, np.ndarray]:
        result = super().batch(query_group_indices)
        states = result["state_index"].astype(np.int64)
        table = self.features
        result.update({
            "interface_kinds": table.interface_kinds[states],
            "continuous": table.continuous[states],
            "interface_counts": table.interface_counts[states],
            "top_kind": table.top_kind[states],
            "top_alpha": table.top_alpha[states],
            "top_relative_ior": table.top_relative_ior[states],
            "top_eta": table.top_eta[states],
            "top_k": table.top_k[states],
            "top_color": table.top_color[states],
            "top_rotation": table.top_rotation[states],
        })
        return result
