from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.data import QUERY_ROLE_NAMES, ReferenceDataset, SPLIT_NAMES


PARTITION_POLICY_IDS = (
    "ncls.source-state-split@1",
    "ncls.query-role-within-state@1",
    "ncls.source-state-and-query-role@1",
)

from .features import StackFeatureTable, load_feature_table


class ReferenceQueryStore:
    """只解释公共 query/response 合同，不依赖任何源材质表示。"""

    def __init__(self, dataset_path: Path | str, *, verify_hashes: bool = True) -> None:
        self.dataset = ReferenceDataset.open(dataset_path, verify_hashes=verify_hashes)
        self.query_group_count = self.dataset.query_group_count
        self.source_split_indices = {
            name: self.dataset.group_indices(source_split=name)
            for name in SPLIT_NAMES
        }
        self.query_role_indices = {
            name: self.dataset.group_indices(query_role=name)
            for name in QUERY_ROLE_NAMES
        }

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> "ReferenceQueryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def partition_indices(self, policy_id: str, lifecycle_role: str) -> np.ndarray:
        if policy_id not in PARTITION_POLICY_IDS:
            raise ValueError(f"unsupported dataset partition policy: {policy_id}")
        if lifecycle_role not in SPLIT_NAMES:
            raise ValueError(f"lifecycle role must be one of {SPLIT_NAMES}")
        source_split = lifecycle_role if policy_id != "ncls.query-role-within-state@1" else None
        query_role = lifecycle_role if policy_id != "ncls.source-state-split@1" else None
        return self.dataset.group_indices(source_split=source_split, query_role=query_role)

    @staticmethod
    def sample_batch_indices(candidates: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if not len(candidates):
            raise ValueError("dataset lifecycle partition is empty")
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
