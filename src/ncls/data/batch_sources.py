from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from ncls.core.identity import sha256_json
from ncls.core.material import LayerStackIR
from ncls.data.reference import FalcorReferenceEvaluator
from ncls.data.training_batch import TrainingBatch
from ncls.data.stores import ReferenceCorpusStore, ReferenceQueryStore


class BatchSource(Protocol):
    kind: str
    identity: str
    source_contracts: tuple[Mapping[str, Any], ...]
    source_state_ids: tuple[str, ...]
    device: torch.device

    def next_batch(self, batch_size: int) -> TrainingBatch: ...
    def close(self) -> None: ...


class OfflineBatchSource:
    """把 frozen reference corpus 变成唯一 TrainingBatch；HDF5 只存在于该 producer。"""

    def __init__(
        self,
        store: ReferenceQueryStore | ReferenceCorpusStore,
        candidates: np.ndarray,
        *,
        device: torch.device | str,
        seed: int,
    ) -> None:
        self.kind = "offline"
        self.store = store
        self.candidates = np.asarray(candidates, dtype=np.int64)
        if not len(self.candidates):
            raise ValueError("offline batch source requires a nonempty partition")
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.identity = store.data_id
        families = tuple(sorted(set(map(str, store.state_strings("family_id").tolist()))))
        self.source_contracts = tuple(
            {"family_id": family, "source_contract_version": 1} for family in families
        )
        self.source_state_ids = tuple(map(str, store.state_strings("state_id").tolist()))

    def next_batch(self, batch_size: int) -> TrainingBatch:
        selected = self.store.sample_batch_indices(self.candidates, batch_size, self.rng)
        raw = self.store.batch(selected)
        source_index = np.asarray(raw["state_index"], dtype=np.int64)
        state_ids = tuple(self.source_state_ids[index] for index in source_index)
        family_values = np.asarray(self.store.state_strings("family_id"), dtype=object)[source_index]
        families = set(map(str, family_values.tolist()))
        if len(families) != 1:
            raise ValueError("one rectangular TrainingBatch may only contain one source family")
        tensors = {
            "source_index": torch.as_tensor(source_index, dtype=torch.int64, device=self.device),
            "wo": torch.as_tensor(raw["wo"], dtype=torch.float32, device=self.device),
            "wi": torch.as_tensor(raw["wi"], dtype=torch.float32, device=self.device),
            "target": torch.as_tensor(raw["mean"], dtype=torch.float32, device=self.device),
            "solid_angle_weight": torch.as_tensor(raw["solid_angle_weight"], dtype=torch.float32, device=self.device),
            "reference_pdf": torch.as_tensor(raw["reference_pdf"], dtype=torch.float32, device=self.device),
            "sample_count": torch.as_tensor(raw["sample_count"], dtype=torch.int64, device=self.device),
            "rng_seed": torch.as_tensor(raw["rng_seed"].astype(np.int64, copy=False), dtype=torch.int64, device=self.device),
            "query_role": torch.as_tensor(raw["query_role"], dtype=torch.int64, device=self.device),
        }
        return TrainingBatch(
            next(iter(families)), state_ids,
            "rgb-bsdf-times-absolute-shading-normal-light-cosine",
            tensors, {"producer": "offline", "data_source_identity": self.identity},
        )

    def close(self) -> None:
        self.store.close()


@dataclass
class _LiveBatchLease:
    owner: "LiveReferenceBatchSource"
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.owner._release(self)


class LiveReferenceBatchSource:
    """在线生成 LayerStack reference target；Falcor 输出不经过 CPU/HDF5。"""

    def __init__(
        self,
        materials: Sequence[LayerStackIR],
        source_state_ids: Sequence[str],
        *,
        light_count: int = 64,
        samples_per_replica: int = 64,
        max_depth: int = 64,
        max_batch_size: int = 64,
        seed: int = 0,
        device: torch.device | str = "cuda:0",
    ) -> None:
        if len(materials) != len(source_state_ids) or not materials:
            raise ValueError("live source materials and source_state_ids must be nonempty and aligned")
        if light_count < 1 or samples_per_replica < 1 or max_batch_size < 1:
            raise ValueError("live source sizes must be positive")
        self.kind = "live"
        self.materials = tuple(materials)
        self.source_state_ids = tuple(map(str, source_state_ids))
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("LiveReferenceBatchSource requires a CUDA device")
        self.light_count = int(light_count)
        self.samples_per_replica = int(samples_per_replica)
        self.max_batch_size = int(max_batch_size)
        self.rng = np.random.default_rng(seed)
        initial_lights = self._hemisphere((self.light_count,))
        self.evaluator = FalcorReferenceEvaluator(
            initial_lights,
            max_depth=max_depth,
            max_query_group_batch=max_batch_size,
        )
        self.identity = sha256_json(
            {
                "producer": "live-reference",
                "family_id": "ncls.layer-stack@1",
                "source_state_ids": list(self.source_state_ids),
                "light_count": self.light_count,
                "samples_per_replica": self.samples_per_replica,
                "max_depth": max_depth,
                "seed": seed,
            }
        )
        self.source_contracts = (
            {"family_id": "ncls.layer-stack@1", "source_contract_version": 1},
        )
        self._active_lease: _LiveBatchLease | None = None

    def _hemisphere(self, shape: tuple[int, ...]) -> np.ndarray:
        u = self.rng.random(shape, dtype=np.float32)
        phi = self.rng.random(shape, dtype=np.float32) * np.float32(2.0 * np.pi)
        radius = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - u * u))
        return np.stack((radius * np.cos(phi), radius * np.sin(phi), u), axis=-1).astype(np.float32)

    def _release(self, lease: _LiveBatchLease) -> None:
        if self._active_lease is not lease:
            raise RuntimeError("live batch lease does not belong to the active dispatch")
        self._active_lease = None

    def next_batch(self, batch_size: int) -> TrainingBatch:
        if self._active_lease is not None:
            raise RuntimeError("release the active live TrainingBatch before requesting another batch")
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"live batch_size must lie in [1, {self.max_batch_size}]")
        source_index = self.rng.integers(0, len(self.materials), size=batch_size, dtype=np.int64)
        wo = self._hemisphere((batch_size,))
        wi = self._hemisphere((batch_size, self.light_count))
        seeds = self.rng.integers(0, np.iinfo(np.uint32).max, size=batch_size, dtype=np.uint32)
        mean_a, _, mean_b, _ = self.evaluator.evaluate_query_groups_torch(
            [self.materials[index] for index in source_index],
            wo,
            sample_count_per_replica=self.samples_per_replica,
            query_group_seeds=seeds,
            light_directions=wi,
        )
        if mean_a.device != self.device:
            raise RuntimeError(f"Falcor interop returned {mean_a.device}, expected {self.device}")
        target = (mean_a + mean_b) * 0.5
        scalar_shape = (batch_size, self.light_count)
        lease = _LiveBatchLease(self)
        self._active_lease = lease
        rows = tuple(self.source_state_ids[index] for index in source_index)
        return TrainingBatch(
            "ncls.layer-stack@1",
            rows,
            "rgb-bsdf-times-absolute-shading-normal-light-cosine",
            {
                "source_index": torch.as_tensor(source_index, dtype=torch.int64, device=self.device),
                "wo": torch.as_tensor(wo, dtype=torch.float32, device=self.device),
                "wi": torch.as_tensor(wi, dtype=torch.float32, device=self.device),
                "target": target,
                "solid_angle_weight": torch.full(scalar_shape, 2.0 * np.pi / self.light_count, device=self.device),
                "reference_pdf": torch.full(scalar_shape, 1.0 / (2.0 * np.pi), device=self.device),
                "sample_count": torch.full(scalar_shape, 2 * self.samples_per_replica, dtype=torch.int64, device=self.device),
                "rng_seed": torch.as_tensor(np.broadcast_to(seeds[:, None], scalar_shape).copy().astype(np.int64), device=self.device),
                "query_role": torch.zeros(batch_size, dtype=torch.int64, device=self.device),
            },
            {
                "producer": "live-reference",
                "data_source_identity": self.identity,
                "host_readback": False,
                "synchronization": "wait_for_falcor",
            },
            lease,
        )

    def close(self) -> None:
        if self._active_lease is not None:
            raise RuntimeError("cannot close a live source while a TrainingBatch lease is active")
