from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from ncls.core.identity import sha256_json
from ncls.core.material import LayerStackIR
from ncls.data.reference import FalcorReferenceEvaluator
from ncls.data.native_features import (
    DenseNativeFeaturePyramid,
    NativeFeaturePyramid,
    MaterialXNativeFeaturePyramid,
    encode_layer_stack_native_features,
    layer_stack_native_feature_layout,
)
from ncls.data.contract import SourceState
from ncls.data.providers.materialx import MaterialXGpuQueryRuntime, MaterialXProvider
from ncls.data.providers.mdl import MdlGpuQueryRuntime, MdlProvider
from ncls.data.providers.base import implementation_hash
from ncls.data.training_batch import TrainingBatch, TrainingRouteRequest
from ncls.data.stores import ReferenceCorpusStore, ReferenceQueryStore
from ncls.paths import PROJECT_ROOT


_LIVE_PRODUCER_IMPLEMENTATION_SHA256 = implementation_hash(
    (
        Path(__file__),
        PROJECT_ROOT / "src/ncls/data/native_features.py",
        PROJECT_ROOT / "src/ncls/data/training_batch.py",
        PROJECT_ROOT / "src/ncls/data/reference.py",
    )
)


def _uniform_hemisphere(count: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.random(count, dtype=np.float32)
    phi = rng.random(count, dtype=np.float32) * np.float32(2.0 * np.pi)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1).astype(np.float32)


def _half_difference_directions(
    count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    views = np.empty((count, 3), dtype=np.float32)
    lights = np.empty((count, 3), dtype=np.float32)
    jacobian = np.empty(count, dtype=np.float32)
    filled = 0
    while filled < count:
        candidate_count = max(1024, 2 * (count - filled))
        half = _uniform_hemisphere(candidate_count, rng)
        difference = _uniform_hemisphere(candidate_count, rng)
        cosine_theta, sine_theta = half[:, 2], np.sqrt(
            np.maximum(0.0, 1.0 - half[:, 2] * half[:, 2])
        )
        phi = np.arctan2(half[:, 1], half[:, 0])
        cosine_phi, sine_phi = np.cos(phi), np.sin(phi)

        def rotate(local: np.ndarray) -> np.ndarray:
            x = cosine_theta * local[:, 0] + sine_theta * local[:, 2]
            y = local[:, 1]
            z = -sine_theta * local[:, 0] + cosine_theta * local[:, 2]
            return np.stack(
                (cosine_phi * x - sine_phi * y, sine_phi * x + cosine_phi * y, z),
                axis=1,
            ).astype(np.float32)

        first = rotate(difference)
        reflected_difference = difference.copy()
        reflected_difference[:, :2] *= -1.0
        second = rotate(reflected_difference)
        valid = (first[:, 2] > 0.0) & (second[:, 2] > 0.0)
        selected = np.flatnonzero(valid)[: count - filled]
        take = len(selected)
        views[filled : filled + take] = first[selected]
        lights[filled : filled + take] = second[selected]
        jacobian[filled : filled + take] = 4.0 * difference[selected, 2]
        filled += take
    proposal_pdf = 1.0 / np.maximum(4.0 * np.pi * np.pi * jacobian, 1e-12)
    return (
        views,
        lights,
        proposal_pdf.astype(np.float32),
        (1.0 / proposal_pdf).astype(np.float32),
    )


class BatchSource(Protocol):
    kind: str
    identity: str
    source_contracts: tuple[Mapping[str, Any], ...]
    source_state_ids: tuple[str, ...]
    device: torch.device

    def next_batch(self, request: TrainingRouteRequest) -> TrainingBatch: ...
    def materialization_features(self) -> NativeFeaturePyramid: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
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
        self.seed = int(seed)
        self._rng_by_route: dict[str, np.random.Generator] = {}
        self._request_count: dict[str, int] = {}
        self.identity = store.data_id
        families = tuple(sorted(set(map(str, store.state_strings("family_id").tolist()))))
        self.source_contracts = tuple(
            {"family_id": family, "source_contract_version": 1} for family in families
        )
        self.source_state_ids = tuple(map(str, store.state_strings("state_id").tolist()))

    def _rng(self, request: TrainingRouteRequest) -> np.random.Generator:
        if request.name not in self._rng_by_route:
            route_seed = np.random.SeedSequence((self.seed, request.seed))
            self._rng_by_route[request.name] = np.random.default_rng(route_seed)
        return self._rng_by_route[request.name]

    def next_batch(self, request: TrainingRouteRequest) -> TrainingBatch:
        selected = self.store.sample_batch_indices(
            self.candidates, request.batch_size, self._rng(request)
        )
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
        if tensors["wi"].shape[1] != request.direction_count:
            raise ValueError("offline shard direction count disagrees with training route")
        for name in ("uv", "uv_dx", "uv_dy"):
            if name in raw:
                tensors[name] = torch.as_tensor(raw[name], dtype=torch.float32, device=self.device)
        count = self._request_count.get(request.name, 0)
        self._request_count[request.name] = count + 1
        return TrainingBatch(
            next(iter(families)), state_ids,
            "rgb-bsdf-times-absolute-shading-normal-light-cosine",
            tensors, {
                "producer": "offline", "data_source_identity": self.identity,
                "route_name": request.name, "request_index": count,
                "global_step": request.global_step,
            },
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        raise RuntimeError("offline corpus does not contain a native feature pyramid")

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "rng_by_route": {
                name: generator.bit_generator.state
                for name, generator in self._rng_by_route.items()
            },
            "request_count": dict(self._request_count),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"rng_by_route", "request_count"}:
            raise ValueError("offline batch source state fields are invalid")
        generators: dict[str, np.random.Generator] = {}
        for name, value in dict(state["rng_by_route"]).items():
            generator = np.random.default_rng()
            generator.bit_generator.state = value
            generators[str(name)] = generator
        self._rng_by_route = generators
        self._request_count = {
            str(name): int(value) for name, value in dict(state["request_count"]).items()
        }

    def close(self) -> None:
        self.store.close()


@dataclass
class _LiveBatchLease:
    owner: Any
    route_name: str
    slot_index: int = -1
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
        self.seed = int(seed)
        self._rng_by_route: dict[str, np.random.Generator] = {}
        self._request_count: dict[str, int] = {}
        initial_lights = self._hemisphere((self.light_count,))
        self._initial_lights = initial_lights
        self._max_depth = int(max_depth)
        self._evaluators: dict[str, FalcorReferenceEvaluator] = {}
        self.identity = sha256_json(
            {
                "producer": "live-reference",
                "family_id": "ncls.layer-stack@1",
                "source_state_ids": list(self.source_state_ids),
                "light_count": self.light_count,
                "samples_per_replica": self.samples_per_replica,
                "max_depth": max_depth,
                "seed": seed,
                "training_producer_implementation_sha256": _LIVE_PRODUCER_IMPLEMENTATION_SHA256,
            }
        )
        self.source_contracts = (
            {"family_id": "ncls.layer-stack@1", "source_contract_version": 1},
        )
        self._active_leases: dict[str, _LiveBatchLease] = {}

    def _rng(self, route_name: str, route_seed: int = 0) -> np.random.Generator:
        if route_name not in self._rng_by_route:
            self._rng_by_route[route_name] = np.random.default_rng(
                np.random.SeedSequence((self.seed, route_seed))
            )
        return self._rng_by_route[route_name]

    def _hemisphere(
        self, shape: tuple[int, ...], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        generator = self._rng("initial") if rng is None else rng
        u = generator.random(shape, dtype=np.float32)
        phi = generator.random(shape, dtype=np.float32) * np.float32(2.0 * np.pi)
        radius = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - u * u))
        return np.stack((radius * np.cos(phi), radius * np.sin(phi), u), axis=-1).astype(np.float32)

    def _release(self, lease: _LiveBatchLease) -> None:
        if self._active_leases.get(lease.route_name) is not lease:
            raise RuntimeError("live batch lease does not belong to the active dispatch")
        del self._active_leases[lease.route_name]

    def next_batch(self, request: TrainingRouteRequest) -> TrainingBatch:
        if request.name in self._active_leases:
            raise RuntimeError("release the active route TrainingBatch before requesting another batch")
        if request.direction_count != self.light_count:
            raise ValueError("live route direction_count disagrees with configured light_count")
        batch_size = request.batch_size
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"live batch_size must lie in [1, {self.max_batch_size}]")
        rng = self._rng(request.name, request.seed)
        source_index = rng.integers(0, len(self.materials), size=batch_size, dtype=np.int64)
        proposal = request.options.get(
            "direction_proposal", "fixed-uniform-hemisphere-grid@1"
        )
        if proposal == "uniform-half-difference@1":
            if self.light_count != 1:
                raise ValueError("faithful half/difference LayerStack route requires direction_count=1")
            wo, light, reference_pdf, solid_angle_weight = _half_difference_directions(
                batch_size, rng
            )
            wi = light[:, None, :]
            reference_pdf = reference_pdf[:, None]
            solid_angle_weight = solid_angle_weight[:, None]
        elif proposal == "uniform-hemisphere-conditioning@1":
            wo = _uniform_hemisphere(batch_size, rng)
            wi = _uniform_hemisphere(batch_size, rng)[:, None, :]
            reference_pdf = np.full(
                (batch_size, 1), 1.0 / (2.0 * np.pi), dtype=np.float32
            )
            solid_angle_weight = np.full(
                (batch_size, 1), 2.0 * np.pi, dtype=np.float32
            )
        elif proposal == "fixed-uniform-hemisphere-grid@1":
            wo = _uniform_hemisphere(batch_size, rng)
            wi = np.broadcast_to(
                self._initial_lights[None, :, :],
                (batch_size, self.light_count, 3),
            ).copy()
            reference_pdf = np.full(
                (batch_size, self.light_count),
                1.0 / (2.0 * np.pi),
                dtype=np.float32,
            )
            solid_angle_weight = np.full(
                (batch_size, self.light_count), 2.0 * np.pi, dtype=np.float32
            )
        else:
            raise ValueError("LayerStack route direction proposal is unsupported")
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=batch_size, dtype=np.uint32)
        target_estimator = request.options.get("target_estimator", "reference")
        if target_estimator == "reference":
            evaluator = self._evaluators.get(request.name)
            if evaluator is None:
                evaluator = FalcorReferenceEvaluator(
                    self._initial_lights,
                    max_depth=self._max_depth,
                    max_query_group_batch=self.max_batch_size,
                )
                self._evaluators[request.name] = evaluator
            mean_a, _, mean_b, _ = evaluator.evaluate_query_groups_torch(
                [self.materials[index] for index in source_index],
                wo,
                sample_count_per_replica=self.samples_per_replica,
                query_group_seeds=seeds,
                light_directions=wi,
            )
            if mean_a.device != self.device:
                raise RuntimeError(f"Falcor interop returned {mean_a.device}, expected {self.device}")
            target = (mean_a + mean_b) * 0.5
            sample_count = 2 * self.samples_per_replica
        elif target_estimator == "learned-sampler":
            target = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.device)
            sample_count = 1
        else:
            raise ValueError("LayerStack route target estimator is unsupported")
        scalar_shape = (batch_size, self.light_count)
        lease = _LiveBatchLease(self, request.name)
        self._active_leases[request.name] = lease
        rows = tuple(self.source_state_ids[index] for index in source_index)
        features = np.stack(
            [encode_layer_stack_native_features(self.materials[index]) for index in source_index]
        )
        layout = layer_stack_native_feature_layout()
        request_index = self._request_count.get(request.name, 0)
        self._request_count[request.name] = request_index + 1
        return TrainingBatch(
            "ncls.layer-stack@1",
            rows,
            "rgb-bsdf-times-absolute-shading-normal-light-cosine",
            {
                "source_index": torch.as_tensor(source_index, dtype=torch.int64, device=self.device),
                "wo": torch.as_tensor(wo, dtype=torch.float32, device=self.device),
                "wi": torch.as_tensor(wi, dtype=torch.float32, device=self.device),
                "target": target,
                "solid_angle_weight": torch.as_tensor(solid_angle_weight, device=self.device),
                "reference_pdf": torch.as_tensor(reference_pdf, device=self.device),
                "sample_count": torch.full(scalar_shape, sample_count, dtype=torch.int64, device=self.device),
                "rng_seed": torch.as_tensor(np.broadcast_to(seeds[:, None], scalar_shape).copy().astype(np.int64), device=self.device),
                "query_role": torch.full((batch_size,), request.query_role, dtype=torch.int64, device=self.device),
                "uv": torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device),
                "uv_dx": torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device),
                "uv_dy": torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device),
                "mip_level": torch.zeros(batch_size, dtype=torch.float32, device=self.device),
                "native_features": torch.as_tensor(features, dtype=torch.float32, device=self.device),
                "sample_u": torch.as_tensor(rng.random((batch_size, 2), dtype=np.float32), device=self.device),
            },
            {
                "producer": "live-reference",
                "data_source_identity": self.identity,
                "host_readback": False,
                "synchronization": "wait_for_falcor",
                "route_name": request.name,
                "request_index": request_index,
                "global_step": request.global_step,
                "native_feature_layout_id": layout.layout_id,
                "source_adaptation_id": "layer-stack-uniform-1x1@1",
                "direction_proposal": proposal,
                "target_estimator": target_estimator,
            },
            lease,
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        values = np.stack(
            [encode_layer_stack_native_features(material) for material in self.materials]
        )
        if len(values) != 1:
            raise RuntimeError("faithful NVIDIA materialization trains one source snapshot per run")
        level = torch.as_tensor(values[0:1, None, :], dtype=torch.float32)
        return DenseNativeFeaturePyramid((level,))

    def state_dict(self) -> Mapping[str, Any]:
        if self._active_leases:
            raise RuntimeError("cannot checkpoint a live source while a TrainingBatch lease is active")
        return {
            "rng_by_route": {
                name: generator.bit_generator.state
                for name, generator in self._rng_by_route.items()
            },
            "request_count": dict(self._request_count),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._active_leases:
            raise RuntimeError("cannot restore a live source while a TrainingBatch lease is active")
        if set(state) != {"rng_by_route", "request_count"}:
            raise ValueError("live batch source state fields are invalid")
        generators: dict[str, np.random.Generator] = {}
        for name, value in dict(state["rng_by_route"]).items():
            generator = np.random.default_rng()
            generator.bit_generator.state = value
            generators[str(name)] = generator
        self._rng_by_route = generators
        self._request_count = {
            str(name): int(value) for name, value in dict(state["request_count"]).items()
        }

    def close(self) -> None:
        if self._active_leases:
            raise RuntimeError("cannot close a live source while a TrainingBatch lease is active")


class MaterialXLiveReferenceBatchSource:
    """论文几何的MaterialX spatial online producer；reference结果保持GPU-resident。"""

    def __init__(
        self,
        provider: MaterialXProvider,
        state: SourceState,
        *,
        max_batch_size: int,
        query_tile_size: int = 262_144,
        seed: int = 0,
        device: torch.device | str = "cuda:0",
    ) -> None:
        self.kind = "live"
        self.provider = provider
        self.state = state
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MaterialX live training requires a CUDA device")
        if state.family_id != "materialx.document@1.39.4":
            raise ValueError("MaterialX live training requires a MaterialX source snapshot")
        if max_batch_size < 1 or query_tile_size < 1:
            raise ValueError("MaterialX live batch and query tile sizes must be positive")
        self.max_batch_size = int(max_batch_size)
        self.query_tile_size = int(query_tile_size)
        self.seed = int(seed)
        runtime = state.runtime_state
        self._feature_pyramid = MaterialXNativeFeaturePyramid.from_textures(
            runtime.inputs,
            base_color=runtime.base_color,
            roughness=runtime.roughness,
            metalness=runtime.metalness,
            normal=runtime.normal,
        )
        self._runtime = MaterialXGpuQueryRuntime(
            provider, state, query_capacity=self.query_tile_size, slot_count=2
        )
        self.source_state_ids = (state.state_id,)
        self.source_contracts = (
            {"family_id": state.family_id, "source_contract_version": 1},
        )
        self.identity = sha256_json(
            {
                "producer": "materialx-live-reference",
                "source_state_id": state.state_id,
                "native_feature_layout_id": self._feature_pyramid.layout_id,
                "query_tile_size": self.query_tile_size,
                "seed": self.seed,
                "reference_implementation_sha256": provider.descriptor.implementation_sha256,
                "training_producer_implementation_sha256": _LIVE_PRODUCER_IMPLEMENTATION_SHA256,
            }
        )
        self._rng_by_route: dict[str, np.random.Generator] = {}
        self._request_count: dict[str, int] = {}
        self._active_leases: dict[str, _LiveBatchLease] = {}
        self._free_slots = [0, 1]

    def _rng(self, request: TrainingRouteRequest) -> np.random.Generator:
        if request.name not in self._rng_by_route:
            self._rng_by_route[request.name] = np.random.default_rng(
                np.random.SeedSequence((self.seed, request.seed))
            )
        return self._rng_by_route[request.name]

    @staticmethod
    def _request_generator(rng: np.random.Generator, device: torch.device) -> tuple[torch.Generator, int]:
        seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator, seed

    @staticmethod
    def _uniform_hemisphere(
        count: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        random_values = torch.rand(
            (count, 2), dtype=torch.float32, device=device, generator=generator
        )
        z = random_values[:, 0]
        phi = random_values[:, 1] * (2.0 * math.pi)
        radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=1)

    @classmethod
    def _half_difference_directions_torch(
        cls,
        count: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        views = torch.empty((count, 3), dtype=torch.float32, device=device)
        lights = torch.empty_like(views)
        jacobian = torch.empty(count, dtype=torch.float32, device=device)
        filled = 0
        while filled < count:
            candidate_count = max(1024, 2 * (count - filled))
            half = cls._uniform_hemisphere(candidate_count, generator, device)
            difference = cls._uniform_hemisphere(candidate_count, generator, device)
            cosine_theta = half[:, 2]
            sine_theta = torch.sqrt(torch.clamp(1.0 - cosine_theta * cosine_theta, min=0.0))
            phi = torch.atan2(half[:, 1], half[:, 0])
            cosine_phi, sine_phi = torch.cos(phi), torch.sin(phi)

            def rotate(local: torch.Tensor) -> torch.Tensor:
                x = cosine_theta * local[:, 0] + sine_theta * local[:, 2]
                y = local[:, 1]
                z = -sine_theta * local[:, 0] + cosine_theta * local[:, 2]
                return torch.stack(
                    (cosine_phi * x - sine_phi * y, sine_phi * x + cosine_phi * y, z),
                    dim=1,
                )

            first = rotate(difference)
            reflected_difference = difference * torch.tensor(
                (-1.0, -1.0, 1.0), dtype=torch.float32, device=device
            )
            second = rotate(reflected_difference)
            selected = torch.nonzero(
                (first[:, 2] > 0.0) & (second[:, 2] > 0.0), as_tuple=False
            ).flatten()[: count - filled]
            take = int(selected.numel())
            views[filled : filled + take] = first[selected]
            lights[filled : filled + take] = second[selected]
            jacobian[filled : filled + take] = 4.0 * difference[selected, 2]
            filled += take
        proposal_pdf = 1.0 / torch.clamp(
            4.0 * math.pi * math.pi * jacobian, min=1e-12
        )
        return views, lights, proposal_pdf, 1.0 / proposal_pdf

    @staticmethod
    def _half_difference_directions(
        count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """保留独立CPU几何oracle；formal producer使用CUDA实现。"""

        return _half_difference_directions(count, rng)

    @staticmethod
    def _mollified_views(
        views: torch.Tensor,
        count: int,
        angle_degrees: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if count == 1 or angle_degrees <= 0.0:
            return views
        repeated = torch.repeat_interleave(views, count, dim=0)
        random_values = torch.rand(
            (len(repeated), 2),
            dtype=torch.float32,
            device=views.device,
            generator=generator,
        )
        cosine_max = math.cos(math.radians(angle_degrees))
        cosine_theta = 1.0 - random_values[:, 0] * (1.0 - cosine_max)
        sine_theta = torch.sqrt(torch.clamp(1.0 - cosine_theta * cosine_theta, min=0.0))
        phi = random_values[:, 1] * (2.0 * math.pi)
        helper = torch.zeros_like(repeated)
        use_x = torch.abs(repeated[:, 2]) > 0.9
        helper[~use_x, 2] = 1.0
        helper[use_x, 0] = 1.0
        tangent = torch.linalg.cross(helper, repeated)
        tangent /= torch.clamp(torch.linalg.vector_norm(tangent, dim=1, keepdim=True), min=1e-12)
        bitangent = torch.linalg.cross(repeated, tangent)
        result = (
            repeated * cosine_theta[:, None]
            + tangent * (sine_theta * torch.cos(phi))[:, None]
            + bitangent * (sine_theta * torch.sin(phi))[:, None]
        )
        return result

    def _spatial_samples(
        self,
        uv: torch.Tensor,
        mip_level: torch.Tensor,
        request: TrainingRouteRequest,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        options = request.options
        rate = float(options.get("spatial_samples_per_texel_area", 1.0))
        cap = int(options.get("maximum_spatial_samples", 64))
        if rate <= 0.0 or cap < 1:
            raise ValueError("MaterialX spatial sampling recipe is invalid")
        counts = torch.clamp(
            torch.round(rate * torch.pow(4.0, mip_level)).long(), 1, cap
        )
        group_count = int(counts.sum().item())
        groups = torch.repeat_interleave(
            torch.arange(len(uv), dtype=torch.int64, device=self.device),
            counts,
            output_size=group_count,
        )
        base = uv[groups]
        selected_mip = mip_level[groups]
        random_values = torch.randn(
            (len(groups), 2),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        height, width = self._feature_pyramid.level_shapes[0]
        sigma = torch.pow(2.0, selected_mip) * 0.5
        offsets = random_values * torch.stack((sigma / width, sigma / height), dim=1)
        sampled_uv = torch.remainder(base + offsets, 1.0)
        features = self._feature_pyramid.sample_torch(sampled_uv, selected_mip)
        accumulated = torch.zeros(
            (len(uv), features.shape[1]), dtype=torch.float32, device=self.device
        )
        accumulated.index_add_(0, groups, features)
        accumulated /= counts[:, None]
        return sampled_uv, selected_mip, groups, accumulated, counts

    def _reference_target(
        self,
        slot_index: int,
        views: torch.Tensor,
        lights: torch.Tensor,
        uv: torch.Tensor,
        gradients: torch.Tensor,
        mip_level: torch.Tensor,
        request: TrainingRouteRequest,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_uv, _, spatial_groups, features, spatial_counts = self._spatial_samples(
            uv, mip_level, request, generator
        )
        spatial_views = views[spatial_groups]
        spatial_lights = lights[spatial_groups]
        spatial_gradients = gradients[spatial_groups]
        recipe = dict(request.options["mollification"])
        mollification_steps = int(recipe["steps"])
        if request.global_step < mollification_steps:
            position = request.global_step / max(mollification_steps, 1)
            angle = 0.5 * float(recipe["start_degrees"]) * (
                1.0 + math.cos(math.pi * position)
            )
            mollification_count = int(recipe["samples"])
        else:
            angle = 0.0
            mollification_count = 1
        if mollification_count > self.query_tile_size:
            raise ValueError("MaterialX query tile must hold one complete mollification estimator")
        target = torch.zeros((len(views), 3), dtype=torch.float32, device=self.device)
        sample_counts = spatial_counts * mollification_count
        spatial_tile_size = max(1, self.query_tile_size // mollification_count)
        for begin in range(0, len(spatial_views), spatial_tile_size):
            end = min(begin + spatial_tile_size, len(spatial_views))
            query_views = self._mollified_views(
                spatial_views[begin:end], mollification_count, angle, generator
            )
            query_lights = torch.repeat_interleave(
                spatial_lights[begin:end], mollification_count, dim=0
            )
            query_uv = torch.repeat_interleave(
                sampled_uv[begin:end], mollification_count, dim=0
            )
            query_gradients = torch.repeat_interleave(
                spatial_gradients[begin:end], mollification_count, dim=0
            )
            query_groups = torch.repeat_interleave(
                spatial_groups[begin:end], mollification_count, dim=0
            )
            values = self._runtime.evaluate_torch(
                slot_index,
                query_views,
                query_lights,
                query_uv,
                query_gradients,
            )
            target.index_add_(0, query_groups, values)
        target /= sample_counts.to(dtype=torch.float32)[:, None]
        return target[:, None, :], sample_counts, features

    def _release(self, lease: _LiveBatchLease) -> None:
        if self._active_leases.get(lease.route_name) is not lease:
            raise RuntimeError("MaterialX batch lease does not belong to the active dispatch")
        del self._active_leases[lease.route_name]
        if lease.slot_index >= 0:
            self._free_slots.append(lease.slot_index)
            self._free_slots.sort()
        if not self._active_leases:
            # A training iteration is one Falcor frame. Closing it rotates the
            # transient heap and retires submissions accumulated by tiled
            # reference queries before the next CUDA/Falcor hand-off begins.
            self._runtime.device.end_frame()

    def next_batch(self, request: TrainingRouteRequest) -> TrainingBatch:
        if request.name in self._active_leases:
            raise RuntimeError("release the active route TrainingBatch before requesting another batch")
        if request.batch_size > self.max_batch_size or request.direction_count != 1:
            raise ValueError("MaterialX faithful route requires batch within capacity and direction_count=1")
        proposal = request.options.get("direction_proposal")
        if proposal not in {
            "uniform-half-difference@1", "uniform-hemisphere-conditioning@1"
        }:
            raise ValueError("MaterialX faithful route has an unsupported direction proposal")
        rng = self._rng(request)
        generator, request_seed = self._request_generator(rng, self.device)
        batch_size = request.batch_size
        if proposal == "uniform-half-difference@1":
            views, lights, reference_pdf, solid_angle_weight = self._half_difference_directions_torch(
                batch_size, generator, self.device
            )
        else:
            views = self._uniform_hemisphere(batch_size, generator, self.device)
            lights = self._uniform_hemisphere(batch_size, generator, self.device)
            reference_pdf = torch.full(
                (batch_size,), 1.0 / (2.0 * math.pi), dtype=torch.float32, device=self.device
            )
            solid_angle_weight = torch.full(
                (batch_size,), 2.0 * math.pi, dtype=torch.float32, device=self.device
            )
        uv = torch.rand(
            (batch_size, 2), dtype=torch.float32, device=self.device, generator=generator
        )
        scale = float(request.options.get("mip_exponential_scale", 1.0))
        if scale <= 0.0:
            raise ValueError("MaterialX mip exponential scale must be positive")
        mip_level = torch.clamp(
            torch.floor(
                -torch.log1p(
                    -torch.rand(
                        batch_size,
                        dtype=torch.float32,
                        device=self.device,
                        generator=generator,
                    )
                )
                * scale
            ),
            max=len(self._feature_pyramid.level_shapes) - 1,
        )
        height, width = self._feature_pyramid.level_shapes[0]
        footprint = torch.pow(2.0, mip_level)
        gradients = torch.zeros(
            (batch_size, 4), dtype=torch.float32, device=self.device
        )
        gradients[:, 0] = footprint / width
        gradients[:, 3] = footprint / height
        target_estimator = request.options.get("target_estimator", "reference")
        slot_index = -1
        if target_estimator == "reference":
            if not self._free_slots:
                raise RuntimeError("MaterialX live reference has no free in-flight query slot")
            slot_index = self._free_slots.pop(0)
            target, sample_counts, native_features = self._reference_target(
                slot_index, views, lights, uv, gradients, mip_level, request, generator
            )
        elif target_estimator == "learned-sampler":
            sampled_uv, _, _, native_features, _ = self._spatial_samples(
                uv, mip_level, request, generator
            )
            del sampled_uv
            target = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.device)
            sample_counts = torch.ones(batch_size, dtype=torch.int64, device=self.device)
        else:
            raise ValueError("MaterialX route target_estimator is unsupported")
        lease = _LiveBatchLease(self, request.name, slot_index)
        self._active_leases[request.name] = lease
        request_index = self._request_count.get(request.name, 0)
        self._request_count[request.name] = request_index + 1
        seeds = torch.randint(
            0,
            np.iinfo(np.int32).max,
            (batch_size, 1),
            dtype=torch.int64,
            device=self.device,
            generator=generator,
        )
        return TrainingBatch(
            self.state.family_id,
            self.source_state_ids * batch_size,
            "rgb-bsdf-times-absolute-shading-normal-light-cosine",
            {
                "source_index": torch.zeros(batch_size, dtype=torch.int64, device=self.device),
                "wo": views,
                "wi": lights[:, None, :],
                "target": target,
                "solid_angle_weight": solid_angle_weight[:, None],
                "reference_pdf": reference_pdf[:, None],
                "sample_count": sample_counts[:, None],
                "rng_seed": seeds,
                "query_role": torch.full(
                    (batch_size,), request.query_role, dtype=torch.int64, device=self.device
                ),
                "uv": uv,
                "uv_dx": gradients[:, :2],
                "uv_dy": gradients[:, 2:],
                "mip_level": mip_level,
                "native_features": native_features,
                "sample_u": torch.rand(
                    (batch_size, 2),
                    dtype=torch.float32,
                    device=self.device,
                    generator=generator,
                ),
            },
            {
                "producer": "materialx-live-reference",
                "data_source_identity": self.identity,
                "host_readback": False,
                "synchronization": "wait_for_falcor",
                "route_name": request.name,
                "request_index": request_index,
                "global_step": request.global_step,
                "native_feature_layout_id": self._feature_pyramid.layout_id,
                "source_adaptation_id": "materialx-standard-surface-spatial@1",
                "direction_proposal": proposal,
                "target_estimator": target_estimator,
                "gpu_request_seed": request_seed,
                "gpu_online_sampling": True,
            },
            lease,
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        return self._feature_pyramid

    def state_dict(self) -> Mapping[str, Any]:
        if self._active_leases:
            raise RuntimeError("cannot checkpoint MaterialX source with active batch leases")
        return {
            "rng_by_route": {
                name: generator.bit_generator.state
                for name, generator in self._rng_by_route.items()
            },
            "request_count": dict(self._request_count),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._active_leases or set(state) != {"rng_by_route", "request_count"}:
            raise ValueError("MaterialX batch source state is invalid")
        generators = {}
        for name, value in dict(state["rng_by_route"]).items():
            generator = np.random.default_rng()
            generator.bit_generator.state = value
            generators[str(name)] = generator
        self._rng_by_route = generators
        self._request_count = {
            str(name): int(value) for name, value in dict(state["request_count"]).items()
        }

    def close(self) -> None:
        if self._active_leases:
            raise RuntimeError("cannot close MaterialX source with active batch leases")
        self._runtime.close()
        self.provider.close()


class MdlLiveReferenceBatchSource:
    """原生 MDL online producer；正式 target 由 current Falcor shared buffers 产生。"""

    def __init__(
        self,
        provider: MdlProvider,
        state: SourceState,
        *,
        max_batch_size: int,
        query_tile_size: int = 262_144,
        seed: int = 0,
        device: torch.device | str = "cuda:0",
    ) -> None:
        self.kind = "live"
        self.provider = provider
        self.state = state
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MDL live training requires a CUDA device")
        if state.family_id != "mdl.program@1":
            raise ValueError("MDL live training requires an mdl.program@1 source snapshot")
        if max_batch_size < 1 or query_tile_size < 1:
            raise ValueError("MDL live batch and query tile sizes must be positive")
        self.max_batch_size = int(max_batch_size)
        self.query_tile_size = int(query_tile_size)
        self.seed = int(seed)
        artifact = state.runtime_state.artifact
        self._runtime = MdlGpuQueryRuntime(
            artifact,
            sdk_root=provider.provider_config.sdk_root,
            query_capacity=self.query_tile_size,
            slot_count=2,
        )
        self._feature_values, self._feature_fields = self._parameter_features(state)
        self._feature_pyramid = DenseNativeFeaturePyramid(
            (torch.as_tensor(self._feature_values[None, None, :], dtype=torch.float32),)
        )
        self._layout_id = sha256_json(
            {
                "family_id": state.family_id,
                "source_contract_version": 1,
                "fields": self._feature_fields,
                "spatial": False,
            }
        )
        self.source_state_ids = (state.state_id,)
        self.source_contracts = (
            {"family_id": state.family_id, "source_contract_version": 1},
        )
        self.identity = sha256_json(
            {
                "producer": "mdl-live-reference",
                "source_state_id": state.state_id,
                "native_feature_layout_id": self._layout_id,
                "query_tile_size": self.query_tile_size,
                "seed": self.seed,
                "reference_implementation_sha256": provider.descriptor.implementation_sha256,
                "training_producer_implementation_sha256": _LIVE_PRODUCER_IMPLEMENTATION_SHA256,
            }
        )
        self._rng_by_route: dict[str, np.random.Generator] = {}
        self._request_count: dict[str, int] = {}
        self._active_leases: dict[str, _LiveBatchLease] = {}
        self._free_slots = [0, 1]

    @staticmethod
    def _parameter_features(state: SourceState) -> tuple[np.ndarray, list[dict[str, Any]]]:
        payload = json.loads(state.snapshot.native_payload.decode("utf-8"))
        values = []
        fields = []
        for name, descriptor in sorted(payload.get("arguments", {}).items()):
            value = descriptor.get("value")
            if isinstance(value, bool):
                components = [float(value)]
            elif isinstance(value, (int, float)):
                components = [float(value)]
            elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                components = [float(item) for item in value]
            elif isinstance(value, dict) and set(value) >= {"name", "value"}:
                components = [float(value["value"])]
            else:
                continue
            values.extend(components)
            fields.append(
                {
                    "name": name,
                    "mdl_type": str(descriptor["mdl_type"]),
                    "channels": len(components),
                    "filter_rule": "constant",
                }
            )
        if not values:
            values = [0.0]
            fields = [{"name": "no-numeric-arguments", "channels": 1, "filter_rule": "constant"}]
        result = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("MDL numeric source arguments must be finite")
        return result, fields

    def _rng(self, request: TrainingRouteRequest) -> np.random.Generator:
        if request.name not in self._rng_by_route:
            self._rng_by_route[request.name] = np.random.default_rng(
                np.random.SeedSequence((self.seed, request.seed))
            )
        return self._rng_by_route[request.name]

    def _release(self, lease: _LiveBatchLease) -> None:
        if self._active_leases.get(lease.route_name) is not lease:
            raise RuntimeError("MDL batch lease does not belong to the active dispatch")
        del self._active_leases[lease.route_name]
        if lease.slot_index >= 0:
            self._free_slots.append(lease.slot_index)
            self._free_slots.sort()
        if not self._active_leases:
            self._runtime._device.end_frame()

    def next_batch(self, request: TrainingRouteRequest) -> TrainingBatch:
        if request.name in self._active_leases:
            raise RuntimeError("release the active route TrainingBatch before requesting another batch")
        if request.batch_size > self.max_batch_size or request.direction_count != 1:
            raise ValueError("MDL V1 live route requires batch within capacity and direction_count=1")
        proposal = request.options.get("direction_proposal")
        if proposal not in {
            "uniform-half-difference@1",
            "uniform-hemisphere-conditioning@1",
        }:
            raise ValueError("MDL live route has an unsupported direction proposal")
        rng = self._rng(request)
        generator, request_seed = MaterialXLiveReferenceBatchSource._request_generator(
            rng, self.device
        )
        batch_size = request.batch_size
        if proposal == "uniform-half-difference@1":
            views, lights, proposal_pdf, solid_angle_weight = (
                MaterialXLiveReferenceBatchSource._half_difference_directions_torch(
                    batch_size, generator, self.device
                )
            )
        else:
            views = MaterialXLiveReferenceBatchSource._uniform_hemisphere(
                batch_size, generator, self.device
            )
            lights = MaterialXLiveReferenceBatchSource._uniform_hemisphere(
                batch_size, generator, self.device
            )
            proposal_pdf = torch.full(
                (batch_size,),
                1.0 / (2.0 * math.pi),
                dtype=torch.float32,
                device=self.device,
            )
            solid_angle_weight = torch.full(
                (batch_size,),
                2.0 * math.pi,
                dtype=torch.float32,
                device=self.device,
            )
        uv = torch.rand(
            (batch_size, 2), dtype=torch.float32, device=self.device, generator=generator
        )
        # MDL V1 与冻结的 falcor2 oracle 都采用 ExplicitLod(0)。这些字段
        # 仍保留在统一 TrainingBatch 合同中，但不能伪装成 reference 已消费
        # 的 footprint/mip 输入。
        mip_level = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        gradients = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)
        target_estimator = request.options.get("target_estimator", "reference")
        slot_index = -1
        if target_estimator == "reference":
            if not self._free_slots:
                raise RuntimeError("MDL live reference has no free in-flight query slot")
            slot_index = self._free_slots.pop(0)
            target, _ = self._runtime.evaluate_torch(
                slot_index,
                views,
                lights,
                uv,
                gradients,
            )
            target = target[:, None, :]
        elif target_estimator == "learned-sampler":
            target = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=self.device)
        else:
            raise ValueError("MDL route target_estimator is unsupported")
        lease = _LiveBatchLease(self, request.name, slot_index)
        self._active_leases[request.name] = lease
        request_index = self._request_count.get(request.name, 0)
        self._request_count[request.name] = request_index + 1
        source_features = torch.as_tensor(
            self._feature_values, dtype=torch.float32, device=self.device
        ).expand(batch_size, -1)
        seeds = torch.randint(
            0,
            np.iinfo(np.int32).max,
            (batch_size, 1),
            dtype=torch.int64,
            device=self.device,
            generator=generator,
        )
        return TrainingBatch(
            self.state.family_id,
            self.source_state_ids * batch_size,
            "rgb-bsdf-times-absolute-shading-normal-light-cosine",
            {
                "source_index": torch.zeros(batch_size, dtype=torch.int64, device=self.device),
                "wo": views,
                "wi": lights[:, None, :],
                "target": target,
                "solid_angle_weight": solid_angle_weight[:, None],
                "reference_pdf": proposal_pdf[:, None],
                "sample_count": torch.ones((batch_size, 1), dtype=torch.int64, device=self.device),
                "rng_seed": seeds,
                "query_role": torch.full(
                    (batch_size,), request.query_role, dtype=torch.int64, device=self.device
                ),
                "uv": uv,
                "uv_dx": gradients[:, :2],
                "uv_dy": gradients[:, 2:],
                "mip_level": mip_level,
                "native_features": source_features,
                "sample_u": torch.rand(
                    (batch_size, 2),
                    dtype=torch.float32,
                    device=self.device,
                    generator=generator,
                ),
            },
            {
                "producer": "mdl-live-reference",
                "data_source_identity": self.identity,
                "host_readback": False,
                "synchronization": "wait_for_falcor",
                "route_name": request.name,
                "request_index": request_index,
                "global_step": request.global_step,
                "native_feature_layout_id": self._layout_id,
                "source_adaptation_id": "mdl-class-compiled-parameters-and-uv@1",
                "texture_filtering": "explicit-lod0",
                "uv_derivatives_consumed": False,
                "direction_proposal": proposal,
                "target_estimator": target_estimator,
                "gpu_request_seed": request_seed,
                "gpu_online_sampling": True,
            },
            lease,
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        return self._feature_pyramid

    def state_dict(self) -> Mapping[str, Any]:
        if self._active_leases:
            raise RuntimeError("cannot checkpoint MDL source with active batch leases")
        return {
            "rng_by_route": {
                name: generator.bit_generator.state
                for name, generator in self._rng_by_route.items()
            },
            "request_count": dict(self._request_count),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._active_leases or set(state) != {"rng_by_route", "request_count"}:
            raise ValueError("MDL batch source state is invalid")
        generators = {}
        for name, value in dict(state["rng_by_route"]).items():
            generator = np.random.default_rng()
            generator.bit_generator.state = value
            generators[str(name)] = generator
        self._rng_by_route = generators
        self._request_count = {
            str(name): int(value) for name, value in dict(state["request_count"]).items()
        }

    def close(self) -> None:
        if self._active_leases:
            raise RuntimeError("cannot close MDL source with active batch leases")
        self._runtime.close()
        self.provider.close()
