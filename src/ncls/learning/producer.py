from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from ncls.core.identity import sha256_json
from ncls.core.source import create_source_family
from ncls.learning.batches import (
    EvaluatorBatch,
    MethodSamplerBatch,
    OnlineTrainingBatch,
    TrainingConditioning,
    TrainingRouteRequest,
)
from ncls.learning.method import MethodDefinition
from ncls.learning.source_adaptation import NativeFeaturePyramid
from ncls.learning.source_adapters import create_method_source_adapter
from ncls.learning.training.config import TrainingConfig
from ncls.references.programs import get_reference_program_for_source
from ncls.references.query import ReferenceQueryDispatcher, ScatteringQuery


def _uniform_hemisphere(
    count: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    z = torch.rand(count, generator=generator, device=device)
    phi = torch.rand(count, generator=generator, device=device) * (2.0 * math.pi)
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=1)


def _half_difference_directions(
    count: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    views = torch.empty((count, 3), dtype=torch.float32, device=device)
    lights = torch.empty_like(views)
    filled = 0
    while filled < count:
        candidates = max(1024, 2 * (count - filled))
        half = _uniform_hemisphere(candidates, generator, device)
        difference = _uniform_hemisphere(candidates, generator, device)
        cosine_theta = half[:, 2]
        sine_theta = torch.sqrt(torch.clamp(1.0 - cosine_theta * cosine_theta, min=0.0))
        phi = torch.atan2(half[:, 1], half[:, 0])
        cosine_phi = torch.cos(phi)
        sine_phi = torch.sin(phi)

        def rotate(local: torch.Tensor) -> torch.Tensor:
            x = cosine_theta * local[:, 0] + sine_theta * local[:, 2]
            y = local[:, 1]
            z = -sine_theta * local[:, 0] + cosine_theta * local[:, 2]
            return torch.stack(
                (cosine_phi * x - sine_phi * y, sine_phi * x + cosine_phi * y, z),
                dim=1,
            )

        first = rotate(difference)
        reflected = difference.clone()
        reflected[:, :2] *= -1.0
        second = rotate(reflected)
        selected = torch.nonzero(
            (first[:, 2] > 0.0) & (second[:, 2] > 0.0), as_tuple=False
        )[: count - filled, 0]
        take = int(selected.shape[0])
        views[filled : filled + take] = first[selected]
        lights[filled : filled + take] = second[selected]
        filled += take
    return views, lights


class OnlineTrainingProducer:
    """唯一 online producer；source 差异只存在于 registry-selected adapters。"""

    def __init__(
        self,
        definition: MethodDefinition,
        config: TrainingConfig,
        *,
        dispatcher: ReferenceQueryDispatcher | None = None,
    ) -> None:
        self.device = torch.device(config.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("online training producer requires a CUDA device")
        family = create_source_family(str(config.source["family_id"]))
        snapshots = tuple(
            family.load_snapshot(material["locator"])
            for material in config.source["materials"]
        )
        for snapshot in snapshots:
            family.validate_snapshot(snapshot)
            definition.descriptor.adaptation_contract(snapshot)
        reference = get_reference_program_for_source(
            family.descriptor.family_id,
            family.descriptor.source_contract_version,
            source_descriptor=family.descriptor,
        )
        capacity = max(
            route.batch_size * route.direction_count for route in config.routes
        )
        self.dispatcher = dispatcher or ReferenceQueryDispatcher(
            reference,
            snapshots,
            query_capacity=capacity,
            device=self.device,
        )
        if self.dispatcher.snapshots != snapshots:
            raise ValueError("online producer dispatcher snapshots disagree with source locators")
        self.snapshots = snapshots
        self.source_snapshot_ids = tuple(snapshot.snapshot_id for snapshot in snapshots)
        self.source_contracts = (
            {
                "family_id": family.descriptor.family_id,
                "source_contract_version": family.descriptor.source_contract_version,
            },
        )
        self.reference_program_identity = self.dispatcher.reference_program_identity
        self.adapter = create_method_source_adapter(
            definition.descriptor.method_key, snapshots, self.device
        )
        self.config = config
        self.query_stream_identity = sha256_json(
            {
                "schema": "ncls.online-query-stream@1",
                "source": dict(config.source),
                "source_snapshot_ids": list(self.source_snapshot_ids),
                "reference_program_identity": self.reference_program_identity,
                "adapter_identity": self.adapter.identity,
                "online_query": dict(config.online_query),
                "routes": [route.to_dict() for route in config.routes],
                "seed": config.seed,
            }
        )
        self._generators: dict[str, torch.Generator] = {}
        self._request_count: dict[str, int] = {}
        self._closed = False

    def _generator(self, request: TrainingRouteRequest) -> torch.Generator:
        generator = self._generators.get(request.name)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(request.seed)
            self._generators[request.name] = generator
        return generator

    def _conditioning(
        self, request: TrainingRouteRequest
    ) -> tuple[TrainingConditioning, torch.Generator, torch.Tensor | None]:
        generator = self._generator(request)
        source_index = torch.randint(
            0,
            len(self.snapshots),
            (request.batch_size,),
            generator=generator,
            device=self.device,
            dtype=torch.int64,
        )
        if request.kind == "reference-evaluator":
            proposal = request.options.get("direction_proposal")
            if proposal != "uniform-half-difference@1" or request.direction_count != 1:
                raise ValueError(
                    "reference-evaluator route requires one half/difference direction"
                )
            wo, evaluator_wi = _half_difference_directions(
                request.batch_size, generator, self.device
            )
        else:
            proposal = request.options.get("direction_proposal")
            if proposal != "uniform-hemisphere-conditioning@1" or request.direction_count != 1:
                raise ValueError(
                    "method-sampler route requires hemisphere conditioning and direction_count=1"
                )
            wo = _uniform_hemisphere(request.batch_size, generator, self.device)
            evaluator_wi = None
        adapted, provenance = self.adapter.sample_tensors(source_index, generator)
        request_index = self._request_count.get(request.name, 0)
        self._request_count[request.name] = request_index + 1
        conditioning = TrainingConditioning(
            self.snapshots[0].family_id,
            self.source_snapshot_ids,
            {"source_index": source_index, "wo": wo, **adapted},
            {
                "producer": "generic-online",
                "host_response_readback": False,
                "route_name": request.name,
                "route_kind": request.kind,
                "request_index": request_index,
                "global_step": request.global_step,
                "query_stream_identity": self.query_stream_identity,
                "reference_program_identity": self.reference_program_identity,
                "source_adapter_identity": self.adapter.identity,
                **provenance,
            },
        )
        return conditioning, generator, evaluator_wi

    @staticmethod
    def _query(conditioning: TrainingConditioning) -> ScatteringQuery:
        tensors = conditioning.tensors
        return ScatteringQuery(
            tensors["source_index"],
            tensors["wo"],
            uv=tensors.get("uv"),
            uv_dx=tensors.get("uv_dx"),
            uv_dy=tensors.get("uv_dy"),
        )

    def next_batch(self, request: TrainingRouteRequest) -> OnlineTrainingBatch:
        if self._closed:
            raise RuntimeError("online training producer is closed")
        if request.kind == "method-sampler":
            conditioning, generator, _ = self._conditioning(request)
            sample_u = torch.rand(
                (request.batch_size, 2), generator=generator, device=self.device
            )
            return MethodSamplerBatch(conditioning, sample_u)
        return self._evaluator_batch(request)

    def _evaluator_batch(self, request: TrainingRouteRequest) -> EvaluatorBatch:
        """在 GPU 上压实有效 reference 查询，材质局部 horizon 不构成批次失败。"""

        accepted_conditioning: dict[str, list[torch.Tensor]] = {}
        accepted_wi: list[torch.Tensor] = []
        accepted_f: list[torch.Tensor] = []
        first_conditioning: TrainingConditioning | None = None
        candidate_count = 0
        accepted_count = 0
        rejection_rounds = 0
        maximum_rounds = int(request.options.get("maximum_rejection_rounds", 64))
        if maximum_rounds < 1:
            raise ValueError("maximum_rejection_rounds must be positive")
        evaluation_samples = int(
            request.options.get(
                "evaluation_samples",
                self.config.online_query.get("evaluation_samples", 1),
            )
        )
        while accepted_count < request.batch_size:
            if rejection_rounds >= maximum_rounds:
                raise RuntimeError(
                    "reference evaluator could not fill a valid online batch within "
                    f"{maximum_rounds} rejection rounds"
                )
            remaining = request.batch_size - accepted_count
            candidate_request = TrainingRouteRequest(
                request.name,
                request.kind,
                remaining,
                request.direction_count,
                request.global_step,
                request.seed,
                request.options,
            )
            conditioning, generator, evaluator_wi = self._conditioning(
                candidate_request
            )
            if first_conditioning is None:
                first_conditioning = conditioning
            if evaluator_wi is None:
                raise AssertionError("reference-evaluator route did not create wi")
            wi = evaluator_wi[:, None, :]
            seeds = torch.randint(
                0,
                2**31 - 1,
                (remaining, request.direction_count),
                generator=generator,
                device=self.device,
                dtype=torch.int64,
            )
            result = self.dispatcher.evaluate(
                self._query(conditioning),
                wi,
                seeds,
                evaluation_samples=evaluation_samples,
            )
            try:
                selected = torch.nonzero(
                    result.valid.all(dim=1), as_tuple=False
                ).flatten()
                take = int(selected.shape[0])
                if take:
                    for name, value in conditioning.tensors.items():
                        accepted_conditioning.setdefault(name, []).append(
                            value.index_select(0, selected)
                        )
                    accepted_wi.append(wi.index_select(0, selected))
                    accepted_f.append(result.f.index_select(0, selected))
                    accepted_count += take
            finally:
                result.lease.release()
            candidate_count += remaining
            rejection_rounds += 1
        if first_conditioning is None:
            raise AssertionError("reference-evaluator route did not create conditioning")
        tensors = {
            name: torch.cat(values, dim=0)[: request.batch_size]
            for name, values in accepted_conditioning.items()
        }
        provenance = {
            **first_conditioning.provenance,
            "candidate_count": candidate_count,
            "rejected_count": candidate_count - request.batch_size,
            "rejection_rounds": rejection_rounds,
        }
        compacted = TrainingConditioning(
            first_conditioning.source_family_id,
            first_conditioning.source_snapshot_ids,
            tensors,
            provenance,
        )
        return EvaluatorBatch(
            compacted,
            torch.cat(accepted_wi, dim=0)[: request.batch_size],
            torch.cat(accepted_f, dim=0)[: request.batch_size],
        )

    def materialization_features(self) -> NativeFeaturePyramid:
        return self.adapter.materialization_features()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "query_stream_identity": self.query_stream_identity,
            "generator_states": {
                name: generator.get_state()
                for name, generator in self._generators.items()
            },
            "request_count": dict(self._request_count),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {
            "query_stream_identity",
            "generator_states",
            "request_count",
        }:
            raise ValueError("online query stream state fields are invalid")
        if state["query_stream_identity"] != self.query_stream_identity:
            raise ValueError("online query stream state identity mismatch")
        generators: dict[str, torch.Generator] = {}
        for name, generator_state in dict(state["generator_states"]).items():
            if not isinstance(generator_state, torch.Tensor):
                raise ValueError("online query generator state must be a tensor")
            generator = torch.Generator(device=self.device)
            generator.set_state(generator_state.cpu())
            generators[str(name)] = generator
        self._generators = generators
        self._request_count = {
            str(name): int(value)
            for name, value in dict(state["request_count"]).items()
        }

    def end_iteration(self) -> None:
        self.dispatcher.end_iteration()

    def close(self) -> None:
        if self._closed:
            return
        self.dispatcher.close()
        self._closed = True


__all__ = ["OnlineTrainingProducer"]
