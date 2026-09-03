from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Mapping

import torch

from ncls.core.identity import sha256_json
from ncls.core.source import create_source_family
from ncls.learning.batches import (
    AssetTileBatch,
    EvaluatorBatch,
    MethodSamplerBatch,
    OnlineTrainingBatch,
    TrainingConditioning,
    TrainingRouteRequest,
)
from ncls.learning.method import MethodDefinition
from ncls.learning.source_adaptation import NativeAssetCollection
from ncls.learning.source_adapters import create_method_source_adapter
from ncls.learning.source_states import expand_source_states
from ncls.learning.training.config import TrainingConfig
from ncls.references.programs import get_reference_program_for_source
from ncls.references.backend import ReferenceBackendCapability, create_reference_backend
from ncls.references.plan import (
    ReferenceExecutionGroup,
    compile_single_program_plan,
)
from ncls.references.query import ReferenceBackendSession, ScatteringQuery


def _group_block_sequence(
    groups: tuple[ReferenceExecutionGroup, ...], plan_identity: str
) -> tuple[ReferenceExecutionGroup, ...]:
    """Return a deterministic record-weighted cycle with full-group prefix coverage."""

    prefix = list(groups)
    remainder = [
        (group, ordinal)
        for group in groups
        for ordinal in range(1, len(group.records))
    ]
    remainder.sort(
        key=lambda item: hashlib.sha256(
            f"{plan_identity}:{item[0].group_id}:{item[1]}".encode("ascii")
        ).digest()
    )
    return tuple(prefix + [group for group, _ in remainder])


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
        backend: ReferenceBackendCapability | None = None,
    ) -> None:
        # DDP keeps a shared CUDA visibility list; each rank consumes its
        # remapped local device while the config identity remains identical.
        local_rank = os.environ.get("NCLS_DDP_LOCAL_RANK")
        self.ddp_rank = int(os.environ.get("RANK", "0"))
        self.ddp_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        raw_gpu_list = os.environ.get("NCLS_DDP_GPU_LIST", "")
        self.ddp_gpu_indices = tuple(
            int(value) for value in raw_gpu_list.split(",") if value != ""
        )
        if self.ddp_world_size > 1 and len(self.ddp_gpu_indices) != self.ddp_world_size:
            raise RuntimeError("DDP GPU list length must match WORLD_SIZE")
        if local_rank is not None:
            try:
                rank = int(local_rank)
            except ValueError as error:
                raise RuntimeError("NCLS_DDP_LOCAL_RANK must be an integer") from error
            if rank < 0:
                raise RuntimeError("NCLS_DDP_LOCAL_RANK must be nonnegative")
            device_index = int(os.environ.get("NCLS_DDP_DEVICE_INDEX", str(rank)))
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = torch.device(config.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("online training producer requires a CUDA device")
        family = create_source_family(str(config.source["family_id"]))
        base_snapshots = tuple(
            family.load_snapshot(material["locator"])
            for material in config.source["materials"]
        )
        expanded_states = expand_source_states(
            family,
            base_snapshots,
            config.online_query.get("typed_state_recipe"),
        )
        snapshots = expanded_states.snapshots
        for snapshot in snapshots:
            family.validate_snapshot(snapshot)
            definition.descriptor.adaptation_contract(snapshot)
        reference = get_reference_program_for_source(
            family.descriptor.family_id,
            family.descriptor.source_contract_version,
            source_descriptor=family.descriptor,
        )
        plan_recipe = {
            **config.online_query,
            "typed_state_pool_identity": expanded_states.identity,
            "typed_state_recipe_schema": expanded_states.recipe_schema,
        }
        self.plan = compile_single_program_plan(
            reference,
            snapshots,
            query_recipe=plan_recipe,
        )
        capacity = max(
            (
                route.batch_size * route.direction_count
                for route in config.all_routes
                if route.kind != "asset-tile"
            ),
            default=1,
        )
        self.backend = backend or create_reference_backend()
        schedule = config.online_query.get("group_schedule")
        self._group_schedule_recipe = (
            str(schedule.get("recipe")) if isinstance(schedule, Mapping) else None
        )
        self._group_block_steps = (
            int(schedule.get("block_steps", 0)) if isinstance(schedule, Mapping) else 0
        )
        self._group_validation_offset_blocks = (
            int(schedule.get("validation_offset_blocks", 0))
            if isinstance(schedule, Mapping)
            else 0
        )
        if self._group_schedule_recipe is None:
            self._group_sequence = self.plan.groups
        elif (
            self._group_schedule_recipe == "group-block-balanced@1"
            and schedule.get("weight") == "record-count"
            and self._group_block_steps >= 1
            and self._group_validation_offset_blocks >= 1
        ):
            self._group_sequence = _group_block_sequence(
                self.plan.groups, self.plan.identity
            )
        else:
            raise ValueError("online query group_schedule is unsupported")
        self.session: ReferenceBackendSession = self.backend.open(
            self.plan,
            query_capacity=capacity,
            device=self.device,
            requested_operations=("evaluate",),
        )
        if self.session.snapshots != snapshots:
            raise ValueError("online producer backend session disagrees with source locators")
        self.snapshots = snapshots
        self.base_source_snapshot_ids = expanded_states.base_snapshot_ids
        self.typed_state_pool_identity = expanded_states.identity
        self.typed_state_recipe_schema = expanded_states.recipe_schema
        self.source_snapshot_ids = tuple(snapshot.snapshot_id for snapshot in snapshots)
        self.source_contracts = (
            {
                "family_id": family.descriptor.family_id,
                "source_contract_version": family.descriptor.source_contract_version,
            },
        )
        self.reference_program_identity = self.session.reference_program_identity
        self.reference_execution_plan_identity = self.plan.identity
        self.reference_backend_identity = self.session.backend_identity
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
                "reference_execution_plan_identity": self.reference_execution_plan_identity,
                "reference_backend_identity": self.reference_backend_identity,
                "adapter_identity": self.adapter.identity,
                "online_query": plan_recipe,
                "routes": [route.to_dict() for route in config.all_routes],
                "seed": config.seed,
                "partition": {
                    "world_size": self.ddp_world_size,
                    "gpu_indices": list(self.ddp_gpu_indices),
                    "recipe": "rank-strided-route-seed@1",
                },
            }
        )
        self._generators: dict[str, torch.Generator] = {}
        self._request_count: dict[str, int] = {}
        self._group_cursor: dict[str, int] = {}
        self._asset_tile_cursor: dict[str, int] = {}
        self._closed = False

    def _generator(self, request: TrainingRouteRequest) -> torch.Generator:
        generator = self._generators.get(request.name)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(request.seed)
            self._generators[request.name] = generator
        return generator

    def _select_group(self, request: TrainingRouteRequest) -> ReferenceExecutionGroup:
        if self._group_schedule_recipe == "group-block-balanced@1":
            block = request.global_step // self._group_block_steps
            if bool(request.options.get("validation", False)):
                block += self._group_validation_offset_blocks
            sequence_index = block * self.ddp_world_size + self.ddp_rank
            return self._group_sequence[sequence_index % len(self._group_sequence)]
        cursor = self._group_cursor.get(request.name, 0)
        self._group_cursor[request.name] = cursor + 1
        return self.plan.groups[cursor % len(self.plan.groups)]

    def _conditioning(
        self, request: TrainingRouteRequest, group: ReferenceExecutionGroup
    ) -> tuple[TrainingConditioning, torch.Generator, torch.Tensor | None]:
        generator = self._generator(request)
        local_source_index = torch.randint(
            0,
            len(group.records),
            (request.batch_size,),
            generator=generator,
            device=self.device,
            dtype=torch.int64,
        )
        group_indices = torch.tensor(
            group.global_source_indices,
            dtype=torch.int64,
            device=self.device,
        )
        source_index = group_indices.index_select(0, local_source_index)
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
        adapted, provenance = self.adapter.sample_tensors(
            source_index, generator, request.options
        )
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
                "reference_backend_identity": self.reference_backend_identity,
                "reference_execution_plan_identity": self.reference_execution_plan_identity,
                "reference_execution_group_id": group.group_id,
                "group_schedule_recipe": self._group_schedule_recipe or "route-round-robin@1",
                "group_schedule_block_steps": self._group_block_steps,
                "group_schedule_validation_offset_blocks": (
                    self._group_validation_offset_blocks
                ),
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
            str(conditioning.provenance["reference_execution_group_id"]),
            uv=tensors.get("uv"),
            uv_dx=tensors.get("uv_dx"),
            uv_dy=tensors.get("uv_dy"),
        )

    def next_batch(self, request: TrainingRouteRequest) -> OnlineTrainingBatch:
        if self._closed:
            raise RuntimeError("online training producer is closed")
        if request.kind == "asset-tile":
            return self._asset_tile_batch(request)
        group = self._select_group(request)
        if request.kind == "method-sampler":
            conditioning, generator, _ = self._conditioning(request, group)
            sample_u = torch.rand(
                (request.batch_size, 2), generator=generator, device=self.device
            )
            return MethodSamplerBatch(conditioning, sample_u)
        return self._evaluator_batch(request, group)

    def _asset_tile_batch(self, request: TrainingRouteRequest) -> AssetTileBatch:
        assets = self.adapter.native_assets()
        max_core_texels = int(request.options.get("max_core_texels", 65_536))
        halo = int(request.options.get("halo", 0))
        if max_core_texels < 1 or halo < 0:
            raise ValueError("asset-tile route budget and halo are invalid")
        selected_asset_indices = tuple(
            int(value)
            for value in request.options.get(
                "asset_indices", range(len(assets.descriptors))
            )
        )
        if (
            not selected_asset_indices
            or len(set(selected_asset_indices)) != len(selected_asset_indices)
            or any(
                value < 0 or value >= len(assets.descriptors)
                for value in selected_asset_indices
            )
        ):
            raise ValueError("asset-tile route asset_indices are empty, duplicate or out of range")

        def one_cycle():
            streams = []
            for asset_index in selected_asset_indices:
                descriptor = assets.descriptors[asset_index]
                for domain in descriptor.domains:
                    streams.append(
                        iter(
                            assets.iter_tile_requests(
                                asset_index,
                                domain.domain_id,
                                max_core_texels,
                                halo,
                            )
                        )
                    )
            # Deterministic round-robin prevents one large 4K domain from
            # starving other assets/roles/mips for thousands of steps.
            active = streams
            while active:
                following = []
                for stream in active:
                    try:
                        yield next(stream)
                    except StopIteration:
                        continue
                    following.append(stream)
                active = following

        tile_counts: list[int] = []
        for asset_index in selected_asset_indices:
            descriptor = assets.descriptors[asset_index]
            for domain in descriptor.domains:
                for height, width in domain.level_shapes:
                    tile_width = min(width, max_core_texels)
                    tile_height = max(1, min(height, max_core_texels // tile_width))
                    tile_counts.append(
                        ((height + tile_height - 1) // tile_height)
                        * ((width + tile_width - 1) // tile_width)
                    )
        cycle_count = sum(tile_counts)
        if cycle_count < 1:
            raise RuntimeError("native asset collection produced no tiles")
        cursor = self._asset_tile_cursor.get(request.name, 0)
        skip = cursor % cycle_count
        iterator = one_cycle()
        for _ in range(skip):
            next(iterator)
        selected_requests = []
        while len(selected_requests) < request.batch_size:
            try:
                selected_requests.append(next(iterator))
            except StopIteration:
                iterator = one_cycle()
        selected = []
        try:
            for tile_request in selected_requests:
                selected.append(assets.acquire_tile(tile_request, self.device))
        except BaseException:
            for tile in reversed(selected):
                tile.release()
            raise
        self._asset_tile_cursor[request.name] = cursor + request.batch_size
        request_index = self._request_count.get(request.name, 0)
        self._request_count[request.name] = request_index + 1
        return AssetTileBatch(
            assets.descriptors,
            tuple(selected),
            {
                "producer": "generic-online",
                "route_name": request.name,
                "route_kind": request.kind,
                "request_index": request_index,
                "global_step": request.global_step,
                "query_stream_identity": self.query_stream_identity,
                "native_asset_collection_identity": assets.collection_id,
                "max_core_texels": max_core_texels,
                "halo": halo,
                "asset_indices": list(selected_asset_indices),
            },
        )

    def _evaluator_batch(
        self, request: TrainingRouteRequest, group: ReferenceExecutionGroup
    ) -> EvaluatorBatch:
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
        footprint_samples = int(
            request.options.get(
                "footprint_samples",
                self.config.online_query.get("footprint_samples", 1),
            )
        )
        source_execution_mode = str(
            request.options.get(
                "source_execution_mode",
                self.config.online_query.get("source_execution_mode", "authoritative@1"),
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
                candidate_request, group
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
            result = self.session.evaluate(
                self._query(conditioning),
                wi,
                seeds,
                evaluation_samples=evaluation_samples,
                footprint_samples=footprint_samples,
                source_execution_mode=source_execution_mode,
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
            "evaluation_samples": evaluation_samples,
            "footprint_samples": footprint_samples,
            "source_execution_mode": source_execution_mode,
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

    @property
    def native_asset_collection_identity(self) -> str:
        return self.adapter.native_assets().collection_id

    def native_assets(self) -> NativeAssetCollection:
        return self.adapter.native_assets()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "query_stream_identity": self.query_stream_identity,
            "typed_state_pool_identity": self.typed_state_pool_identity,
            "generator_states": {
                name: generator.get_state()
                for name, generator in self._generators.items()
            },
            "request_count": dict(self._request_count),
            "group_cursor": dict(self._group_cursor),
            "asset_tile_cursor": dict(self._asset_tile_cursor),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {
            "query_stream_identity",
            "typed_state_pool_identity",
            "generator_states",
            "request_count",
            "group_cursor",
            "asset_tile_cursor",
        }:
            raise ValueError("online query stream state fields are invalid")
        if state["query_stream_identity"] != self.query_stream_identity:
            raise ValueError("online query stream state identity mismatch")
        if state["typed_state_pool_identity"] != self.typed_state_pool_identity:
            raise ValueError("online typed-state pool identity mismatch")
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
        self._group_cursor = {
            str(name): int(value)
            for name, value in dict(state["group_cursor"]).items()
        }
        self._asset_tile_cursor = {
            str(name): int(value)
            for name, value in dict(state["asset_tile_cursor"]).items()
        }

    def end_iteration(self) -> None:
        self.session.end_iteration()

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, float]:
        return self.session.profile_snapshot(reset=reset)

    def close(self) -> None:
        if self._closed:
            return
        self.session.close()
        self._closed = True


__all__ = ["OnlineTrainingProducer"]
