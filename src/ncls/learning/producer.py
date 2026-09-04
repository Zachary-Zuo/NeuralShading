from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, cast

import torch

from ncls.core.identity import sha256_json
from ncls.core.source import create_source_family
from ncls.data import (
    DataExecutionPlan,
    LogicalReferenceRequest,
    OnlineStepRequest,
    PipelineTrace,
    ReferenceScheduler,
)
from ncls.learning.batches import (
    AssetTileBatch,
    EvaluatorBatch,
    MethodSamplerBatch,
    OnlineTrainingBatch,
    TrainingConditioning,
    TrainingRouteRequest,
)
from ncls.learning.methods.contracts import MethodPlugin
from ncls.learning.source_adaptation import NativeAssetCollection
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


@dataclass(frozen=True)
class _EvaluatorLogicalRequest:
    request: TrainingRouteRequest
    group: ReferenceExecutionGroup
    generator: torch.Generator
    request_index: int
    dispatch_identity: str


class OnlineTrainingProducer:
    """唯一 online producer；source 差异只存在于 registry-selected adapters。"""

    def __init__(
        self,
        plugin: MethodPlugin,
        config: TrainingConfig,
        *,
        backend: ReferenceBackendCapability | None = None,
        execution_context: Any,
        data_execution_plan: DataExecutionPlan,
    ) -> None:
        # Platform/GPU mapping belongs to the launcher. The data source only
        # consumes the already validated rank-local execution context.
        self.ddp_rank = int(execution_context.rank)
        self.ddp_world_size = int(execution_context.world_size)
        self.ddp_gpu_indices = tuple(execution_context.topology.devices)
        self.device = torch.device(execution_context.torch_device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("online training producer requires a CUDA device")
        partition = data_execution_plan.partition
        if (
            partition.rank != self.ddp_rank
            or partition.world_size != self.ddp_world_size
        ):
            raise ValueError(
                "data execution plan partition disagrees with execution context"
            )
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
            plugin.descriptor.adaptation_contract(snapshot)
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
        ) * data_execution_plan.reference_batch_steps
        self.backend = backend or create_reference_backend()
        reference_inflight = data_execution_plan.reference_inflight
        reference_slots = reference_inflight + 1
        maximum_slots = self.backend.descriptor.concurrency.maximum_safe_slots
        if reference_slots > maximum_slots:
            raise ValueError(
                "data execution reference_inflight plus the current session slot "
                f"exceeds backend capability: {reference_inflight}+1 > {maximum_slots}"
            )
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
            slot_count=reference_slots,
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
        self.pipeline_trace = PipelineTrace()
        self.adapter = plugin.data.create_source_adapter(snapshots, self.device)
        self.adapter.configure_data_execution(
            data_execution_plan, self.pipeline_trace
        )
        self.config = config
        self.data_execution_plan_identity = data_execution_plan.identity
        query_stream_manifest = {
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
                    "recipe": "rank-strided-logical-request-seed@1",
                },
            }
        query_stream_manifest["data_execution_plan_identity"] = (
            self.data_execution_plan_identity
        )
        self.query_stream_identity = sha256_json(query_stream_manifest)
        self._request_count: dict[str, int] = {}
        self._group_cursor: dict[str, int] = {}
        self._asset_tile_cursor: dict[str, int] = {}
        self._reference_logical_id = 0
        self._reference_scheduler = ReferenceScheduler[
            _EvaluatorLogicalRequest, EvaluatorBatch
        ](
            self._dispatch_evaluator_requests,
            capability=self.backend.descriptor.concurrency,
            batch_steps=data_execution_plan.reference_batch_steps,
            ready_capacity=data_execution_plan.ready_batches,
            maximum_inflight=data_execution_plan.reference_inflight,
            trace=self.pipeline_trace,
        )
        self._closed = False

    def _reserve_request(
        self, request: TrainingRouteRequest
    ) -> tuple[int, torch.Generator]:
        request_index = self._request_count.get(request.name, 0)
        self._request_count[request.name] = request_index + 1
        digest = hashlib.sha256(
            (
                f"ncls.logical-request-seed@1:{request.name}:"
                f"{request.seed}:{request_index}"
            ).encode("utf-8")
        ).digest()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
        return request_index, generator

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
        self,
        request: TrainingRouteRequest,
        group: ReferenceExecutionGroup,
        generator: torch.Generator,
        request_index: int,
    ) -> tuple[TrainingConditioning, torch.Tensor | None]:
        execution_source_indices = self.adapter.execution_source_indices(
            group.global_source_indices, request
        )
        local_source_index = torch.randint(
            0,
            len(execution_source_indices),
            (request.batch_size,),
            generator=generator,
            device=self.device,
            dtype=torch.int64,
        )
        group_indices = torch.tensor(
            execution_source_indices,
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
            source_index,
            generator,
            request.options,
            execution_source_indices=execution_source_indices,
        )
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
                "execution_source_cohort": list(execution_source_indices),
                "group_schedule_recipe": self._group_schedule_recipe or "route-round-robin@1",
                "group_schedule_block_steps": self._group_block_steps,
                "group_schedule_validation_offset_blocks": (
                    self._group_validation_offset_blocks
                ),
                "source_adapter_identity": self.adapter.identity,
                **provenance,
            },
        )
        return conditioning, evaluator_wi

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

    def _produce_route(self, request: TrainingRouteRequest) -> OnlineTrainingBatch:
        if self._closed:
            raise RuntimeError("online training producer is closed")
        if request.kind == "asset-tile":
            return self._asset_tile_batch(request)
        group = self._select_group(request)
        if request.kind == "method-sampler":
            request_index, generator = self._reserve_request(request)
            conditioning, _ = self._conditioning(
                request, group, generator, request_index
            )
            sample_u = torch.rand(
                (request.batch_size, 2), generator=generator, device=self.device
            )
            return MethodSamplerBatch(conditioning, sample_u)
        raise RuntimeError("reference-evaluator routes require the packed dispatcher")

    def prefetch_steps(self, requests: tuple[OnlineStepRequest, ...]) -> None:
        if self._closed:
            raise RuntimeError("online training producer is closed")
        # Only the block schedule can be inspected without advancing a route
        # cursor. Other schedules remain correct and simply do no early host work.
        if self._group_schedule_recipe != "group-block-balanced@1":
            return
        visited: set[tuple[str, int]] = set()
        for step in requests:
            for route in step.routes.values():
                if not isinstance(route, TrainingRouteRequest):
                    raise TypeError("online producer requires TrainingRouteRequest routes")
                if route.kind == "asset-tile":
                    continue
                group = self._select_group(route)
                identity = (group.group_id, route.global_step)
                if identity in visited:
                    continue
                self.adapter.prefetch_host(group.global_source_indices, route)
                visited.add(identity)

    def produce_steps(
        self, requests: tuple[OnlineStepRequest, ...]
    ) -> tuple[Mapping[str, OnlineTrainingBatch], ...]:
        if self._closed:
            raise RuntimeError("online training producer is closed")
        produced: list[dict[str, OnlineTrainingBatch]] = [
            {} for _ in requests
        ]
        evaluator_slots: dict[int, tuple[int, str]] = {}
        try:
            for step_index, step in enumerate(requests):
                for slot, route in step.routes.items():
                    if not isinstance(route, TrainingRouteRequest):
                        raise TypeError("online producer requires TrainingRouteRequest routes")
                    if route.kind != "reference-evaluator":
                        produced[step_index][slot] = self._produce_route(route)
                        continue
                    group = self._select_group(route)
                    request_index, generator = self._reserve_request(route)
                    dispatch_identity = sha256_json(
                        {
                            "schema": "ncls.reference-packed-dispatch@1",
                            "execution_group_id": group.group_id,
                            "direction_count": route.direction_count,
                            "options": dict(route.options),
                            "online_query": dict(self.config.online_query),
                        }
                    )
                    logical_id = self._reference_logical_id
                    self._reference_logical_id += 1
                    self._reference_scheduler.submit(
                        LogicalReferenceRequest(
                            logical_id,
                            dispatch_identity,
                            _EvaluatorLogicalRequest(
                                route,
                                group,
                                generator,
                                request_index,
                                dispatch_identity,
                            ),
                            {
                                "step_logical_id": step.logical_id,
                                "route_slot": slot,
                                "reference_execution_group_id": group.group_id,
                            },
                        )
                    )
                    evaluator_slots[logical_id] = (step_index, slot)
            while evaluator_slots:
                scheduled = self._reference_scheduler.next_result()
                try:
                    step_index, slot = evaluator_slots.pop(scheduled.logical_id)
                    produced[step_index][slot] = scheduled.payload
                finally:
                    scheduled.release()
            self._reference_scheduler.assert_idle()
        except BaseException:
            try:
                self._reference_scheduler.discard_boundary()
            except BaseException:
                pass
            for batches in reversed(produced):
                for batch in reversed(tuple(batches.values())):
                    batch.release()
            raise
        return tuple(produced)

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

    def _dispatch_evaluator_requests(
        self,
        packed: tuple[
            LogicalReferenceRequest[_EvaluatorLogicalRequest], ...
        ],
    ) -> tuple[EvaluatorBatch, ...]:
        """Pack same-group logical requests while preserving per-request RNG."""

        if not packed:
            return ()
        payloads = tuple(item.payload for item in packed)
        if len({item.dispatch_identity for item in payloads}) != 1:
            raise RuntimeError("packed evaluator requests disagree on dispatch identity")
        states = [
            {
                "accepted_conditioning": {},
                "accepted_wi": [],
                "accepted_f": [],
                "first_conditioning": None,
                "candidate_count": 0,
                "accepted_count": 0,
                "rejection_rounds": 0,
            }
            for _ in payloads
        ]
        active = list(range(len(payloads)))
        while active:
            candidates: list[
                tuple[int, TrainingConditioning, torch.Tensor, torch.Tensor]
            ] = []
            for index in active:
                payload = payloads[index]
                request = payload.request
                state = states[index]
                maximum_rounds = int(
                    request.options.get("maximum_rejection_rounds", 64)
                )
                if maximum_rounds < 1:
                    raise ValueError("maximum_rejection_rounds must be positive")
                if int(state["rejection_rounds"]) >= maximum_rounds:
                    raise RuntimeError(
                        "reference evaluator could not fill a valid online batch within "
                        f"{maximum_rounds} rejection rounds"
                    )
                remaining = request.batch_size - int(state["accepted_count"])
                candidate_request = TrainingRouteRequest(
                    request.name,
                    request.kind,
                    remaining,
                    request.direction_count,
                    request.global_step,
                    request.seed,
                    request.options,
                )
                conditioning, evaluator_wi = self._conditioning(
                    candidate_request,
                    payload.group,
                    payload.generator,
                    payload.request_index,
                )
                if state["first_conditioning"] is None:
                    state["first_conditioning"] = conditioning
                if evaluator_wi is None:
                    raise AssertionError("reference-evaluator route did not create wi")
                wi = evaluator_wi[:, None, :]
                seeds = torch.randint(
                    0,
                    2**31 - 1,
                    (remaining, request.direction_count),
                    generator=payload.generator,
                    device=self.device,
                    dtype=torch.int64,
                )
                candidates.append((index, conditioning, wi, seeds))
            tensor_keys = tuple(candidates[0][1].tensors)
            if any(tuple(item[1].tensors) != tensor_keys for item in candidates):
                raise RuntimeError("packed evaluator conditioning fields disagree")
            combined_conditioning = TrainingConditioning(
                candidates[0][1].source_family_id,
                candidates[0][1].source_snapshot_ids,
                {
                    name: torch.cat(
                        [item[1].tensors[name] for item in candidates], dim=0
                    )
                    for name in tensor_keys
                },
                candidates[0][1].provenance,
            )
            combined_wi = torch.cat([item[2] for item in candidates], dim=0)
            combined_seeds = torch.cat([item[3] for item in candidates], dim=0)
            first_request = payloads[candidates[0][0]].request
            evaluation_samples = int(
                first_request.options.get(
                    "evaluation_samples",
                    self.config.online_query.get("evaluation_samples", 1),
                )
            )
            footprint_samples = int(
                first_request.options.get(
                    "footprint_samples",
                    self.config.online_query.get("footprint_samples", 1),
                )
            )
            source_execution_mode = str(
                first_request.options.get(
                    "source_execution_mode",
                    self.config.online_query.get(
                        "source_execution_mode", "authoritative@1"
                    ),
                )
            )
            result = self.session.evaluate(
                self._query(combined_conditioning),
                combined_wi,
                combined_seeds,
                evaluation_samples=evaluation_samples,
                footprint_samples=footprint_samples,
                source_execution_mode=source_execution_mode,
            )
            try:
                offset = 0
                following: list[int] = []
                for index, conditioning, wi, _ in candidates:
                    state = states[index]
                    count = conditioning.batch_size
                    local_valid = result.valid[offset : offset + count]
                    selected = torch.nonzero(
                        local_valid.all(dim=1), as_tuple=False
                    ).flatten()
                    take = int(selected.shape[0])
                    if take:
                        accepted_conditioning = state["accepted_conditioning"]
                        assert isinstance(accepted_conditioning, dict)
                        for name, value in conditioning.tensors.items():
                            accepted_conditioning.setdefault(name, []).append(
                                value.index_select(0, selected)
                            )
                        accepted_wi = state["accepted_wi"]
                        accepted_f = state["accepted_f"]
                        assert isinstance(accepted_wi, list)
                        assert isinstance(accepted_f, list)
                        accepted_wi.append(wi.index_select(0, selected))
                        accepted_f.append(
                            result.f[offset : offset + count].index_select(
                                0, selected
                            )
                        )
                        state["accepted_count"] = int(state["accepted_count"]) + take
                    state["candidate_count"] = int(state["candidate_count"]) + count
                    state["rejection_rounds"] = int(state["rejection_rounds"]) + 1
                    if int(state["accepted_count"]) < payloads[index].request.batch_size:
                        following.append(index)
                    offset += count
                active = following
            finally:
                result.lease.release()
        batches: list[EvaluatorBatch] = []
        for payload, state in zip(payloads, states, strict=True):
            request = payload.request
            first_conditioning = state["first_conditioning"]
            if not isinstance(first_conditioning, TrainingConditioning):
                raise AssertionError("reference-evaluator route did not create conditioning")
            accepted_conditioning = state["accepted_conditioning"]
            accepted_wi = state["accepted_wi"]
            accepted_f = state["accepted_f"]
            assert isinstance(accepted_conditioning, dict)
            assert isinstance(accepted_wi, list)
            assert isinstance(accepted_f, list)
            tensors = {
                name: torch.cat(values, dim=0)[: request.batch_size]
                for name, values in accepted_conditioning.items()
            }
            provenance = {
                **first_conditioning.provenance,
                "candidate_count": int(state["candidate_count"]),
                "rejected_count": int(state["candidate_count"]) - request.batch_size,
                "rejection_rounds": int(state["rejection_rounds"]),
                "evaluation_samples": evaluation_samples,
                "footprint_samples": footprint_samples,
                "source_execution_mode": source_execution_mode,
                "reference_dispatch_identity": payload.dispatch_identity,
                "reference_dispatch_logical_steps": len(packed),
            }
            compacted = TrainingConditioning(
                first_conditioning.source_family_id,
                first_conditioning.source_snapshot_ids,
                tensors,
                provenance,
            )
            batches.append(
                EvaluatorBatch(
                    compacted,
                    torch.cat(accepted_wi, dim=0)[: request.batch_size],
                    torch.cat(accepted_f, dim=0)[: request.batch_size],
                )
            )
        return tuple(batches)

    @property
    def native_asset_collection_identity(self) -> str:
        return self.adapter.native_assets().collection_id

    def native_assets(self) -> NativeAssetCollection:
        return self.adapter.native_assets()

    def state_dict(self) -> Mapping[str, Any]:
        self._reference_scheduler.assert_idle()
        return {
            "query_stream_identity": self.query_stream_identity,
            "typed_state_pool_identity": self.typed_state_pool_identity,
            "request_count": dict(self._request_count),
            "group_cursor": dict(self._group_cursor),
            "asset_tile_cursor": dict(self._asset_tile_cursor),
            "reference_logical_id": self._reference_logical_id,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._reference_scheduler.assert_idle()
        if set(state) != {
            "query_stream_identity",
            "typed_state_pool_identity",
            "request_count",
            "group_cursor",
            "asset_tile_cursor",
            "reference_logical_id",
        }:
            raise ValueError("online query stream state fields are invalid")
        if state["query_stream_identity"] != self.query_stream_identity:
            raise ValueError("online query stream state identity mismatch")
        if state["typed_state_pool_identity"] != self.typed_state_pool_identity:
            raise ValueError("online typed-state pool identity mismatch")
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
        self._reference_logical_id = int(state["reference_logical_id"])
        if self._reference_logical_id < 0:
            raise ValueError("online reference logical cursor is invalid")

    def end_iteration(self) -> None:
        self._reference_scheduler.assert_idle()
        self.session.end_iteration()

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, float]:
        result = dict(self.session.profile_snapshot(reset=reset))
        data = self.pipeline_trace.snapshot(reset=reset).to_dict()
        for category, values in data.items():
            for name, value in values.items():
                result[f"data/{category}/{name}"] = float(value)
        return result

    def close(self) -> None:
        if self._closed:
            return
        scheduler = self._reference_scheduler
        adapter = self.adapter
        session = self.session
        try:
            scheduler.close()
        finally:
            try:
                adapter.close()
            finally:
                session.close()
                # Release Python owners of CUDA tensors/native sessions while
                # both runtimes are still fully alive, not during interpreter
                # global teardown.
                self.adapter = cast(Any, None)
                self.session = cast(Any, None)
                self._reference_scheduler = cast(Any, None)
                self._closed = True


__all__ = ["OnlineTrainingProducer"]
