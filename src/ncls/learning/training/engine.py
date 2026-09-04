from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Any, Callable, Mapping, cast

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ncls.core.identity import sha256_json
from ncls.data import OnlineDataSession, OnlineStepBatch
from ncls.learning.batches import (
    AssetTileBatch,
    EvaluatorBatch,
    MethodSamplerBatch,
    OnlineTrainingBatch,
    TrainingRouteRequest,
)
from ncls.learning.conformance import (
    validate_gradient_coverage,
    validate_objective_outputs,
    validate_phase_execution,
)
from ncls.learning.methods.contracts import MethodPlugin

from .checkpoint import TrainingCheckpoint
from .config import TrainingConfig, TrainingPhase, TrainingRoute
from .distributed import DistributedContext, DistributedObjective
from .events import TrainingEvent, TrainingEventBus, TrainingEventKind


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint: TrainingCheckpoint | None
    metrics: tuple[Mapping[str, float], ...]


@dataclass
class _PreparedStep:
    global_step: int
    batches: dict[str, OnlineTrainingBatch]
    step_batch: OnlineStepBatch
    preparation_seconds: float


class _CosinePhaseScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule: Mapping[str, Any],
        *,
        local_step: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.start = float(schedule["start"])
        self.end = float(schedule["end"])
        self.total_steps = int(schedule["total_steps"])
        self.offset = int(schedule["offset"])
        self.local_step = int(local_step)
        self._apply()

    def _apply(self) -> None:
        position = min(max(self.offset + self.local_step, 0), self.total_steps)
        fraction = position / self.total_steps
        value = self.end + 0.5 * (self.start - self.end) * (
            1.0 + math.cos(math.pi * fraction)
        )
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> None:
        self.local_step += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "cosine",
            "start": self.start,
            "end": self.end,
            "total_steps": self.total_steps,
            "offset": self.offset,
            "local_step": self.local_step,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        required = {"kind", "start", "end", "total_steps", "offset", "local_step"}
        if set(value) != required or value["kind"] != "cosine":
            raise ValueError("checkpoint phase scheduler state is invalid")
        expected = (self.start, self.end, self.total_steps, self.offset)
        actual = (
            float(value["start"]), float(value["end"]),
            int(value["total_steps"]), int(value["offset"]),
        )
        if actual != expected:
            raise ValueError("checkpoint phase scheduler recipe mismatch")
        self.local_step = int(value["local_step"])
        if self.local_step < 0:
            raise ValueError("checkpoint phase scheduler cursor is invalid")
        self._apply()


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    return value


def _merge_backend_profile(
    target: dict[str, float], source: Mapping[str, float]
) -> None:
    for name, raw_value in source.items():
        value = float(raw_value)
        if name == "resident_groups":
            target[name] = value
        elif name.endswith("_max"):
            target[name] = max(target.get(name, 0.0), value)
        else:
            target[name] = target.get(name, 0.0) + value


def _backend_profile_metrics(
    profile: Mapping[str, float], *, prefix: str
) -> dict[str, float]:
    result = {f"{prefix}{name}": float(value) for name, value in profile.items()}
    requests = float(
        profile.get("session_hits", 0.0) + profile.get("session_misses", 0.0)
    )
    result[f"{prefix}session_miss_rate"] = float(
        profile.get("session_misses", 0.0)
    ) / max(requests, 1.0)
    return result


def _execution_group_code(group_id: str) -> float:
    try:
        prefix = int(group_id[:12], 16)
    except ValueError:
        prefix = int.from_bytes(
            hashlib.sha256(group_id.encode("utf-8")).digest()[:6], "big"
        )
    return float(prefix)


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    return value


class TrainingEngine:
    """执行任意 versioned phase graph 的唯一 online training lifecycle。"""

    def __init__(
        self,
        plugin: MethodPlugin,
        data_session: OnlineDataSession,
        config: TrainingConfig,
        *,
        progress_factory: Callable[..., Any] = tqdm,
        checkpoint_callback: Callable[[TrainingCheckpoint], None] | None = None,
        metric_callback: Callable[[Mapping[str, float]], None] | None = None,
        event_bus: TrainingEventBus | None = None,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        if plugin.descriptor.method_key != config.method_key:
            raise ValueError("training plan method disagrees with MethodPlugin")
        configured_family = str(config.source["family_id"])
        producer_families = {
            str(value.get("family_id", "")) for value in data_session.source_contracts
        }
        if producer_families != {configured_family}:
            raise ValueError("configured source family disagrees with online producer")
        plugin.lifecycle.validate_training_plan(config.to_dict())
        validate_phase_execution(
            plugin.descriptor, (phase.to_dict() for phase in config.phases)
        )
        declared_groups = set(plugin.descriptor.parameter_groups)
        declared_batches = set(plugin.descriptor.training_batch_requirements)
        configured_phases = {phase.name for phase in config.phases}
        for phase in config.phases:
            if not set(phase.parameter_groups).issubset(declared_groups):
                raise ValueError(f"phase {phase.name!r} references unknown parameter groups")
            if not {route.kind for route in phase.routes}.issubset(declared_batches):
                raise ValueError(f"phase {phase.name!r} references unsupported typed routes")
        for component in plugin.descriptor.components:
            if not set(component.active_phases).issubset(configured_phases):
                raise ValueError(
                    f"component {component.component_id!r} references an absent phase"
                )
            if component.required:
                for group in component.parameter_groups:
                    active = any(
                        phase.name in component.active_phases
                        and group in phase.parameter_groups
                        for phase in config.phases
                    )
                    if not active:
                        raise ValueError(
                            f"required component {component.component_id!r} is never trainable"
                        )
        self.plugin = plugin
        self.data_session = data_session
        self.config = config
        self.progress_factory = progress_factory
        self.checkpoint_callback = checkpoint_callback
        self.metric_callback = metric_callback
        self.event_bus = event_bus
        self.distributed = (
            DistributedContext.single(data_session.device)
            if distributed_context is None
            else distributed_context
        )
        if self.distributed.device != data_session.device:
            raise ValueError("distributed context device disagrees with data session")
        self._last_checkpoint_profile: dict[str, float] = {}

    def _emit(
        self,
        kind: str,
        global_step: int,
        *,
        phase_name: str | None = None,
        scalars: Mapping[str, float] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if self.event_bus is None:
            return
        self.event_bus.emit(
            TrainingEvent(
                cast(TrainingEventKind, kind),
                global_step,
                self._ddp_rank(),
                self._ddp_world_size(),
                phase_name,
                {} if scalars is None else scalars,
                details={} if details is None else details,
            )
        )

    def _seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _phase_context(
        self,
        phase_index: int,
        phase_step: int,
        global_step: int,
        *,
        validation: bool = False,
    ) -> dict[str, Any]:
        phase = self.config.phases[phase_index]
        return {
            "name": phase.name,
            "phase_index": phase_index,
            "phase_step": phase_step,
            "phase_steps": phase.steps,
            "global_step": global_step,
            "total_steps": self.config.total_steps,
            "parameter_groups": list(phase.parameter_groups),
            "loss_terms": list(phase.loss_terms),
            "recipes": dict(phase.recipes),
            "validation": validation,
        }

    def _request(
        self,
        phase: TrainingPhase,
        route: TrainingRoute,
        step: int,
        *,
        validation: bool = False,
    ) -> TrainingRouteRequest:
        options = dict(route.options)
        options.update({"recipes": dict(phase.recipes), "validation": validation})
        name = f"validation:{phase.name}:{route.name}" if validation else f"{phase.name}:{route.name}"
        # Give every rank a deterministic, disjoint query subsequence. The
        # shared identity freezes the partition recipe, while rank-local state
        # is persisted in the checkpoint envelope for exact resume.
        rank_stride = self._ddp_rank() * 1_000_003
        seed_offset = route.seed_offset + rank_stride + (1_000_000_007 if validation else 0)
        return TrainingRouteRequest(
            name,
            route.kind,
            route.batch_size,
            route.direction_count,
            step,
            self.config.seed + seed_offset,
            options,
        )

    def _ddp_rank(self) -> int:
        return self.distributed.rank

    def _ddp_world_size(self) -> int:
        return self.distributed.world_size

    def _ddp_select_state(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("schema") != "ncls.ddp-rank-state@1":
            return value
        world = int(value.get("world_size", 0))
        if not self.distributed.is_distributed or world != self.distributed.world_size:
            raise ValueError("DDP checkpoint world size mismatch")
        rank = self.distributed.rank
        entries = [item for item in value.get("rank_states", ()) if int(item.get("rank", -1)) == rank]
        if len(entries) != 1:
            raise ValueError("DDP checkpoint does not contain this rank state")
        return entries[0]["state"]

    def _ddp_report(
        self,
        loss: torch.Tensor,
        metrics: Mapping[str, torch.Tensor | float],
        *,
        scope: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """一次 packed collective 聚合 loss 与全部 scalar metrics。"""
        return self.distributed.reduce_report(loss, metrics, scope=scope)

    @staticmethod
    def _validate_batch_type(route: TrainingRoute, batch: OnlineTrainingBatch) -> None:
        expected = {
            "asset-tile": AssetTileBatch,
            "reference-evaluator": EvaluatorBatch,
            "method-sampler": MethodSamplerBatch,
        }[route.kind]
        if not isinstance(batch, expected):
            raise TypeError(f"{route.kind} route returned the wrong batch type")

    def _route_requests(
        self,
        phase: TrainingPhase,
        step: int,
        *,
        validation: bool = False,
    ) -> dict[str, TrainingRouteRequest]:
        return {
            route.name: self._request(phase, route, step, validation=validation)
            for route in phase.routes
        }

    def _acquire_submitted_step(
        self,
        phase: TrainingPhase,
        step: int,
        logical_id: int,
    ) -> _PreparedStep:
        started = time.perf_counter()
        step_batch = self.data_session.acquire_step(logical_id)
        try:
            result = dict(step_batch.batches)
            for route in phase.routes:
                batch = result[route.name]
                self._validate_batch_type(route, batch)
        except BaseException:
            step_batch.release()
            raise
        identities = {
            (batch.provenance.get("route_name"), batch.provenance.get("request_index"))
            for batch in result.values()
        }
        if len(identities) != len(result):
            step_batch.release()
            raise RuntimeError("training routes reused one query stream request")
        return _PreparedStep(
            step,
            result,
            step_batch,
            max(step_batch.consumer_wait_seconds, time.perf_counter() - started),
        )

    def _release_prepared(self, prepared: _PreparedStep) -> None:
        prepared.step_batch.release()

    def _active_named_parameters(
        self, model: nn.Module, phase: TrainingPhase
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        named = dict(model.named_parameters())
        result = tuple(
            (name, named[name])
            for group in phase.parameter_groups
            for name in self.plugin.descriptor.parameter_groups[group]
        )
        if not result:
            raise ValueError("training phase has no active parameters")
        return result

    def _create_phase_optimization(
        self,
        model: nn.Module,
        phase: TrainingPhase,
        *,
        phase_step: int,
        state: Mapping[str, Any] | None = None,
        overlap_state: Mapping[str, Any] | None = None,
    ) -> tuple[
        torch.optim.Optimizer,
        _CosinePhaseScheduler,
        torch.amp.GradScaler,
        tuple[tuple[str, nn.Parameter], ...],
    ]:
        self.plugin.lifecycle.configure_phase(model, phase.to_dict())
        active = self._active_named_parameters(model, phase)
        parameters = tuple(parameter for _, parameter in active)
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(phase.schedule["start"]),
            betas=tuple(float(value) for value in phase.optimizer["betas"]),
            eps=float(phase.optimizer["epsilon"]),
            weight_decay=float(phase.optimizer["weight_decay"]),
            fused=all(parameter.device.type == "cuda" for parameter in parameters),
        )
        scheduler = _CosinePhaseScheduler(
            optimizer, phase.schedule, local_step=phase_step
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(phase.precision["gradient_scaler"])
        )
        if state is not None:
            if state.get("phase_name") != phase.name:
                raise ValueError("checkpoint optimizer state belongs to another phase")
            self._restore_optimizer(optimizer, active, state["optimizer"], overlap_only=False)
            scheduler.load_state_dict(state["scheduler"])
            precision = dict(state["precision"])
            if precision.get("config") != dict(phase.precision):
                raise ValueError("checkpoint precision recipe mismatch")
            scaler.load_state_dict(dict(precision.get("scaler", {})))
        elif overlap_state is not None and phase.optimizer_state_policy == "carry-overlap":
            self._restore_optimizer(
                optimizer, active, overlap_state["optimizer"], overlap_only=True
            )
        return optimizer, scheduler, scaler, active

    @staticmethod
    def _serialize_optimizer(
        optimizer: torch.optim.Optimizer,
        active: tuple[tuple[str, nn.Parameter], ...],
    ) -> dict[str, Any]:
        state = {
            name: _tree_to_cpu(optimizer.state[parameter])
            for name, parameter in active
            if parameter in optimizer.state
        }
        return {
            "parameter_names": [name for name, _ in active],
            "state_by_name": state,
        }

    @staticmethod
    def _restore_optimizer(
        optimizer: torch.optim.Optimizer,
        active: tuple[tuple[str, nn.Parameter], ...],
        value: Mapping[str, Any],
        *,
        overlap_only: bool,
    ) -> None:
        if set(value) != {"parameter_names", "state_by_name"}:
            raise ValueError("checkpoint named optimizer state fields are invalid")
        saved_names = tuple(str(name) for name in value["parameter_names"])
        active_names = tuple(name for name, _ in active)
        if not overlap_only and saved_names != active_names:
            raise ValueError("checkpoint optimizer parameter order mismatch")
        states = dict(value["state_by_name"])
        if not set(states).issubset(saved_names):
            raise ValueError("checkpoint optimizer contains an undeclared parameter state")
        for name, parameter in active:
            if name in states:
                optimizer.state[parameter] = _tree_to_device(states[name], parameter.device)

    def _optimization_state(
        self,
        phase: TrainingPhase,
        optimizer: torch.optim.Optimizer,
        scheduler: _CosinePhaseScheduler,
        scaler: torch.amp.GradScaler,
        active: tuple[tuple[str, nn.Parameter], ...],
    ) -> dict[str, Any]:
        return {
            "phase_name": phase.name,
            "optimizer": self._serialize_optimizer(optimizer, active),
            "scheduler": scheduler.state_dict(),
            "precision": {
                "config": dict(phase.precision),
                "scaler": _tree_to_cpu(scaler.state_dict()),
            },
        }

    @staticmethod
    def _rng_state() -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            result["cuda"] = torch.cuda.get_rng_state_all()
        return result

    @staticmethod
    def _restore_rng(state: Mapping[str, Any]) -> None:
        torch.set_rng_state(state["torch"].cpu())
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])

    def _validate_resume(self, checkpoint: TrainingCheckpoint) -> None:
        checkpoint.validate_method(self.plugin.descriptor)
        expected = {
            "training_config_sha256": self.config.sha256,
            "reference_program_identity": self.data_session.reference_program_identity,
            "reference_execution_plan_identity": self.data_session.reference_execution_plan_identity,
            "native_asset_collection_identity": self.data_session.native_asset_collection_identity,
            "query_stream_identity": self.data_session.query_stream_identity,
        }
        for name, value in expected.items():
            if getattr(checkpoint, name) != value:
                raise ValueError(f"resume checkpoint {name} mismatch")
        if checkpoint.source_snapshot_ids != self.data_session.source_snapshot_ids:
            raise ValueError("resume checkpoint source snapshot identity mismatch")

    def _autocast(self, phase: TrainingPhase):
        mode = str(phase.precision["autocast"])
        if mode == "fp32":
            return nullcontext()
        dtype = torch.float16 if mode == "float16" else torch.bfloat16
        return torch.autocast(device_type=self.data_session.device.type, dtype=dtype)

    def _gradient_audit(
        self,
        phase: TrainingPhase,
        registry: Mapping[str, tuple[nn.Parameter, ...]],
        snapshots: Mapping[str, tuple[torch.Tensor, ...]],
        coverage: dict[str, dict[str, Any]],
        global_step: int,
    ) -> None:
        group_names: list[str] = []
        checks: list[torch.Tensor] = []
        for group in phase.parameter_groups:
            gradients = [parameter.grad for parameter in registry[group] if parameter.grad is not None]
            if not gradients:
                raise RuntimeError(f"parameter group {group!r} produced no gradients")
            group_names.append(group)
            checks.extend(
                (
                    torch.stack([torch.isfinite(value).all() for value in gradients]).all(),
                    torch.stack([torch.any(value != 0) for value in gradients]).any(),
                    torch.stack([
                        torch.any(parameter.detach() != before)
                        for parameter, before in zip(registry[group], snapshots[group], strict=True)
                    ]).any(),
                )
            )
        values = torch.stack(checks).to(device="cpu", non_blocking=False).tolist()
        for index, group in enumerate(group_names):
            finite, nonzero, updated = (bool(value) for value in values[3 * index : 3 * index + 3])
            if not finite or not nonzero or not updated:
                raise RuntimeError(
                    f"gradient audit failed for {group!r}: finite={finite}, "
                    f"nonzero={nonzero}, updated={updated}"
                )
            item = coverage[group]
            item["finite_observed"] = bool(item["finite_observed"] or finite)
            item["nonzero_gradient_observed"] = bool(
                item["nonzero_gradient_observed"] or nonzero
            )
            item["parameter_update_observed"] = bool(
                item["parameter_update_observed"] or updated
            )
            item["last_audit_step"] = global_step

    def _validation_rows(
        self,
        model: nn.Module,
        phase_index: int,
        phase_step: int,
        global_step: int,
    ) -> list[Mapping[str, float]]:
        phase = self.config.phases[phase_index]
        rows: list[Mapping[str, float]] = []
        for batch_index in range(int(self.config.validation["batches"])):
            logical_id = self.data_session.submit_step(
                self._route_requests(
                    phase, global_step, validation=True
                ),
                boundary_id=(
                    f"validation:{phase.name}:{global_step}:{batch_index}"
                ),
            )
            prepared = self._acquire_submitted_step(
                phase, global_step, logical_id
            )
            try:
                with torch.no_grad(), self._autocast(phase):
                    loss, metrics = self.plugin.objective.compute(
                        model,
                        prepared.batches,
                        self._phase_context(
                            phase_index, phase_step, global_step, validation=True
                        ),
                    )
                    report_loss, report_metrics = self._ddp_report(
                        loss,
                        metrics,
                        scope=f"validation:{phase.name}:metrics",
                    )
                row = {
                    "step": float(global_step),
                    "phase_index": float(phase_index),
                    "validation/loss": float(report_loss.detach()),
                }
                validate_objective_outputs(
                    self.plugin.descriptor, phase.name, metrics
                )
                for name, value in report_metrics.items():
                    row[f"validation/{name}"] = (
                        float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    )
                rows.append(row)
            finally:
                self._release_prepared(prepared)
        return rows

    def _component_manifest(self) -> dict[str, Any]:
        descriptor = self.plugin.descriptor
        return {
            "schema": "ncls.method-components@1",
            "parameter_groups": {
                name: list(values) for name, values in descriptor.parameter_groups.items()
            },
            "components": [component.to_dict() for component in descriptor.components],
        }

    def _checkpoint(
        self,
        model: nn.Module,
        global_step: int,
        optimization_state: (
            Mapping[str, Any] | Callable[[], Mapping[str, Any]]
        ),
        coverage: Mapping[str, Mapping[str, Any]],
        validation_rows: list[Mapping[str, float]],
    ) -> TrainingCheckpoint | None:
        # A checkpoint is a lifecycle boundary. No prefetched host work,
        # reference result or GPU lease may cross it.
        local_state: dict[str, Any] | None = None
        local_state_error: BaseException | None = None
        try:
            self.data_session.drain()
            local_state = {
                "rng": self._rng_state(),
                "query_stream": self.data_session.state_dict(),
            }
        except BaseException as error:
            local_state_error = error
        self.distributed.synchronize_rank_errors(
            f"checkpoint rank state at step {global_step}",
            local_state_error,
        )
        if local_state is None:
            raise RuntimeError("checkpoint rank state was not constructed")
        config_value = self.config.to_dict()
        phase_index, phase_step = self.config.locate_step(global_step)
        phase_name = (
            "complete" if phase_index == len(self.config.phases)
            else self.config.phases[phase_index].name
        )
        rank_states = self.distributed.gather_rank_payload(local_state)
        if rank_states is None:
            return None
        resolved_optimization_state = (
            optimization_state()
            if callable(optimization_state)
            else optimization_state
        )
        if self.distributed.is_distributed:
            rng_state: Mapping[str, Any] = {
                "schema": "ncls.ddp-rank-state@1",
                "world_size": self.distributed.world_size,
                "rank_states": [
                    {
                        "rank": int(value["rank"]),
                        "world_size": int(value["world_size"]),
                        "state": value["state"]["rng"],
                    }
                    for value in rank_states
                ],
            }
            query_stream_state: Mapping[str, Any] = {
                "schema": "ncls.ddp-rank-state@1",
                "world_size": self.distributed.world_size,
                "rank_states": [
                    {
                        "rank": int(value["rank"]),
                        "world_size": int(value["world_size"]),
                        "state": value["state"]["query_stream"],
                    }
                    for value in rank_states
                ],
            }
        else:
            rng_state = local_state["rng"]
            query_stream_state = local_state["query_stream"]
        checkpoint = TrainingCheckpoint(
            self.plugin.descriptor.method_key,
            self.plugin.descriptor.descriptor_sha256,
            self.plugin.descriptor.implementation_sha256,
            self._component_manifest(),
            config_value,
            sha256_json(config_value),
            sha256_json([phase.to_dict() for phase in self.config.phases]),
            self.data_session.reference_program_identity,
            self.data_session.reference_execution_plan_identity,
            self.data_session.native_asset_collection_identity,
            self.data_session.query_stream_identity,
            self.data_session.source_contracts,
            self.data_session.source_snapshot_ids,
            global_step,
            phase_index,
            phase_name,
            phase_step,
            {"policy": self.config.checkpoint_selection, "tail": validation_rows[-1:]},
            self.plugin.checkpoint.encode(model),
            resolved_optimization_state,
            rng_state,
            query_stream_state,
            coverage,
            {"rows": validation_rows},
        )
        checkpoint.validate_method(self.plugin.descriptor)
        return checkpoint

    def _coordinated_checkpoint(
        self,
        model: nn.Module,
        global_step: int,
        optimization_state: (
            Mapping[str, Any] | Callable[[], Mapping[str, Any]]
        ),
        coverage: Mapping[str, Mapping[str, Any]],
        validation_rows: list[Mapping[str, float]],
        *,
        label: str,
        callback: Callable[[TrainingCheckpoint], None] | None = None,
    ) -> TrainingCheckpoint | None:
        assembly_started = time.perf_counter()
        checkpoint: TrainingCheckpoint | None = None
        checkpoint_error: BaseException | None = None
        try:
            checkpoint = self._checkpoint(
                model,
                global_step,
                optimization_state,
                coverage,
                validation_rows,
            )
        except BaseException as error:
            checkpoint_error = error

        def commit() -> TrainingCheckpoint:
            if checkpoint_error is not None:
                raise checkpoint_error
            if checkpoint is None:
                raise RuntimeError("rank0 did not construct a training checkpoint")
            if callback is not None:
                callback(checkpoint)
            return checkpoint

        assembly_seconds = time.perf_counter() - assembly_started
        commit_started = time.perf_counter()
        committed = self.distributed.run_rank_zero(label, commit)
        commit_seconds = time.perf_counter() - commit_started
        local_profile = {
            "profile/checkpoint_rank_state_and_assembly_seconds": assembly_seconds,
            "profile/checkpoint_rank0_commit_wait_seconds": commit_seconds,
        }
        self._last_checkpoint_profile = {
            **local_profile,
            **self.distributed.rank_statistics(
                local_profile,
                scope=f"checkpoint:{global_step}:stage-metrics",
            ),
        }
        return committed if self.distributed.is_rank_zero else None

    def run(
        self,
        *,
        resume: TrainingCheckpoint | None = None,
        stop_at_step: int | None = None,
    ) -> TrainingRunResult:
        self.distributed.validate_descriptor(
            "training:lifecycle",
            {
                "training_config_sha256": self.config.sha256,
                "resume": (
                    None
                    if resume is None
                    else {
                        "global_step": resume.global_step,
                        "method_key": resume.method_key,
                        "training_config_sha256": resume.training_config_sha256,
                    }
                ),
                "stop_at_step": stop_at_step,
                "checkpoint_callback": self.checkpoint_callback is not None,
            },
        )
        self._seed()
        model = self.distributed.run_all_ranks(
            "training model construction",
            lambda: self.plugin.model_factory.create(self.config.model_context).to(
                self.data_session.device
            ),
        )
        if self.distributed.is_distributed and self.data_session.device.type != "cuda":
            raise RuntimeError("DDP training requires a CUDA producer device")
        global_step = 0
        validation_rows: list[Mapping[str, float]] = []
        coverage: dict[str, dict[str, Any]] = {
            group: {
                "finite_observed": False,
                "nonzero_gradient_observed": False,
                "parameter_update_observed": False,
                "last_audit_step": -1,
            }
            for group in self.plugin.descriptor.parameter_groups
        }
        resume_optimization: Mapping[str, Any] | None = None
        def restore_resume() -> None:
            nonlocal global_step, validation_rows, coverage, resume_optimization
            if resume is None:
                return
            self._validate_resume(resume)
            self.plugin.checkpoint.restore(model, resume.model_state)
            self.data_session.load_state_dict(
                self._ddp_select_state(resume.query_stream_state)
            )
            self._restore_rng(self._ddp_select_state(resume.rng_state))
            global_step = resume.global_step
            validation_rows = [
                dict(row) for row in resume.validation_state.get("rows", ())
            ]
            coverage = {
                group: dict(value)
                for group, value in resume.gradient_coverage.items()
            }
            resume_optimization = resume.phase_optimization_state

        self.distributed.run_all_ranks("training resume restore", restore_resume)

        target_step = self.config.total_steps if stop_at_step is None else int(stop_at_step)
        if not global_step <= target_step <= self.config.total_steps:
            raise ValueError("stop_at_step must lie between resume step and total steps")
        if global_step == self.config.total_steps:
            checkpoint = self._coordinated_checkpoint(
                model,
                global_step,
                {},
                coverage,
                validation_rows,
                label=f"final checkpoint assembly at step {global_step}",
            )
            self._emit(
                "run-completed",
                global_step,
                phase_name="complete",
                scalars=self._last_checkpoint_profile,
                details={"already_complete": True},
            )
            return TrainingRunResult(checkpoint, tuple(validation_rows))

        phase_index, phase_step = self.config.locate_step(global_step)
        phase = self.config.phases[phase_index]
        optimizer, scheduler, scaler, active = self.distributed.run_all_ranks(
            f"phase {phase.name} optimization setup",
            lambda: self._create_phase_optimization(
                model,
                phase,
                phase_step=phase_step,
                state=resume_optimization,
            ),
        )
        objective_owner, execution_objective = self.distributed.build_objective(
            self.plugin.objective,
            model,
            phase_name=phase.name,
        )
        registry = self.plugin.lifecycle.parameter_registry(model)
        metric_rows: list[Mapping[str, float]] = []
        global_batch_multiplier = self._ddp_world_size()
        work_units = sum(
            self.config.phases[index].steps
            * sum(route.batch_size * route.direction_count for route in self.config.phases[index].routes)
            * global_batch_multiplier
            for index in range(phase_index)
        ) + phase_step * sum(
            route.batch_size * route.direction_count for route in phase.routes
        ) * global_batch_multiplier
        run_start_work_units = work_units
        run_started = time.perf_counter()
        run_start_step = global_step
        phase_timing_index = phase_index
        phase_timing_started = run_started
        phase_timing_start_step = phase_step
        preparation_window: list[float] = []
        step_wall_window: list[float] = []
        reference_group_codes: set[float] = set()
        reference_last_group_code = 0.0
        reference_candidate_count = 0
        reference_rejected_count = 0
        reference_rejection_rounds = 0
        reference_rejection_rounds_max = 0
        profile_snapshot = getattr(self.data_session, "profile_snapshot", None)
        pending_training_profile: dict[str, float] = {}
        latest_checkpoint: TrainingCheckpoint | None = None
        latest_checkpoint_step = -1
        if callable(profile_snapshot):
            profile_snapshot(reset=True)
        queue: deque[tuple[int, int]] = deque()
        bar = self.progress_factory(
            total=target_step - global_step,
            desc="train",
            unit="step",
            disable=not self.distributed.is_rank_zero,
        )
        self._emit(
            "run-started",
            global_step,
            phase_name=phase.name,
            details={
                "target_step": target_step,
                "resumed": resume is not None,
                "training_config_sha256": self.config.sha256,
            },
        )
        self._emit("phase-started", global_step, phase_name=phase.name)
        try:
            while global_step < target_step:
                phase_index, phase_step = self.config.locate_step(global_step)
                phase = self.config.phases[phase_index]
                if phase_timing_index != phase_index:
                    phase_timing_index = phase_index
                    phase_timing_started = time.perf_counter()
                    phase_timing_start_step = phase_step
                    step_wall_window.clear()
                step_started = time.perf_counter()
                next_validation = (
                    ((global_step // int(self.config.validation["interval"])) + 1)
                    * int(self.config.validation["interval"])
                )
                phase_end = self.config.phase_start_step(phase_index) + phase.steps
                barrier = min(target_step, phase_end, next_validation)
                lookahead = min(
                    self.data_session.submission_capacity,
                    max(
                        phase.prefetch_depth,
                        self.data_session.production_batch_steps,
                    ),
                )
                while len(queue) < lookahead:
                    next_step = global_step + len(queue)
                    if next_step >= barrier:
                        break
                    logical_id = self.data_session.submit_step(
                        self._route_requests(phase, next_step),
                        boundary_id=f"training:{phase.name}:{barrier}",
                    )
                    queue.append((next_step, logical_id))
                if not queue:
                    raise RuntimeError("training data lookahead produced no logical step")
                prepared_step, logical_id = queue.popleft()
                prepared = self._acquire_submitted_step(
                    phase, prepared_step, logical_id
                )
                preparation_window.append(prepared.preparation_seconds)
                for batch in prepared.batches.values():
                    provenance = batch.provenance
                    group_id = provenance.get("reference_execution_group_id")
                    if isinstance(group_id, str) and group_id:
                        reference_last_group_code = _execution_group_code(group_id)
                        reference_group_codes.add(reference_last_group_code)
                    if isinstance(batch, EvaluatorBatch):
                        candidate_count = int(provenance.get("candidate_count", 0))
                        rejected_count = int(provenance.get("rejected_count", 0))
                        rejection_rounds = int(provenance.get("rejection_rounds", 0))
                        reference_candidate_count += candidate_count
                        reference_rejected_count += rejected_count
                        reference_rejection_rounds += rejection_rounds
                        reference_rejection_rounds_max = max(
                            reference_rejection_rounds_max, rejection_rounds
                        )
                if prepared.global_step != global_step:
                    raise RuntimeError("training prefetch queue lost deterministic step order")
                audit = (
                    phase_step == 0
                    or (phase_step + 1) % phase.gradient_audit_interval == 0
                    or phase_step + 1 == phase.steps
                    or global_step + 1 == target_step
                )
                snapshots = (
                    {
                        group: tuple(parameter.detach().clone() for parameter in registry[group])
                        for group in phase.parameter_groups
                    }
                    if audit else {}
                )
                will_log = (
                    global_step + 1 == target_step
                    or phase_step + 1 == phase.steps
                    or (phase_step + 1) % phase.log_interval == 0
                )
                timing: dict[str, float] = {
                    "profile/batch_prepare_wall_seconds": prepared.preparation_seconds,
                    "profile/explicit_syncs": 0.0,
                }
                cuda_events: tuple[torch.cuda.Event, ...] | None = None
                if self.data_session.device.type == "cuda" and (audit or will_log):
                    cuda_events = tuple(
                        torch.cuda.Event(enable_timing=True) for _ in range(4)
                    )
                    cuda_events[0].record()
                forward_started = time.perf_counter()
                try:
                    optimizer.zero_grad(set_to_none=True)
                    with self._autocast(phase):
                        context = self._phase_context(phase_index, phase_step, global_step)
                        loss = execution_objective(
                            prepared.batches,
                            context,
                        )
                        metrics = objective_owner.take_metrics()
                    if cuda_events is not None:
                        cuda_events[1].record()
                    forward_finished = time.perf_counter()
                    validate_objective_outputs(
                        self.plugin.descriptor, phase.name, metrics
                    )
                    if loss.ndim != 0:
                        raise RuntimeError("training objective must return one scalar loss")
                    finite_loss = torch.isfinite(loss)
                    if finite_loss.device.type == "cuda":
                        torch._assert_async(finite_loss)
                    elif not bool(finite_loss):
                        raise RuntimeError("training objective returned a non-finite loss")
                    scaler.scale(loss).backward()
                    if cuda_events is not None:
                        cuda_events[2].record()
                    backward_finished = time.perf_counter()
                    scaler.unscale_(optimizer)
                    gradients = [
                        parameter.grad for _, parameter in active if parameter.grad is not None
                    ]
                    if not gradients:
                        raise RuntimeError("training objective produced no active gradients")
                    finite_gradients = torch.stack(
                        [torch.isfinite(value).all() for value in gradients]
                    ).all()
                    if finite_gradients.device.type == "cuda":
                        torch._assert_async(finite_gradients)
                    elif not bool(finite_gradients):
                        raise RuntimeError("training objective produced non-finite gradients")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if cuda_events is not None:
                        cuda_events[3].record()
                        cuda_events[3].synchronize()
                        timing.update(
                            {
                                "profile/forward_gpu_seconds": (
                                    cuda_events[0].elapsed_time(cuda_events[1]) / 1000.0
                                ),
                                "profile/backward_gpu_seconds": (
                                    cuda_events[1].elapsed_time(cuda_events[2]) / 1000.0
                                ),
                                "profile/backward_reducer_gpu_seconds": (
                                    cuda_events[1].elapsed_time(cuda_events[2]) / 1000.0
                                ),
                                "profile/optimizer_gpu_seconds": (
                                    cuda_events[2].elapsed_time(cuda_events[3]) / 1000.0
                                ),
                                "profile/explicit_syncs": 1.0,
                            }
                        )
                    else:
                        timing.update(
                            {
                                "profile/forward_wall_seconds": (
                                    forward_finished - forward_started
                                ),
                                "profile/backward_wall_seconds": (
                                    backward_finished - forward_finished
                                ),
                                "profile/backward_reducer_wall_seconds": (
                                    backward_finished - forward_finished
                                ),
                                "profile/optimizer_wall_seconds": (
                                    time.perf_counter() - backward_finished
                                ),
                            }
                        )
                    if audit:
                        self._gradient_audit(
                            phase, registry, snapshots, coverage, global_step
                        )
                finally:
                    self._release_prepared(prepared)
                global_step += 1
                phase_step += 1
                work_units += sum(
                    route.batch_size * route.direction_count for route in phase.routes
                ) * global_batch_multiplier
                step_wall_window.append(time.perf_counter() - step_started)
                should_log = (
                    global_step == target_step
                    or phase_step == phase.steps
                    or phase_step % phase.log_interval == 0
                )
                if should_log:
                    elapsed = time.perf_counter() - run_started
                    completed = global_step - run_start_step
                    speed = completed / max(elapsed, 1e-12)
                    phase_elapsed = time.perf_counter() - phase_timing_started
                    phase_completed = phase_step - phase_timing_start_step
                    phase_speed = phase_completed / max(phase_elapsed, 1e-12)
                    wall_values = np.asarray(step_wall_window, dtype=np.float64)
                    rolling_speed = len(step_wall_window) / max(
                        float(wall_values.sum()), 1e-12
                    )
                    phase_work_units_per_step = sum(
                        route.batch_size * route.direction_count
                        for route in phase.routes
                    )
                    completed_global_work_units = work_units - run_start_work_units
                    row: dict[str, float] = {
                        "step": float(global_step),
                        "phase_index": float(phase_index),
                        "phase_step": float(phase_step),
                        "loss": float(loss.detach()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "work_units": float(work_units),
                        "global_batch_multiplier": float(global_batch_multiplier),
                        "global_work_units_per_second": (
                            completed_global_work_units / max(elapsed, 1e-12)
                        ),
                        "local_work_units_per_second": (
                            completed_global_work_units
                            / global_batch_multiplier
                            / max(elapsed, 1e-12)
                        ),
                        "phase_global_work_units_per_second": (
                            phase_speed
                            * phase_work_units_per_step
                            * global_batch_multiplier
                        ),
                        "rolling_global_work_units_per_second": (
                            rolling_speed
                            * phase_work_units_per_step
                            * global_batch_multiplier
                        ),
                        "elapsed_seconds": elapsed,
                        "steps_per_second": speed,
                        "phase_steps_per_second": phase_speed,
                        "rolling_steps_per_second": rolling_speed,
                        "eta_seconds": (target_step - global_step) / max(speed, 1e-12),
                        "phase_eta_seconds": (
                            phase.steps - phase_step
                        ) / max(phase_speed, 1e-12),
                        "rolling_eta_seconds": (
                            target_step - global_step
                        ) / max(rolling_speed, 1e-12),
                    }
                    if self.data_session.device.type == "cuda":
                        row["peak_memory_bytes"] = float(
                            torch.cuda.max_memory_allocated(self.data_session.device)
                        )
                        row["reserved_memory_bytes"] = float(
                            torch.cuda.memory_reserved(self.data_session.device)
                        )
                    report_loss, report_metrics = self._ddp_report(
                        loss,
                        metrics,
                        scope=f"training:{phase.name}:metrics",
                    )
                    row["loss"] = float(report_loss)
                    for name, value in report_metrics.items():
                        row[name] = (
                            float(value.detach())
                            if isinstance(value, torch.Tensor)
                            else float(value)
                        )
                    row.update(timing)
                    preparation_values = np.asarray(preparation_window, dtype=np.float64)
                    row.update(
                        {
                            "profile/step_wall_window_count": float(
                                len(step_wall_window)
                            ),
                            "profile/step_wall_seconds_mean": float(
                                wall_values.mean()
                            ),
                            "profile/step_wall_seconds_median": float(
                                np.median(wall_values)
                            ),
                            "profile/step_wall_seconds_p90": float(
                                np.quantile(wall_values, 0.9)
                            ),
                            "profile/step_wall_seconds_max": float(
                                wall_values.max()
                            ),
                            "profile/batch_prepare_window_count": float(
                                len(preparation_window)
                            ),
                            "profile/batch_prepare_wall_seconds_mean": float(
                                preparation_values.mean()
                            ),
                            "profile/batch_prepare_wall_seconds_median": float(
                                np.median(preparation_values)
                            ),
                            "profile/batch_prepare_wall_seconds_p90": float(
                                np.quantile(preparation_values, 0.9)
                            ),
                            "profile/batch_prepare_wall_seconds_max": float(
                                preparation_values.max()
                            ),
                            "profile/reference_execution_group_count": float(
                                len(reference_group_codes)
                            ),
                            "profile/reference_last_group_id_u48": (
                                reference_last_group_code
                            ),
                            "profile/reference_candidate_count": float(
                                reference_candidate_count
                            ),
                            "profile/reference_rejected_count": float(
                                reference_rejected_count
                            ),
                            "profile/reference_rejection_rate": float(
                                reference_rejected_count
                                / max(reference_candidate_count, 1)
                            ),
                            "profile/reference_rejection_rounds": float(
                                reference_rejection_rounds
                            ),
                            "profile/reference_rejection_rounds_max": float(
                                reference_rejection_rounds_max
                            ),
                        }
                    )
                    if callable(profile_snapshot):
                        backend_profile = profile_snapshot(reset=True)
                        _merge_backend_profile(
                            pending_training_profile, backend_profile
                        )
                        row.update(
                            _backend_profile_metrics(
                                pending_training_profile,
                                prefix="profile/reference_",
                            )
                        )
                        pending_training_profile.clear()
                    row.update(
                        self.distributed.ddp_logging_metrics(execution_objective)
                    )
                    stage_names = tuple(
                        name
                        for name in (
                            "profile/step_wall_seconds_mean",
                            "profile/step_wall_seconds_max",
                            "profile/batch_prepare_wall_seconds_mean",
                            "profile/batch_prepare_wall_seconds_max",
                            "profile/forward_gpu_seconds",
                            "profile/backward_reducer_gpu_seconds",
                            "profile/optimizer_gpu_seconds",
                            "profile/reference_last_group_id_u48",
                            "profile/reference_group_build_seconds_max",
                            "profile/reference_rejection_rounds_max",
                        )
                        if name in row
                    )
                    row.update(
                        self.distributed.rank_statistics(
                            {name: row[name] for name in stage_names},
                            scope=f"training:{phase.name}:stage-metrics",
                        )
                    )
                    preparation_window.clear()
                    step_wall_window.clear()
                    reference_group_codes.clear()
                    reference_last_group_code = 0.0
                    reference_candidate_count = 0
                    reference_rejected_count = 0
                    reference_rejection_rounds = 0
                    reference_rejection_rounds_max = 0
                    metric_rows.append(row)
                    if self.metric_callback is not None:
                        self.metric_callback(row)
                    self._emit(
                        "step-completed",
                        global_step,
                        phase_name=phase.name,
                        scalars={
                            name: value
                            for name, value in row.items()
                            if name not in {"step", "phase_index", "phase_step"}
                        },
                    )
                    bar.set_postfix(
                        {
                            "phase": phase.name,
                            "loss": f"{row['loss']:.6g}",
                            "queries": work_units,
                        }
                    )
                bar.update(1)

                boundary = phase_step == phase.steps
                if boundary:
                    if queue:
                        raise RuntimeError("prefetch queue crossed a phase boundary")
                    previous_state = (
                        self._optimization_state(
                            phase,
                            optimizer,
                            scheduler,
                            scaler,
                            active,
                        )
                        if global_step < self.config.total_steps
                        else None
                    )
                    del execution_objective, objective_owner
                    if phase.transition is not None:
                        self.distributed.run_all_ranks(
                            f"phase {phase.name} transition {phase.transition}",
                            lambda: self.plugin.lifecycle.apply_transition(
                                model,
                                phase.transition,
                                self.data_session.native_assets(),
                            ),
                        )
                    if global_step < self.config.total_steps:
                        next_index, next_step = self.config.locate_step(global_step)
                        next_phase = self.config.phases[next_index]
                        optimizer, scheduler, scaler, active = (
                            self.distributed.run_all_ranks(
                                f"phase {next_phase.name} optimization setup",
                                lambda: self._create_phase_optimization(
                                    model,
                                    next_phase,
                                    phase_step=next_step,
                                    overlap_state=previous_state,
                                ),
                            )
                        )
                        objective_owner, execution_objective = (
                            self.distributed.build_objective(
                                self.plugin.objective,
                                model,
                                phase_name=next_phase.name,
                            )
                        )
                        registry = self.plugin.lifecycle.parameter_registry(model)
                        self._emit(
                            "phase-started",
                            global_step,
                            phase_name=next_phase.name,
                        )

                needs_validation = (
                    global_step % int(self.config.validation["interval"]) == 0
                    or global_step == self.config.total_steps
                )
                if needs_validation:
                    if queue:
                        raise RuntimeError("prefetch queue crossed a validation boundary")
                    if callable(profile_snapshot):
                        _merge_backend_profile(
                            pending_training_profile,
                            profile_snapshot(reset=True),
                        )
                    if global_step == self.config.total_steps:
                        validation_phase_index = len(self.config.phases) - 1
                        validation_phase_step = self.config.phases[-1].steps
                    else:
                        validation_phase_index, validation_phase_step = self.config.locate_step(global_step)
                    validation_started = time.perf_counter()
                    validation_backend_profile: Mapping[str, float] = {}
                    try:
                        new_rows = self._validation_rows(
                            model,
                            validation_phase_index,
                            validation_phase_step,
                            global_step,
                        )
                    finally:
                        if callable(profile_snapshot):
                            validation_backend_profile = profile_snapshot(reset=True)
                    validation_seconds = time.perf_counter() - validation_started
                    new_rows = [
                        {
                            **row,
                            "profile/validation_wall_seconds": (
                                validation_seconds if index == 0 else 0.0
                            ),
                            **(
                                _backend_profile_metrics(
                                    validation_backend_profile,
                                    prefix="profile/validation_reference_",
                                )
                                if index == 0 and validation_backend_profile
                                else {}
                            ),
                        }
                        for index, row in enumerate(new_rows)
                    ]
                    validation_rows.extend(new_rows)
                    if self.metric_callback is not None:
                        for row in new_rows:
                            self.metric_callback(row)
                    validation_scalars: dict[str, float] = {}
                    for row in new_rows:
                        for name, value in row.items():
                            if name != "step":
                                validation_scalars[name] = float(value)
                    self._emit(
                        "validation-completed",
                        global_step,
                        phase_name=phase.name,
                        scalars=validation_scalars,
                    )

                checkpoint_boundary = boundary and phase.checkpoint_boundary
                if global_step == self.config.total_steps:
                    validate_gradient_coverage(
                        self.plugin.descriptor, coverage
                    )
                if needs_validation or checkpoint_boundary:
                    if global_step == self.config.total_steps:
                        optimization_state: (
                            Mapping[str, Any]
                            | Callable[[], Mapping[str, Any]]
                        ) = {}
                    else:
                        current_index, _ = self.config.locate_step(global_step)
                        current_phase = self.config.phases[current_index]
                        optimization_state = lambda: self._optimization_state(
                            current_phase,
                            optimizer,
                            scheduler,
                            scaler,
                            active,
                        )
                    if self.checkpoint_callback is not None:
                        latest_checkpoint = self._coordinated_checkpoint(
                            model,
                            global_step,
                            optimization_state,
                            coverage,
                            validation_rows,
                            label=f"periodic checkpoint commit at step {global_step}",
                            callback=self.checkpoint_callback,
                        )
                        latest_checkpoint_step = global_step
                    self._emit(
                        "checkpoint-committed",
                        global_step,
                        phase_name=(
                            "complete"
                            if global_step == self.config.total_steps
                            else self.config.phases[
                                self.config.locate_step(global_step)[0]
                            ].name
                        ),
                        details={
                            "periodic_callback": self.checkpoint_callback is not None
                        },
                        scalars=self._last_checkpoint_profile,
                    )
        except BaseException as error:
            self._emit(
                "run-failed",
                global_step,
                phase_name=phase.name,
                details={"error_type": type(error).__name__, "message": str(error)},
            )
            raise
        finally:
            if queue:
                queue.clear()
                self.data_session.cancel_pending()
            bar.close()

        if global_step == self.config.total_steps:
            final_optimization: (
                Mapping[str, Any] | Callable[[], Mapping[str, Any]]
            ) = {}
        else:
            current_index, _ = self.config.locate_step(global_step)
            current_phase = self.config.phases[current_index]
            final_optimization = lambda: self._optimization_state(
                current_phase, optimizer, scheduler, scaler, active
            )
        checkpoint = latest_checkpoint
        if latest_checkpoint_step != global_step:
            checkpoint = self._coordinated_checkpoint(
                model,
                global_step,
                final_optimization,
                coverage,
                validation_rows,
                label=f"final checkpoint assembly at step {global_step}",
            )
        self._emit(
            "run-completed",
            global_step,
            phase_name=(
                "complete"
                if global_step == self.config.total_steps
                else self.config.phases[self.config.locate_step(global_step)[0]].name
            ),
            scalars=self._last_checkpoint_profile,
            details={"complete": global_step == self.config.total_steps},
        )
        return TrainingRunResult(checkpoint, tuple(metric_rows + validation_rows))
