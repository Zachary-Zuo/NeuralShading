from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import math
import random
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from ncls.core.identity import sha256_json
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
from ncls.learning.method import MethodDefinition
from ncls.learning.source_adaptation import NativeAssetCollection

from .checkpoint import TrainingCheckpoint
from .config import TrainingConfig, TrainingPhase, TrainingRoute


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint: TrainingCheckpoint
    metrics: tuple[Mapping[str, float], ...]


class OnlineTrainingProducer(Protocol):
    reference_program_identity: str
    reference_execution_plan_identity: str
    native_asset_collection_identity: str
    query_stream_identity: str
    source_contracts: tuple[Mapping[str, Any], ...]
    source_snapshot_ids: tuple[str, ...]
    device: torch.device

    def next_batch(self, request: TrainingRouteRequest) -> OnlineTrainingBatch: ...
    def native_assets(self) -> NativeAssetCollection: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
    def end_iteration(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class _PreparedStep:
    global_step: int
    batches: dict[str, OnlineTrainingBatch]
    iteration_ended: bool
    preparation_seconds: float


class _DDPObjective(nn.Module):
    """DDP forward shell for objectives implemented as model helper calls."""

    def __init__(self, definition: MethodDefinition, model: nn.Module) -> None:
        super().__init__()
        self.definition = definition
        self.model = model
        self._batches: Mapping[str, OnlineTrainingBatch] | None = None
        self._phase: Mapping[str, Any] | None = None
        self.last_metrics: Mapping[str, torch.Tensor | float] = {}

    def set_inputs(
        self,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> None:
        self._batches = batches
        self._phase = phase

    def forward(self) -> torch.Tensor:
        if self._batches is None or self._phase is None:
            raise RuntimeError("DDP objective inputs were not set")
        loss, metrics = self.definition.training_objective(
            self.model, self._batches, self._phase
        )
        self.last_metrics = metrics
        return loss


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


class TrainingRunner:
    """执行任意versioned phase graph的唯一online training orchestration。"""

    def __init__(
        self,
        definition: MethodDefinition,
        producer: OnlineTrainingProducer,
        config: TrainingConfig,
        *,
        progress_factory: Callable[..., Any] = tqdm,
        checkpoint_callback: Callable[[TrainingCheckpoint], None] | None = None,
        metric_callback: Callable[[Mapping[str, float]], None] | None = None,
    ) -> None:
        if definition.descriptor.method_key != config.method_key:
            raise ValueError("training config method_key disagrees with MethodDefinition")
        configured_family = str(config.source["family_id"])
        producer_families = {
            str(value.get("family_id", "")) for value in producer.source_contracts
        }
        if producer_families != {configured_family}:
            raise ValueError("configured source family disagrees with online producer")
        definition.validate_training_config(config.to_dict())
        validate_phase_execution(
            definition.descriptor, (phase.to_dict() for phase in config.phases)
        )
        declared_groups = set(definition.descriptor.parameter_groups)
        declared_batches = set(definition.descriptor.training_batch_requirements)
        configured_phases = {phase.name for phase in config.phases}
        for phase in config.phases:
            if not set(phase.parameter_groups).issubset(declared_groups):
                raise ValueError(f"phase {phase.name!r} references unknown parameter groups")
            if not {route.kind for route in phase.routes}.issubset(declared_batches):
                raise ValueError(f"phase {phase.name!r} references unsupported typed routes")
        for component in definition.descriptor.components:
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
        self.definition = definition
        self.producer = producer
        self.config = config
        self.progress_factory = progress_factory
        self.checkpoint_callback = checkpoint_callback
        self.metric_callback = metric_callback

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
        # Keep every rank's producer cursor identical so the single rank-0
        # checkpoint remains resumable by all ranks. DDP synchronizes gradients;
        # the authoritative online query stream is intentionally shared.
        seed_offset = route.seed_offset + (1_000_000_007 if validation else 0)
        return TrainingRouteRequest(
            name,
            route.kind,
            route.batch_size,
            route.direction_count,
            step,
            self.config.seed + seed_offset,
            options,
        )

    @staticmethod
    def _ddp_rank() -> int:
        return int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0

    @staticmethod
    def _ddp_barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _validate_batch_type(route: TrainingRoute, batch: OnlineTrainingBatch) -> None:
        expected = {
            "asset-tile": AssetTileBatch,
            "reference-evaluator": EvaluatorBatch,
            "method-sampler": MethodSamplerBatch,
        }[route.kind]
        if not isinstance(batch, expected):
            raise TypeError(f"{route.kind} route returned the wrong batch type")

    def _batches(
        self,
        phase: TrainingPhase,
        step: int,
        *,
        validation: bool = False,
    ) -> dict[str, OnlineTrainingBatch]:
        result: dict[str, OnlineTrainingBatch] = {}
        try:
            for route in phase.routes:
                batch = self.producer.next_batch(
                    self._request(phase, route, step, validation=validation)
                )
                self._validate_batch_type(route, batch)
                result[route.name] = batch
        except BaseException:
            for batch in reversed(tuple(result.values())):
                batch.release()
            raise
        identities = {
            (batch.provenance.get("route_name"), batch.provenance.get("request_index"))
            for batch in result.values()
        }
        if len(identities) != len(result):
            for batch in reversed(tuple(result.values())):
                batch.release()
            raise RuntimeError("training routes reused one query stream request")
        return result

    @staticmethod
    def _is_detached(batches: Mapping[str, OnlineTrainingBatch]) -> bool:
        return all(getattr(batch, "lease", None) is None for batch in batches.values())

    def _prepare_step(self, phase: TrainingPhase, step: int) -> _PreparedStep:
        started = time.perf_counter()
        batches = self._batches(phase, step)
        preparation_seconds = time.perf_counter() - started
        detached = self._is_detached(batches)
        if detached:
            self.producer.end_iteration()
        return _PreparedStep(step, batches, detached, preparation_seconds)

    def _release_prepared(self, prepared: _PreparedStep) -> None:
        for batch in reversed(tuple(prepared.batches.values())):
            batch.release()
        if not prepared.iteration_ended:
            self.producer.end_iteration()

    def _active_named_parameters(
        self, model: nn.Module, phase: TrainingPhase
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        named = dict(model.named_parameters())
        result = tuple(
            (name, named[name])
            for group in phase.parameter_groups
            for name in self.definition.descriptor.parameter_groups[group]
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
        self.definition.configure_phase(model, phase.to_dict())
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
        checkpoint.validate_method(self.definition.descriptor)
        expected = {
            "training_config_sha256": self.config.sha256,
            "reference_program_identity": self.producer.reference_program_identity,
            "reference_execution_plan_identity": self.producer.reference_execution_plan_identity,
            "native_asset_collection_identity": self.producer.native_asset_collection_identity,
            "query_stream_identity": self.producer.query_stream_identity,
        }
        for name, value in expected.items():
            if getattr(checkpoint, name) != value:
                raise ValueError(f"resume checkpoint {name} mismatch")
        if checkpoint.source_snapshot_ids != self.producer.source_snapshot_ids:
            raise ValueError("resume checkpoint source snapshot identity mismatch")

    def _autocast(self, phase: TrainingPhase):
        mode = str(phase.precision["autocast"])
        if mode == "fp32":
            return nullcontext()
        dtype = torch.float16 if mode == "float16" else torch.bfloat16
        return torch.autocast(device_type=self.producer.device.type, dtype=dtype)

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
        for _ in range(int(self.config.validation["batches"])):
            batches = self._batches(phase, global_step, validation=True)
            prepared = _PreparedStep(global_step, batches, False, 0.0)
            try:
                with torch.no_grad(), self._autocast(phase):
                    loss, metrics = self.definition.training_objective(
                        model,
                        batches,
                        self._phase_context(
                            phase_index, phase_step, global_step, validation=True
                        ),
                    )
                row = {
                    "step": float(global_step),
                    "phase_index": float(phase_index),
                    "validation/loss": float(loss.detach()),
                }
                validate_objective_outputs(
                    self.definition.descriptor, phase.name, metrics
                )
                for name, value in metrics.items():
                    row[f"validation/{name}"] = (
                        float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    )
                rows.append(row)
            finally:
                self._release_prepared(prepared)
        return rows

    def _component_manifest(self) -> dict[str, Any]:
        descriptor = self.definition.descriptor
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
        optimization_state: Mapping[str, Any],
        coverage: Mapping[str, Mapping[str, Any]],
        validation_rows: list[Mapping[str, float]],
    ) -> TrainingCheckpoint:
        config_value = self.config.to_dict()
        phase_index, phase_step = self.config.locate_step(global_step)
        phase_name = (
            "complete" if phase_index == len(self.config.phases)
            else self.config.phases[phase_index].name
        )
        checkpoint = TrainingCheckpoint(
            self.definition.descriptor.method_key,
            self.definition.descriptor.descriptor_sha256,
            self.definition.descriptor.implementation_sha256,
            self._component_manifest(),
            config_value,
            sha256_json(config_value),
            sha256_json([phase.to_dict() for phase in self.config.phases]),
            self.producer.reference_program_identity,
            self.producer.reference_execution_plan_identity,
            self.producer.native_asset_collection_identity,
            self.producer.query_stream_identity,
            self.producer.source_contracts,
            self.producer.source_snapshot_ids,
            global_step,
            phase_index,
            phase_name,
            phase_step,
            {"policy": self.config.checkpoint_selection, "tail": validation_rows[-1:]},
            self.definition.export_training_state(model),
            optimization_state,
            self._rng_state(),
            self.producer.state_dict(),
            coverage,
            {"rows": validation_rows},
        )
        checkpoint.validate_method(self.definition.descriptor)
        return checkpoint

    def run(
        self,
        *,
        resume: TrainingCheckpoint | None = None,
        stop_at_step: int | None = None,
    ) -> TrainingRunResult:
        self._seed()
        model = self.definition.create_trainable(self.config.model_context).to(
            self.producer.device
        )
        execution_model: nn.Module = model
        ddp_shell: _DDPObjective | None = None
        if dist.is_available() and dist.is_initialized():
            if self.producer.device.type != "cuda":
                raise RuntimeError("DDP training requires a CUDA producer device")
            local_rank = self.producer.device.index
            if local_rank is None:
                raise RuntimeError("DDP CUDA device index is required")
            ddp_shell = _DDPObjective(self.definition, model)
            execution_model = DistributedDataParallel(
                ddp_shell,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        global_step = 0
        validation_rows: list[Mapping[str, float]] = []
        coverage: dict[str, dict[str, Any]] = {
            group: {
                "finite_observed": False,
                "nonzero_gradient_observed": False,
                "parameter_update_observed": False,
                "last_audit_step": -1,
            }
            for group in self.definition.descriptor.parameter_groups
        }
        resume_optimization: Mapping[str, Any] | None = None
        if resume is not None:
            self._validate_resume(resume)
            self.definition.restore_training_state(model, resume.model_state)
            self.producer.load_state_dict(resume.query_stream_state)
            self._restore_rng(resume.rng_state)
            global_step = resume.global_step
            validation_rows = [dict(row) for row in resume.validation_state.get("rows", ())]
            coverage = {
                group: dict(value) for group, value in resume.gradient_coverage.items()
            }
            resume_optimization = resume.phase_optimization_state

        target_step = self.config.total_steps if stop_at_step is None else int(stop_at_step)
        if not global_step <= target_step <= self.config.total_steps:
            raise ValueError("stop_at_step must lie between resume step and total steps")
        if global_step == self.config.total_steps:
            checkpoint = self._checkpoint(model, global_step, {}, coverage, validation_rows)
            return TrainingRunResult(checkpoint, tuple(validation_rows))

        phase_index, phase_step = self.config.locate_step(global_step)
        phase = self.config.phases[phase_index]
        optimizer, scheduler, scaler, active = self._create_phase_optimization(
            model,
            phase,
            phase_step=phase_step,
            state=resume_optimization,
        )
        registry = self.definition.parameter_registry(model)
        metric_rows: list[Mapping[str, float]] = []
        work_units = sum(
            self.config.phases[index].steps
            * sum(route.batch_size * route.direction_count for route in self.config.phases[index].routes)
            for index in range(phase_index)
        ) + phase_step * sum(
            route.batch_size * route.direction_count for route in phase.routes
        )
        run_started = time.perf_counter()
        run_start_step = global_step
        queue: deque[_PreparedStep] = deque()
        bar = self.progress_factory(
            total=target_step - global_step, desc="train", unit="step"
        )
        try:
            while global_step < target_step:
                phase_index, phase_step = self.config.locate_step(global_step)
                phase = self.config.phases[phase_index]
                next_validation = (
                    ((global_step // int(self.config.validation["interval"])) + 1)
                    * int(self.config.validation["interval"])
                )
                phase_end = self.config.phase_start_step(phase_index) + phase.steps
                barrier = min(target_step, phase_end, next_validation)
                while len(queue) < phase.prefetch_depth:
                    next_step = global_step + len(queue)
                    if next_step >= barrier:
                        break
                    prepared = self._prepare_step(phase, next_step)
                    queue.append(prepared)
                    if not prepared.iteration_ended:
                        break
                if not queue:
                    queue.append(self._prepare_step(phase, global_step))
                prepared = queue.popleft()
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
                if self.producer.device.type == "cuda" and (audit or will_log):
                    cuda_events = tuple(
                        torch.cuda.Event(enable_timing=True) for _ in range(4)
                    )
                    cuda_events[0].record()
                forward_started = time.perf_counter()
                try:
                    optimizer.zero_grad(set_to_none=True)
                    with self._autocast(phase):
                        context = self._phase_context(phase_index, phase_step, global_step)
                        if ddp_shell is None:
                            loss, metrics = self.definition.training_objective(
                                model, prepared.batches, context
                            )
                        else:
                            ddp_shell.set_inputs(prepared.batches, context)
                            loss = execution_model()
                            metrics = ddp_shell.last_metrics
                    if cuda_events is not None:
                        cuda_events[1].record()
                    forward_finished = time.perf_counter()
                    validate_objective_outputs(
                        self.definition.descriptor, phase.name, metrics
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
                )
                should_log = (
                    global_step == target_step
                    or phase_step == phase.steps
                    or phase_step % phase.log_interval == 0
                )
                if should_log:
                    elapsed = time.perf_counter() - run_started
                    completed = global_step - run_start_step
                    speed = completed / max(elapsed, 1e-12)
                    row: dict[str, float] = {
                        "step": float(global_step),
                        "phase_index": float(phase_index),
                        "phase_step": float(phase_step),
                        "loss": float(loss.detach()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "work_units": float(work_units),
                        "elapsed_seconds": elapsed,
                        "steps_per_second": speed,
                        "eta_seconds": (target_step - global_step) / max(speed, 1e-12),
                    }
                    if self.producer.device.type == "cuda":
                        row["peak_memory_bytes"] = float(
                            torch.cuda.max_memory_allocated(self.producer.device)
                        )
                    for name, value in metrics.items():
                        row[name] = (
                            float(value.detach())
                            if isinstance(value, torch.Tensor)
                            else float(value)
                        )
                    row.update(timing)
                    metric_rows.append(row)
                    if self.metric_callback is not None:
                        self.metric_callback(row)
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
                    previous_state = self._optimization_state(
                        phase, optimizer, scheduler, scaler, active
                    )
                    if phase.transition is not None:
                        self.definition.apply_phase_transition(
                            model, phase.transition, self.producer.native_assets()
                        )
                    if global_step < self.config.total_steps:
                        next_index, next_step = self.config.locate_step(global_step)
                        next_phase = self.config.phases[next_index]
                        optimizer, scheduler, scaler, active = self._create_phase_optimization(
                            model,
                            next_phase,
                            phase_step=next_step,
                            overlap_state=previous_state,
                        )
                        registry = self.definition.parameter_registry(model)

                needs_validation = (
                    global_step % int(self.config.validation["interval"]) == 0
                    or global_step == self.config.total_steps
                )
                if needs_validation:
                    if queue:
                        raise RuntimeError("prefetch queue crossed a validation boundary")
                    if global_step == self.config.total_steps:
                        validation_phase_index = len(self.config.phases) - 1
                        validation_phase_step = self.config.phases[-1].steps
                    else:
                        validation_phase_index, validation_phase_step = self.config.locate_step(global_step)
                    validation_started = time.perf_counter()
                    new_rows = self._validation_rows(
                        model,
                        validation_phase_index,
                        validation_phase_step,
                        global_step,
                    )
                    validation_seconds = time.perf_counter() - validation_started
                    new_rows = [
                        {
                            **row,
                            "profile/validation_wall_seconds": (
                                validation_seconds if index == 0 else 0.0
                            ),
                        }
                        for index, row in enumerate(new_rows)
                    ]
                    validation_rows.extend(new_rows)
                    if self.metric_callback is not None:
                        for row in new_rows:
                            self.metric_callback(row)

                checkpoint_boundary = boundary and phase.checkpoint_boundary
                if global_step == self.config.total_steps:
                    validate_gradient_coverage(
                        self.definition.descriptor, coverage
                    )
                if needs_validation or checkpoint_boundary:
                    if global_step == self.config.total_steps:
                        optimization_state: Mapping[str, Any] = {}
                    else:
                        current_index, _ = self.config.locate_step(global_step)
                        current_phase = self.config.phases[current_index]
                        optimization_state = self._optimization_state(
                            current_phase, optimizer, scheduler, scaler, active
                        )
                    self._ddp_barrier()
                    if self.checkpoint_callback is not None:
                        self.checkpoint_callback(
                            self._checkpoint(
                                model,
                                global_step,
                                optimization_state,
                                coverage,
                                validation_rows,
                            )
                        )
                    self._ddp_barrier()
        finally:
            while queue:
                self._release_prepared(queue.popleft())
            bar.close()

        if global_step == self.config.total_steps:
            final_optimization: Mapping[str, Any] = {}
        else:
            current_index, _ = self.config.locate_step(global_step)
            final_optimization = self._optimization_state(
                self.config.phases[current_index], optimizer, scheduler, scaler, active
            )
        checkpoint = self._checkpoint(
            model, global_step, final_optimization, coverage, validation_rows
        )
        return TrainingRunResult(checkpoint, tuple(metric_rows + validation_rows))
