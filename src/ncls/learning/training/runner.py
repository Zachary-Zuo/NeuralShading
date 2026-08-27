from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ncls.core.identity import sha256_json
from ncls.data.batch_sources import BatchSource
from ncls.data.training_batch import TrainingBatch, TrainingRouteRequest
from ncls.learning.method import MethodDefinition

from .checkpoint import TrainingCheckpoint
from .config import TrainingConfig, TrainingRoute


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint: TrainingCheckpoint
    metrics: tuple[Mapping[str, float], ...]


class TrainingRunner:
    """统一双 route、单 optimizer、可恢复 lifecycle orchestration。"""

    def __init__(
        self,
        definition: MethodDefinition,
        source: BatchSource,
        config: TrainingConfig,
        *,
        progress_factory: Callable[..., Any] = tqdm,
        checkpoint_callback: Callable[[TrainingCheckpoint], None] | None = None,
        metric_callback: Callable[[Mapping[str, float]], None] | None = None,
    ) -> None:
        if definition.descriptor.method_key != config.method_key:
            raise ValueError("training config method_key disagrees with MethodDefinition")
        if source.kind != str(config.batch_source["kind"]):
            raise ValueError("configured batch source kind disagrees with producer")
        definition.validate_training_config(config.to_dict())
        self.definition = definition
        self.source = source
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

    @staticmethod
    def _finite_gradients(model: nn.Module) -> None:
        active = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        gradients = [value for value in active if value is not None]
        if not gradients:
            raise RuntimeError("training objective produced no active gradients")
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError("training objective produced non-finite gradients")
        if not any(bool(torch.any(value != 0)) for value in gradients):
            raise RuntimeError("training objective produced only zero gradients")

    def _lifecycle(self, step: int) -> dict[str, Any]:
        stage = "bootstrap" if step < self.config.materialization_step else "finetune"
        return {
            "stage": stage,
            "global_step": step,
            "materialization_step": self.config.materialization_step,
            "total_steps": self.config.total_steps,
        }

    def _request(
        self,
        route: TrainingRoute,
        step: int,
        *,
        validation: bool = False,
    ) -> TrainingRouteRequest:
        options = dict(route.options)
        options.update(
            {
                "filtering": dict(self.config.filtering),
                "mollification": dict(self.config.mollification),
                "validation": validation,
            }
        )
        name = f"validation:{route.name}" if validation else route.name
        seed_offset = route.seed_offset + (1_000_000_007 if validation else 0)
        return TrainingRouteRequest(
            name,
            route.batch_size,
            route.direction_count,
            step,
            route.query_role,
            self.config.seed + seed_offset,
            options,
        )

    def _batches(self, step: int, *, validation: bool = False) -> dict[str, TrainingBatch]:
        result: dict[str, TrainingBatch] = {}
        try:
            for route in self.config.routes:
                result[route.name] = self.source.next_batch(
                    self._request(route, step, validation=validation)
                )
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
    def _release_batches(batches: Mapping[str, TrainingBatch]) -> None:
        for batch in reversed(tuple(batches.values())):
            batch.release()

    def _optimizer_and_scheduler(
        self, model: nn.Module
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        start = float(self.config.schedule["start"])
        end = float(self.config.schedule["end"])
        total = self.config.total_steps
        parameters = tuple(model.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=start,
            betas=tuple(float(value) for value in self.config.optimizer["betas"]),
            eps=float(self.config.optimizer["epsilon"]),
            weight_decay=float(self.config.optimizer["weight_decay"]),
            fused=any(parameter.device.type == "cuda" for parameter in parameters),
        )

        def multiplier(step: int) -> float:
            position = min(max(step, 0), total) / total
            learning_rate = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * position))
            return learning_rate / start

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

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
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in state["cuda"]]
            )

    def _validate_resume(self, checkpoint: TrainingCheckpoint) -> None:
        checkpoint.validate_method(self.definition.descriptor)
        if checkpoint.training_config_sha256 != self.config.sha256:
            raise ValueError("resume checkpoint training config identity mismatch")
        if checkpoint.data_source_identity != self.source.identity:
            raise ValueError("resume checkpoint data source identity mismatch")
        if not 0 <= checkpoint.step <= self.config.total_steps:
            raise ValueError("resume checkpoint step is outside the configured lifecycle")

    def _validation_rows(
        self,
        model: nn.Module,
        step: int,
    ) -> list[Mapping[str, float]]:
        rows: list[Mapping[str, float]] = []
        for _ in range(int(self.config.validation["batches"])):
            batches = self._batches(step, validation=True)
            try:
                with torch.no_grad():
                    loss, metrics = self.definition.training_objective(
                        model, batches, self._lifecycle(step)
                    )
                row = {"step": float(step), "validation/loss": float(loss)}
                for name, value in metrics.items():
                    scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    row[f"validation/{name}"] = scalar
                rows.append(row)
            finally:
                self._release_batches(batches)
        return rows

    def _checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        global_step: int,
        validation_rows: list[Mapping[str, float]],
    ) -> TrainingCheckpoint:
        config_value = self.config.to_dict()
        phase = "complete" if global_step == self.config.total_steps else self._lifecycle(global_step)["stage"]
        checkpoint = TrainingCheckpoint(
            self.definition.descriptor.method_key,
            self.definition.descriptor.descriptor_sha256,
            self.definition.descriptor.implementation_sha256,
            config_value,
            sha256_json(config_value),
            self.source.identity,
            self.source.source_contracts,
            self.source.source_state_ids,
            global_step,
            phase,
            {"policy": self.config.checkpoint_selection, "tail": validation_rows[-1:]},
            self.definition.export_training_state(model),
            optimizer.state_dict(),
            scheduler.state_dict(),
            {},
            self._rng_state(),
            self._lifecycle(global_step),
            self.source.state_dict(),
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
        model = self.definition.create_trainable(self.config.model_context).to(self.source.device)
        self.definition.configure_lifecycle(model, self._lifecycle(0))
        optimizer, scheduler = self._optimizer_and_scheduler(model)
        global_step = 0
        validation_rows: list[Mapping[str, float]] = []
        if resume is not None:
            self._validate_resume(resume)
            self.definition.restore_training_state(model, resume.model_state)
            optimizer.load_state_dict(resume.optimizer_state)
            scheduler.load_state_dict(resume.scheduler_state)
            self.source.load_state_dict(resume.batch_source_state)
            self._restore_rng(resume.rng_state)
            global_step = resume.step
            validation_rows = [dict(row) for row in resume.validation_state.get("rows", ())]

        expected = (
            self._lifecycle(global_step)["stage"]
            if global_step < self.config.total_steps else "finetune"
        )
        model_stage = str(getattr(model, "lifecycle_stage", expected))
        if model_stage != expected:
            raise ValueError("checkpoint lifecycle stage disagrees with global step")

        metric_rows: list[Mapping[str, float]] = []
        target_step = self.config.total_steps if stop_at_step is None else int(stop_at_step)
        if not global_step <= target_step <= self.config.total_steps:
            raise ValueError("stop_at_step must lie between resume step and total steps")
        remaining = target_step - global_step
        run_start_step = global_step
        run_started = time.perf_counter()
        work_per_step = sum(
            route.batch_size * route.direction_count for route in self.config.routes
        )
        bar = self.progress_factory(total=remaining, desc="train", unit="step")
        try:
            while global_step < target_step:
                lifecycle = self._lifecycle(global_step)
                self.definition.configure_lifecycle(model, lifecycle)
                batches = self._batches(global_step)
                try:
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = self.definition.training_objective(model, batches, lifecycle)
                    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                        raise RuntimeError("training objective must return one finite scalar loss")
                    loss.backward()
                    self._finite_gradients(model)
                    optimizer.step()
                    scheduler.step()
                finally:
                    self._release_batches(batches)
                global_step += 1
                if global_step == self.config.materialization_step:
                    self.definition.materialize_latent(
                        model, self.source.materialization_features()
                    )
                row: dict[str, float] = {
                    "step": float(global_step),
                    "loss": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "work_units": float(global_step * work_per_step),
                }
                if self.source.device.type == "cuda":
                    row["peak_memory_bytes"] = float(
                        torch.cuda.max_memory_allocated(self.source.device)
                    )
                for name, value in metrics.items():
                    row[name] = (
                        float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    )
                elapsed_seconds = time.perf_counter() - run_started
                completed_steps = global_step - run_start_step
                steps_per_second = completed_steps / max(elapsed_seconds, 1e-12)
                row["elapsed_seconds"] = elapsed_seconds
                row["steps_per_second"] = steps_per_second
                row["eta_seconds"] = (
                    (target_step - global_step) / max(steps_per_second, 1e-12)
                )
                metric_rows.append(row)
                if self.metric_callback is not None:
                    self.metric_callback(row)
                interval = int(self.config.validation["interval"])
                if global_step % interval == 0 or global_step == self.config.total_steps:
                    new_validation_rows = self._validation_rows(model, global_step)
                    validation_rows.extend(new_validation_rows)
                    if self.metric_callback is not None:
                        for validation_row in new_validation_rows:
                            self.metric_callback(validation_row)
                    if self.checkpoint_callback is not None:
                        self.checkpoint_callback(
                            self._checkpoint(
                                model, optimizer, scheduler, global_step, validation_rows
                            )
                        )
                bar.set_postfix(
                    {
                        "stage": lifecycle["stage"],
                        "loss": f"{row['loss']:.6g}",
                        "queries": global_step * work_per_step,
                    }
                )
                bar.update(1)
        finally:
            bar.close()

        checkpoint = self._checkpoint(
            model, optimizer, scheduler, global_step, validation_rows
        )
        return TrainingRunResult(checkpoint, tuple(metric_rows + validation_rows))
