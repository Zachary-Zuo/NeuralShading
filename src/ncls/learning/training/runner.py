from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ncls.core.identity import sha256_json
from ncls.data.batch_sources import BatchSource
from ncls.learning.method import MethodDefinition

from .checkpoint import TrainingCheckpoint
from .config import TrainingConfig


@dataclass(frozen=True)
class TrainingRunResult:
    checkpoint: TrainingCheckpoint
    metrics: tuple[Mapping[str, float], ...]


class TrainingRunner:
    """唯一 optimizer/checkpoint orchestration；方法和 batch producer 都通过合同注入。"""

    def __init__(
        self,
        definition: MethodDefinition,
        source: BatchSource,
        config: TrainingConfig,
        *,
        progress_factory: Callable[..., Any] = tqdm,
    ) -> None:
        if definition.descriptor.method_key != config.method_key:
            raise ValueError("training config method_key disagrees with MethodDefinition")
        expected_kind = str(config.batch_source["kind"])
        if source.kind != expected_kind:
            raise ValueError("configured batch source kind disagrees with producer")
        self.definition = definition
        self.source = source
        self.config = config
        self.progress_factory = progress_factory

    def _seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    @staticmethod
    def _finite_gradients(model: nn.Module) -> None:
        active = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not active or any(value is None for value in active):
            raise RuntimeError("training objective left an active parameter without a gradient")
        gradients = [value for value in active if value is not None]
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError("training objective produced non-finite gradients")
        if not any(bool(torch.any(value != 0)) for value in gradients):
            raise RuntimeError("training objective produced only zero gradients")

    def run(self) -> TrainingRunResult:
        self._seed()
        model = self.definition.create_trainable(self.config.model_context).to(self.source.device)
        metric_rows: list[Mapping[str, float]] = []
        global_step = 0
        final_optimizer: Mapping[str, Any] = {}
        final_phase = "complete"
        for phase in self.config.phases:
            self.definition.configure_phase(model, phase.name)
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not parameters:
                raise RuntimeError(f"training phase {phase.name!r} has no active parameters")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=phase.learning_rate,
                weight_decay=float(self.config.optimizer["weight_decay"]),
            )
            bar = self.progress_factory(total=phase.steps, desc=f"train:{phase.name}", unit="step")
            try:
                for _ in range(phase.steps):
                    batch = self.source.next_batch(self.config.batch_size)
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        loss, metrics = self.definition.training_objective(model, batch, phase.name)
                        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                            raise RuntimeError("training objective must return one finite scalar loss")
                        loss.backward()
                        self._finite_gradients(model)
                        optimizer.step()
                        global_step += 1
                        row = {"step": float(global_step), "loss": float(loss.detach())}
                        for name, value in metrics.items():
                            row[name] = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                        metric_rows.append(row)
                        bar.set_postfix({"loss": f"{row['loss']:.6g}"})
                        bar.update(1)
                    finally:
                        batch.release()
            finally:
                bar.close()
            final_optimizer = optimizer.state_dict()
            final_phase = phase.name
        state = self.definition.export_training_state(model)
        config_value = self.config.to_dict()
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
            final_phase,
            {"policy": self.config.checkpoint_selection, "observed_metrics": metric_rows[-1:]},
            state,
            final_optimizer,
            rng_state={
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        )
        checkpoint.validate_method(self.definition.descriptor)
        return TrainingRunResult(checkpoint, tuple(metric_rows))
