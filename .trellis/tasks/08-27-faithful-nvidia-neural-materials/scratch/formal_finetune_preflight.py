from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from ncls.cli import _batch_source
from ncls.learning.methods import get_method
from ncls.learning.training import TrainingConfig, TrainingRunner


config = TrainingConfig.load(Path("configs/learning/nvidia-rta2024-materialx-formal.json"))
definition = get_method(config.method_key)
source = _batch_source(config)
runner = TrainingRunner(definition, source, config)
runner._seed()
model = definition.create_trainable(config.model_context).to(source.device)
definition.configure_lifecycle(model, runner._lifecycle(0))
optimizer, _ = runner._optimizer_and_scheduler(model)
torch.cuda.reset_peak_memory_stats(source.device)


def mark(start: float) -> tuple[float, float]:
    torch.cuda.synchronize(source.device)
    now = time.perf_counter()
    return now, now - start


rows = []
try:
    started = time.perf_counter()
    definition.materialize_latent(model, source.materialization_features())
    point, materialization_seconds = mark(started)
    for global_step in (config.materialization_step, config.materialization_step + 1):
        batches = None
        try:
            batches = runner._batches(global_step)
            next_point, data_seconds = mark(point)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = definition.training_objective(
                model, batches, runner._lifecycle(global_step)
            )
            point, forward_seconds = mark(next_point)
            loss.backward()
            runner._finite_gradients(model)
            next_point, backward_seconds = mark(point)
            optimizer.step()
            point, optimizer_seconds = mark(next_point)
            rows.append(
                {
                    "global_step": global_step,
                    "data_seconds": data_seconds,
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "loss": float(loss.detach()),
                    "metrics": {
                        name: float(value.detach())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                        for name, value in metrics.items()
                    },
                }
            )
        finally:
            if batches is not None:
                runner._release_batches(batches)
    print(
        json.dumps(
            {
                "kind": "diagnostic-finetune-preflight",
                "config_sha256": config.sha256,
                "materialization_seconds": materialization_seconds,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(source.device),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
finally:
    source.close()
