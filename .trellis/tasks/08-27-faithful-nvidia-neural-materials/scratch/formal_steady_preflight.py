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
torch.empty(1, device=source.device)
torch.cuda.reset_peak_memory_stats(source.device)
runner = TrainingRunner(definition, source, config)
runner._seed()
model = definition.create_trainable(config.model_context).to(source.device)
definition.configure_lifecycle(model, runner._lifecycle(20_000))
optimizer, _ = runner._optimizer_and_scheduler(model)


def mark(start: float) -> tuple[float, float]:
    torch.cuda.synchronize(source.device)
    now = time.perf_counter()
    return now, now - start


rows = []
try:
    for global_step in (20_000, 20_001):
        batches = None
        try:
            started = time.perf_counter()
            batches = runner._batches(global_step)
            point, data_seconds = mark(started)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = definition.training_objective(
                model, batches, runner._lifecycle(global_step)
            )
            next_point, forward_seconds = mark(point)
            loss.backward()
            runner._finite_gradients(model)
            point, backward_seconds = mark(next_point)
            optimizer.step()
            _, optimizer_seconds = mark(point)
            rows.append({
                "global_step": global_step,
                "data_seconds": data_seconds,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "reference_subqueries": int(
                    batches["evaluator"].tensors["sample_count"].sum()
                ),
                "loss": float(loss.detach()),
                "metrics": {
                    name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    for name, value in metrics.items()
                },
            })
        finally:
            if batches is not None:
                runner._release_batches(batches)
    print(json.dumps({
        "kind": "diagnostic-steady-preflight",
        "config_sha256": config.sha256,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(source.device),
        "rows": rows,
    }, ensure_ascii=False, indent=2))
finally:
    source.close()
