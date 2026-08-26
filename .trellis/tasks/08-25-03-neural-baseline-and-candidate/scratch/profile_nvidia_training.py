from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from ncls.learning.evaluation.evaluator import MODEL_BATCH_FIELDS, tensor_batch
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint
from ncls.learning.training.config import TrainingConfig
from ncls.learning.training.runner import _gradient_statistics, _parameter_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = PROJECT_ROOT / "artifacts/corpus/layer-stack-p1-mollification-training-v1.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts/runs/unified-scattering-03/formal-nvidia-original-seed-20260824/checkpoints/last.pt"
)


def _timed(name: str, function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return name, time.perf_counter() - start, result


def main() -> None:
    device = torch.device("cuda")
    checkpoint = load_checkpoint(CHECKPOINT_PATH, map_location=device)
    config = TrainingConfig.from_dict(checkpoint["training_config"])
    pipeline = create_pipeline(str(checkpoint["pipeline"]))
    store = pipeline.open_store(str(DATA_PATH))
    try:
        pipeline.load_training_state(checkpoint["fitted_training_state"])
        model = pipeline.create_model(config.model).to(device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        indices = store.partition_indices(
            pipeline.descriptor.partition_policy_id, "train"
        )
        rng = np.random.default_rng(config.seed + 991)

        def batches(step: int):
            main_raw = store.training_batch(
                indices,
                config.batch_size,
                rng,
                step=step,
                total_steps=config.steps,
            )
            auxiliary_raw = pipeline.auxiliary_training_batch(
                store,
                indices,
                config.batch_size,
                rng,
                step=step,
                total_steps=config.steps,
            )
            return main_raw, auxiliary_raw

        for warm_step in (21_001, 21_002):
            main_raw, auxiliary_raw = batches(warm_step)
            main_batch = tensor_batch(main_raw, device, fields=MODEL_BATCH_FIELDS)
            auxiliary_batch = tensor_batch(
                auxiliary_raw, device, fields=MODEL_BATCH_FIELDS
            )
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = pipeline.training_objective(
                model, main_batch, auxiliary_batch, store, device
            )
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        rows = []
        for offset in range(5):
            step = 21_003 + offset
            phases: dict[str, float] = {}
            name, elapsed, raw = _timed("cpu_data", lambda: batches(step))
            phases[name] = elapsed
            main_raw, auxiliary_raw = raw
            name, elapsed, batch_pair = _timed(
                "tensor_transfer",
                lambda: (
                    tensor_batch(main_raw, device, fields=MODEL_BATCH_FIELDS),
                    tensor_batch(auxiliary_raw, device, fields=MODEL_BATCH_FIELDS),
                ),
            )
            phases[name] = elapsed
            main_batch, auxiliary_batch = batch_pair
            optimizer.zero_grad(set_to_none=True)
            name, elapsed, objective = _timed(
                "objective_forward",
                lambda: pipeline.training_objective(
                    model, main_batch, auxiliary_batch, store, device
                ),
            )
            phases[name] = elapsed
            _, loss, _ = objective
            name, elapsed, _ = _timed("backward", loss.backward)
            phases[name] = elapsed
            name, elapsed, _ = _timed(
                "gradient_statistics", lambda: _gradient_statistics(trainable)
            )
            phases[name] = elapsed
            name, elapsed, _ = _timed(
                "method_gradient_evidence", lambda: pipeline.gradient_evidence(model)
            )
            phases[name] = elapsed
            name, elapsed, _ = _timed(
                "gradient_clipping",
                lambda: torch.nn.utils.clip_grad_norm_(
                    trainable, config.gradient_clip, error_if_nonfinite=True
                ),
            )
            phases[name] = elapsed
            name, elapsed, _ = _timed("optimizer", optimizer.step)
            phases[name] = elapsed
            name, elapsed, _ = _timed(
                "parameter_statistics", lambda: _parameter_statistics(trainable)
            )
            phases[name] = elapsed
            phases["total"] = sum(phases.values())
            rows.append(phases)
        summary = {
            name: {
                "median_ms": 1000.0 * float(np.median([row[name] for row in rows])),
                "minimum_ms": 1000.0 * float(np.min([row[name] for row in rows])),
                "maximum_ms": 1000.0 * float(np.max([row[name] for row in rows])),
            }
            for name in rows[0]
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
