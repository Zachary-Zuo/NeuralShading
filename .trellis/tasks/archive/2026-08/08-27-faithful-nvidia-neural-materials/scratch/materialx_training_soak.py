from __future__ import annotations

import argparse

import torch
from tqdm.auto import tqdm

from ncls.cli import _batch_source
from ncls.learning.methods import get_method
from ncls.learning.training import TrainingConfig, TrainingRunner
from ncls.paths import PROJECT_ROOT


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--start-step", type=int, default=20000)
args = parser.parse_args()

config = TrainingConfig.load(
    PROJECT_ROOT / "configs/learning/nvidia-rta2024-materialx-formal.json"
)
definition = get_method(config.method_key)
source = _batch_source(config)
runner = TrainingRunner(definition, source, config)
runner._seed()
model = definition.create_trainable(config.model_context).to(source.device)
definition.configure_lifecycle(model, runner._lifecycle(args.start_step))
optimizer, _ = runner._optimizer_and_scheduler(model)

try:
    progress = tqdm(range(args.iterations), unit="step", dynamic_ncols=True)
    for offset in progress:
        global_step = args.start_step + offset
        batches = None
        try:
            batches = runner._batches(global_step)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = definition.training_objective(
                model, batches, runner._lifecycle(global_step)
            )
            loss.backward()
            runner._finite_gradients(model)
            optimizer.step()
            if (offset + 1) % 20 == 0:
                progress.set_postfix(
                    loss=f"{float(loss.detach()):.6f}",
                    evaluator=f"{float(metrics['evaluator_log1p_l1']):.6f}",
                )
        finally:
            if batches is not None:
                runner._release_batches(batches)
    torch.cuda.synchronize(source.device)
    print("materialx-training-soak-ok", args.iterations, float(loss.detach()))
finally:
    source.close()
