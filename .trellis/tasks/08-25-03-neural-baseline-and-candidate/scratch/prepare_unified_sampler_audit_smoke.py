from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ncls.learning.evaluation.sampler_correctness import (
    _load_sampler_model,
    _select_cases,
)


DATA = "artifacts/corpus/layer-stack-p1-mollification-training-v1.json"
EVALUATOR = "artifacts/runs/unified-scattering-03/smoke-direct-v1/checkpoints/best.pt"
ROOT = Path(".trellis/tasks/08-25-03-neural-baseline-and-candidate/scratch")


for sampler, checkpoint, name in (
    (
        "nvidia-diffuse-ggx9",
        "artifacts/runs/unified-scattering-03/smoke-direct-ggx9-v3/checkpoints/best.pt",
        "ggx",
    ),
    (
        "ltc-k2",
        "artifacts/runs/unified-scattering-03/smoke-direct-ltc-v1/checkpoints/best.pt",
        "ltc",
    ),
):
    device = torch.device("cuda")
    _, _, _, pipeline, store, model = _load_sampler_model(
        DATA, EVALUATOR, checkpoint, device
    )
    try:
        states, views, prepared = _select_cases(store, pipeline, model, sampler, device)
        np.savez(ROOT / f"sampler-audit-{name}-cases.npz", state_indices=states, views=views, prepared=prepared)
    finally:
        store.close()
