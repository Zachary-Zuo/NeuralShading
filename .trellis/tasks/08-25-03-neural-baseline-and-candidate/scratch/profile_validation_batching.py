from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

from ncls.learning.evaluation.evaluator import evaluate_model
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint
from ncls.learning.training.config import TrainingConfig
from ncls.learning.training.selection import directional_summary


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = PROJECT_ROOT / "artifacts/corpus/layer-stack-p1-mollification-training-v1.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts/runs/unified-scattering-03/formal-nvidia-original-seed-20260824/checkpoints/best.pt"
)


def _hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        indices = store.select_indices(
            pipeline.lifecycle_indices(store, "validation"),
            config.dataset_selection,
        )
        evaluate_model(
            model,
            pipeline,
            store,
            indices,
            device,
            batch_size=16,
            evaluation_role="validation",
            max_query_groups=16,
        )
        results = []
        for batch_size in (16, 32, 64, 128):
            torch.cuda.synchronize()
            start = time.perf_counter()
            report = evaluate_model(
                model,
                pipeline,
                store,
                indices,
                device,
                batch_size=batch_size,
                evaluation_role="validation",
            )
            torch.cuda.synchronize()
            results.append({
                "batch_size": batch_size,
                "seconds": time.perf_counter() - start,
                "directional_summary": list(directional_summary(report)),
                "report_sha256": _hash(report),
            })
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
