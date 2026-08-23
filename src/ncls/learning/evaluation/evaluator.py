from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ncls.data import SPLIT_NAMES
from ncls.learning.data import ReferenceQueryStore
from ncls.learning.pipelines import LearningPipeline, create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint
from ncls.learning.training.config import TrainingConfig

from .metrics import summarize


def tensor_batch(raw: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(values, device=device) for name, values in raw.items()}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    pipeline: LearningPipeline,
    store: ReferenceQueryStore,
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    max_query_groups: int | None = None,
) -> dict[str, Any]:
    if len(indices) == 0:
        raise ValueError("cannot evaluate an empty split")
    if max_query_groups is not None and len(indices) > max_query_groups:
        selected = indices[np.linspace(0, len(indices) - 1, max_query_groups, dtype=np.int64)]
    else:
        selected = indices
    weighted_loss = 0.0
    metric_parts: dict[str, list[np.ndarray]] = {}
    model.eval()
    for start in range(0, len(selected), batch_size):
        batch = tensor_batch(store.batch(selected[start : start + batch_size]), device)
        prediction = pipeline.predict(model, batch, store, device)
        count = len(batch["mean"])
        weighted_loss += float(pipeline.training_loss(prediction, batch)) * count
        for name, values in pipeline.metric_distributions(prediction, batch).items():
            metric_parts.setdefault(name, []).append(np.asarray(values))
    result: dict[str, Any] = {
        "query_group_count": int(len(selected)),
        "loss": weighted_loss / len(selected),
    }
    result.update({name: summarize(np.concatenate(parts)) for name, parts in metric_parts.items()})
    return result


def evaluate_checkpoint(
    dataset_path: Path | str,
    checkpoint_path: Path | str,
    *,
    split: str,
    output_path: Path | str | None = None,
    device_name: str | None = None,
    max_query_groups: int | None = None,
) -> dict[str, Any]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = TrainingConfig.from_dict(checkpoint["training_config"])
    pipeline_id = str(checkpoint.get("pipeline_id", ""))
    pipeline = create_pipeline(pipeline_id)
    if checkpoint.get("pipeline_contract_sha256") != pipeline.descriptor.sha256:
        raise ValueError("checkpoint learning pipeline contract is unsupported")
    store = pipeline.open_store(str(dataset_path))
    if checkpoint["dataset_id"] != store.dataset.manifest.dataset_id:
        raise ValueError("checkpoint dataset_id does not match the requested dataset")
    if checkpoint["feature_contract_id"] != pipeline.descriptor.feature_transform_id:
        raise ValueError("checkpoint feature contract is unsupported")
    model = pipeline.create_model(config.model_parameters).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_model(
        model,
        pipeline,
        store,
        pipeline.lifecycle_indices(store, split),
        device,
        batch_size=config.batch_size,
        max_query_groups=max_query_groups,
    )
    result = {
        "dataset_id": store.dataset.manifest.dataset_id,
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_step": int(checkpoint["step"]),
        "split": split,
        "partition_policy_id": pipeline.descriptor.partition_policy_id,
        "metrics": metrics,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
