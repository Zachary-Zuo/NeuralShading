from __future__ import annotations

import hashlib
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


EVALUATION_ROLE_NAMES = (*SPLIT_NAMES, "adversarial_probe")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        for name, values in pipeline.additional_metric_distributions(
            model, batch, store, device
        ).items():
            metric_parts.setdefault(name, []).append(np.asarray(values))
    result: dict[str, Any] = {
        "query_group_count": int(len(selected)),
        "loss": weighted_loss / len(selected),
    }
    concatenated = {name: np.concatenate(parts) for name, parts in metric_parts.items()}
    result.update({name: summarize(values) for name, values in concatenated.items()})
    query_states = np.asarray(store.dataset.stream["queries/state_index"], dtype=np.int64)[selected]
    state_ids = store.dataset.state_strings("state_id")
    state_assets = store.dataset.state_strings("asset_id")
    state_families = store.dataset.state_strings("family_id")
    state_splits = np.asarray(store.dataset.stream["states/split"], dtype=np.int64)
    selected_families = state_families[query_states]
    result["by_family"] = {
        family_id: {
            name: summarize(values[selected_families == family_id])
            for name, values in concatenated.items()
        }
        for family_id in sorted(set(map(str, selected_families.tolist())))
    }
    result["by_state"] = {
        str(state_ids[state_index]): {
            "asset_id": str(state_assets[state_index]),
            "family_id": str(state_families[state_index]),
            "source_split": SPLIT_NAMES[int(state_splits[state_index])],
            "query_group_count": int(np.count_nonzero(query_states == state_index)),
            "metrics": {
                name: summarize(values[query_states == state_index])
                for name, values in concatenated.items()
            },
        }
        for state_index in sorted(set(map(int, query_states.tolist())))
    }
    selected_splits = np.asarray(
        [SPLIT_NAMES[int(state_splits[state_index])] for state_index in query_states]
    )
    result["by_source_split"] = {
        split_name: {
            name: summarize(values[selected_splits == split_name])
            for name, values in concatenated.items()
        }
        for split_name in SPLIT_NAMES
        if np.any(selected_splits == split_name)
    }
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
    if split not in EVALUATION_ROLE_NAMES:
        raise ValueError(f"evaluation role must be one of {EVALUATION_ROLE_NAMES}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = TrainingConfig.from_dict(checkpoint["training_config"])
    if checkpoint.get("training_config_sha256") != config.resolved_sha256:
        raise ValueError("checkpoint training config hash is unsupported")
    pipeline_id = str(checkpoint.get("pipeline_id", ""))
    pipeline = create_pipeline(pipeline_id)
    if checkpoint.get("pipeline_contract_sha256") != pipeline.descriptor.sha256:
        raise ValueError("checkpoint learning pipeline contract is unsupported")
    store = pipeline.open_store(str(dataset_path))
    try:
        if checkpoint["dataset_id"] != store.dataset.manifest.dataset_id:
            raise ValueError("checkpoint dataset_id does not match the requested dataset")
        if checkpoint["feature_contract_id"] != pipeline.descriptor.feature_transform_id:
            raise ValueError("checkpoint feature contract is unsupported")
        if (
            checkpoint.get("architecture_id") != pipeline.descriptor.architecture_id
            or checkpoint.get("representation_id") != pipeline.descriptor.representation_id
        ):
            raise ValueError("checkpoint model identity is unsupported")
        fitted_training_state = checkpoint.get("fitted_training_state")
        if not isinstance(fitted_training_state, dict):
            raise ValueError("checkpoint fitted training state is missing")
        if checkpoint.get("fitted_training_state_sha256") != _sha256_json(fitted_training_state):
            raise ValueError("checkpoint fitted training state hash is unsupported")
        pipeline.load_training_state(fitted_training_state)
        model = pipeline.create_model(config.model_parameters).to(device)
        model.load_state_dict(checkpoint["model_state"])
        selected_indices = store.select_indices(
            pipeline.evaluation_indices(store, split),
            config.dataset_selection,
        )
        metrics = evaluate_model(
            model,
            pipeline,
            store,
            selected_indices,
            device,
            batch_size=config.batch_size,
            max_query_groups=max_query_groups,
        )
        result = {
            "dataset_id": store.dataset.manifest.dataset_id,
            "checkpoint": str(Path(checkpoint_path)),
            "checkpoint_step": int(checkpoint["step"]),
            "evaluation_role": split,
            "split": split,
            "partition_policy_id": pipeline.descriptor.partition_policy_id,
            "metrics": metrics,
        }
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return result
    finally:
        store.close()
