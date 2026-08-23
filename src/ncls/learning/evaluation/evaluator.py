from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ncls.data import SPLIT_NAMES
from ncls.learning.data import LayerStackReferenceStore
from ncls.learning.features import FEATURE_CONTRACT_ID
from ncls.learning.models import create_model
from ncls.learning.prediction import predict_legacy_ltc_k2_response
from ncls.learning.training.checkpoint import load_checkpoint
from ncls.learning.training.config import TrainingConfig

from .metrics import directional_relative_l1, response_loss, summarize


def tensor_batch(raw: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(values, device=device) for name, values in raw.items()}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    store: LayerStackReferenceStore,
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
    lights = torch.as_tensor(store.lights, dtype=torch.float32, device=device)
    losses: list[float] = []
    relative_parts: list[np.ndarray] = []
    noise_parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(selected), batch_size):
        batch = tensor_batch(store.batch(selected[start : start + batch_size]), device)
        prediction = predict_legacy_ltc_k2_response(model, batch, lights)
        target = batch["mean"].float()
        standard_error = batch["standard_error"].float()
        losses.append(float(response_loss(prediction, target, standard_error)))
        relative_parts.append(directional_relative_l1(prediction, target).cpu().numpy())
        replica_noise = torch.sum(
            torch.abs(batch["replica_mean_a"].float() - batch["replica_mean_b"].float()),
            dim=(1, 2),
        ) / torch.clamp(torch.sum(torch.abs(target), dim=(1, 2)), min=1e-8)
        noise_parts.append(replica_noise.cpu().numpy())
    relative = np.concatenate(relative_parts)
    replica_noise = np.concatenate(noise_parts)
    return {
        "query_group_count": int(len(selected)),
        "loss": float(np.mean(losses)),
        "relative_l1": summarize(relative),
        "replica_relative_l1": summarize(replica_noise),
    }


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
    store = LayerStackReferenceStore(dataset_path)
    if checkpoint["dataset_id"] != store.dataset.manifest.dataset_id:
        raise ValueError("checkpoint dataset_id does not match the requested dataset")
    if checkpoint["feature_contract_id"] != FEATURE_CONTRACT_ID:
        raise ValueError("checkpoint feature contract is unsupported")
    model = create_model(config.architecture_id, width=config.width).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_model(
        model,
        store,
        store.split_indices[split],
        device,
        batch_size=config.batch_size,
        max_query_groups=max_query_groups,
    )
    result = {
        "dataset_id": store.dataset.manifest.dataset_id,
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_step": int(checkpoint["step"]),
        "split": split,
        "metrics": metrics,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
