from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ncls.data import QUERY_ROLE_NAMES, SPLIT_NAMES
from ncls.learning.data import ReferenceQueryStore
from ncls.learning.pipelines import LearningPipeline, create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint
from ncls.learning.training.config import TrainingConfig

from .quality import (
    build_quality_report,
    finalize_quality_report,
    quality_metric_rows,
    write_quality_report,
)


EVALUATION_ROLE_NAMES = (*SPLIT_NAMES, *QUERY_ROLE_NAMES[3:])


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_batch(raw: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(values, device=device) for name, values in raw.items()}


def _append_rows(target: dict[str, list[np.ndarray]], rows: Mapping[str, np.ndarray]) -> None:
    for name, values in rows.items():
        target.setdefault(name, []).append(np.asarray(values))


def _reciprocal_raw_batch(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    group_count, direction_count = np.asarray(raw["wi"]).shape[:2]
    directional = {
        "wi", "lights", "proposal_pdf", "solid_angle_weight", "rng_seed",
        "mean", "variance", "standard_error", "replica_mean_a", "replica_mean_b",
        "sample_count", "valid", "event_flags", "reference_pdf",
        "reciprocal_mean", "reciprocal_variance", "reciprocal_standard_error",
        "reciprocal_sample_count",
    }
    result: dict[str, np.ndarray] = {}
    for name, source in raw.items():
        values = np.asarray(source)
        if name in directional:
            result[name] = values.reshape(group_count * direction_count, 1, *values.shape[2:])
        elif values.ndim and values.shape[0] == group_count:
            result[name] = np.repeat(values, direction_count, axis=0)
        else:
            result[name] = values
    original_views = np.repeat(np.asarray(raw["wo"]), direction_count, axis=0)
    reciprocal_views = np.asarray(raw["wi"]).reshape(group_count * direction_count, 3).copy()
    transmission = reciprocal_views[:, 2] < 0.0
    reciprocal_views[transmission] *= -1.0
    original_views[transmission] *= -1.0
    result["wo"] = reciprocal_views
    result["view"] = reciprocal_views
    result["wi"] = original_views[:, None, :]
    result["lights"] = result["wi"]
    result["mean"] = np.asarray(raw["reciprocal_mean"]).reshape(
        group_count * direction_count, 1, 3
    )
    result["standard_error"] = np.asarray(raw["reciprocal_standard_error"]).reshape(
        group_count * direction_count, 1, 3
    )
    return result


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    pipeline: LearningPipeline,
    store: ReferenceQueryStore,
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    evaluation_role: str,
    max_query_groups: int | None = None,
    provenance_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """用固定 quality-v1 计算线性 `f`；候选不能替换指标。"""

    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim not in {1, 2} or not len(requested):
        raise ValueError("cannot evaluate an empty selection")
    selected = requested
    complete = True
    if max_query_groups is not None and len(requested) > max_query_groups:
        selected = requested[np.linspace(0, len(requested) - 1, max_query_groups, dtype=np.int64)]
        complete = False
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    weighted_loss = 0.0
    for raw in store.iter_batches(selected, batch_size):
        batch = tensor_batch(raw, device)
        prediction_f = pipeline.predict_f(model, batch, store, device)
        if tuple(prediction_f.shape) != tuple(batch["mean"].shape):
            raise ValueError("pipeline predict_f must match the reference [group,direction,RGB] shape")
        count = len(raw["mean"])
        weighted_loss += float(pipeline.training_loss(prediction_f, batch)) * count
        reciprocal_raw = _reciprocal_raw_batch(raw)
        reciprocal_batch = tensor_batch(reciprocal_raw, device)
        reciprocal_prediction = pipeline.predict_f(
            model, reciprocal_batch, store, device
        ).reshape(*raw["mean"].shape)
        rows = quality_metric_rows(
            prediction_f.detach().cpu().numpy(),
            raw,
            reciprocal_prediction.detach().cpu().numpy(),
        )
        _append_rows(parts, rows)
    rows = {name: np.concatenate(values) for name, values in parts.items()}
    tags = tuple(
        tuple(json.loads(value)) for value in store.state_strings("difficulty_tags_json")
    )
    checks = {
        "dataset_hash_verified": True,
        "complete_evaluation_role": complete,
        **store.sanity_checks(),
        **dict(provenance_checks or {}),
    }
    report = build_quality_report(
        rows,
        state_ids=store.state_strings("state_id"),
        family_ids=store.state_strings("family_id"),
        structure_family_ids=store.state_strings("structure_family_id"),
        difficulty_classes=store.state_strings("difficulty_class"),
        difficulty_tags=tags,
        evaluation_cohorts=store.state_strings("evaluation_cohort"),
        data_id=store.data_id,
        evaluation_role=evaluation_role,
        provenance_checks=checks,
    )
    report["query_group_count"] = int(len(selected))
    report["training_loss_diagnostic"] = weighted_loss / len(selected)
    return finalize_quality_report(report)


def evaluate_checkpoint(
    data_path: Path | str,
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
    pipeline = create_pipeline(str(checkpoint.get("pipeline", "")))
    if checkpoint.get("pipeline_sha256") != pipeline.descriptor.sha256:
        raise ValueError("checkpoint learning pipeline identity is unsupported")
    store = pipeline.open_store(str(data_path))
    try:
        if checkpoint.get("data_id") != store.data_id:
            raise ValueError("checkpoint data_id does not match the requested dataset")
        fitted_state = checkpoint.get("fitted_training_state")
        fitted_hash_ok = isinstance(fitted_state, dict) and (
            checkpoint.get("fitted_training_state_sha256") == _sha256_json(fitted_state)
        )
        if not fitted_hash_ok:
            raise ValueError("checkpoint fitted training state is missing or invalid")
        pipeline.load_training_state(fitted_state)
        model = pipeline.create_model(config.model).to(device)
        model.load_state_dict(checkpoint["model_state"])
        selected = store.select_indices(
            pipeline.evaluation_indices(store, split),
            config.dataset_selection,
        )
        report = evaluate_model(
            model,
            pipeline,
            store,
            selected,
            device,
            batch_size=config.batch_size,
            evaluation_role=split,
            max_query_groups=max_query_groups,
            provenance_checks={
                "checkpoint_recovered": True,
                "fitted_state_hash": fitted_hash_ok,
                "fitted_state_train_only": bool(checkpoint.get("fitted_state_train_only")),
            },
        )
        report["checkpoint"] = {
            "uri": str(Path(checkpoint_path)),
            "step": int(checkpoint["step"]),
            "pipeline": pipeline.descriptor.to_dict(),
            "pipeline_sha256": pipeline.descriptor.sha256,
        }
        report["training"] = {
            "stage": config.stage,
            "capacity": config.capacity,
            "steps": config.steps,
            "seed": config.seed,
            "dataset_selection": {
                name: list(values) for name, values in config.dataset_selection.items()
            },
        }
        report["cost"] = dict(pipeline.parameter_costs(model))
        report = finalize_quality_report(report)
        if output_path is not None:
            write_quality_report(output_path, report)
        return report
    finally:
        store.close()
