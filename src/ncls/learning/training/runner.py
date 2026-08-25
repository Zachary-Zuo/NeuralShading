from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ncls.learning.evaluation.evaluator import (
    MODEL_BATCH_FIELDS,
    evaluate_model,
    tensor_batch,
)
from ncls.learning.pipelines import LearningPipeline, create_pipeline

from .checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint_atomic,
    sha256_file,
)
from .config import TrainingConfig
from .selection import checkpoint_score, directional_summary


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    pipeline: LearningPipeline,
    store,
    *,
    step: int,
    validation_metrics: dict[str, Any] | None,
    numpy_rng: np.random.Generator,
    fitted_training_state: dict[str, Any],
    initialization: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_name": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "pipeline": pipeline.descriptor.name,
        "pipeline_descriptor": pipeline.descriptor.to_dict(),
        "pipeline_sha256": pipeline.descriptor.sha256,
        "data_id": store.data_id,
        "training_config": config.to_dict(),
        "training_config_sha256": config.resolved_sha256,
        "fitted_training_state": fitted_training_state,
        "fitted_training_state_sha256": _sha256_json(fitted_training_state),
        "fitted_state_train_only": True,
        "initialization": initialization,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": numpy_rng.bit_generator.state,
    }


def train(
    data_path: Path | str,
    run_dir: Path | str,
    config: TrainingConfig,
) -> dict[str, Any]:
    """训练一个明确命名的研究 baseline；训练期间绝不读取 held-out test 指标。"""

    output = Path(run_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("training run directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    (output / "tensorboard").mkdir()
    (output / "training_config.json").write_text(config.to_json(), encoding="utf-8")

    torch.manual_seed(config.seed)
    np_rng = np.random.default_rng(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pipeline = create_pipeline(config.pipeline)
    if config.stage != pipeline.descriptor.stage:
        raise ValueError("training config stage does not match the registered pipeline")
    store = pipeline.open_store(str(data_path))
    lifecycle_indices = {
        role: store.select_indices(
            pipeline.lifecycle_indices(store, role),
            config.dataset_selection,
        )
        for role in ("train", "validation", "test")
    }
    if len(lifecycle_indices["train"]) == 0 or len(lifecycle_indices["validation"]) == 0:
        raise ValueError("training requires nonempty train and validation lifecycle partitions")
    fitted_training_state = dict(pipeline.fit_training_state(store, lifecycle_indices["train"]))
    pipeline.load_training_state(fitted_training_state)
    model = pipeline.create_model(config.model).to(device)
    initialization: dict[str, Any] | None = None
    if config.initialization_checkpoint is not None:
        initialization_path = Path(config.initialization_checkpoint)
        if not initialization_path.is_absolute():
            initialization_path = PROJECT_ROOT / initialization_path
        initialization_path = initialization_path.resolve()
        source_checkpoint = load_checkpoint(initialization_path, map_location=device)
        if source_checkpoint.get("data_id") != store.data_id:
            raise ValueError("initialization checkpoint data_id does not match training data")
        details = dict(pipeline.initialize_model_from_checkpoint(model, source_checkpoint))
        initialization = {
            "uri": config.initialization_checkpoint,
            "sha256": sha256_file(initialization_path),
            "source_pipeline": source_checkpoint.get("pipeline"),
            "source_run_step": source_checkpoint.get("step"),
            **details,
        }
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("training pipeline exposed no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.steps,
            eta_min=config.learning_rate * config.final_learning_rate_fraction,
        )
        if config.learning_rate_schedule == "cosine"
        else None
    )
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = hashlib.sha256(
        f"{store.data_id}\0{config.resolved_sha256}\0{created_at}".encode("utf-8")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "format_name": "training-run",
        "format_version": 1,
        "run_id": run_id,
        "status": "running",
        "created_at": created_at,
        "completed_at": None,
        "source_git_commit": _git_commit(),
        "data_id": store.data_id,
        "pipeline": pipeline.descriptor.name,
        "pipeline_descriptor": pipeline.descriptor.to_dict(),
        "pipeline_sha256": pipeline.descriptor.sha256,
        "capacity": config.capacity,
        "dataset_selection": {
            name: list(values) for name, values in config.dataset_selection.items()
        },
        "fitted_training_state": fitted_training_state,
        "fitted_training_state_sha256": _sha256_json(fitted_training_state),
        "initialization": initialization,
        "training_config_sha256": config.resolved_sha256,
        "model_costs": dict(pipeline.parameter_costs(model)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "device": str(device),
        "selection_split": "validation",
        "partition_policy_id": pipeline.descriptor.partition_policy_id,
        "lifecycle_query_group_counts": {
            role: int(len(indices)) for role, indices in lifecycle_indices.items()
        },
        "lifecycle_source_state_counts": {
            role: int(len(np.unique(store.query_state_indices(indices))))
            for role, indices in lifecycle_indices.items()
        },
        "held_out_test_accessed": False,
        "checkpoints": {},
    }
    _write_json_atomic(output / "run_manifest.json", manifest)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"))
    validation_history: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    minimum_p95 = math.inf
    best_hash: str | None = None
    last_hash: str | None = None
    start_time = time.perf_counter()
    latest_validation: dict[str, Any] | None = None
    validations_without_improvement = 0
    completed_step = 0
    early_stopped = False
    try:
        for step in range(1, config.steps + 1):
            completed_step = step
            stop_requested = False
            model.train()
            indices = store.sample_batch_indices(lifecycle_indices["train"], config.batch_size, np_rng)
            batch = tensor_batch(
                store.batch(indices, fields=MODEL_BATCH_FIELDS), device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = pipeline.predict_f(model, batch, store, device)
            loss = pipeline.training_loss(prediction, batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config.gradient_clip
            )
            optimizer.step()
            writer.add_scalar("train/loss", float(loss.detach()), step)
            writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            if scheduler is not None:
                scheduler.step()

            should_validate = step % config.validation_interval == 0 or step == config.steps
            if should_validate:
                latest_validation = evaluate_model(
                    model,
                    pipeline,
                    store,
                    lifecycle_indices["validation"],
                    device,
                    batch_size=config.batch_size,
                    evaluation_role="validation",
                )
                if not latest_validation["valid"]:
                    raise RuntimeError("validation quality-v1 sanity failed")
                record = {"step": step, **latest_validation}
                validation_history.append(record)
                writer.add_scalar(
                    "validation/loss",
                    latest_validation["training_loss_diagnostic"],
                    step,
                )
                for metric_name, summary in latest_validation.items():
                    if not isinstance(summary, dict):
                        continue
                    for summary_name, value in summary.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"validation/{metric_name}_{summary_name}", value, step
                            )
                minimum_p95 = min(minimum_p95, directional_summary(record)[1])
                score = checkpoint_score(record, minimum_p95, config.checkpoint_selection)
                if best_record is None or score < checkpoint_score(
                    best_record, minimum_p95, config.checkpoint_selection
                ):
                    best_record = record
                    best_hash = save_checkpoint_atomic(
                        output / "checkpoints" / "best.pt",
                        _checkpoint_payload(
                            model,
                            optimizer,
                            config,
                            pipeline,
                            store,
                            step=step,
                            validation_metrics=latest_validation,
                            numpy_rng=np_rng,
                            fitted_training_state=fitted_training_state,
                            initialization=initialization,
                        ),
                    )
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
                stop_requested = bool(
                    config.early_stopping_patience is not None
                    and step >= config.minimum_steps
                    and validations_without_improvement >= config.early_stopping_patience
                )

            if step % config.checkpoint_interval == 0 or step == config.steps or stop_requested:
                last_hash = save_checkpoint_atomic(
                    output / "checkpoints" / "last.pt",
                    _checkpoint_payload(
                        model,
                        optimizer,
                        config,
                        pipeline,
                        store,
                        step=step,
                        validation_metrics=latest_validation,
                        numpy_rng=np_rng,
                        fitted_training_state=fitted_training_state,
                        initialization=initialization,
                    ),
                )
            if stop_requested:
                early_stopped = True
                break
        writer.flush()
        (output / "validation_history.json").write_text(
            json.dumps(validation_history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["seconds"] = time.perf_counter() - start_time
        manifest["completed_steps"] = completed_step
        manifest["early_stopped"] = early_stopped
        manifest["best_validation"] = {
            "selection": f"directional_l1_by_state.{config.checkpoint_selection}",
            "step": None if best_record is None else best_record["step"],
            "value": None if best_record is None else list(directional_summary(best_record)),
        }
        manifest["checkpoints"] = {
            "best": {"uri": "checkpoints/best.pt", "sha256": best_hash},
            "last": {"uri": "checkpoints/last.pt", "sha256": last_hash},
        }
        _write_json_atomic(output / "run_manifest.json", manifest)
    except Exception:
        manifest["status"] = "failed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["seconds"] = time.perf_counter() - start_time
        _write_json_atomic(output / "run_manifest.json", manifest)
        raise
    finally:
        writer.close()
        store.close()
    return manifest
