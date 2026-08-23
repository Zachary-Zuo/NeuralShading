from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ncls.learning.evaluation.evaluator import evaluate_model, tensor_batch
from ncls.learning.pipelines import LearningPipeline, create_pipeline

from .checkpoint import CHECKPOINT_FORMAT, CHECKPOINT_VERSION, save_checkpoint_atomic
from .config import TrainingConfig


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
) -> dict[str, Any]:
    return {
        "format_name": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "pipeline_id": pipeline.descriptor.pipeline_id,
        "pipeline_contract": pipeline.descriptor.to_dict(),
        "pipeline_contract_sha256": pipeline.descriptor.sha256,
        "architecture_id": pipeline.descriptor.architecture_id,
        "representation_id": pipeline.descriptor.representation_id,
        "feature_contract_id": pipeline.descriptor.feature_transform_id,
        "dataset_id": store.dataset.manifest.dataset_id,
        "training_config": config.to_dict(),
        "training_config_sha256": config.resolved_sha256,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_metrics": validation_metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": numpy_rng.bit_generator.state,
    }


def train(
    dataset_path: Path | str,
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
    pipeline = create_pipeline(config.pipeline_id)
    if config.research_stage != pipeline.descriptor.research_role:
        raise ValueError("training config research_stage does not match the registered pipeline role")
    store = pipeline.open_store(str(dataset_path))
    lifecycle_indices = {
        role: pipeline.lifecycle_indices(store, role)
        for role in ("train", "validation", "test")
    }
    if len(lifecycle_indices["train"]) == 0 or len(lifecycle_indices["validation"]) == 0:
        raise ValueError("training requires nonempty train and validation lifecycle partitions")
    model = pipeline.create_model(config.model_parameters).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = hashlib.sha256(
        f"{store.dataset.manifest.dataset_id}\0{config.resolved_sha256}\0{created_at}".encode("utf-8")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "format_name": "ncls.training-run",
        "format_version": 2,
        "run_id": run_id,
        "status": "running",
        "created_at": created_at,
        "completed_at": None,
        "source_git_commit": _git_commit(),
        "dataset_id": store.dataset.manifest.dataset_id,
        "pipeline_id": pipeline.descriptor.pipeline_id,
        "candidate_id": pipeline.descriptor.candidate_id,
        "research_role": pipeline.descriptor.research_role,
        "pipeline_contract": pipeline.descriptor.to_dict(),
        "pipeline_contract_sha256": pipeline.descriptor.sha256,
        "architecture_id": pipeline.descriptor.architecture_id,
        "representation_id": pipeline.descriptor.representation_id,
        "feature_contract": dict(pipeline.feature_contract),
        "training_config_sha256": config.resolved_sha256,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "selection_split": "validation",
        "partition_policy_id": pipeline.descriptor.partition_policy_id,
        "lifecycle_query_group_counts": {
            role: int(len(indices)) for role, indices in lifecycle_indices.items()
        },
        "held_out_test_accessed": False,
        "checkpoints": {},
    }
    _write_json_atomic(output / "run_manifest.json", manifest)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"))
    validation_history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_hash: str | None = None
    last_hash: str | None = None
    start_time = time.perf_counter()
    latest_validation: dict[str, Any] | None = None
    try:
        for step in range(1, config.steps + 1):
            model.train()
            indices = store.sample_batch_indices(lifecycle_indices["train"], config.batch_size, np_rng)
            batch = tensor_batch(store.batch(indices), device)
            optimizer.zero_grad(set_to_none=True)
            prediction = pipeline.predict(model, batch, store, device)
            loss = pipeline.training_loss(prediction, batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            writer.add_scalar("train/loss", float(loss.detach()), step)
            writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)

            should_validate = step % config.validation_interval == 0 or step == config.steps
            if should_validate:
                latest_validation = evaluate_model(
                    model,
                    pipeline,
                    store,
                    lifecycle_indices["validation"],
                    device,
                    batch_size=config.batch_size,
                    max_query_groups=config.max_validation_query_groups,
                )
                record = {"step": step, **latest_validation}
                validation_history.append(record)
                writer.add_scalar("validation/loss", latest_validation["loss"], step)
                for metric_name, summary in latest_validation.items():
                    if not isinstance(summary, dict):
                        continue
                    for summary_name, value in summary.items():
                        writer.add_scalar(f"validation/{metric_name}_{summary_name}", value, step)
                metric_name, summary_name = config.selection_metric.split(".", 1)
                score = float(latest_validation[metric_name][summary_name])
                if score < best_score:
                    best_score = score
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
                        ),
                    )

            if step % config.checkpoint_interval == 0 or step == config.steps:
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
                    ),
                )
        writer.flush()
        (output / "validation_history.json").write_text(
            json.dumps(validation_history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["seconds"] = time.perf_counter() - start_time
        manifest["best_validation"] = {
            "selection_metric": config.selection_metric,
            "value": best_score,
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
    return manifest
