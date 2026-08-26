from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

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


def _write_json_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def current_git_commit() -> str:
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


def _parameter_statistics(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    """每步强制检查权重有限；收敛报告只在validation step持久化摘要。"""

    maximum = 0.0
    for parameter in parameters:
        detached = parameter.detach()
        if not bool(torch.isfinite(detached).all()):
            raise RuntimeError("training parameters became non-finite")
        if detached.numel() > 0:
            maximum = max(maximum, float(detached.abs().max()))
    return {"all_finite": True, "maximum_absolute_value": maximum}


def _gradient_statistics(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    has_nonzero = False
    maximum = 0.0
    gradient_count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_count += 1
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("training gradients became non-finite")
        if gradient.numel() > 0:
            local_maximum = float(gradient.detach().abs().max())
            maximum = max(maximum, local_maximum)
            has_nonzero = has_nonzero or local_maximum > 0.0
    if gradient_count == 0 or not has_nonzero:
        raise RuntimeError("training step produced no finite non-zero gradient")
    return {
        "all_finite": True,
        "parameter_tensor_count": gradient_count,
        "has_nonzero": True,
        "maximum_absolute_value": maximum,
    }


def capture_implementation_identity(
    pipeline: LearningPipeline,
    *,
    sampler: str | None = None,
) -> dict[str, Any]:
    """把dirty正式run绑定到实际文件内容；最终commit只在selection收口时追加。"""

    result: dict[str, Any] = {
        "pipeline_descriptor_sha256": pipeline.descriptor.sha256,
    }
    compiler = pipeline.descriptor.runtime.get("compiler")
    if compiler not in {
        "unified-slang-core-v1",
        "nvidia-neural-appearance-slang-v1",
    }:
        result["identity_sha256"] = _sha256_json(result)
        return result
    if compiler == "unified-slang-core-v1":
        from ncls.learning.slang import (
            unified_layout_sha256 as layout_sha256,
            unified_slang_implementation_sha256 as slang_sha256,
        )

        relative_paths = (
            "src/ncls/learning/data.py",
            "src/ncls/learning/models/unified_neural.py",
            "src/ncls/learning/pipelines/appearance_loss.py",
            "src/ncls/learning/pipelines/p1_evaluator.py",
            "src/ncls/learning/pipelines/sampler_objective.py",
            "src/ncls/learning/pipelines/unified_neural.py",
            "src/ncls/learning/training/runner.py",
            "src/ncls/learning/training/sampler_config.py",
            "src/ncls/learning/training/sampler_runner.py",
            "src/ncls/learning/slang/layout.py",
            "src/ncls/learning/slang/session.py",
            "src/ncls/learning/abi/unified_neural_layout_v1.json",
            "shaders/ncls/backends/unified_neural/unified_neural_core.slang",
            "shaders/ncls/backends/unified_neural/unified_neural_layout.slang",
        )
    else:
        from ncls.learning.slang import (
            nvidia_neural_appearance_implementation_sha256 as slang_sha256,
            nvidia_neural_appearance_layout_sha256 as layout_sha256,
        )

        relative_paths = (
            "src/ncls/learning/data.py",
            "src/ncls/learning/models/nvidia_neural_appearance.py",
            "src/ncls/learning/pipelines/base.py",
            "src/ncls/learning/pipelines/nvidia_neural_appearance.py",
            "src/ncls/learning/pipelines/sampler_objective.py",
            "src/ncls/learning/training/runner.py",
            "src/ncls/learning/training/sampler_config.py",
            "src/ncls/learning/training/sampler_runner.py",
            "src/ncls/learning/slang/nvidia_layout.py",
            "src/ncls/learning/slang/session.py",
            "src/ncls/learning/artifact_packing.py",
            "src/ncls/learning/nvidia_neural_artifacts.py",
            "src/ncls/learning/abi/nvidia_neural_appearance_layout_v1.json",
            "shaders/ncls/backends/nvidia_neural_appearance/"
            "nvidia_neural_appearance_core.slang",
            "shaders/ncls/backends/nvidia_neural_appearance/"
            "nvidia_neural_appearance_layout.slang",
            "shaders/ncls/backends/nvidia_neural_appearance/"
            "nvidia_neural_appearance_mlp.slang",
            "shaders/ncls/backends/nvidia_neural_appearance/"
            "nvidia_neural_appearance_pack.slang",
        )
        if sampler == "ltc-k2":
            from ncls.learning.slang import nvidia_matched_ltc_implementation_sha256

            relative_paths += (
                "src/ncls/learning/models/nvidia_matched_ltc.py",
                "shaders/ncls/backends/nvidia_neural_appearance/"
                "nvidia_matched_ltc_sampler.slang",
            )
            result["sampler_adaptation"] = "nvidia-frozen-evaluator-ltc-k2-v1"
            result["sampler_adaptation_implementation_sha256"] = (
                nvidia_matched_ltc_implementation_sha256()
            )
        elif sampler not in {None, "nvidia-diffuse-ggx9"}:
            raise ValueError("unsupported NVIDIA implementation sampler identity")
    files = {
        path: sha256_file(PROJECT_ROOT / path)
        for path in relative_paths
    }
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("cannot capture method implementation git status")
    result.update({
        "slang_implementation_sha256": slang_sha256(),
        "layout_sha256": layout_sha256(),
        "source_files": files,
        "source_tree_dirty_for_identity_files": bool(status.stdout.strip()),
    })
    result["identity_sha256"] = _sha256_json(result)
    return result


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
    implementation_identity: Mapping[str, Any],
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
        "implementation_identity": dict(implementation_identity),
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
    *,
    progress: Callable[[str], None] | None = None,
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
    implementation_identity = capture_implementation_identity(pipeline)
    store = pipeline.open_store(str(data_path))
    training_lifecycle = (
        dict(store.training_lifecycle_contract(config.steps))
        if hasattr(store, "training_lifecycle_contract")
        else None
    )
    early_stopping_floor = max(
        config.minimum_steps,
        0 if training_lifecycle is None else int(training_lifecycle["early_stopping_floor_step"]),
    )
    lifecycle_indices = {
        role: store.select_indices(
            pipeline.lifecycle_indices(store, role),
            config.dataset_selection,
        )
        for role in ("train", "validation", "test")
    }
    if len(lifecycle_indices["train"]) == 0 or len(lifecycle_indices["validation"]) == 0:
        raise ValueError("training requires nonempty train and validation lifecycle partitions")
    if hasattr(store, "prepare_training_partition"):
        store.prepare_training_partition(lifecycle_indices["train"])
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
    optimizer_type = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(
        trainable_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
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
        "source_git_commit": current_git_commit(),
        "implementation_identity": implementation_identity,
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
        "training_lifecycle": training_lifecycle,
        "effective_early_stopping_floor_step": early_stopping_floor,
        "training_target_source_counts": {},
        "auxiliary_training_target_source_counts": {},
        "initialization_validation": None,
        "optimization_trace": None,
        "checkpoints": {},
        "progress": {
            "completed_steps": 0,
            "total_steps": config.steps,
            "last_validation_step": None,
            "updated_at": created_at,
        },
    }
    _write_json_atomic(output / "run_manifest.json", manifest)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"))
    validation_history: list[dict[str, Any]] = []
    optimization_trace: list[dict[str, Any]] = []
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
        if progress is not None:
            progress(f"initial-validation step=0/{config.steps}")
        initial_validation = evaluate_model(
            model,
            pipeline,
            store,
            lifecycle_indices["validation"],
            device,
            batch_size=config.batch_size,
            evaluation_role="validation",
        )
        if not initial_validation["valid"]:
            raise RuntimeError("initial validation quality-v1 sanity failed")
        initial_record = {"step": 0, **initial_validation}
        _write_json_atomic(output / "initialization_validation.json", initial_record)
        manifest["initialization_validation"] = {
            "uri": "initialization_validation.json",
            "sha256": sha256_file(output / "initialization_validation.json"),
            "directional_l1_by_state": list(directional_summary(initial_record)),
        }
        manifest["progress"] = {
            "completed_steps": 0,
            "total_steps": config.steps,
            "last_validation_step": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(output / "run_manifest.json", manifest)
        if progress is not None:
            progress(f"step=0/{config.steps} initialization-validation=complete")
        for step in range(1, config.steps + 1):
            completed_step = step
            stop_requested = False
            model.train()
            if hasattr(store, "training_batch"):
                raw_batch = store.training_batch(
                    lifecycle_indices["train"],
                    config.batch_size,
                    np_rng,
                    step=step,
                    total_steps=config.steps,
                )
            else:
                indices = store.sample_batch_indices(
                    lifecycle_indices["train"], config.batch_size, np_rng
                )
                raw_batch = store.batch(indices, fields=MODEL_BATCH_FIELDS)
            for source in np.asarray(
                raw_batch.get("target_source", np.full(config.batch_size, "base-v5", dtype=object))
            ).tolist():
                counts = manifest["training_target_source_counts"]
                counts[str(source)] = int(counts.get(str(source), 0)) + 1
            batch = tensor_batch(raw_batch, device, fields=MODEL_BATCH_FIELDS)
            auxiliary_raw = pipeline.auxiliary_training_batch(
                store,
                lifecycle_indices["train"],
                config.batch_size,
                np_rng,
                step=step,
                total_steps=config.steps,
            )
            auxiliary_batch = None
            if auxiliary_raw is not None:
                for source in np.asarray(
                    auxiliary_raw.get(
                        "target_source",
                        np.full(config.batch_size, "base-v5", dtype=object),
                    )
                ).tolist():
                    counts = manifest["auxiliary_training_target_source_counts"]
                    counts[str(source)] = int(counts.get(str(source), 0)) + 1
                auxiliary_batch = tensor_batch(
                    auxiliary_raw, device, fields=MODEL_BATCH_FIELDS
                )
            optimizer.zero_grad(set_to_none=True)
            prediction, loss, objective_components = pipeline.training_objective(
                model,
                batch,
                auxiliary_batch,
                store,
                device,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training objective became non-finite")
            loss.backward()
            should_validate = step % config.validation_interval == 0 or step == config.steps
            if should_validate:
                gradient_statistics = _gradient_statistics(trainable_parameters)
                method_gradient_evidence = dict(pipeline.gradient_evidence(model))
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if should_validate:
                parameter_statistics = _parameter_statistics(trainable_parameters)
            if scheduler is not None:
                scheduler.step()

            if should_validate:
                scalar_objective_components = {
                    name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    for name, value in objective_components.items()
                }
                writer.add_scalar("train/loss", float(loss.detach()), step)
                for name, value in scalar_objective_components.items():
                    writer.add_scalar(f"train/objective_{name}", value, step)
                writer.add_scalar("train/gradient_norm", float(gradient_norm), step)
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], step
                )
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
                optimization_trace.append({
                    "step": step,
                    "objective": float(loss.detach()),
                    "gradient_norm_before_clipping": float(gradient_norm),
                    "gradient": gradient_statistics,
                    "method_gradient_evidence": method_gradient_evidence,
                    "parameters": parameter_statistics,
                    "objective_components": scalar_objective_components,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "optimizer_step_skipped": False,
                })
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
                            implementation_identity=implementation_identity,
                        ),
                    )
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
                stop_requested = bool(
                    config.early_stopping_patience is not None
                    and step >= early_stopping_floor
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
                        implementation_identity=implementation_identity,
                    ),
                )
            if should_validate:
                _write_json_atomic(output / "validation_history.json", validation_history)
                _write_json_atomic(output / "optimization_trace.json", optimization_trace)
                manifest["validation_history"] = {
                    "uri": "validation_history.json",
                    "sha256": sha256_file(output / "validation_history.json"),
                    "record_count": len(validation_history),
                    "evaluation_role": "validation",
                }
                manifest["optimization_trace"] = {
                    "uri": "optimization_trace.json",
                    "sha256": sha256_file(output / "optimization_trace.json"),
                    "record_count": len(optimization_trace),
                    "all_finite": True,
                }
                manifest["checkpoints"] = {
                    "best": {"uri": "checkpoints/best.pt", "sha256": best_hash},
                    "last": {"uri": "checkpoints/last.pt", "sha256": last_hash},
                }
                manifest["progress"] = {
                    "completed_steps": step,
                    "total_steps": config.steps,
                    "last_validation_step": step,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_json_atomic(output / "run_manifest.json", manifest)
                if progress is not None:
                    elapsed = time.perf_counter() - start_time
                    progress(
                        f"step={step}/{config.steps} elapsed_seconds={elapsed:.1f} "
                        f"steps_per_second={step / max(elapsed, 1e-9):.2f}"
                    )
            if stop_requested:
                early_stopped = True
                break
        writer.flush()
        _write_json_atomic(output / "validation_history.json", validation_history)
        _write_json_atomic(output / "optimization_trace.json", optimization_trace)
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
        manifest["optimization_trace"] = {
            "uri": "optimization_trace.json",
            "sha256": sha256_file(output / "optimization_trace.json"),
            "record_count": len(optimization_trace),
            "all_finite": all(
                record["gradient"]["all_finite"]
                and record["parameters"]["all_finite"]
                and not record["optimizer_step_skipped"]
                for record in optimization_trace
            ),
        }
        manifest["validation_history"] = {
            "uri": "validation_history.json",
            "sha256": sha256_file(output / "validation_history.json"),
            "record_count": len(validation_history),
            "evaluation_role": "validation",
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
