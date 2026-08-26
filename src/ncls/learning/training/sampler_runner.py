from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ncls.learning.evaluation.evaluator import MODEL_BATCH_FIELDS, tensor_batch
from ncls.learning.models import (
    NvidiaNeuralAppearanceLtcAdaptationModel,
    NvidiaNeuralAppearanceModel,
    UnifiedNeuralModel,
    adapt_nvidia_model_for_sampler,
)
from ncls.learning.pipelines import create_pipeline
from ncls.learning.pipelines.sampler_objective import (
    sampler_cross_entropy,
    sampler_cross_entropy_by_group,
)

from .checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint_atomic,
    sha256_file,
)
from .config import TrainingConfig
from .sampler_config import SamplerTrainingConfig
from .runner import capture_implementation_identity, current_git_commit


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_sampler_validation(
    model: (
        NvidiaNeuralAppearanceModel
        | NvidiaNeuralAppearanceLtcAdaptationModel
        | UnifiedNeuralModel
    ),
    pipeline,
    store,
    indices: np.ndarray,
    device: torch.device,
    config: SamplerTrainingConfig,
) -> dict[str, Any]:
    totals = np.zeros(2, dtype=np.float64)
    state_totals = np.zeros((store.state_count, 2), dtype=np.float64)
    state_counts = np.zeros(store.state_count, dtype=np.int64)
    count = 0
    model.eval()
    for raw in store.iter_batches(indices, config.batch_size, fields=MODEL_BATCH_FIELDS):
        batch = tensor_batch(raw, device, fields=MODEL_BATCH_FIELDS)
        evaluator_f = pipeline.predict_f(model, batch, store, device)
        proposal_pdf = model.sampler_pdf(
            batch["state_index"].long(), batch["wo"].float(), batch["wi"].float(), config.sampler
        )
        cross_entropy_rows, relative_kl_rows = sampler_cross_entropy_by_group(
            evaluator_f, batch["wi"], batch["solid_angle_weight"], proposal_pdf
        )
        batch_count = len(raw["mean"])
        rows = np.stack(
            (
                cross_entropy_rows.detach().cpu().numpy(),
                relative_kl_rows.detach().cpu().numpy(),
            ),
            axis=1,
        ).astype(np.float64, copy=False)
        totals += rows.sum(axis=0)
        state_index = np.asarray(raw["state_index"], dtype=np.int64)
        np.add.at(state_totals, state_index, rows)
        np.add.at(state_counts, state_index, 1)
        count += batch_count
    if count == 0 or np.any(state_counts == 0):
        raise ValueError("sampler validation must cover every evaluator state")
    means = state_totals / state_counts[:, None]
    state_ids = list(map(str, store.state_strings("state_id").tolist()))
    return {
        "evaluation_role": "validation",
        "valid": True,
        "evaluator_relative_cross_entropy": float(totals[0] / count),
        "evaluator_relative_kl": float(totals[1] / count),
        "states": {
            state_id: {
                "evaluator_relative_cross_entropy": float(means[index, 0]),
                "evaluator_relative_kl": float(means[index, 1]),
            }
            for index, state_id in enumerate(state_ids)
        },
    }


def train_sampler(
    data_path: Path | str,
    run_dir: Path | str,
    config: SamplerTrainingConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """从冻结best evaluator只训练一个detached sampler head。"""
    output = Path(run_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("sampler run directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    (output / "training_config.json").write_text(config.to_json(), encoding="utf-8")
    evaluator_checkpoint_path = Path(config.evaluator_checkpoint)
    if not evaluator_checkpoint_path.is_absolute():
        evaluator_checkpoint_path = PROJECT_ROOT / evaluator_checkpoint_path
    evaluator_checkpoint_path = evaluator_checkpoint_path.resolve()

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True)
    device = torch.device(config.device)
    source = load_checkpoint(evaluator_checkpoint_path, map_location=device)
    if source.get("pipeline") != config.evaluator_pipeline:
        raise ValueError("sampler source pipeline identity mismatch")
    source_config = TrainingConfig.from_dict(source["training_config"])
    pipeline = create_pipeline(config.evaluator_pipeline)
    store = pipeline.open_store(str(data_path))
    try:
        if source.get("data_id") != store.data_id:
            raise ValueError("sampler source data identity mismatch")
        pipeline.load_training_state(source["fitted_training_state"])
        source_model = pipeline.create_model(source_config.model).to(device)
        source_model.load_state_dict(source["model_state"])
        model = (
            adapt_nvidia_model_for_sampler(source_model, config.sampler).to(device)
            if isinstance(source_model, NvidiaNeuralAppearanceModel)
            else source_model
        )
        if not isinstance(
            model,
            (
                NvidiaNeuralAppearanceModel,
                NvidiaNeuralAppearanceLtcAdaptationModel,
                UnifiedNeuralModel,
            ),
        ):
            raise TypeError("sampler training requires a registered neural evaluator")
        model.reset_sampler_parameters(config.sampler)
        model.set_sampler_training(config.sampler)
        implementation_identity = capture_implementation_identity(
            pipeline, sampler=config.sampler
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        expected = model.sampler_parameter_count(config.sampler)
        if sum(parameter.numel() for parameter in trainable) != expected:
            raise ValueError("sampler detach contract exposed unexpected trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.steps,
            eta_min=config.learning_rate * config.final_learning_rate_fraction,
        )
        train_indices = store.partition_indices(pipeline.descriptor.partition_policy_id, "train")
        validation_indices = store.partition_indices(
            pipeline.descriptor.partition_policy_id, "validation"
        )
        store.prepare_training_partition(train_indices)
        created_at = datetime.now(timezone.utc).isoformat()
        run_id = hashlib.sha256(
            f"{store.data_id}\0{config.resolved_sha256}\0{created_at}".encode("utf-8")
        ).hexdigest()
        manifest: dict[str, Any] = {
            "format_name": "unified-sampler-training-run",
            "format_version": 1,
            "run_id": run_id,
            "status": "running",
            "created_at": created_at,
            "completed_at": None,
            "source_git_commit": current_git_commit(),
            "implementation_identity": implementation_identity,
            "data_id": store.data_id,
            "evaluator_pipeline": config.evaluator_pipeline,
            "evaluator_checkpoint": {
                "uri": config.evaluator_checkpoint,
                "sha256": sha256_file(evaluator_checkpoint_path),
                "step": int(source["step"]),
            },
            "sampler": config.sampler,
            "training_config_sha256": config.resolved_sha256,
            "steps": config.steps,
            "seed": config.seed,
            "trainable_parameter_count": expected,
            "shared_evaluator_detached": True,
            "target_head_reinitialized": True,
            "held_out_test_accessed": False,
            "initialization_validation": None,
            "validation_history": None,
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
        if progress is not None:
            progress(f"initial-validation step=0/{config.steps}")
        initial_validation = {
            "step": 0,
            **evaluate_sampler_validation(
                model, pipeline, store, validation_indices, device, config
            ),
        }
        _write_json_atomic(
            output / "initialization_validation.json", initial_validation
        )
        manifest["initialization_validation"] = {
            "uri": "initialization_validation.json",
            "sha256": sha256_file(output / "initialization_validation.json"),
            "evaluation_role": "validation",
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
        best_value = math.inf
        best_hash = None
        last_hash = None
        validation_history: list[dict[str, Any]] = []
        optimization_trace: list[dict[str, Any]] = []
        start = time.perf_counter()
        for step in range(1, config.steps + 1):
            model.train()
            raw = store.base_training_batch(train_indices, config.batch_size, rng)
            batch = tensor_batch(raw, device, fields=MODEL_BATCH_FIELDS)
            with torch.no_grad():
                evaluator_f = pipeline.predict_f(model, batch, store, device)
            proposal_pdf, head_output = model.sampler_pdf_with_head(
                batch["state_index"].long(), batch["wo"].float(), batch["wi"].float(), config.sampler
            )
            if not head_output.requires_grad or not proposal_pdf.requires_grad:
                raise RuntimeError("sampler head/PDF did not enter the autograd graph")
            loss, _ = sampler_cross_entropy(
                evaluator_f, batch["wi"], batch["solid_angle_weight"], proposal_pdf
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            target_gradients = [parameter.grad for parameter in trainable]
            if any(gradient is None or not torch.isfinite(gradient).all() for gradient in target_gradients):
                raise RuntimeError("sampler target head produced missing or nonfinite gradients")
            if not any(bool(torch.any(gradient != 0.0)) for gradient in target_gradients if gradient is not None):
                raise RuntimeError("sampler target head produced only zero gradients")
            frozen_gradients = [
                parameter.grad for parameter in model.parameters() if not parameter.requires_grad
            ]
            if any(gradient is not None for gradient in frozen_gradients):
                raise RuntimeError("sampler backward modified a frozen evaluator parameter")
            gradient_maximum = max(
                float(gradient.detach().abs().max())
                for gradient in target_gradients
                if gradient is not None
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            parameter_maximum = 0.0
            for parameter in trainable:
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError("sampler training produced non-finite parameters")
                parameter_maximum = max(
                    parameter_maximum, float(parameter.detach().abs().max())
                )
            scheduler.step()
            validation = None
            if step % config.validation_interval == 0 or step == config.steps:
                validation = {
                    "step": step,
                    **evaluate_sampler_validation(
                    model, pipeline, store, validation_indices, device, config
                    ),
                }
                validation_history.append(validation)
                optimization_trace.append({
                    "step": step,
                    "objective": float(loss.detach()),
                    "gradient_norm_before_clipping": float(gradient_norm),
                    "gradient": {
                        "all_finite": True,
                        "has_nonzero": gradient_maximum > 0.0,
                        "maximum_absolute_value": gradient_maximum,
                    },
                    "parameters": {
                        "all_finite": True,
                        "maximum_absolute_value": parameter_maximum,
                    },
                    "optimizer_step_skipped": False,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                })
                cross_entropy = float(validation["evaluator_relative_cross_entropy"])
                if cross_entropy < best_value:
                    best_value = cross_entropy
                    best_hash = save_checkpoint_atomic(
                        output / "checkpoints" / "best.pt",
                        _sampler_checkpoint(
                            model,
                            optimizer,
                            source,
                            config,
                            step,
                            validation,
                            implementation_identity,
                        ),
                    )
            if step % config.checkpoint_interval == 0 or step == config.steps:
                last_hash = save_checkpoint_atomic(
                    output / "checkpoints" / "last.pt",
                    _sampler_checkpoint(
                        model,
                        optimizer,
                        source,
                        config,
                        step,
                        validation,
                        implementation_identity,
                    ),
                )
            if validation is not None:
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
                    elapsed = time.perf_counter() - start
                    progress(
                        f"step={step}/{config.steps} elapsed_seconds={elapsed:.1f} "
                        f"steps_per_second={step / max(elapsed, 1e-9):.2f}"
                    )
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["seconds"] = time.perf_counter() - start
        manifest["completed_steps"] = config.steps
        manifest["best_validation_cross_entropy"] = best_value
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
            "all_finite": all(
                record["gradient"]["all_finite"]
                and record["parameters"]["all_finite"]
                and not record["optimizer_step_skipped"]
                for record in optimization_trace
            ),
        }
        manifest["checkpoints"] = {
            "best": {"uri": "checkpoints/best.pt", "sha256": best_hash},
            "last": {"uri": "checkpoints/last.pt", "sha256": last_hash},
        }
        _write_json_atomic(output / "run_manifest.json", manifest)
        return manifest
    except Exception:
        if "manifest" in locals():
            manifest["status"] = "failed"
            manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
            if "start" in locals():
                manifest["seconds"] = time.perf_counter() - start
            _write_json_atomic(output / "run_manifest.json", manifest)
        raise
    finally:
        store.close()


def _sampler_checkpoint(
    model: (
        NvidiaNeuralAppearanceModel
        | NvidiaNeuralAppearanceLtcAdaptationModel
        | UnifiedNeuralModel
    ),
    optimizer: torch.optim.Optimizer,
    source: Mapping[str, Any],
    config: SamplerTrainingConfig,
    step: int,
    validation: Mapping[str, Any] | None,
    implementation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_name": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "checkpoint_role": "unified-sampler-head",
        "pipeline": config.evaluator_pipeline,
        "pipeline_descriptor": source["pipeline_descriptor"],
        "pipeline_sha256": source["pipeline_sha256"],
        "data_id": source["data_id"],
        "fitted_training_state": source["fitted_training_state"],
        "fitted_training_state_sha256": source["fitted_training_state_sha256"],
        "fitted_state_train_only": True,
        "implementation_identity": dict(implementation_identity),
        "sampler_training_config": config.to_dict(),
        "sampler_training_config_sha256": config.resolved_sha256,
        "source_evaluator_checkpoint_sha256": sha256_file(
            (PROJECT_ROOT / config.evaluator_checkpoint).resolve()
            if not Path(config.evaluator_checkpoint).is_absolute()
            else Path(config.evaluator_checkpoint)
        ),
        "sampler": config.sampler,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_metrics": dict(validation or {}),
    }
