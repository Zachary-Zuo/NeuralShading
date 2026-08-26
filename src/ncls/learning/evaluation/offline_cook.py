from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ncls.learning.models import UnifiedNeuralModel
from ncls.learning.pipelines import create_pipeline
from ncls.learning.slang import (
    unified_layout_sha256,
    unified_slang_implementation_sha256,
)
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig
from ncls.learning.unified_artifacts import pack_unified_record

from .evaluator import MODEL_BATCH_FIELDS, evaluate_model, tensor_batch


@dataclass(frozen=True)
class UnifiedOfflineCookConfig:
    """冻结shared decoder后，只从base-v5 train query重拟合每state latent。"""

    name: str
    seed: int
    steps: int
    batch_size: int
    learning_rate: float
    gradient_clip: float
    latent_initialization: str
    evaluation_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.name != "unified-offline-cook-v1":
            raise ValueError("unsupported unified offline cook config name")
        if self.seed != 20260824 or self.steps != 2_000 or self.batch_size != 16:
            raise ValueError("unified offline cook budget or seed drifted")
        if self.learning_rate != 0.01 or self.gradient_clip != 5.0:
            raise ValueError("unified offline cook optimizer contract drifted")
        if self.latent_initialization != "zero-v1":
            raise ValueError("unified offline cook latent initialization is unsupported")
        if self.evaluation_roles != ("validation", "dense_slice"):
            raise ValueError("unified offline cook may evaluate only validation and dense_slice")

    @classmethod
    def load(cls, path: Path | str) -> "UnifiedOfflineCookConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(value) != {
            "schema",
            "name",
            "seed",
            "steps",
            "batch_size",
            "learning_rate",
            "gradient_clip",
            "latent_initialization",
            "evaluation_roles",
        }:
            raise ValueError("unified offline cook config fields are not frozen v1")
        if value["schema"] != {"name": "unified-offline-cook", "version": 1}:
            raise ValueError("unsupported unified offline cook schema")
        return cls(
            name=str(value["name"]),
            seed=int(value["seed"]),
            steps=int(value["steps"]),
            batch_size=int(value["batch_size"]),
            learning_rate=float(value["learning_rate"]),
            gradient_clip=float(value["gradient_clip"]),
            latent_initialization=str(value["latent_initialization"]),
            evaluation_roles=tuple(map(str, value["evaluation_roles"])),
        )

    @property
    def sha256(self) -> str:
        return _sha256_json({
            "schema": {"name": "unified-offline-cook", "version": 1},
            "name": self.name,
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "latent_initialization": self.latent_initialization,
            "evaluation_roles": list(self.evaluation_roles),
        })


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _freeze_for_offline_cook(model: UnifiedNeuralModel) -> dict[str, torch.Tensor]:
    frozen: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "latent")
        if name != "latent":
            frozen[name] = parameter
    return frozen


def run_unified_offline_cook(
    data_path: Path | str,
    evaluator_checkpoint_path: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
    *,
    device_name: str = "cuda",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """执行target-visible资产cook；test role从入口和实现中都被排除。"""

    config = UnifiedOfflineCookConfig.load(config_path)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("offline cook output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("formal unified offline cook requires CUDA")
    checkpoint_path = Path(evaluator_checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    training_config = TrainingConfig.from_dict(checkpoint["training_config"])
    pipeline = create_pipeline(str(checkpoint.get("pipeline", "")))
    if checkpoint.get("pipeline_sha256") != pipeline.descriptor.sha256:
        raise ValueError("offline cook checkpoint pipeline identity mismatch")
    if not pipeline.descriptor.name.startswith((
        "nvidia-frame-two-lobe-", "core-frame-neural-"
    )):
        raise ValueError("offline cook requires a unified neural evaluator")
    store = pipeline.open_store(str(data_path))
    report_path = output / "offline_cook_report.json"
    try:
        if checkpoint.get("data_id") != store.data_id:
            raise ValueError("offline cook data identity mismatch")
        fitted_state = checkpoint.get("fitted_training_state")
        if not isinstance(fitted_state, Mapping) or fitted_state.get("train_only") is not True:
            raise ValueError("offline cook requires train-only fitted state")
        pipeline.load_training_state(fitted_state)
        model = pipeline.create_model(training_config.model).to(device)
        model.load_state_dict(checkpoint["model_state"])
        if not isinstance(model, UnifiedNeuralModel):
            raise TypeError("offline cook requires UnifiedNeuralModel")
        frozen = _freeze_for_offline_cook(model)
        shared_before = _tensor_mapping_sha256(frozen)
        latent_before_checkpoint = _tensor_mapping_sha256({"latent": model.latent})
        with torch.no_grad():
            model.latent.zero_()
        latent_initial = _tensor_mapping_sha256({"latent": model.latent})
        optimizer = torch.optim.AdamW(
            [model.latent], lr=config.learning_rate, weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.steps, eta_min=config.learning_rate * 0.05
        )
        rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        train_indices = store.select_indices(
            pipeline.lifecycle_indices(store, "train"),
            training_config.dataset_selection,
        )
        losses: list[dict[str, float | int]] = []
        for step in range(1, config.steps + 1):
            model.train()
            indices = store.sample_batch_indices(train_indices, config.batch_size, rng)
            raw = store.batch(indices, fields=MODEL_BATCH_FIELDS)
            batch = tensor_batch(raw, device, fields=MODEL_BATCH_FIELDS)
            optimizer.zero_grad(set_to_none=True)
            prediction = pipeline.predict_f(model, batch, store, device)
            loss = pipeline.training_loss(prediction, batch)
            loss.backward()
            if model.latent.grad is None or not torch.all(torch.isfinite(model.latent.grad)):
                raise RuntimeError("offline cook latent gradient is missing or non-finite")
            if step == 1 and not torch.any(model.latent.grad != 0.0):
                raise RuntimeError("offline cook latent gradient is identically zero")
            if any(parameter.grad is not None for parameter in frozen.values()):
                raise RuntimeError("offline cook modified a frozen shared parameter")
            gradient_norm = torch.nn.utils.clip_grad_norm_([model.latent], config.gradient_clip)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 100 == 0 or step == config.steps:
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "latent_gradient_norm": float(gradient_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                losses.append(record)
                if progress is not None:
                    progress(
                        f"offline-cook step={step}/{config.steps} "
                        f"loss={record['loss']:.6f}"
                    )
        shared_after = _tensor_mapping_sha256(frozen)
        if shared_after != shared_before:
            raise RuntimeError("offline cook changed frozen shared parameters")
        if not torch.all(torch.isfinite(model.latent)):
            raise RuntimeError("offline cook produced a non-finite latent")
        model.eval()
        role_reports: dict[str, Any] = {}
        for role in config.evaluation_roles:
            indices = store.select_indices(
                pipeline.evaluation_indices(store, role),
                training_config.dataset_selection,
            )
            role_reports[role] = evaluate_model(
                model,
                pipeline,
                store,
                indices,
                device,
                batch_size=training_config.batch_size,
                evaluation_role=role,
                provenance_checks={
                    "offline_cook_shared_frozen": True,
                    "offline_cook_test_not_accessed": True,
                },
            )
        latent_path = output / "cooked_latent_fp32.npy"
        latent = model.latent.detach().cpu().numpy().astype(np.float32)
        np.save(latent_path, latent)
        response_scale = model.response_scale.detach().cpu().numpy().astype(np.float32)
        rows = fitted_state["direct_top"]["rows"]
        flags = 1 if pipeline.descriptor.name == "core-frame-neural-v1" else 0
        records = b"".join(
            pack_unified_record(row, latent[index], response_scale[index], flags)
            for index, row in enumerate(rows)
        )
        materials_path = output / "cooked_materials.bin"
        materials_path.write_bytes(records)
        report: dict[str, Any] = {
            "format_name": "unified-offline-cook-report",
            "format_version": 1,
            "status": "complete",
            "cook_contract": "target-visible-latent-cook@1",
            "config": {
                "uri": str(Path(config_path).resolve()),
                "sha256": config.sha256,
                "steps": config.steps,
                "seed": config.seed,
            },
            "data_id": store.data_id,
            "source_evaluator_checkpoint": {
                "uri": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "step": int(checkpoint["step"]),
                "pipeline": pipeline.descriptor.name,
            },
            "slang_implementation_sha256": unified_slang_implementation_sha256(),
            "layout_sha256": unified_layout_sha256(),
            "training": {
                "query_role": "train",
                "query_group_count": int(len(train_indices)),
                "target_source": "base-v5",
                "optimized_parameters": ["compiled_material.latent_z16"],
                "response_scale_source": "checkpoint train-only fitted state",
                "shared_parameter_sha256_before": shared_before,
                "shared_parameter_sha256_after": shared_after,
                "shared_unchanged": True,
                "latent_checkpoint_sha256": latent_before_checkpoint,
                "latent_initial_sha256": latent_initial,
                "latent_final_sha256": _tensor_mapping_sha256({"latent": model.latent}),
                "trajectory": losses,
            },
            "evaluation_roles": list(config.evaluation_roles),
            "held_out_test_accessed": False,
            "roles": role_reports,
            "assets": {
                "cooked_latent_fp32.npy": sha256_file(latent_path),
                "cooked_materials.bin": hashlib.sha256(records).hexdigest(),
                "record_stride": len(records) // len(rows),
                "state_count": len(rows),
            },
        }
        report["cook_id"] = _sha256_json(report)
        _write_json_atomic(report_path, report)
        return report
    finally:
        store.close()


__all__ = [
    "UnifiedOfflineCookConfig",
    "run_unified_offline_cook",
]
