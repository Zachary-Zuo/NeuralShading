from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ncls.learning.models import UnifiedNeuralModel
from ncls.learning.models.unified_neural import UNIFIED_TOP_FLOAT_FIELDS
from ncls.learning.pipelines import create_pipeline
from ncls.learning.slang import (
    UNIFIED_LAYOUT,
    unified_layout_sha256,
    unified_slang_implementation_sha256,
)
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig
from ncls.learning.unified_artifacts import pack_unified_record


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parameter_values(
    model: UnifiedNeuralModel,
    parameter_layout: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    state = model.state_dict()
    count = sum(int(record["element_count"]) for record in parameter_layout.values())
    result = np.empty(count, dtype=np.float32)
    for name, record in parameter_layout.items():
        if name not in state:
            raise ValueError(f"compiled parameter {name!r} is absent from the model")
        values = state[name].detach().cpu().numpy().astype(np.float32, copy=False)
        shape = list(map(int, record["shape"]))
        offset = int(record["offset_elements"])
        element_count = int(record["element_count"])
        if list(values.shape) != shape or values.size != element_count:
            raise ValueError(f"compiled parameter {name!r} shape drifted")
        result[offset : offset + element_count] = values.reshape(-1)
    return result


def _shader_offsets(
    parameter_layout: Mapping[str, Mapping[str, Any]], sampler: str
) -> dict[str, int]:
    prefix = "nvidia_sampler" if sampler == "nvidia-diffuse-ggx9" else "ltc_sampler"
    names = {
        "gPrepareW0Offset": "prepare_w0",
        "gPrepareB0Offset": "prepare_b0",
        "gPrepareW1Offset": "prepare_w1",
        "gPrepareB1Offset": "prepare_b1",
        "gEvaluatorStateWOffset": "evaluator_state_w",
        "gEvaluatorStateBOffset": "evaluator_state_b",
        "gSamplerWOffset": f"{prefix}_w",
        "gSamplerBOffset": f"{prefix}_b",
        "gEvaluateW0Offset": "evaluate_w0",
        "gEvaluateB0Offset": "evaluate_b0",
        "gEvaluateW1Offset": "evaluate_w1",
        "gEvaluateB1Offset": "evaluate_b1",
        "gEvaluateOutWOffset": "evaluate_out_w",
        "gEvaluateOutBOffset": "evaluate_out_b",
    }
    result = {
        shader_name: int(parameter_layout[parameter_name]["offset_elements"])
        for shader_name, parameter_name in names.items()
    }
    result["gEvaluateW2Offset"] = int(
        parameter_layout.get("evaluate_w2", {"offset_elements": 0})["offset_elements"]
    )
    result["gEvaluateB2Offset"] = int(
        parameter_layout.get("evaluate_b2", {"offset_elements": 0})["offset_elements"]
    )
    return result


def _top_arrays(rows: list[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.asarray([int(row["interface_kind"]) for row in rows], dtype=np.uint32)
    values = []
    for row in rows:
        source = {
            "alphaX": row["alpha"][0],
            "alphaY": row["alpha"][1],
            "relativeIor": row["relative_ior"],
            "etaR": row["eta"][0],
            "etaG": row["eta"][1],
            "etaB": row["eta"][2],
            "kR": row["k"][0],
            "kG": row["k"][1],
            "kB": row["k"][2],
            "colorR": row["color"][0],
            "colorG": row["color"][1],
            "colorB": row["color"][2],
            "tangentRotation": row["tangent_rotation"],
            "reserved": 0.0,
        }
        values.extend(float(source[name]) for name in UNIFIED_TOP_FLOAT_FIELDS)
    return kinds, np.asarray(values, dtype=np.float32)


def _selected_validation_groups(store: Any, pipeline: Any) -> np.ndarray:
    indices = store.partition_indices(pipeline.descriptor.partition_policy_id, "validation")
    raw = _batch_across_shards(store, indices, fields=("state_index", "wo"))
    states = np.asarray(raw["state_index"], dtype=np.int64)
    views = np.asarray(raw["wo"], dtype=np.float32)
    selected: list[np.ndarray] = []
    for state_index in range(store.state_count):
        available = np.flatnonzero(states == state_index)
        if len(available) != 16:
            raise ValueError("unified parity requires 16 validation views per state")
        selected.append(indices[available[np.argmin(views[available, 2])]].copy())
    return np.asarray(selected, dtype=np.int64)


def _batch_across_shards(
    store: Any,
    references: np.ndarray,
    *,
    fields: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """保持reference顺序合并各shard矩形batch，供跨shard parity case选择。"""

    requested = np.asarray(references, dtype=np.int64)
    if requested.ndim != 2 or requested.shape[1] != 2 or not len(requested):
        raise ValueError("unified parity requires nonempty [shard, group] references")
    merged: dict[str, np.ndarray] = {}
    for shard_index in sorted(set(map(int, requested[:, 0].tolist()))):
        positions = np.flatnonzero(requested[:, 0] == shard_index)
        raw = store.batch(requested[positions], fields=fields)
        for name, source in raw.items():
            values = np.asarray(source)
            if name not in merged:
                merged[name] = np.empty(
                    (len(requested), *values.shape[1:]), dtype=values.dtype
                )
            merged[name][positions] = values
    return merged


@torch.no_grad()
def _expected(
    model: UnifiedNeuralModel,
    state_index: torch.Tensor,
    wo: torch.Tensor,
    wi: torch.Tensor,
    sampler: str,
) -> np.ndarray:
    prediction = model(state_index, wo, wi)
    pdf = model.sampler_pdf(state_index, wo, wi, sampler)
    return np.concatenate(
        (
            prediction.detach().cpu().numpy().reshape(-1, 3),
            pdf.detach().cpu().numpy().reshape(-1, 1),
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def _comparison(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), atol)
    return {
        "finite": bool(np.isfinite(actual).all()),
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_error": float(np.max(absolute)),
        "maximum_relative_error": float(np.max(relative)),
        "passed": bool(
            np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=rtol, atol=atol)
        ),
    }


def run_unified_checkpoint_parity(
    data_path: Path | str,
    evaluator_checkpoint_path: Path | str,
    sampler_checkpoint_path: Path | str,
    compiled_manifest_path: Path | str,
    output_path: Path | str,
    *,
    device_name: str = "cuda",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """同一checkpoint在SlangPy与Falcor执行FP32/FP16-packed完整前向。"""

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("formal unified parity requires CUDA")
    evaluator_path = Path(evaluator_checkpoint_path).resolve()
    sampler_path = Path(sampler_checkpoint_path).resolve()
    compiled_path = Path(compiled_manifest_path).resolve()
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    if compiled.get("format_name") != "unified-compiled-material-set":
        raise ValueError("unified parity requires a compiled-material set")
    if compiled.get("layout_sha256") != unified_layout_sha256():
        raise ValueError("compiled set layout identity mismatch")
    if compiled.get("slang_implementation_sha256") != unified_slang_implementation_sha256():
        raise ValueError("compiled set Slang implementation identity mismatch")
    if compiled.get("evaluator_checkpoint_sha256") != sha256_file(evaluator_path):
        raise ValueError("compiled set evaluator checkpoint identity mismatch")
    if compiled.get("sampler_checkpoint_sha256") != sha256_file(sampler_path):
        raise ValueError("compiled set sampler checkpoint identity mismatch")
    compiled_dir = compiled_path.parent
    materials_path = compiled_dir / "compiled_materials.bin"
    weights_path = compiled_dir / "shared_weights_fp16.bin"
    for name, path in {
        "compiled_materials.bin": materials_path,
        "shared_weights_fp16.bin": weights_path,
    }.items():
        if sha256_file(path) != compiled["files"][name]:
            raise ValueError(f"compiled set file hash mismatch: {name}")

    evaluator = load_checkpoint(evaluator_path, map_location=device)
    sampler_checkpoint = load_checkpoint(sampler_path, map_location=device)
    if sampler_checkpoint.get("checkpoint_role") != "unified-sampler-head":
        raise ValueError("unified parity requires a sampler checkpoint")
    config = TrainingConfig.from_dict(evaluator["training_config"])
    pipeline = create_pipeline(str(evaluator["pipeline"]))
    pipeline.load_training_state(evaluator["fitted_training_state"])
    model = pipeline.create_model(config.model).to(device)
    model.load_state_dict(sampler_checkpoint["model_state"])
    if not isinstance(model, UnifiedNeuralModel):
        raise TypeError("unified parity requires UnifiedNeuralModel")
    model.eval()
    sampler = str(sampler_checkpoint["sampler"])
    if compiled.get("pipeline") != evaluator["pipeline"] or compiled.get("sampler") != sampler:
        raise ValueError("compiled set method identity mismatch")
    parameter_layout = compiled.get("parameter_layout")
    if not isinstance(parameter_layout, Mapping):
        raise ValueError("compiled set parameter layout is missing")
    store = pipeline.open_store(str(data_path))
    try:
        if store.data_id != compiled["data_id"] or evaluator["data_id"] != store.data_id:
            raise ValueError("unified parity data identity mismatch")
        selected = _selected_validation_groups(store, pipeline)
        raw = _batch_across_shards(
            store, selected, fields=("state_index", "wo", "wi")
        )
        state_group = np.asarray(raw["state_index"], dtype=np.int64)
        wo_group = np.asarray(raw["wo"], dtype=np.float32)
        wi_group = np.asarray(raw["wi"], dtype=np.float32)
        direction_count = wi_group.shape[1]
        state_flat = np.repeat(state_group, direction_count).astype(np.uint32)
        wo_flat = np.repeat(wo_group, direction_count, axis=0)
        wi_flat = wi_group.reshape(-1, 3)
        state_tensor = torch.as_tensor(state_group, dtype=torch.long, device=device)
        wo_tensor = torch.as_tensor(wo_group, dtype=torch.float32, device=device)
        wi_tensor = torch.as_tensor(wi_group, dtype=torch.float32, device=device)
        expected_fp32 = _expected(model, state_tensor, wo_tensor, wi_tensor, sampler)
        weights_fp32 = _parameter_values(model, parameter_layout)
        latent_fp32 = model.latent.detach().cpu().numpy().astype(np.float32)
        scale_fp32 = model.response_scale.detach().cpu().numpy().astype(np.float32)
        rows = evaluator["fitted_training_state"]["direct_top"]["rows"]
        top_kind, top_fields = _top_arrays(rows)

        stride = int(compiled["record_stride"])
        records = materials_path.read_bytes()
        if len(records) != stride * int(compiled["state_count"]):
            raise ValueError("compiled material byte length is invalid")
        fields = UNIFIED_LAYOUT["compiled_material"]["fields"]
        latent_fp16 = np.stack([
            np.frombuffer(
                records,
                dtype="<f2",
                count=16,
                offset=index * stride + int(fields["latent"]["offset"]),
            ).astype(np.float32)
            for index in range(int(compiled["state_count"]))
        ])
        scale_packed = np.stack([
            np.frombuffer(
                records,
                dtype="<f4",
                count=3,
                offset=index * stride + int(fields["response_scale"]["offset"]),
            ).copy()
            for index in range(int(compiled["state_count"]))
        ])
        flags = (1 if pipeline.descriptor.name == "core-frame-neural-v1" else 0) | (
            2 if sampler == "ltc-k2" else 0
        )
        for index, row in enumerate(rows):
            expected_record = pack_unified_record(
                row, latent_fp16[index], scale_packed[index], flags
            )
            if records[index * stride : (index + 1) * stride] != expected_record:
                raise ValueError("compiled material record does not match checkpoint assets")
        weights_fp16 = np.frombuffer(weights_path.read_bytes(), dtype="<f2").astype(np.float32)
        if len(weights_fp16) != len(weights_fp32):
            raise ValueError("compiled shared weight count is invalid")
        with torch.no_grad():
            for name, record in parameter_layout.items():
                offset = int(record["offset_elements"])
                count = int(record["element_count"])
                values = torch.as_tensor(
                    weights_fp16[offset : offset + count].reshape(record["shape"]),
                    device=device,
                )
                dict(model.named_parameters())[name].copy_(values)
            model.latent.copy_(torch.as_tensor(latent_fp16, device=device))
            model.response_scale.copy_(torch.as_tensor(scale_packed, device=device))
        expected_fp16 = _expected(model, state_tensor, wo_tensor, wi_tensor, sampler)

        output_target = Path(output_path).resolve()
        work = output_target.with_name(output_target.stem + "-work")
        if work.exists():
            raise ValueError("unified parity work directory must be new")
        work.mkdir(parents=True)
        shader_config = {
            "paper": pipeline.spec.runtime_class == "diagnostic",
            "core": pipeline.spec.evaluator == "core-frame-neural-v1",
            "sampler": 0 if sampler == "nvidia-diffuse-ggx9" else 1,
            "offsets": _shader_offsets(parameter_layout, sampler),
        }
        config_path = work / "shader_config.json"
        config_path.write_text(
            json.dumps(shader_config, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        actual: dict[str, np.ndarray] = {}
        for mode, weights, latent, scale in (
            ("fp32", weights_fp32, latent_fp32, scale_fp32),
            ("fp16_packed", weights_fp16, latent_fp16, scale_packed),
        ):
            input_path = work / f"{mode}_input.npz"
            result_path = work / f"{mode}_falcor.npy"
            np.savez(
                input_path,
                weights=weights,
                latent=latent.reshape(-1),
                response_scale=scale.reshape(-1),
                top_kind=top_kind,
                top_fields=top_fields,
                state_index=state_flat,
                wo=wo_flat,
                wi=wi_flat,
            )
            command = [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(PROJECT_ROOT / "scripts/run_falcor_python.ps1"),
                "-m", "ncls.learning.evaluation.parity_falcor_worker",
                "--input", str(input_path), "--config", str(config_path),
                "--output", str(result_path),
            ]
            if progress is not None:
                progress(f"unified-parity mode={mode} cases={len(state_flat)}")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            actual[mode] = np.load(result_path, allow_pickle=False)

        fp32_parity = _comparison(expected_fp32, actual["fp32"], rtol=2e-5, atol=1e-7)
        fp16_falcor_parity = _comparison(
            expected_fp16, actual["fp16_packed"], rtol=2e-5, atol=1e-7
        )
        deployment_evaluator = _comparison(
            expected_fp32[:, :3], expected_fp16[:, :3], rtol=2e-3, atol=2e-6
        )
        report: dict[str, Any] = {
            "format_name": "unified-checkpoint-parity-report",
            "format_version": 1,
            "data_id": store.data_id,
            "pipeline": pipeline.descriptor.name,
            "sampler": sampler,
            "case_count": len(state_flat),
            "coverage": "30 validation states x lowest-wo.z view x 64 directions",
            "evaluator_checkpoint_sha256": sha256_file(evaluator_path),
            "sampler_checkpoint_sha256": sha256_file(sampler_path),
            "compiled_set_id": compiled["compiled_set_id"],
            "slang_implementation_sha256": unified_slang_implementation_sha256(),
            "layout_sha256": unified_layout_sha256(),
            "fp32_slangpy_falcor": fp32_parity,
            "fp16_packed_slangpy_falcor": fp16_falcor_parity,
            "fp16_packed_vs_fp32_evaluator": deployment_evaluator,
            "held_out_test_accessed": False,
        }
        report["passed"] = bool(
            fp32_parity["passed"]
            and fp16_falcor_parity["passed"]
            and deployment_evaluator["passed"]
        )
        report["report_sha256"] = _sha256_json(report)
        _write_json_atomic(output_target, report)
        return report
    finally:
        store.close()


__all__ = ["run_unified_checkpoint_parity"]
