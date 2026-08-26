from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np

from ncls.core.scattering import (
    BackendCapability,
    BackendCostModel,
    BackendDescriptor,
    StateStorage,
)
from ncls.learning.artifact_packing import pack_fp16_parameters
from ncls.learning.models import NvidiaNeuralAppearanceModel
from ncls.learning.pipelines import create_pipeline
from ncls.learning.slang import (
    NVIDIA_NEURAL_APPEARANCE_LAYOUT,
    nvidia_neural_appearance_implementation_sha256,
    nvidia_neural_appearance_layout_sha256,
)
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig


NVIDIA_PARAMETER_FIELDS: tuple[tuple[str, str], ...] = (
    ("frameWeight", "frame_w"),
    ("evaluateWeight0", "evaluate_w0"),
    ("evaluateBias0", "evaluate_b0"),
    ("evaluateWeight1", "evaluate_w1"),
    ("evaluateBias1", "evaluate_b1"),
    ("evaluateWeight2", "evaluate_w2"),
    ("evaluateBias2", "evaluate_b2"),
    ("evaluateOutWeight", "evaluate_out_w"),
    ("evaluateOutBias", "evaluate_out_b"),
    ("samplerWeight0", "sampler_w0"),
    ("samplerBias0", "sampler_b0"),
    ("samplerWeight1", "sampler_w1"),
    ("samplerBias1", "sampler_b1"),
    ("samplerWeight2", "sampler_w2"),
    ("samplerBias2", "sampler_b2"),
    ("samplerOutWeight", "sampler_out_w"),
    ("samplerOutBias", "sampler_out_b"),
)


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def pack_nvidia_neural_record(latent: np.ndarray, *, flags: int = 0) -> bytes:
    """编码原方法私有的 32 B z8 record，不混入 candidate 字段。"""

    layout = NVIDIA_NEURAL_APPEARANCE_LAYOUT["compiled_material"]
    fields = layout["fields"]
    values = np.asarray(latent, dtype="<f2")
    if values.shape != (int(layout["latent_count"]),):
        raise ValueError("NVIDIA compiled material requires exactly one z8 latent")
    if not np.isfinite(values).all():
        raise ValueError("NVIDIA compiled material latent must be finite")
    result = bytearray(int(layout["total_bytes"]))
    latent_field = fields["latent"]
    latent_bytes = values.tobytes()
    offset = int(latent_field["offset"])
    result[offset : offset + len(latent_bytes)] = latent_bytes
    struct.pack_into("<I", result, int(fields["layout_version"]["offset"]), 1)
    struct.pack_into("<I", result, int(fields["flags"]["offset"]), int(flags))
    return bytes(result)


def pack_nvidia_neural_shared_parameters(
    model: NvidiaNeuralAppearanceModel,
) -> tuple[bytes, dict[str, dict[str, Any]], dict[str, int]]:
    """把唯一反射布局映射到生产 Slang params 字段。"""

    parameter_names = tuple(name for _, name in NVIDIA_PARAMETER_FIELDS)
    state_names = set(model.state_dict())
    expected = set(parameter_names) | {"latent"}
    if state_names != expected:
        raise ValueError(
            "NVIDIA parameter identity drifted: "
            f"missing={sorted(expected - state_names)}, "
            f"extra={sorted(state_names - expected)}"
        )
    shared, parameter_layout = pack_fp16_parameters(model, parameter_names)
    shader_offsets: dict[str, int] = {}
    for field_name, parameter_name in NVIDIA_PARAMETER_FIELDS:
        parameter_layout[parameter_name]["slang_field"] = field_name
        shader_offsets[field_name] = int(
            parameter_layout[parameter_name]["offset_elements"]
        )
    return shared, parameter_layout, shader_offsets


def _runtime_adapter(
    shader_offsets: Mapping[str, int],
    *,
    record_stride: int,
    state_stride: int,
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    define_fields = {
        "NCLS_NVIDIA_FRAME_WEIGHT_OFFSET": "frameWeight",
        "NCLS_NVIDIA_EVALUATE_WEIGHT0_OFFSET": "evaluateWeight0",
        "NCLS_NVIDIA_EVALUATE_BIAS0_OFFSET": "evaluateBias0",
        "NCLS_NVIDIA_EVALUATE_WEIGHT1_OFFSET": "evaluateWeight1",
        "NCLS_NVIDIA_EVALUATE_BIAS1_OFFSET": "evaluateBias1",
        "NCLS_NVIDIA_EVALUATE_WEIGHT2_OFFSET": "evaluateWeight2",
        "NCLS_NVIDIA_EVALUATE_BIAS2_OFFSET": "evaluateBias2",
        "NCLS_NVIDIA_EVALUATE_OUT_WEIGHT_OFFSET": "evaluateOutWeight",
        "NCLS_NVIDIA_EVALUATE_OUT_BIAS_OFFSET": "evaluateOutBias",
        "NCLS_NVIDIA_SAMPLER_WEIGHT0_OFFSET": "samplerWeight0",
        "NCLS_NVIDIA_SAMPLER_BIAS0_OFFSET": "samplerBias0",
        "NCLS_NVIDIA_SAMPLER_WEIGHT1_OFFSET": "samplerWeight1",
        "NCLS_NVIDIA_SAMPLER_BIAS1_OFFSET": "samplerBias1",
        "NCLS_NVIDIA_SAMPLER_WEIGHT2_OFFSET": "samplerWeight2",
        "NCLS_NVIDIA_SAMPLER_BIAS2_OFFSET": "samplerBias2",
        "NCLS_NVIDIA_SAMPLER_OUT_WEIGHT_OFFSET": "samplerOutWeight",
        "NCLS_NVIDIA_SAMPLER_OUT_BIAS_OFFSET": "samplerOutBias",
    }
    missing = sorted(set(define_fields.values()) - set(shader_offsets))
    if missing:
        raise ValueError(f"NVIDIA runtime adapter is missing reflected offsets: {missing}")
    descriptor = BackendDescriptor(
        backend_id="nvidia-neural-appearance-v1",
        backend_version=1,
        supported_ir_ids=("ncls.layer-stack-ir@1",),
        capabilities=(
            BackendCapability.PREPARE
            | BackendCapability.EVALUATE
            | BackendCapability.SAMPLE
            | BackendCapability.PDF
            | BackendCapability.ANISOTROPIC_FRAME
            | BackendCapability.REVERSE_PDF
        ),
        state_storage=StateStorage.STRUCTURED,
        state_stride=state_stride,
        state_alignment=16,
        deterministic_eval=True,
        bounded_execution=True,
        shader_entry_points={
            "prepare": "INclsScatteringBackend.prepare",
            "evaluate": "INclsScatteringState.evaluate",
            "sample": "INclsScatteringState.sample",
            "pdf": "INclsScatteringState.pdf",
        },
        cost_model=BackendCostModel(
            compiled_material_bytes=record_stride,
            state_bytes_per_pixel=state_stride,
            prepare_parameter_count=int(cost["C_prepare_macs"]),
            data_dependent_loops=False,
        ),
    )
    return {
        "architecture_id": "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1",
        "shader_module": (
            "ncls/backends/nvidia_neural_appearance/"
            "nvidia_neural_appearance.slang"
        ),
        "shader_defines": {
            define: str(shader_offsets[field])
            for define, field in define_fields.items()
        },
        "compiled_material_stride": record_stride,
        "packed_state_stride": state_stride,
        "shared_weight_storage": "float16-little-endian",
        "backend_descriptor": descriptor.to_dict(),
    }


def export_nvidia_neural_compiled_set(
    checkpoint_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """从 LayerStack 离线预算适配 checkpoint 导出原规模网络资产。"""

    checkpoint_file = Path(checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("NVIDIA compiled set output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint.get("pipeline") != "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1":
        raise ValueError("compiled set requires the exact NVIDIA offline adaptation pipeline")
    pipeline = create_pipeline(str(checkpoint["pipeline"]))
    if checkpoint.get("pipeline_sha256") != pipeline.descriptor.sha256:
        raise ValueError("NVIDIA checkpoint pipeline identity mismatch")
    fitted = checkpoint.get("fitted_training_state")
    if not isinstance(fitted, Mapping) or fitted.get("train_only") is not True:
        raise ValueError("NVIDIA compiled set requires a train-only fitted state")
    pipeline.load_training_state(fitted)
    config = TrainingConfig.from_dict(checkpoint["training_config"])
    model = pipeline.create_model(config.model)
    if not isinstance(model, NvidiaNeuralAppearanceModel):
        raise TypeError("NVIDIA pipeline did not create its private model")
    model.load_state_dict(checkpoint["model_state"])
    latent = model.latent.detach().cpu().numpy()
    state_ids = list(map(str, fitted["state_ids"]))
    if latent.shape != (len(state_ids), 8):
        raise ValueError("NVIDIA checkpoint latent/state identity mismatch")
    records = b"".join(pack_nvidia_neural_record(value) for value in latent)
    materials_path = output / "compiled_materials.bin"
    materials_path.write_bytes(records)
    shared, parameter_layout, shader_offsets = pack_nvidia_neural_shared_parameters(model)
    weights_path = output / "shared_weights_fp16.bin"
    weights_path.write_bytes(shared)
    cost = dict(pipeline.parameter_costs(model))
    if len(shared) != int(cost["B_shared"]):
        raise ValueError("packed NVIDIA shared bytes disagree with static cost accounting")
    record_stride = int(
        NVIDIA_NEURAL_APPEARANCE_LAYOUT["compiled_material"]["total_bytes"]
    )
    state_stride = int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["state"]["stride_bytes"])
    manifest: dict[str, Any] = {
        "format_name": "nvidia-neural-appearance-compiled-set",
        "format_version": 1,
        "method_identity": "nvidia-frame-two-lobe-layer-stack-budget-adapted-v1",
        "runtime_class": cost["runtime_class"],
        "pipeline": checkpoint["pipeline"],
        "sampler": "nvidia-diffuse-ggx9",
        "data_id": checkpoint["data_id"],
        "layout_sha256": nvidia_neural_appearance_layout_sha256(),
        "slang_implementation_sha256": (
            nvidia_neural_appearance_implementation_sha256()
        ),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "implementation_identity": checkpoint.get("implementation_identity"),
        "record_stride": record_stride,
        "prepared_state_stride": state_stride,
        "state_count": len(state_ids),
        "state_ids": state_ids,
        "parameter_layout": parameter_layout,
        "shader_offsets": shader_offsets,
        "files": {
            "compiled_materials.bin": hashlib.sha256(records).hexdigest(),
            "shared_weights_fp16.bin": hashlib.sha256(shared).hexdigest(),
        },
        "cost": {
            **cost,
            "B_shared": len(shared),
            "packed_shared_parameter_count": len(shared) // 2,
        },
    }
    manifest["runtime_adapter"] = _runtime_adapter(
        shader_offsets,
        record_stride=record_stride,
        state_stride=state_stride,
        cost=manifest["cost"],
    )
    manifest["compiled_set_id"] = _sha256_json(manifest)
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


__all__ = [
    "NVIDIA_PARAMETER_FIELDS",
    "export_nvidia_neural_compiled_set",
    "pack_nvidia_neural_record",
    "pack_nvidia_neural_shared_parameters",
]
