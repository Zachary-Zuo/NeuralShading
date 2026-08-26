from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np
import torch

from ncls.core.scattering import (
    BackendCapability,
    BackendCostModel,
    BackendDescriptor,
    StateStorage,
)
from ncls.core.material import (
    DiffuseInterface,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
    pack_layer_interface,
)
from ncls.learning.artifact_packing import pack_fp16_parameters
from ncls.learning.pipelines import create_pipeline
from ncls.learning.slang import (
    UNIFIED_LAYOUT,
    unified_layout_sha256,
    unified_slang_implementation_sha256,
)
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _top_interface(row: Mapping[str, Any]):
    kind = int(row["interface_kind"])
    alpha = tuple(map(float, row["alpha"]))
    color = tuple(map(float, row["color"]))
    if kind == 0:
        return RoughDielectricInterface(
            alpha[0], alpha[1], float(row["relative_ior"]), float(row["tangent_rotation"])
        )
    if kind == 1:
        return RoughConductorInterface(
            alpha[0], alpha[1], tuple(map(float, row["eta"])),
            tuple(map(float, row["k"])), float(row["tangent_rotation"]),
        )
    if kind == 2:
        return DiffuseInterface(color)
    if kind == 3:
        return SheenInterface(color, alpha[0])
    raise ValueError("unsupported compiled top-interface kind")


def pack_unified_record(
    top_row: Mapping[str, Any], latent: np.ndarray, response_scale: np.ndarray, flags: int
) -> bytes:
    layout = UNIFIED_LAYOUT["compiled_material"]
    fields = layout["fields"]
    result = bytearray(int(layout["total_bytes"]))
    top = pack_layer_interface(_top_interface(top_row))
    top_field = fields["top_interface"]
    result[int(top_field["offset"]): int(top_field["offset"]) + int(top_field["size"])] = top
    latent_field = fields["latent"]
    latent_bytes = np.asarray(latent, dtype="<f2").tobytes()
    result[int(latent_field["offset"]): int(latent_field["offset"]) + len(latent_bytes)] = latent_bytes
    scale_field = fields["response_scale"]
    scale_bytes = np.asarray(response_scale, dtype="<f4").tobytes()
    result[int(scale_field["offset"]): int(scale_field["offset"]) + len(scale_bytes)] = scale_bytes
    struct.pack_into("<I", result, int(fields["layout_version"]["offset"]), 1)
    struct.pack_into("<I", result, int(fields["flags"]["offset"]), int(flags))
    if len(result) != int(layout["total_bytes"]):
        raise AssertionError("compiled-material record size drifted from the ABI")
    return bytes(result)


def pack_unified_shared_parameters(
    model: torch.nn.Module,
    sampler_name: str,
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """从实际model反射生成连续FP16权重与offset manifest，不维护手写布局。"""

    if sampler_name not in {"nvidia-diffuse-ggx9", "ltc-k2"}:
        raise ValueError("unsupported unified sampler parameter pack")
    selected_head = (
        "nvidia_sampler_" if sampler_name == "nvidia-diffuse-ggx9" else "ltc_sampler_"
    )
    parameter_names = {
        name for name, _ in model.named_parameters()
        if name != "latent" and (
            not name.startswith(("nvidia_sampler_", "ltc_sampler_"))
            or name.startswith(selected_head)
        )
    }
    return pack_fp16_parameters(model, sorted(parameter_names))


def _runtime_adapter(
    parameter_layout: Mapping[str, Mapping[str, Any]],
    *,
    record_stride: int,
    state_stride: int,
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    """生成 MethodBundle 可直接消费的标准散射接口 specialization 描述。"""

    offset_names = {
        "NCLS_UNIFIED_PREPARE_W0_OFFSET": "prepare_w0",
        "NCLS_UNIFIED_PREPARE_B0_OFFSET": "prepare_b0",
        "NCLS_UNIFIED_PREPARE_W1_OFFSET": "prepare_w1",
        "NCLS_UNIFIED_PREPARE_B1_OFFSET": "prepare_b1",
        "NCLS_UNIFIED_EVALUATOR_STATE_W_OFFSET": "evaluator_state_w",
        "NCLS_UNIFIED_EVALUATOR_STATE_B_OFFSET": "evaluator_state_b",
        "NCLS_UNIFIED_SAMPLER_W_OFFSET": "nvidia_sampler_w",
        "NCLS_UNIFIED_SAMPLER_B_OFFSET": "nvidia_sampler_b",
        "NCLS_UNIFIED_EVALUATE_W0_OFFSET": "evaluate_w0",
        "NCLS_UNIFIED_EVALUATE_B0_OFFSET": "evaluate_b0",
        "NCLS_UNIFIED_EVALUATE_W1_OFFSET": "evaluate_w1",
        "NCLS_UNIFIED_EVALUATE_B1_OFFSET": "evaluate_b1",
        "NCLS_UNIFIED_EVALUATE_OUT_W_OFFSET": "evaluate_out_w",
        "NCLS_UNIFIED_EVALUATE_OUT_B_OFFSET": "evaluate_out_b",
    }
    missing = sorted(set(offset_names.values()) - set(parameter_layout))
    if missing:
        raise ValueError(f"unified runtime adapter is missing reflected parameters: {missing}")
    descriptor = BackendDescriptor(
        backend_id="unified-neural-v1",
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
        "architecture_id": "core-frame-neural-v1",
        "shader_module": "ncls/backends/unified_neural/unified_neural.slang",
        "shader_defines": {
            define: str(parameter_layout[name]["offset_elements"])
            for define, name in offset_names.items()
        },
        "compiled_material_stride": record_stride,
        "packed_state_stride": state_stride,
        "shared_weight_storage": "float16-little-endian",
        "backend_descriptor": descriptor.to_dict(),
    }


def export_unified_compiled_set(
    evaluator_checkpoint_path: Path | str,
    sampler_checkpoint_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """导出03冻结资产；它不是04拥有的generic MethodBundle。"""
    evaluator_path = Path(evaluator_checkpoint_path).resolve()
    sampler_path = Path(sampler_checkpoint_path).resolve()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("compiled set output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    evaluator = load_checkpoint(evaluator_path)
    sampler = load_checkpoint(sampler_path)
    if sampler.get("checkpoint_role") != "unified-sampler-head":
        raise ValueError("compiled set requires a unified sampler-head checkpoint")
    if sampler.get("pipeline") != evaluator.get("pipeline") or sampler.get("data_id") != evaluator.get("data_id"):
        raise ValueError("compiled evaluator and sampler identities disagree")
    if sampler.get("source_evaluator_checkpoint_sha256") != sha256_file(evaluator_path):
        raise ValueError("sampler checkpoint was not trained from this evaluator checkpoint")
    config = TrainingConfig.from_dict(evaluator["training_config"])
    pipeline = create_pipeline(str(evaluator["pipeline"]))
    pipeline.load_training_state(evaluator["fitted_training_state"])
    model = pipeline.create_model(config.model)
    model.load_state_dict(sampler["model_state"])
    state = model.state_dict()
    latent = state["latent"].detach().cpu().numpy()
    response_scale = state["response_scale"].detach().cpu().numpy()
    rows = evaluator["fitted_training_state"]["direct_top"]["rows"]
    state_ids = evaluator["fitted_training_state"]["state_ids"]
    sampler_name = str(sampler["sampler"])
    flags = (1 if evaluator["pipeline"] == "core-frame-neural-v1" else 0) | (
        2 if sampler_name == "ltc-k2" else 0
    )
    records = b"".join(
        pack_unified_record(row, latent[index], response_scale[index], flags)
        for index, row in enumerate(rows)
    )
    materials_path = output / "compiled_materials.bin"
    materials_path.write_bytes(records)
    shared_bytes, parameter_layout = pack_unified_shared_parameters(model, sampler_name)
    shared_path = output / "shared_weights_fp16.bin"
    shared_path.write_bytes(shared_bytes)
    packed_cost = dict(pipeline.parameter_costs(model))
    packed_cost["B_shared"] = len(shared_bytes)
    packed_cost["packed_shared_parameter_count"] = len(shared_bytes) // 2
    packed_cost["selected_sampler_head"] = sampler_name
    record_stride = int(UNIFIED_LAYOUT["compiled_material"]["total_bytes"])
    state_stride = int(packed_cost["state_bytes_per_pixel"])
    manifest: dict[str, Any] = {
        "format_name": "unified-compiled-material-set",
        "format_version": 1,
        "runtime_class": pipeline.parameter_costs(model)["runtime_class"],
        "pipeline": evaluator["pipeline"],
        "sampler": sampler_name,
        "data_id": evaluator["data_id"],
        "layout_sha256": unified_layout_sha256(),
        "slang_implementation_sha256": unified_slang_implementation_sha256(),
        "evaluator_checkpoint_sha256": sha256_file(evaluator_path),
        "sampler_checkpoint_sha256": sha256_file(sampler_path),
        "evaluator_implementation_identity": evaluator.get("implementation_identity"),
        "sampler_implementation_identity": sampler.get("implementation_identity"),
        "record_stride": record_stride,
        "state_count": len(state_ids),
        "state_ids": list(state_ids),
        "parameter_layout": parameter_layout,
        "files": {
            "compiled_materials.bin": hashlib.sha256(records).hexdigest(),
            "shared_weights_fp16.bin": hashlib.sha256(shared_bytes).hexdigest(),
        },
        "cost": packed_cost,
        "runtime_adapter": _runtime_adapter(
            parameter_layout,
            record_stride=record_stride,
            state_stride=state_stride,
            cost=packed_cost,
        ),
    }
    manifest["compiled_set_id"] = _sha256_json(manifest)
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest
