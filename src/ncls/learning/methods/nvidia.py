from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ncls.core.scattering import BackendCapability
from ncls.core.source import SourceSnapshot
from ncls.data.training_batch import TrainingBatch
from ncls.learning.artifact_packing import pack_fp16_parameters
from ncls.learning.method import (
    MaterialPayload,
    MethodDefinition,
    MethodDescriptor,
    RuntimePayload,
    SourceAdaptationContract,
    TensorField,
)
from ncls.learning.models.nvidia_neural_appearance import NvidiaNeuralAppearanceModel
from ncls.learning.objectives import sampler_cross_entropy
from ncls.learning.slang.nvidia_layout import NVIDIA_NEURAL_APPEARANCE_LAYOUT
from ncls.paths import PROJECT_ROOT


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

_DEFINE_FIELDS = {
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

_TENSOR_SHAPES: tuple[tuple[str, tuple[int | str, ...]], ...] = (
    ("latent", ("state", 8)),
    ("frame_w", (12, 8)),
    ("evaluate_w0", (64, 20)),
    ("evaluate_b0", (64,)),
    ("evaluate_w1", (64, 64)),
    ("evaluate_b1", (64,)),
    ("evaluate_w2", (64, 64)),
    ("evaluate_b2", (64,)),
    ("evaluate_out_w", (3, 64)),
    ("evaluate_out_b", (3,)),
    ("sampler_w0", (32, 11)),
    ("sampler_b0", (32,)),
    ("sampler_w1", (32, 32)),
    ("sampler_b1", (32,)),
    ("sampler_w2", (32, 32)),
    ("sampler_b2", (32,)),
    ("sampler_out_w", (9, 32)),
    ("sampler_out_b", (9,)),
)


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "src/ncls/learning/models/nvidia_neural_appearance.py",
        PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang",
        PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_mlp.slang",
        PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance.slang",
        PROJECT_ROOT / "shaders/ncls/scattering/nvidia_proposal.slang",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little") + relative)
        digest.update(len(payload).to_bytes(8, "little") + payload)
    return digest.hexdigest()


def _module_closure(entry: Path) -> dict[str, bytes]:
    include_pattern = re.compile(rb'^\s*#include\s+"([^"]+)"', re.MULTILINE)
    shader_root = PROJECT_ROOT / "shaders"
    pending = [entry.resolve()]
    result: dict[str, bytes] = {}
    while pending:
        path = pending.pop()
        try:
            relative = path.relative_to(shader_root).as_posix()
        except ValueError as error:
            raise ValueError(f"NVIDIA shader dependency escapes shader root: {path}") from error
        if relative in result:
            continue
        payload = path.read_bytes()
        result[relative] = payload
        for match in include_pattern.finditer(payload):
            include = match.group(1).decode("utf-8")
            dependency = (path.parent / include).resolve()
            if dependency.is_file():
                pending.append(dependency)
    return result


def _pack_record(latent: torch.Tensor) -> bytes:
    layout = NVIDIA_NEURAL_APPEARANCE_LAYOUT["compiled_material"]
    values = latent.detach().cpu().numpy().astype("<f2", copy=False)
    if values.shape != (int(layout["latent_count"]),) or not np.isfinite(values).all():
        raise ValueError("NVIDIA compiled material requires one finite z8 latent")
    result = bytearray(int(layout["total_bytes"]))
    offset = int(layout["fields"]["latent"]["offset"])
    result[offset : offset + values.nbytes] = values.tobytes()
    struct.pack_into("<I", result, int(layout["fields"]["layout_version"]["offset"]), 1)
    struct.pack_into("<I", result, int(layout["fields"]["flags"]["offset"]), 0)
    return bytes(result)


class NvidiaMethodDefinition(MethodDefinition):
    descriptor = MethodDescriptor(
        "nvidia-neural-appearance",
        1,
        "NVIDIA neural appearance（当前预算适配诊断）",
        _implementation_sha256(),
        (SourceAdaptationContract("ncls.layer-stack@1", 1, ("/",), "recompile"),),
        (
            "source_index", "wo", "wi", "target", "solid_angle_weight",
            "reference_pdf", "sample_count", "rng_seed", "query_role",
        ),
        tuple(TensorField(name, "float32", shape) for name, shape in _TENSOR_SHAPES),
        "ncls.scattering-backend@1",
        int(
            BackendCapability.PREPARE
            | BackendCapability.EVALUATE
            | BackendCapability.SAMPLE
            | BackendCapability.PDF
            | BackendCapability.ANISOTROPIC_FRAME
            | BackendCapability.REVERSE_PDF
        ),
        {
            "maximum_prepare_steps": 4,
            "maximum_evaluate_steps": 4,
            "maximum_state_bytes": int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["state"]["stride_bytes"]),
            "maximum_reads": 24,
        },
        {
            "runtime_class": "diagnostic",
            "B_asset": int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["compiled_material"]["total_bytes"]),
            "B_shared": 2 * (9859 + 96 + 2793),
            "C_prepare_macs": int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["evaluator"]["frame_macs"] + NVIDIA_NEURAL_APPEARANCE_LAYOUT["sampler"]["prepare_macs"]),
            "C_eval_macs": int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["evaluator"]["evaluate_macs"]),
        },
    )

    def create_trainable(self, context: Mapping[str, Any]) -> nn.Module:
        if set(context) != {"state_count"}:
            raise ValueError("NVIDIA trainable context must contain only state_count")
        return NvidiaNeuralAppearanceModel(state_count=int(context["state_count"]))

    def configure_phase(self, model: nn.Module, phase: str) -> None:
        if not isinstance(model, NvidiaNeuralAppearanceModel):
            raise TypeError("NVIDIA method requires NvidiaNeuralAppearanceModel")
        if phase not in {"evaluator", "joint", "sampler"}:
            raise ValueError(f"unsupported NVIDIA training phase {phase!r}")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                phase == "joint"
                or (phase == "sampler" and name.startswith("sampler_"))
                or (phase == "evaluator" and not name.startswith("sampler_"))
            )

    def training_objective(
        self,
        model: nn.Module,
        batch: TrainingBatch,
        phase: str,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if not isinstance(model, NvidiaNeuralAppearanceModel):
            raise TypeError("NVIDIA method requires NvidiaNeuralAppearanceModel")
        values = batch.tensors
        prediction_f = model(values["source_index"], values["wo"], values["wi"])
        cosine = torch.clamp(values["wi"][..., 2:3], min=1e-4)
        evaluator_loss = torch.mean(
            torch.abs(torch.log1p(prediction_f * cosine) - torch.log1p(torch.clamp(values["target"], min=0.0)))
        )
        if phase == "evaluator":
            return evaluator_loss, {"evaluator_log1p_l1": evaluator_loss.detach()}
        proposal_pdf, _ = model.sampler_pdf_with_head(
            values["source_index"], values["wo"], values["wi"], "nvidia-diffuse-ggx9"
        )
        sampler_loss, relative_kl = sampler_cross_entropy(
            prediction_f.detach(), values["wi"], values["solid_angle_weight"], proposal_pdf
        )
        loss = sampler_loss if phase == "sampler" else evaluator_loss + sampler_loss
        return loss, {
            "evaluator_log1p_l1": evaluator_loss.detach(),
            "sampler_cross_entropy": sampler_loss.detach(),
            "sampler_relative_kl": relative_kl.detach(),
        }

    def export_training_state(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        if not isinstance(model, NvidiaNeuralAppearanceModel):
            raise TypeError("NVIDIA method requires NvidiaNeuralAppearanceModel")
        result = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        expected = {field.name for field in self.descriptor.tensor_state_schema}
        if set(result) != expected:
            raise ValueError("NVIDIA training tensor mapping disagrees with descriptor")
        return result

    def restore_training_state(self, model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        if not isinstance(model, NvidiaNeuralAppearanceModel):
            raise TypeError("NVIDIA method requires NvidiaNeuralAppearanceModel")
        expected = {field.name for field in self.descriptor.tensor_state_schema}
        if set(state) != expected:
            raise ValueError("NVIDIA checkpoint tensor mapping disagrees with descriptor")
        model.load_state_dict(dict(state), strict=True)

    def compile_runtime(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        state = checkpoint.get("model_state")
        if not isinstance(state, Mapping) or not isinstance(state.get("latent"), torch.Tensor):
            raise ValueError("NVIDIA runtime compilation requires validated model_state tensors")
        model = self.create_trainable({"state_count": int(state["latent"].shape[0])})
        self.restore_training_state(model, state)
        names = tuple(name for _, name in NVIDIA_PARAMETER_FIELDS)
        shared, layout = pack_fp16_parameters(model, names)
        offsets = {field: int(layout[name]["offset_elements"]) for field, name in NVIDIA_PARAMETER_FIELDS}
        defines = {define: str(offsets[field]) for define, field in _DEFINE_FIELDS.items()}
        module = "ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance.slang"
        closure = _module_closure(PROJECT_ROOT / "shaders" / module)
        return RuntimePayload(
            module,
            closure,
            {"shared-weights": shared},
            {
                "shared-weights": {
                    "dtype": "float16",
                    "shape": [len(shared) // 2],
                    "stride": 2,
                    "alignment": 4,
                    "usage": "gNclsRuntimeWeights",
                }
            },
            self.descriptor.capabilities,
            defines,
        )

    def compile_material(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> MaterialPayload:
        self.descriptor.adaptation_contract(snapshot)
        state_ids = checkpoint.get("source_state_ids")
        model_state = checkpoint.get("model_state")
        if not isinstance(state_ids, (list, tuple)) or not isinstance(model_state, Mapping):
            raise ValueError("NVIDIA material compilation requires source_state_ids and model_state")
        try:
            index = list(map(str, state_ids)).index(snapshot.snapshot_id)
        except ValueError as error:
            raise ValueError("NVIDIA checkpoint has no latent for this source snapshot") from error
        latent = model_state.get("latent")
        if not isinstance(latent, torch.Tensor) or index >= latent.shape[0]:
            raise ValueError("NVIDIA checkpoint latent mapping is invalid")
        payload = _pack_record(latent[index])
        return MaterialPayload(
            snapshot.snapshot_id,
            {"compiled-material": payload},
            {
                "compiled-material": {
                    "dtype": "uint8",
                    "shape": [len(payload)],
                    "stride": len(payload),
                    "alignment": 16,
                    "usage": "gNclsCompiledMaterials",
                }
            },
        )


METHOD_DEFINITION = NvidiaMethodDefinition()
