from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ncls.core.scattering import BackendCapability
from ncls.bundle.typed_texture import RGBA16F_DDS_DTYPE, encode_rgba16f_dds
from ncls.core.source import SourceSnapshot
from ncls.learning.batches import EvaluatorBatch, MethodSamplerBatch, OnlineTrainingBatch
from ncls.learning.source_adaptation import NativeAssetCollection
from ncls.learning.artifact_packing import pack_fp16_parameters
from ncls.learning.method import (
    ComponentContract,
    MaterialPayload,
    Method,
    MethodDescriptor,
    RuntimePayload,
    SourceAdaptationContract,
    TensorField,
)
from ncls.learning.methods.nvidia.model import NvidiaModel
from ncls.learning.method import Method
from ncls.learning.source_adapters import MethodSourceAdapter
from ncls.learning.methods.nvidia.data import NvidiaLayerStackSourceAdapter, NvidiaMaterialXSourceAdapter, NvidiaMdlFixedSourceAdapter
from ncls.learning.objectives import sampler_forward_kl_score
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

_TENSOR_SHAPES: tuple[tuple[str, str, tuple[int | str, ...]], ...] = (
    ("encoder_w0", "float32", (64, "native_feature")),
    ("encoder_b0", "float32", (64,)),
    ("encoder_w1", "float32", (64, 64)),
    ("encoder_b1", "float32", (64,)),
    ("encoder_w2", "float32", (64, 64)),
    ("encoder_b2", "float32", (64,)),
    ("encoder_w3", "float32", (64, 64)),
    ("encoder_b3", "float32", (64,)),
    ("encoder_out_w", "float32", (8, 64)),
    ("encoder_out_b", "float32", (8,)),
    ("latent_texels", "float32", ("latent_texel", 8)),
    ("latent_mip_offsets", "int64", ("mip_plus_one",)),
    ("latent_mip_shapes", "int64", ("mip", 2)),
    ("phase_code", "int64", (1,)),
    ("frame_w", "float32", (12, 8)),
    ("evaluate_w0", "float32", (64, 20)),
    ("evaluate_b0", "float32", (64,)),
    ("evaluate_w1", "float32", (64, 64)),
    ("evaluate_b1", "float32", (64,)),
    ("evaluate_w2", "float32", (64, 64)),
    ("evaluate_b2", "float32", (64,)),
    ("evaluate_out_w", "float32", (3, 64)),
    ("evaluate_out_b", "float32", (3,)),
    ("sampler_w0", "float32", (32, 11)),
    ("sampler_b0", "float32", (32,)),
    ("sampler_w1", "float32", (32, 32)),
    ("sampler_b1", "float32", (32,)),
    ("sampler_w2", "float32", (32, 32)),
    ("sampler_b2", "float32", (32,)),
    ("sampler_out_w", "float32", (9, 32)),
    ("sampler_out_b", "float32", (9,)),
)

_PARAMETER_NAMES = tuple(
    name for name, _, _ in _TENSOR_SHAPES
    if name not in {"latent_texels", "latent_mip_offsets", "latent_mip_shapes", "phase_code"}
)

_PARITY_VIEW = (0.0, 0.0, 1.0)
_PARITY_LIGHTS = ((0.0, 0.0, 1.0), (0.6, 0.0, 0.8), (0.0, 0.8, 0.6), (-0.55, 0.35, 0.757))

def _fp16_fma_dense(
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: np.ndarray,
    *,
    activate: bool,
) -> np.ndarray:
    """模拟 regular FP16 shader 的逐列 half FMA 与 half accumulator。"""

    packed_weights = weights.detach().cpu().numpy().astype(np.float16)
    if packed_weights.ndim != 2 or packed_weights.shape[1] != len(inputs):
        raise ValueError("NVIDIA FP16 parity dense shape is invalid")
    if bias is None:
        values = np.zeros(packed_weights.shape[0], dtype=np.float16)
    else:
        values = bias.detach().cpu().numpy().astype(np.float16)
        if values.shape != (packed_weights.shape[0],):
            raise ValueError("NVIDIA FP16 parity bias shape is invalid")
    packed_inputs = np.asarray(inputs, dtype=np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        for column in range(packed_weights.shape[1]):
            # Native half FMA performs the multiply and add before the half
            # destination rounding. The deployment tolerance also covers a
            # backend that lowers this expression to separate half ops.
            values = (
                values.astype(np.float32)
                + packed_weights[:, column].astype(np.float32)
                * np.float32(packed_inputs[column])
            ).astype(np.float16)
    if activate:
        values = np.maximum(values, np.float16(0.0))
    return values


def _normalize_float3(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    length_squared = np.sum(vector * vector, dtype=np.float32)
    if not math.isfinite(float(length_squared)) or length_squared <= 0.0:
        raise ValueError("NVIDIA FP16 parity direction/frame is degenerate")
    return np.asarray(vector / np.float32(np.sqrt(length_squared)), dtype=np.float32)


def _project_frame(
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        (
            np.dot(direction, tangent),
            np.dot(direction, bitangent),
            np.dot(direction, normal),
        ),
        dtype=np.float32,
    )


def _packed_fp16_parity_f(
    state: Mapping[str, torch.Tensor],
    view: tuple[float, float, float],
    lights: tuple[tuple[float, float, float], ...],
) -> list[list[float]]:
    """独立模拟 DDS latent、FP16 weights/MLP 的线性 RGB f 输出。"""

    texels = state.get("latent_texels")
    shapes = state.get("latent_mip_shapes")
    offsets = state.get("latent_mip_offsets")
    if not all(isinstance(value, torch.Tensor) for value in (texels, shapes, offsets)):
        raise ValueError("NVIDIA FP16 parity requires a validated latent hierarchy")
    assert isinstance(texels, torch.Tensor)
    assert isinstance(shapes, torch.Tensor)
    assert isinstance(offsets, torch.Tensor)
    if shapes.ndim != 2 or tuple(shapes.shape[1:]) != (2,) or len(shapes) < 1:
        raise ValueError("NVIDIA FP16 parity latent mip shapes are invalid")
    height, width = map(int, shapes[0].tolist())
    if height < 1 or width < 1 or int(offsets[0]) != 0:
        raise ValueError("NVIDIA FP16 parity base latent extent is invalid")
    corner_indices = torch.tensor(
        (height * width - 1, (height - 1) * width, width - 1, 0),
        dtype=torch.int64,
        device=texels.device,
    )
    corners = texels.index_select(0, corner_indices).detach().cpu().numpy()
    if corners.shape != (4, 8):
        raise ValueError("NVIDIA FP16 parity latent tensor shape is invalid")
    # PackageParity uses uv=(0,0), explicit mip 0 and a wrap/linear sampler.
    # DDS first quantizes every corner to half; the exact 0.5/0.5 footprint
    # then averages the four corners in the texture filtering unit.
    latent = np.mean(corners.astype(np.float16).astype(np.float32), axis=0, dtype=np.float32)
    latent = latent.astype(np.float16).astype(np.float32)

    frame_weights = state.get("frame_w")
    if not isinstance(frame_weights, torch.Tensor):
        raise ValueError("NVIDIA FP16 parity requires frame weights")
    frame_raw = _fp16_fma_dense(frame_weights, None, latent, activate=False).astype(np.float32)
    if frame_raw.shape != (12,):
        raise ValueError("NVIDIA FP16 parity frame shape is invalid")
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    projected_view: list[np.ndarray] = []
    normalized_view = _normalize_float3(np.asarray(view, dtype=np.float32))
    for frame_index in range(2):
        raw = frame_raw[6 * frame_index : 6 * frame_index + 6]
        normal = _normalize_float3(raw[:3] + np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
        tangent = _normalize_float3(raw[3:] + np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
        bitangent = np.asarray(np.cross(normal, tangent), dtype=np.float32)
        frames.append((tangent, bitangent, normal))
        projected_view.append(_project_frame(tangent, bitangent, normal, normalized_view))

    layer_names = (
        ("evaluate_w0", "evaluate_b0"),
        ("evaluate_w1", "evaluate_b1"),
        ("evaluate_w2", "evaluate_b2"),
        ("evaluate_out_w", "evaluate_out_b"),
    )
    expected: list[list[float]] = []
    for light_value in lights:
        light = _normalize_float3(np.asarray(light_value, dtype=np.float32))
        inputs: list[float] = []
        for projected_wo, frame in zip(projected_view, frames, strict=True):
            inputs.extend(float(value) for value in projected_wo)
            inputs.extend(float(value) for value in _project_frame(*frame, light))
        inputs.extend(float(value) for value in latent)
        hidden = np.asarray(inputs, dtype=np.float16)
        for layer_index, (weight_name, bias_name) in enumerate(layer_names):
            weights = state.get(weight_name)
            bias = state.get(bias_name)
            if not isinstance(weights, torch.Tensor) or not isinstance(bias, torch.Tensor):
                raise ValueError(f"NVIDIA FP16 parity requires {weight_name}/{bias_name}")
            hidden = _fp16_fma_dense(
                weights, bias, hidden, activate=layer_index < len(layer_names) - 1
            )
        with np.errstate(over="ignore", invalid="ignore"):
            evaluate_f = np.exp(
                hidden.astype(np.float32) - np.float32(3.0)
            ).astype(np.float16).astype(np.float32)
        if (
            evaluate_f.shape != (3,)
            or not np.isfinite(evaluate_f).all()
            or np.any(evaluate_f < 0.0)
        ):
            raise ValueError("NVIDIA FP16 parity oracle produced an invalid f")
        expected.append([float(value) for value in evaluate_f])
    return expected


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "src/ncls/learning/methods/nvidia/model.py",
        PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_core.slang",
        PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_mlp.slang",
        PROJECT_ROOT / "shaders/ncls/backends/nvidia_neural_appearance/nvidia_neural_appearance_fp16.slang",
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


def _pack_record(width: int, height: int, mip_count: int) -> bytes:
    layout = NVIDIA_NEURAL_APPEARANCE_LAYOUT["compiled_material"]
    if min(width, height, mip_count) < 1:
        raise ValueError("NVIDIA compiled material latent extent must be positive")
    result = bytearray(int(layout["total_bytes"]))
    fields = layout["fields"]
    struct.pack_into("<I", result, int(fields["width"]["offset"]), width)
    struct.pack_into("<I", result, int(fields["height"]["offset"]), height)
    struct.pack_into("<I", result, int(fields["mip_count"]["offset"]), mip_count)
    struct.pack_into("<I", result, int(fields["layout_version"]["offset"]), 2)
    struct.pack_into("<I", result, int(layout["fields"]["flags"]["offset"]), 0)
    return bytes(result)


class NvidiaMethod(Method):
    key = "nvidia"

    def create_source_adapter(self, snapshots, device):
        return _create_source_adapter(snapshots, device)

    descriptor = MethodDescriptor(
        "nvidia-neural-appearance",
        3,
        "NVIDIA Real-Time Neural Appearance functional reproduction",
        _implementation_sha256(),
        (
            SourceAdaptationContract("ncls.layer-stack@1", 1, ("/",), "recompile"),
            SourceAdaptationContract("materialx.document@1.39.4", 1, ("/",), "recompile"),
            SourceAdaptationContract("mdl.program@1", 1, ("/arguments",), "recompile"),
        ),
        {
            "reference-evaluator": (
                "source_index", "wo", "wi", "target_f", "uv",
                "mip_level", "native_features",
            ),
            "method-sampler": (
                "source_index", "wo", "sample_u", "uv",
                "mip_level", "native_features",
            ),
        },
        tuple(TensorField(name, dtype, shape) for name, dtype, shape in _TENSOR_SHAPES),
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
            "runtime_class": "functional-reproduction",
            "B_asset_record": int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["compiled_material"]["total_bytes"]),
            "B_asset_bytes_per_texel": 16,
            "B_shared": 2 * (9859 + 96 + 2793),
            "C_prepare_macs": int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["evaluator"]["frame_macs"] + NVIDIA_NEURAL_APPEARANCE_LAYOUT["sampler"]["prepare_macs"]),
            "C_eval_macs": int(NVIDIA_NEURAL_APPEARANCE_LAYOUT["evaluator"]["evaluate_macs"]),
        },
        {
            "encoder": tuple(name for name in _PARAMETER_NAMES if name.startswith("encoder_")),
            "asset": ("latent_texels",),
            "evaluator": tuple(
                name
                for name in _PARAMETER_NAMES
                if name == "frame_w" or name.startswith("evaluate_")
            ),
            "sampler": tuple(name for name in _PARAMETER_NAMES if name.startswith("sampler_")),
        },
        (
            ComponentContract(
                "native-encoder",
                True,
                ("encoder",),
                ("bootstrap",),
                ("reference-evaluator",),
                ("evaluator_log1p_l1",),
                ("checkpoint:model_state",),
                (),
            ),
            ComponentContract(
                "latent-asset",
                True,
                ("asset",),
                ("finetune",),
                ("reference-evaluator", "method-sampler"),
                ("evaluator_log1p_l1", "sampler_forward_kl"),
                (
                    "asset:latent0.dds",
                    "asset:latent1.dds",
                    "asset-sampler:nvidia-latent",
                ),
                (),
            ),
            ComponentContract(
                "learned-evaluator",
                True,
                ("evaluator",),
                ("bootstrap", "finetune"),
                ("reference-evaluator", "method-sampler"),
                ("evaluator_log1p_l1",),
                ("program:shared-weights",),
                ("nclsNvidiaNeuralPrepareFp16", "nclsNvidiaNeuralEvaluateFFp16"),
            ),
            ComponentContract(
                "matched-sampler",
                True,
                ("sampler",),
                ("bootstrap", "finetune"),
                ("method-sampler",),
                ("sampler_forward_kl", "sampler_valid_fraction"),
                ("program:shared-weights",),
                ("nclsSampleNvidiaNeuralPrepared", "nclsNvidiaNeuralPreparedPdf"),
            ),
        ),
    )

    def create_trainable(self, context: Mapping[str, Any]) -> nn.Module:
        required = {
            "native_feature_count", "latent_width", "latent_height", "latent_mip_count"
        }
        if set(context) != required:
            raise ValueError(f"NVIDIA trainable context fields must be exactly {sorted(required)}")
        return NvidiaModel(
            native_feature_count=int(context["native_feature_count"]),
            latent_width=int(context["latent_width"]),
            latent_height=int(context["latent_height"]),
            latent_mip_count=int(context["latent_mip_count"]),
        )

    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        for phase in config["phases"]:
            routes = {route["name"]: route["kind"] for route in phase["routes"]}
            if routes != {"evaluator": "reference-evaluator", "sampler": "method-sampler"}:
                raise ValueError("NVIDIA objective 需要 evaluator 和 sampler 两条 route")

    def configure_phase(self, model: nn.Module, phase: Mapping[str, Any]) -> None:
        if not isinstance(model, NvidiaModel):
            raise TypeError("NVIDIA method requires NvidiaModel")
        super().configure_phase(model, phase)
        model.set_training_phase("bootstrap" if "encoder" in phase["parameter_groups"] else "finetune")

    def materialize_assets(
        self,
        model: nn.Module,
        native_assets: NativeAssetCollection,
    ) -> None:
        if not isinstance(model, NvidiaModel):
            raise TypeError("NVIDIA method requires NvidiaModel")
        model.materialize(native_assets)

    def apply_phase_transition(
        self,
        model: nn.Module,
        transition: str,
        native_assets: NativeAssetCollection,
    ) -> None:
        if transition != "materialize-assets":
            raise ValueError(f"unsupported NVIDIA phase transition {transition!r}")
        self.materialize_assets(model, native_assets)

    def training_objective(
        self,
        model: nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if not isinstance(model, NvidiaModel):
            raise TypeError("NVIDIA method requires NvidiaModel")
        evaluator_values = dict(batches["evaluator"].tensors)
        sampler_values = dict(batches["sampler"].tensors)
        evaluator_latent = model.latent_for_batch(evaluator_values)
        prediction_f = model.evaluate_f(
            evaluator_latent, evaluator_values["wo"], evaluator_values["wi"]
        )
        evaluator_loss = torch.mean(
            torch.abs(
                torch.log1p(prediction_f)
                - torch.log1p(torch.clamp(evaluator_values["target_f"], min=0.0))
            )
        )
        sampler_latent = model.latent_for_batch(sampler_values)
        sampled_wi, proposal_pdf, _, valid = model.sampler_sample_with_head(
            sampler_latent,
            sampler_values["wo"],
            sampler_values["sample_u"],
            "nvidia-diffuse-ggx9",
        )
        learned_f = model.evaluate_f(
            sampler_latent.detach(), sampler_values["wo"], sampled_wi
        ).detach()
        sampler_loss, valid_fraction = sampler_forward_kl_score(
            learned_f,
            sampled_wi,
            proposal_pdf,
            valid,
        )
        loss = evaluator_loss + sampler_loss
        return loss, {
            "evaluator_log1p_l1": evaluator_loss.detach(),
            "sampler_forward_kl": sampler_loss.detach(),
            "sampler_valid_fraction": valid_fraction.detach(),
            "loss/optimization_total": loss.detach(),
            "loss/appearance": evaluator_loss.detach(),
            "loss/proposal": sampler_loss.detach(),
            "loss/proposal_weight": 1.0,
        }

    def export_training_state(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        if not isinstance(model, NvidiaModel):
            raise TypeError("NVIDIA method requires NvidiaModel")
        named_parameters = dict(model.named_parameters())
        result = {
            name: named_parameters[name].detach().cpu().contiguous()
            for name in _PARAMETER_NAMES
        }
        levels = [
            level.detach().permute(1, 2, 0).reshape(-1, 8).cpu().contiguous()
            for level in model.latent_levels
        ]
        offsets = [0]
        for level in levels:
            offsets.append(offsets[-1] + int(level.shape[0]))
        result.update(
            {
                "latent_texels": torch.cat(levels, dim=0),
                "latent_mip_offsets": torch.tensor(offsets, dtype=torch.int64),
                "latent_mip_shapes": torch.tensor(model.mip_shapes, dtype=torch.int64),
                "phase_code": torch.tensor(
                    [0 if model.phase_name == "bootstrap" else 1], dtype=torch.int64
                ),
            }
        )
        expected = {field.name for field in self.descriptor.tensor_state_schema}
        if set(result) != expected:
            raise ValueError("NVIDIA training tensor mapping disagrees with descriptor")
        return result

    def restore_training_state(self, model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        if not isinstance(model, NvidiaModel):
            raise TypeError("NVIDIA method requires NvidiaModel")
        expected = {field.name for field in self.descriptor.tensor_state_schema}
        if set(state) != expected:
            raise ValueError("NVIDIA checkpoint tensor mapping disagrees with descriptor")
        named_parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name in _PARAMETER_NAMES:
                if state[name].shape != named_parameters[name].shape or state[name].dtype != named_parameters[name].dtype:
                    raise ValueError(f"NVIDIA checkpoint tensor {name!r} shape/dtype mismatch")
            for name in _PARAMETER_NAMES:
                named_parameters[name].copy_(state[name].to(named_parameters[name].device))
            expected_shapes = torch.tensor(model.mip_shapes, dtype=torch.int64)
            shapes = state["latent_mip_shapes"].cpu()
            if not torch.equal(shapes, expected_shapes):
                raise ValueError("NVIDIA checkpoint latent mip shapes disagree with model context")
            offsets = state["latent_mip_offsets"].cpu().tolist()
            if len(offsets) != model.latent_mip_count + 1 or offsets[0] != 0:
                raise ValueError("NVIDIA checkpoint latent mip offsets are invalid")
            flat = state["latent_texels"]
            for index, level in enumerate(model.latent_levels):
                begin, end = int(offsets[index]), int(offsets[index + 1])
                height, width = model.mip_shapes[index]
                values = flat[begin:end].reshape(height, width, 8).permute(2, 0, 1)
                level.copy_(values.to(level.device))
        stage_value = int(state["phase_code"].item())
        if stage_value not in {0, 1}:
            raise ValueError("NVIDIA checkpoint phase code is invalid")
        model.set_training_phase("bootstrap" if stage_value == 0 else "finetune")

    def prepare_export(self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        if int(checkpoint["model_state"]["phase_code"].item()) != 0:
            return checkpoint
        model = self.create_trainable(checkpoint["training_config"]["model_context"])
        self.restore_training_state(model, checkpoint["model_state"])
        adapter = self.create_source_adapter((snapshot,), torch.device("cpu"))
        try:
            model.materialize(adapter.native_assets())
            return {**checkpoint, "model_state": self.export_training_state(model)}
        finally:
            adapter.close()

    def package_validation(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.descriptor.adaptation_contract(snapshot)
        state_ids = checkpoint.get("source_snapshot_ids")
        state = checkpoint.get("model_state")
        if not isinstance(state_ids, (list, tuple)) or snapshot.snapshot_id not in map(str, state_ids):
            raise ValueError("NVIDIA package parity source does not occur in the checkpoint")
        if not isinstance(state, Mapping):
            raise ValueError("NVIDIA package parity requires validated model_state tensors")
        return {
            "status": "gpu-parity-required",
            "parity": {
                "oracle": "nvidia-rta2024-packed-fp16-cpu-emulation@1",
                "view": list(_PARITY_VIEW),
                "lights": [list(value) for value in _PARITY_LIGHTS],
                "expected_f": _packed_fp16_parity_f(
                    state, _PARITY_VIEW, _PARITY_LIGHTS
                ),
                # 64-term half accumulations have an O(10^-2) conservative
                # relative error envelope across native-FMA and split-op
                # lowering. The absolute term keeps dark channels testable.
                "relative_tolerance": 2e-2,
                "absolute_tolerance": 2e-4,
            },
        }

    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        state = checkpoint.get("model_state")
        training_config = checkpoint.get("training_config")
        if not isinstance(state, Mapping) or not isinstance(training_config, Mapping):
            raise ValueError("NVIDIA runtime compilation requires validated model_state tensors")
        context = training_config.get("model_context")
        if not isinstance(context, Mapping):
            raise ValueError("NVIDIA runtime compilation requires model_context provenance")
        model = self.create_trainable(context)
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
                    "dtype": "packed-float16x2-uint32@1",
                    "shape": [len(shared) // 4],
                    "stride": 4,
                    "alignment": 4,
                    "usage": "gNclsRuntimeWeights",
                }
            },
            self.descriptor.capabilities,
            defines,
        )

    def compile_asset(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> MaterialPayload:
        self.descriptor.adaptation_contract(snapshot)
        state_ids = checkpoint.get("source_snapshot_ids")
        model_state = checkpoint.get("model_state")
        if not isinstance(state_ids, (list, tuple)) or not isinstance(model_state, Mapping):
            raise ValueError("NVIDIA material compilation requires source_snapshot_ids and model_state")
        try:
            index = list(map(str, state_ids)).index(snapshot.snapshot_id)
        except ValueError as error:
            raise ValueError("NVIDIA checkpoint has no latent for this source snapshot") from error
        del index
        shapes = model_state.get("latent_mip_shapes")
        offsets = model_state.get("latent_mip_offsets")
        texels = model_state.get("latent_texels")
        if (
            not isinstance(shapes, torch.Tensor)
            or not isinstance(offsets, torch.Tensor)
            or not isinstance(texels, torch.Tensor)
        ):
            raise ValueError("NVIDIA checkpoint latent hierarchy is invalid")
        shape_values = [(int(height), int(width)) for height, width in shapes.tolist()]
        offset_values = [int(value) for value in offsets.tolist()]
        if len(offset_values) != len(shape_values) + 1 or offset_values[0] != 0:
            raise ValueError("NVIDIA checkpoint latent mip offsets are invalid")
        levels = []
        for index, (height, width) in enumerate(shape_values):
            begin, end = offset_values[index], offset_values[index + 1]
            level = texels[begin:end].reshape(height, width, 8).detach().cpu().numpy()
            if not np.isfinite(level).all():
                raise ValueError("NVIDIA checkpoint latent hierarchy contains non-finite texels")
            levels.append(level)
        width, height = shape_values[0][1], shape_values[0][0]
        mip_count = len(shape_values)
        payload = _pack_record(width, height, mip_count)
        latent0 = encode_rgba16f_dds([level[..., :4] for level in levels])
        latent1 = encode_rgba16f_dds([level[..., 4:] for level in levels])
        descriptor_shape = [width, height, mip_count, 4]
        return MaterialPayload(
            snapshot.snapshot_id,
            {"compiled-material": payload},
            {
                "compiled-material": {
                    "dtype": "ncls-nvidia-compiled-material@1",
                    "shape": [1],
                    "stride": len(payload),
                    "alignment": 16,
                    "usage": "gNclsCompiledMaterials",
                }
            },
            {
                "latent0.dds": latent0,
                "latent1.dds": latent1,
            },
            {
                "latent0.dds": {
                    "dtype": RGBA16F_DDS_DTYPE,
                    "shape": descriptor_shape,
                    "stride": 8,
                    "alignment": 16,
                    "usage": "gNclsNvidiaLatent0",
                },
                "latent1.dds": {
                    "dtype": RGBA16F_DDS_DTYPE,
                    "shape": descriptor_shape,
                    "stride": 8,
                    "alignment": 16,
                    "usage": "gNclsNvidiaLatent1",
                },
            },
            {
                "nvidia-latent": {
                    "kind": "sampler",
                    "usage": "gNclsNvidiaLatentSampler",
                    "filter": "linear",
                    "address_mode": "wrap",
                }
            },
        )


def _create_source_adapter(
    snapshots: Sequence[SourceSnapshot], device: torch.device
) -> MethodSourceAdapter:
    values = tuple(snapshots)
    if not values:
        raise ValueError("NVIDIA data facet requires source snapshots")
    contracts = {
        (snapshot.family_id, snapshot.source_contract_version)
        for snapshot in values
    }
    if len(contracts) != 1:
        raise ValueError("one NVIDIA data session cannot mix source contracts")
    family_id, version = next(iter(contracts))
    factories = {
        (
            NvidiaLayerStackSourceAdapter.family_id,
            NvidiaLayerStackSourceAdapter.source_contract_version,
        ): NvidiaLayerStackSourceAdapter,
        (
            NvidiaMaterialXSourceAdapter.family_id,
            NvidiaMaterialXSourceAdapter.source_contract_version,
        ): NvidiaMaterialXSourceAdapter,
        (
            NvidiaMdlFixedSourceAdapter.family_id,
            NvidiaMdlFixedSourceAdapter.source_contract_version,
        ): NvidiaMdlFixedSourceAdapter,
    }
    try:
        factory = factories[(family_id, version)]
    except KeyError as error:
        raise ValueError(
            f"NVIDIA data facet has no source adapter for {family_id}@{version}"
        ) from error
    return factory(values, device)


METHOD = NvidiaMethod()
