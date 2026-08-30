from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.core.scattering import BackendCapability, MaterialPayload, RuntimePayload
from ncls.core.source import SourceSnapshot
from ncls.learning.batches import AssetTileBatch, EvaluatorBatch, OnlineTrainingBatch
from ncls.learning.method import (
    ComponentContract,
    MethodDefinition,
    MethodDescriptor,
    SourceAdaptationContract,
    TensorField,
)
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalFusedNeuralMaterialModel,
    metal_fused_parameter_groups,
)
from ncls.learning.models.metal_fused_profile import (
    METAL_FUSED_FULL_PROFILE,
    METAL_FUSED_LAYOUT_PATH,
    load_metal_fused_layout,
)
from ncls.learning.source_adaptation import NativeAssetCollection
from ncls.paths import PROJECT_ROOT


_CODEC_GROUPS = (
    "codec_role_stems",
    "codec_encoder",
    "codec_decoder",
    "codec_semantic_heads",
    "asset_adapter",
    "quantization",
)
_JOINT_GROUPS = (
    *_CODEC_GROUPS,
    "typed_compiler",
    "optimized_state_teacher",
    "prepared_model",
    "angular_bank",
    "analytic_core",
    "hybrid_evaluator",
)


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "src/ncls/learning/models/metal_fused.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_texture_codec.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_typed_compiler.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_directional_evaluator.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_fused_profile.py",
        PROJECT_ROOT / "src/ncls/learning/metal_asset_cook.py",
        PROJECT_ROOT / "src/ncls/learning/mdl_metal_assets.py",
        PROJECT_ROOT / "src/ncls/learning/source_adapters.py",
        METAL_FUSED_LAYOUT_PATH,
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little") + relative)
        digest.update(len(payload).to_bytes(8, "little") + payload)
    return digest.hexdigest()


def _state_schema() -> tuple[TensorField, ...]:
    with torch.device("meta"):
        model = MetalFusedNeuralMaterialModel.from_context(
            METAL_FUSED_REQUIRED_CONTEXT
        )
    fields = []
    for name, value in model.state_dict().items():
        dtype = str(value.dtype).removeprefix("torch.")
        fields.append(TensorField(name, dtype, tuple(int(item) for item in value.shape)))
    return tuple(fields)


def _component(
    component_id: str,
    groups: tuple[str, ...],
    phases: tuple[str, ...],
    dependencies: tuple[str, ...],
    outputs: tuple[str, ...],
) -> ComponentContract:
    return ComponentContract(
        component_id,
        True,
        groups,
        phases,
        dependencies,
        outputs,
        ("checkpoint:model_state",),
        (),
    )


_COMPONENTS = (
    _component(
        "role-aware-texture-stems",
        ("codec_role_stems",),
        ("codec-warmup", "joint-appearance"),
        ("asset-tile",),
        ("codec_role_stems_trace",),
    ),
    _component(
        "bundle-set-shared-unet-encoder",
        ("codec_encoder",),
        ("codec-warmup", "joint-appearance"),
        ("asset-tile",),
        ("codec_encoder_trace", "codec_bundle_attention_trace"),
    ),
    _component(
        "independent-high-low-qat-grids",
        ("quantization",),
        ("codec-warmup", "joint-appearance"),
        ("asset-tile",),
        ("codec_quantization_trace", "codec_qat_loss"),
    ),
    _component(
        "shared-structured-decoder",
        ("codec_decoder",),
        ("codec-warmup", "joint-appearance"),
        ("asset-tile",),
        ("codec_decoder_trace", "codec_structured_head_trace"),
    ),
    _component(
        "training-semantic-heads",
        ("codec_semantic_heads",),
        ("codec-warmup", "joint-appearance"),
        ("asset-tile",),
        ("codec_semantic_heads_trace", "codec_semantic_loss"),
    ),
    _component(
        "bounded-rank8-asset-adapter",
        ("asset_adapter",),
        ("codec-warmup", "joint-appearance"),
        ("asset-tile",),
        ("codec_adapter_trace",),
    ),
    _component(
        "pure-typed-set-compiler",
        ("typed_compiler",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("typed_compiler_trace", "compiler_distillation_loss"),
    ),
    _component(
        "target-visible-optimized-state-control",
        ("optimized_state_teacher",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("optimized_teacher_trace", "teacher_response_loss"),
    ),
    _component(
        "deterministic-spatial-access-two-mip-prepare",
        ("prepared_model",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("spatial_access_trace", "adjacent_mip_trace"),
    ),
    _component(
        "learned-lobe-frames-and-view-prepare",
        ("prepared_model",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("prepared_frames_trace", "prepared_view_trace"),
    ),
    _component(
        "raw-cartesian-direction",
        ("hybrid_evaluator",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("direction_raw_trace",),
    ),
    _component(
        "stable-half-difference-direction",
        ("hybrid_evaluator",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("direction_half_difference_trace",),
    ),
    _component(
        "shared-warped-angular-bank",
        ("angular_bank",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("angular_multiscale_trace", "angular_difference_trace"),
    ),
    _component(
        "six-slot-source-aware-analytic-core",
        ("analytic_core", "typed_compiler"),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("analytic_core_trace", "analytic_core_loss"),
    ),
    _component(
        "bounded-multiplicative-correction",
        ("hybrid_evaluator",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("multiplicative_correction_trace",),
    ),
    _component(
        "four-positive-residual-lobes",
        ("hybrid_evaluator", "typed_compiler"),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("positive_residual_lobes_trace",),
    ),
    _component(
        "free-positive-rgb-tail",
        ("hybrid_evaluator",),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("free_positive_tail_trace",),
    ),
    _component(
        "eleven-component-proposal-state-reservation",
        ("typed_compiler", "prepared_model"),
        ("joint-appearance",),
        ("reference-evaluator",),
        ("proposal_state_trace",),
    ),
)


class MetalFusedMethodDefinition(MethodDefinition):
    _layout = load_metal_fused_layout()
    descriptor = MethodDescriptor(
        "metal-fused-neural-material",
        1,
        "vMaterials Metal quality-first fused neural evaluator slice",
        _implementation_sha256(),
        (
            SourceAdaptationContract(
                "mdl.program@1", 1, ("/arguments",), "runtime-patch"
            ),
        ),
        {
            "asset-tile": (
                "asset_descriptors",
                "tiles",
                "role_values",
                "mip_level",
            ),
            "reference-evaluator": (
                "source_index",
                "wo",
                "wi",
                "target_f",
                "uv",
                "uv_dx",
                "uv_dy",
                "mip_level",
                "metal_texture_patches",
                "metal_texture_slot_mask",
                "metal_texture_role_class",
                "metal_mip_fraction",
                "metal_graph_index",
                "metal_schema_index",
                "metal_recipe_index",
                "metal_identity_index",
                "metal_finish_index",
                "metal_asset_index",
                "metal_typed_semantic_id",
                "metal_typed_type_id",
                "metal_typed_responsibility_id",
                "metal_typed_discrete",
                "metal_typed_continuous",
                "metal_typed_presence",
                "metal_canonical_optical",
                "metal_access_state",
                "metal_frame_state",
            ),
        },
        _state_schema(),
        "ncls.metal-fused-evaluator-slice@1",
        int(BackendCapability.PREPARE | BackendCapability.EVALUATE),
        {
            "maximum_prepare_steps": int(
                _layout["bounded_execution"]["maximum_prepare_steps"]
            ),
            "maximum_evaluate_steps": int(
                _layout["bounded_execution"]["maximum_evaluate_steps"]
            ),
            "maximum_state_bytes": int(
                _layout["bounded_execution"]["maximum_prepared_state_bytes"]
            ),
            "maximum_reads": int(_layout["bounded_execution"]["maximum_reads"]),
        },
        {
            "runtime_class": "quality-first-evaluator-slice",
            "profile_id": METAL_FUSED_FULL_PROFILE.profile_id,
            "B_prepared_max": METAL_FUSED_FULL_PROFILE.maximum_state_bytes,
            "B_grid_texel": 16,
            "maximum_texture_reads": METAL_FUSED_FULL_PROFILE.maximum_reads,
            "matched_sampler_status": "reserved-downstream-child",
            "observed_quality_gate": False,
        },
        metal_fused_parameter_groups(),
        _COMPONENTS,
    )

    def create_trainable(self, context: Mapping[str, Any]) -> nn.Module:
        return MetalFusedNeuralMaterialModel.from_context(context)

    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        if config.get("correspondence_id") != "metal-fused-full-evaluator@1":
            raise ValueError("Metal fused training requires its frozen evaluator correspondence")
        if config.get("source_adaptation_id") != "metal-fused.mdl-vmaterials2-metal@1":
            raise ValueError("Metal fused training requires the Metal registry adapter")
        if dict(config.get("model_context", {})) != dict(METAL_FUSED_REQUIRED_CONTEXT):
            raise ValueError("Metal fused training cannot shrink or alter the full profile")
        source = config.get("source")
        if not isinstance(source, Mapping) or source.get("family_id") != "mdl.program@1":
            raise ValueError("Metal fused training requires native MDL source semantics")
        phases = config.get("phases")
        if not isinstance(phases, list) or [item.get("name") for item in phases] != [
            "codec-warmup",
            "joint-appearance",
        ]:
            raise ValueError("Metal evaluator slice requires codec and joint phases")
        expected_groups = {
            "codec-warmup": list(_CODEC_GROUPS),
            "joint-appearance": list(_JOINT_GROUPS),
        }
        expected_routes = {
            "codec-warmup": {"asset": "asset-tile"},
            "joint-appearance": {
                "asset": "asset-tile",
                "evaluator": "reference-evaluator",
            },
        }
        expected_losses = {
            "codec-warmup": [
                "semantic-reconstruction",
                "normal-angular",
                "structured-state",
                "mip-consistency",
                "grid-qat",
            ],
            "joint-appearance": [
                "codec-full",
                "response-robust",
                "linear-energy",
                "peak-support",
                "reciprocity",
                "analytic-core-preservation",
                "teacher-response",
                "compiler-functional-distillation",
            ],
        }
        for phase in phases:
            name = str(phase["name"])
            if phase.get("parameter_groups") != expected_groups[name]:
                raise ValueError("Metal phase parameter groups drifted from the full method")
            if phase.get("loss_terms") != expected_losses[name]:
                raise ValueError("Metal phase loss terms drifted from the full method")
            routes = {
                str(item["name"]): str(item["kind"])
                for item in phase.get("routes", ())
            }
            if routes != expected_routes[name]:
                raise ValueError("Metal phase typed routes are incomplete")
            recipes = phase.get("recipes")
            if not isinstance(recipes, Mapping) or recipes.get("profile_id") != "metal_fused_full_v1":
                raise ValueError("Metal phase must freeze the full profile recipe")
            if phase.get("transition") is not None:
                raise ValueError("Metal evaluator phases cannot hide an asset lifecycle transition")
        evaluator = next(
            item for item in phases[1]["routes"] if item["name"] == "evaluator"
        )
        options = evaluator.get("options", {})
        if (
            options.get("direction_proposal") != "uniform-half-difference@1"
            or int(options.get("source_patch_size", 0)) < 8
            or not bool(options.get("asset_tile_coherent", False))
        ):
            raise ValueError("Metal joint route requires coherent real source patches")

    def configure_phase(self, model: nn.Module, phase: Mapping[str, Any]) -> None:
        if not isinstance(model, MetalFusedNeuralMaterialModel):
            raise TypeError("Metal fused method requires MetalFusedNeuralMaterialModel")
        super().configure_phase(model, phase)

    @staticmethod
    def _trace_metric(
        trace: Mapping[str, torch.Tensor], name: str
    ) -> torch.Tensor:
        try:
            return trace[name].detach()
        except KeyError as error:
            raise RuntimeError(f"Metal full execution omitted trace {name!r}") from error

    def training_objective(
        self,
        model: nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if not isinstance(model, MetalFusedNeuralMaterialModel):
            raise TypeError("Metal fused method requires MetalFusedNeuralMaterialModel")
        phase_name = str(phase.get("name"))
        if "asset" not in batches or not isinstance(batches["asset"], AssetTileBatch):
            raise ValueError("Metal phases require the canonical asset-tile route")
        codec_loss, codec_metrics = model.codec_objective(batches["asset"])
        metrics: dict[str, torch.Tensor | float] = {
            name: value.detach() for name, value in codec_metrics.items()
        }
        if phase_name == "codec-warmup":
            if set(batches) != {"asset"}:
                raise ValueError("Metal codec warmup accepts only its asset route")
            return codec_loss, metrics
        if phase_name != "joint-appearance" or set(batches) != {"asset", "evaluator"}:
            raise ValueError("Metal joint appearance requires asset and evaluator routes")
        evaluator_batch = batches["evaluator"]
        if not isinstance(evaluator_batch, EvaluatorBatch):
            raise ValueError("Metal evaluator route returned the wrong typed batch")
        values = evaluator_batch.tensors
        spatial = model.spatial_state(values)
        pure_program, teacher_program = model.compile_program_states(values)
        pure_prepared = model.prepare_from_components(pure_program, spatial, values)
        teacher_prepared = model.prepare_from_components(teacher_program, spatial, values)
        pure = model.evaluate_prepared(
            pure_prepared, values["wo"], values["wi"]
        )
        teacher = model.evaluate_prepared(
            teacher_prepared, values["wo"], values["wi"]
        )
        target = values["target_f"]
        log_error = torch.log1p(pure.f) - torch.log1p(target)
        response_loss = torch.sqrt(log_error.square() + 1e-6).mean()
        cosine = torch.clamp(values["wi"][..., 2:3], min=0.0)
        energy_loss = torch.mean(torch.abs(pure.f - target) * cosine)
        predicted_luminance = torch.sum(
            pure.f * pure.f.new_tensor((0.2126, 0.7152, 0.0722)), dim=-1
        )
        target_luminance = torch.sum(
            target * target.new_tensor((0.2126, 0.7152, 0.0722)), dim=-1
        )
        peak_loss = torch.mean(
            torch.abs(
                torch.log1p(predicted_luminance)
                - torch.log1p(target_luminance)
            )
            * (1.0 + target_luminance / torch.clamp(target_luminance.mean(), min=1e-4))
        )
        reverse_wo = values["wi"][:, 0, :]
        reverse_prepared = model.prepare_from_components(
            pure_program, spatial, values, wo=reverse_wo
        )
        reverse = model.evaluate_prepared(
            reverse_prepared, reverse_wo, values["wo"][:, None, :]
        )
        reciprocity_loss = torch.mean(
            torch.abs(torch.log1p(pure.f) - torch.log1p(reverse.f))
        )
        analytic_loss = torch.mean(
            torch.abs(torch.log1p(pure.core_f) - torch.log1p(target))
        )
        teacher_response_loss = torch.mean(
            torch.abs(torch.log1p(teacher.f) - torch.log1p(target))
        )
        compiler_distillation = F.smooth_l1_loss(
            pure_program.compiler_latent,
            teacher_program.compiler_latent.detach(),
        ) + 0.25 * F.smooth_l1_loss(pure.f, teacher.f.detach())
        footprint_loss = 0.01 * spatial.trace["adjacent_mip_interpolation"]
        loss = (
            0.35 * codec_loss
            + response_loss
            + 0.2 * energy_loss
            + 0.15 * peak_loss
            + 0.05 * reciprocity_loss
            + 0.1 * analytic_loss
            + 0.25 * teacher_response_loss
            + 0.1 * compiler_distillation
            + footprint_loss
        )
        trace = {**pure_prepared.trace, **pure.trace}
        metrics.update(
            {
                "response_robust_loss": response_loss.detach(),
                "linear_energy_loss": energy_loss.detach(),
                "peak_support_loss": peak_loss.detach(),
                "reciprocity_loss": reciprocity_loss.detach(),
                "analytic_core_loss": analytic_loss.detach(),
                "teacher_response_loss": teacher_response_loss.detach(),
                "compiler_distillation_loss": compiler_distillation.detach(),
                "typed_compiler_trace": self._trace_metric(trace, "pure_compiler"),
                "optimized_teacher_trace": self._trace_metric(
                    teacher_program.trace, "optimized_teacher"
                ),
                "spatial_access_trace": self._trace_metric(trace, "spatial_access"),
                "adjacent_mip_trace": self._trace_metric(
                    trace, "adjacent_mip_interpolation"
                ),
                "prepared_frames_trace": self._trace_metric(trace, "prepared_frames"),
                "prepared_view_trace": self._trace_metric(trace, "prepared_view"),
                "proposal_state_trace": self._trace_metric(trace, "proposal_state"),
                "direction_raw_trace": self._trace_metric(trace, "direction_raw"),
                "direction_half_difference_trace": self._trace_metric(
                    trace, "direction_half_difference"
                ),
                "angular_multiscale_trace": self._trace_metric(
                    trace, "angular_multiscale"
                ),
                "angular_difference_trace": self._trace_metric(
                    trace, "angular_difference"
                ),
                "analytic_core_trace": self._trace_metric(trace, "analytic_core"),
                "multiplicative_correction_trace": self._trace_metric(
                    trace, "multiplicative_correction"
                ),
                "positive_residual_lobes_trace": self._trace_metric(
                    trace, "positive_residual_lobes"
                ),
                "free_positive_tail_trace": self._trace_metric(
                    trace, "free_positive_tail"
                ),
            }
        )
        return loss, metrics

    def export_training_state(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        if not isinstance(model, MetalFusedNeuralMaterialModel):
            raise TypeError("Metal fused method requires MetalFusedNeuralMaterialModel")
        result = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        expected = {field.name for field in self.descriptor.tensor_state_schema}
        if set(result) != expected:
            raise ValueError("Metal model state disagrees with its descriptor")
        return result

    def restore_training_state(
        self, model: nn.Module, state: Mapping[str, torch.Tensor]
    ) -> None:
        if not isinstance(model, MetalFusedNeuralMaterialModel):
            raise TypeError("Metal fused method requires MetalFusedNeuralMaterialModel")
        expected = {field.name for field in self.descriptor.tensor_state_schema}
        if set(state) != expected:
            raise ValueError("Metal checkpoint tensor state is incomplete")
        current = model.state_dict()
        restored = {}
        for name, target in current.items():
            value = state[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Metal checkpoint tensor {name!r} shape/dtype drifted")
            restored[name] = value.to(target.device)
        model.load_state_dict(restored, strict=True)

    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        del checkpoint
        raise RuntimeError(
            "Metal evaluator-slice checkpoints are intentionally non-packageable until "
            "the matched sampler and runtime deployment children freeze SAMPLE/PDF and Slang"
        )

    def compile_asset(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload:
        self.descriptor.adaptation_contract(snapshot)
        del checkpoint
        raise RuntimeError(
            "Metal asset packaging is intentionally fail-closed until the runtime child "
            "consumes the frozen encoder-only asset layout"
        )

    def materialize_assets(
        self,
        model: nn.Module,
        native_assets: NativeAssetCollection,
    ) -> None:
        del model, native_assets
        raise RuntimeError(
            "Metal asset cook is an explicit encoder-only/refinement/control operation, "
            "not a hidden training phase transition"
        )


METHOD_DEFINITION = MetalFusedMethodDefinition()


__all__ = ["MetalFusedMethodDefinition", "METHOD_DEFINITION"]
