from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ncls.core.scattering import (
    BackendCapability,
    InstancePayload,
    MaterialPayload,
    RuntimePayload,
)
from ncls.core.source import SourceSnapshot
from ncls.learning.batches import (
    AssetTileBatch,
    EvaluatorBatch,
    MethodSamplerBatch,
    OnlineTrainingBatch,
)
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
from ncls.learning.metal_asset_cook import MetalAssetCooker
from ncls.learning.metal_runtime import (
    METAL_COMPILED_WORD_COUNT,
    METAL_RAW_OFFSETS,
    METAL_RAW_WORD_COUNT,
    evaluate_metal_cooked_asset,
    fake_quantize_fp16_ste,
    metal_runtime_parameter_names,
    pack_metal_asset,
    pack_metal_compiled_material,
    pack_metal_program,
    pack_metal_raw_parameters,
    quantize_runtime_model,
)
from ncls.learning.objectives import sampler_forward_kl_score
from ncls.learning.source_adapters import MetalFusedMdlSourceAdapter
from ncls.learning.source_adaptation import NativeAssetCollection
from ncls.paths import PROJECT_ROOT
from ncls.source_materials.families.mdl import MdlFamilyDefinition
from ncls.source_materials.mdl import MdlMaterialSource


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
_PROPOSAL_GROUPS = ("proposal_sampler",)
_QAT_REFINE_GROUPS = (
    *_CODEC_GROUPS,
    "typed_compiler",
    "prepared_model",
    "angular_bank",
    "analytic_core",
    "hybrid_evaluator",
    *_PROPOSAL_GROUPS,
)
_METAL_PREPARE_TENSORS = (
    "source_index",
    "wo",
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
)

_PARITY_VIEW = (0.17364818, -0.33682409, 0.92541658)
_PARITY_LIGHTS = (
    (0.0, 0.0, 1.0),
    (0.34202015, 0.16317591, 0.92541658),
    (-0.49240388, 0.41317591, 0.76604444),
    (0.71984631, -0.60402277, 0.34202015),
)


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
            raise ValueError(f"Metal shader dependency escapes shader root: {path}") from error
        if relative in result:
            continue
        payload = path.read_bytes()
        result[relative] = payload
        for match in include_pattern.finditer(payload):
            dependency = (path.parent / match.group(1).decode("utf-8")).resolve()
            if dependency.is_file():
                pending.append(dependency)
    return result


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "src/ncls/learning/models/metal_fused.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_texture_codec.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_typed_compiler.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_directional_evaluator.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_sampler.py",
        PROJECT_ROOT / "src/ncls/learning/models/metal_fused_profile.py",
        PROJECT_ROOT / "src/ncls/learning/metal_asset_cook.py",
        PROJECT_ROOT / "src/ncls/learning/source_adaptation.py",
        PROJECT_ROOT / "src/ncls/learning/metal_runtime.py",
        PROJECT_ROOT / "src/ncls/learning/mdl_metal_assets.py",
        PROJECT_ROOT / "src/ncls/learning/source_adapters.py",
        PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused_common.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused_prepare.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused_evaluator.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused_compiler.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused_compiler_heads.slang",
        PROJECT_ROOT / "shaders/ncls/backends/metal_fused/metal_fused_layout.generated.slang",
        PROJECT_ROOT / "shaders/ncls/scattering/metal_fused_proposal.slang",
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
    runtime_artifacts: tuple[str, ...] = ("checkpoint:model_state",),
    slang_entry_points: tuple[str, ...] = (),
) -> ComponentContract:
    return ComponentContract(
        component_id,
        True,
        groups,
        phases,
        dependencies,
        outputs,
        runtime_artifacts,
        slang_entry_points,
    )


_COMPONENTS = (
    _component(
        "role-aware-texture-stems",
        ("codec_role_stems",),
        ("codec-warmup", "joint-appearance", "qat-refine"),
        ("asset-tile",),
        ("codec_role_stems_trace",),
    ),
    _component(
        "bundle-set-shared-unet-encoder",
        ("codec_encoder",),
        ("codec-warmup", "joint-appearance", "qat-refine"),
        ("asset-tile",),
        ("codec_encoder_trace", "codec_bundle_attention_trace"),
    ),
    _component(
        "independent-high-low-qat-grids",
        ("quantization",),
        ("codec-warmup", "joint-appearance", "qat-refine"),
        ("asset-tile",),
        ("codec_quantization_trace", "codec_qat_loss"),
    ),
    _component(
        "shared-structured-decoder",
        ("codec_decoder",),
        ("codec-warmup", "joint-appearance", "qat-refine"),
        ("asset-tile",),
        ("codec_decoder_trace", "codec_structured_head_trace"),
    ),
    _component(
        "training-semantic-heads",
        ("codec_semantic_heads",),
        ("codec-warmup", "joint-appearance", "qat-refine"),
        ("asset-tile",),
        ("codec_semantic_heads_trace", "codec_semantic_loss"),
    ),
    _component(
        "bounded-rank8-asset-adapter",
        ("asset_adapter",),
        ("codec-warmup", "joint-appearance", "qat-refine"),
        ("asset-tile",),
        ("codec_adapter_trace",),
    ),
    _component(
        "pure-typed-set-compiler",
        ("typed_compiler",),
        ("joint-appearance", "qat-refine"),
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
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("spatial_access_trace", "adjacent_mip_trace"),
    ),
    _component(
        "learned-lobe-frames-and-view-prepare",
        ("prepared_model",),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("prepared_frames_trace", "prepared_view_trace"),
    ),
    _component(
        "raw-cartesian-direction",
        ("hybrid_evaluator",),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("direction_raw_trace",),
    ),
    _component(
        "stable-half-difference-direction",
        ("hybrid_evaluator",),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("direction_half_difference_trace",),
    ),
    _component(
        "shared-warped-angular-bank",
        ("angular_bank",),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("angular_multiscale_trace", "angular_difference_trace"),
    ),
    _component(
        "six-slot-source-aware-analytic-core",
        ("analytic_core", "typed_compiler"),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("analytic_core_trace", "analytic_core_loss"),
    ),
    _component(
        "bounded-multiplicative-correction",
        ("hybrid_evaluator",),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("multiplicative_correction_trace",),
    ),
    _component(
        "four-positive-residual-lobes",
        ("hybrid_evaluator", "typed_compiler"),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("positive_residual_lobes_trace",),
    ),
    _component(
        "free-positive-rgb-tail",
        ("hybrid_evaluator",),
        ("joint-appearance", "qat-refine"),
        ("reference-evaluator",),
        ("free_positive_tail_trace",),
    ),
    _component(
        "eleven-component-matched-proposal-mixture",
        ("proposal_sampler",),
        ("proposal-fit", "qat-refine"),
        ("reference-evaluator", "method-sampler"),
        (
            "proposal_state_trace",
            "proposal_component_pdf_trace",
            "proposal_component_sample_trace",
        ),
        (
            "checkpoint:model_state",
            "slang:ncls/scattering/metal_fused_proposal.slang",
            "slang:ncls/backends/metal_fused/metal_fused_layout.generated.slang",
        ),
        ("nclsMetalFusedProposalPdf", "nclsSampleMetalFusedProposal"),
    ),
    _component(
        "folded-full-hemisphere-support",
        ("proposal_sampler",),
        ("proposal-fit", "qat-refine"),
        ("reference-evaluator", "method-sampler"),
        ("proposal_support_trace", "proposal_fallback_trace"),
        (
            "checkpoint:model_state",
            "slang:ncls/scattering/metal_fused_proposal.slang",
        ),
        ("nclsMetalFusedProposalPdf", "nclsSampleMetalFusedProposal"),
    ),
    _component(
        "sample-pdf-throughput-identity",
        ("proposal_sampler",),
        ("proposal-fit", "qat-refine"),
        ("reference-evaluator", "method-sampler"),
        ("proposal_sample_pdf_trace", "proposal_weight_identity_trace"),
        (
            "checkpoint:model_state",
            "slang:ncls/scattering/metal_fused_proposal.slang",
        ),
        ("nclsMetalFusedProposalPdf", "nclsSampleMetalFusedProposal"),
    ),
)


class _MetalQatRefineExecution(nn.Module):
    """One functional-call boundary for the complete quantized runtime path."""

    def __init__(
        self,
        model: MetalFusedNeuralMaterialModel,
        definition: "MetalFusedMethodDefinition",
    ) -> None:
        super().__init__()
        self.model = model
        self.definition = definition

    def forward(
        self, batches: Mapping[str, OnlineTrainingBatch]
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if set(batches) != {"asset", "evaluator", "sampler"}:
            raise ValueError("Metal QAT refine requires asset, evaluator and sampler routes")
        joint_loss, joint_metrics = self.definition.training_objective(
            self.model,
            {"asset": batches["asset"], "evaluator": batches["evaluator"]},
            {"name": "joint-appearance"},
        )
        proposal_loss, proposal_metrics = self.definition._proposal_objective(
            self.model,
            {"evaluator": batches["evaluator"], "sampler": batches["sampler"]},
        )
        metrics = dict(joint_metrics)
        metrics.update(proposal_metrics)
        metrics.update(
            {
                "qat_refine_appearance_loss": joint_loss.detach(),
                "qat_refine_proposal_loss": proposal_loss.detach(),
            }
        )
        return joint_loss + proposal_loss, metrics


class MetalFusedMethodDefinition(MethodDefinition):
    _layout = load_metal_fused_layout()
    descriptor = MethodDescriptor(
        "metal-fused-neural-material",
        1,
        "vMaterials Metal quality-first fused neural material with matched sampler",
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
                "wi",
                "target_f",
                *_METAL_PREPARE_TENSORS,
            ),
            "method-sampler": ("sample_u", *_METAL_PREPARE_TENSORS),
        },
        _state_schema(),
        "ncls.metal-fused-full-method@1",
        int(
            BackendCapability.PREPARE
            | BackendCapability.EVALUATE
            | BackendCapability.SAMPLE
            | BackendCapability.PDF
        ),
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
            "runtime_class": "quality-first-full-method",
            "profile_id": METAL_FUSED_FULL_PROFILE.profile_id,
            "B_prepared_max": METAL_FUSED_FULL_PROFILE.maximum_state_bytes,
            "B_grid_texel": 16,
            "maximum_texture_reads": METAL_FUSED_FULL_PROFILE.maximum_reads,
            "matched_sampler_status": "python-slang-matched@1",
            "proposal_components": 11,
            "proposal_random_values": 2,
            "proposal_fallback_weight_floor": 0.02,
            "maximum_sample_steps": int(
                _layout["bounded_execution"]["maximum_sample_steps"]
            ),
            "maximum_pdf_steps": int(
                _layout["bounded_execution"]["maximum_pdf_steps"]
            ),
            "maximum_sample_evaluator_calls": int(
                _layout["bounded_execution"]["maximum_sample_evaluator_calls"]
            ),
            "observed_quality_gate": False,
        },
        metal_fused_parameter_groups(),
        _COMPONENTS,
    )

    def _deployment(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.descriptor.adaptation_contract(snapshot)
        cache_key = (snapshot.snapshot_id, id(checkpoint))
        cached = getattr(self, "_deployment_cache", None)
        if isinstance(cached, tuple) and cached[0] == cache_key:
            return cached[1]
        state_ids = checkpoint.get("source_snapshot_ids")
        state = checkpoint.get("model_state")
        training_config = checkpoint.get("training_config")
        if (
            not isinstance(state_ids, (list, tuple))
            or snapshot.snapshot_id not in map(str, state_ids)
            or not isinstance(state, Mapping)
            or not isinstance(training_config, Mapping)
        ):
            raise ValueError("Metal deployment requires a checkpoint containing this source")
        context = training_config.get("model_context")
        if not isinstance(context, Mapping):
            raise ValueError("Metal deployment checkpoint has no model_context")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.create_trainable(context).to(device)
        self.restore_training_state(model, state)
        quantize_runtime_model(model)
        adapter = MetalFusedMdlSourceAdapter((snapshot,), device)
        tensors = adapter.compiler_tensors_for_source(0, device=device)
        asset_index = adapter.asset_index_for_source(0)
        cooked = MetalAssetCooker(
            model,
            adapter.native_assets(),
            max_core_texels=262_144,
            encoder_halo=32,
            encoder_batch_tiles=1,
        ).cook_asset(
            asset_index, mode="encoder-only"
        )
        descriptor = adapter.native_assets().descriptors[asset_index]
        address_modes = {
            domain.domain_id: domain.address_mode for domain in descriptor.domains
        }
        packed_asset = pack_metal_asset(cooked, address_modes=address_modes)
        with torch.no_grad():
            program_state = model.typed_compiler(tensors)
        result = {
            "model": model,
            "adapter": adapter,
            "tensors": tensors,
            "cooked": cooked,
            "packed_asset": packed_asset,
            "program_state": program_state,
            "address_modes": address_modes,
        }
        self._deployment_cache = (cache_key, result)
        return result

    @staticmethod
    def _editor_view(
        snapshot: SourceSnapshot,
        adapter: MetalFusedMdlSourceAdapter,
    ) -> Mapping[str, Any]:
        view = MdlFamilyDefinition().describe_parameters(snapshot).to_dict()
        source = MdlMaterialSource.from_snapshot(snapshot)
        record = adapter.registry.resolve_exact_locator(source.module, source.export)
        by_name = {
            str(parameter["name"]): (index, parameter)
            for index, parameter in enumerate(record.parameters)
        }
        argument_names = set(source.arguments)
        derived: dict[str, list[dict[str, Any]]] = {}

        def normalization_range(
            parameter: Mapping[str, Any],
        ) -> dict[str, float]:
            if "minimum" in parameter and "maximum" in parameter:
                minimum = parameter["minimum"]
                maximum = parameter["maximum"]
            elif "soft_minimum" in parameter and "soft_maximum" in parameter:
                minimum = parameter["soft_minimum"]
                maximum = parameter["soft_maximum"]
            else:
                return {}

            def shared_bound(name: str, value: Any) -> float:
                if isinstance(value, (list, tuple)):
                    if not value or any(component != value[0] for component in value):
                        raise ValueError(
                            f"Metal editor {name} must be shared by all vector components"
                        )
                    value = value[0]
                return float(value)

            return {
                "minimum": shared_bound("minimum", minimum),
                "maximum": shared_bound("maximum", maximum),
            }

        def select(names: tuple[str, ...]) -> str | None:
            return next((name for name in names if name in argument_names), None)

        def add(name: str | None, word: int, operation: str, component: int = 0) -> None:
            if name is not None:
                derived.setdefault(name, []).append(
                    {"word": word, "operation": operation, "component": component}
                )

        color_name = select(("metal_color", "metal_tint", "normal_reflectivity", "color_1"))
        grazing_name = select(("grazing_reflectivity",)) or color_name
        for component in range(3):
            add(color_name, METAL_RAW_OFFSETS["optical"] + component, "copy", component)
            add(grazing_name, METAL_RAW_OFFSETS["optical"] + 3 + component, "copy", component)
        for component, name in enumerate((
            "roughness", "metal_roughness", "reflection_roughness",
            "steel_anisotropy", "brushing_anisotropy", "reflection_brightness",
            "metalness", "paint_roughness", "oxide_roughness", "polish_film_strength",
        )):
            add(select((name,)), METAL_RAW_OFFSETS["optical"] + 6 + component, "copy")
        access = METAL_RAW_OFFSETS["access"]
        for component in range(2):
            add(select(("texture_scale",)), access + component, "copy", component)
            add(select(("texture_translate",)), access + 2 + component, "copy", component)
        add(select(("texture_rotate",)), access + 4, "degrees-cos")
        add(select(("texture_rotate",)), access + 5, "degrees-sin")
        add(select(("infinite_tiling",)), access + 6, "bool")
        add(select(("no_uv",)), access + 7, "bool")
        add(select(("uv_space_index",)), access + 8, "copy")
        add(select(("scale",)), access + 9, "copy")
        frame = METAL_RAW_OFFSETS["frame"]
        add(select(("enable_round_corners", "roundcorners_enable")), frame, "bool")
        add(select(("radius", "radius_mm", "roundcorner_radius", "roundcorners_radius_mm")), frame + 1, "copy")
        add(select(("across_materials", "roundcorners_across_materials")), frame + 2, "bool")
        add(select(("object_scaled_bump",)), frame + 3, "bool")

        def annotate(node: dict[str, Any]) -> None:
            metadata = dict(node.get("metadata", {}))
            name = metadata.get("mdl_name")
            if name in by_name:
                index, parameter = by_name[str(name)]
                metadata["runtime"] = {
                    "token_index": index,
                    "continuous_word": METAL_RAW_OFFSETS["continuous"] + 4 * index,
                    "discrete_word": METAL_RAW_OFFSETS["discrete"] + index,
                    "type_word": METAL_RAW_OFFSETS["type"] + index,
                    "normalization": {
                        # The source parameter view is the canonical typed UI
                        # representation. In particular, MDL enum arguments are
                        # normalized from {"name", "value"} to their choice name.
                        "default": node["value"],
                        **normalization_range(parameter),
                    },
                    "derived_writes": derived.get(str(name), []),
                }
                node["metadata"] = metadata
            for child in node.get("children", []):
                annotate(child)

        annotate(view["root"])
        view["runtime_layout"] = {
            "schema": "ncls.metal-raw-typed-parameters@1",
            "word_count": METAL_RAW_WORD_COUNT,
            "offsets": dict(METAL_RAW_OFFSETS),
        }
        return view

    def create_trainable(self, context: Mapping[str, Any]) -> nn.Module:
        return MetalFusedNeuralMaterialModel.from_context(context)

    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        if config.get("correspondence_id") != "metal-fused-full-method@1":
            raise ValueError("Metal fused training requires its full evaluator/sampler correspondence")
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
            "proposal-fit",
            "qat-refine",
        ]:
            raise ValueError("Metal full method requires codec, joint, proposal and QAT phases")
        expected_groups = {
            "codec-warmup": list(_CODEC_GROUPS),
            "joint-appearance": list(_JOINT_GROUPS),
            "proposal-fit": list(_PROPOSAL_GROUPS),
            "qat-refine": list(_QAT_REFINE_GROUPS),
        }
        expected_routes = {
            "codec-warmup": {"asset": "asset-tile"},
            "joint-appearance": {
                "asset": "asset-tile",
                "evaluator": "reference-evaluator",
            },
            "proposal-fit": {
                "evaluator": "reference-evaluator",
                "sampler": "method-sampler",
            },
            "qat-refine": {
                "asset": "asset-tile",
                "evaluator": "reference-evaluator",
                "sampler": "method-sampler",
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
            "proposal-fit": [
                "proposal-forward-kl",
                "proposal-density-fit",
                "proposal-mode-coverage",
                "proposal-weight-tail",
                "sample-pdf-identity",
            ],
            "qat-refine": [
                "codec-full",
                "response-robust",
                "linear-energy",
                "peak-support",
                "reciprocity",
                "analytic-core-preservation",
                "teacher-response",
                "compiler-functional-distillation",
                "proposal-forward-kl",
                "proposal-density-fit",
                "proposal-mode-coverage",
                "proposal-weight-tail",
                "sample-pdf-identity",
                "runtime-fp16-qat",
            ],
        }
        expected_precision = {
            "codec-warmup": {"autocast": "fp32", "gradient_scaler": False},
            "joint-appearance": {"autocast": "bfloat16", "gradient_scaler": False},
            "proposal-fit": {"autocast": "bfloat16", "gradient_scaler": False},
            "qat-refine": {"autocast": "fp32", "gradient_scaler": False},
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
            if any(
                int(item.get("direction_count", 0)) != 1
                for item in phase.get("routes", ())
            ):
                raise ValueError("Metal online routes require one direction per source state")
            if phase.get("precision") != expected_precision[name]:
                raise ValueError("Metal phase precision drifted from its full method recipe")
            recipes = phase.get("recipes")
            if not isinstance(recipes, Mapping) or recipes.get("profile_id") != "metal_fused_full_v1":
                raise ValueError("Metal phase must freeze the full profile recipe")
            if phase.get("transition") is not None:
                raise ValueError("Metal evaluator phases cannot hide an asset lifecycle transition")
        qat_recipes = phases[3].get("recipes", {})
        if (
            qat_recipes.get("runtime_quantization")
            != "fp16-runtime-ste-int8-grid-qat-sensitive-fp32@1"
        ):
            raise ValueError("Metal QAT refine requires the deployed precision simulation")
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
        proposal_evaluator = next(
            item for item in phases[2]["routes"] if item["name"] == "evaluator"
        )
        proposal_sampler = next(
            item for item in phases[2]["routes"] if item["name"] == "sampler"
        )
        if (
            proposal_evaluator.get("options", {}).get("direction_proposal")
            != "uniform-half-difference@1"
            or proposal_sampler.get("options", {}).get("direction_proposal")
            != "uniform-hemisphere-conditioning@1"
        ):
            raise ValueError("Metal proposal phase requires evaluator and sampler strata")
        qat_routes = {item["name"]: item for item in phases[3]["routes"]}
        if (
            qat_routes["evaluator"].get("options", {}).get("direction_proposal")
            != "uniform-half-difference@1"
            or qat_routes["sampler"].get("options", {}).get("direction_proposal")
            != "uniform-hemisphere-conditioning@1"
            or int(qat_routes["evaluator"].get("options", {}).get("source_patch_size", 0)) < 8
            or not bool(qat_routes["asset"].get("options", {}).get("asset_indices"))
        ):
            raise ValueError("Metal QAT refine requires all deployed asset/evaluator/sampler strata")

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

    @staticmethod
    def _proposal_target(f: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        luminance = torch.sum(
            f * f.new_tensor((0.2126, 0.7152, 0.0722)), dim=-1
        )
        return (
            torch.clamp(luminance, min=0.0)
            * torch.clamp(wi[..., 2], min=0.0)
        ).detach()

    @staticmethod
    def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.where(valid, value, 0.0)) / torch.clamp(
            valid.to(value.dtype).sum(), min=1.0
        )

    def _proposal_objective(
        self,
        model: MetalFusedNeuralMaterialModel,
        batches: Mapping[str, OnlineTrainingBatch],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if set(batches) != {"evaluator", "sampler"}:
            raise ValueError("Metal proposal fit requires evaluator and sampler routes")
        evaluator_batch = batches["evaluator"]
        sampler_batch = batches["sampler"]
        if not isinstance(evaluator_batch, EvaluatorBatch) or not isinstance(
            sampler_batch, MethodSamplerBatch
        ):
            raise ValueError("Metal proposal fit received the wrong typed batches")

        sampler_values = sampler_batch.tensors
        sampler_spatial = model.spatial_state(sampler_values)
        sampler_program = model.typed_compiler(sampler_values)
        sampler_prepared = model.prepare_from_components(
            sampler_program, sampler_spatial, sampler_values
        )
        sampled = model.sample_prepared(
            sampler_prepared,
            sampler_values["wo"],
            sampler_values["sample_u"],
        )
        score_loss, sample_valid_fraction = sampler_forward_kl_score(
            sampled.f,
            sampled.wi,
            sampled.forward_pdf,
            sampled.valid,
        )

        evaluator_values = evaluator_batch.tensors
        evaluator_spatial = model.spatial_state(evaluator_values)
        evaluator_program = model.typed_compiler(evaluator_values)
        evaluator_prepared = model.prepare_from_components(
            evaluator_program, evaluator_spatial, evaluator_values
        )
        evaluator_prediction = model.evaluate_prepared(
            evaluator_prepared,
            evaluator_values["wo"],
            evaluator_values["wi"],
        )
        evaluator_density = model.pdf_prepared(
            evaluator_prepared,
            evaluator_values["wo"],
            evaluator_values["wi"],
        )
        query_target = self._proposal_target(
            evaluator_prediction.f, evaluator_values["wi"]
        )
        query_target = query_target / torch.clamp(query_target.mean(), min=1e-4)
        query_valid = evaluator_prediction.valid & evaluator_density.valid
        density_loss = self._masked_mean(
            -query_target * torch.log(torch.clamp(evaluator_density.forward, min=1e-12)),
            query_valid,
        )

        sample_u = sampler_values["sample_u"]
        wo = sampler_values["wo"]
        reflection = torch.stack((-wo[:, 0], -wo[:, 1], wo[:, 2]), dim=1)
        reflection = F.normalize(reflection, dim=1, eps=1e-8)
        phi = 2.0 * math.pi * sample_u[:, 1]
        grazing_z = 0.01 + 0.09 * sample_u[:, 0]
        grazing_radius = torch.sqrt(torch.clamp(1.0 - grazing_z.square(), min=0.0))
        grazing = torch.stack(
            (
                grazing_radius * torch.cos(phi),
                grazing_radius * torch.sin(phi),
                grazing_z,
            ),
            dim=1,
        )
        cosine_radius = torch.sqrt(sample_u[:, 0])
        cosine_direction = torch.stack(
            (
                cosine_radius * torch.cos(phi),
                cosine_radius * torch.sin(phi),
                torch.sqrt(torch.clamp(1.0 - sample_u[:, 0], min=0.0)),
            ),
            dim=1,
        )
        mode_directions = torch.stack(
            (reflection, grazing, cosine_direction), dim=1
        )
        mode_prediction = model.evaluate_prepared(
            sampler_prepared, wo, mode_directions
        )
        mode_density = model.pdf_prepared(
            sampler_prepared, wo, mode_directions
        )
        mode_target = self._proposal_target(mode_prediction.f, mode_directions)
        mode_target = mode_target / torch.clamp(mode_target.mean(), min=1e-4)
        mode_valid = mode_prediction.valid & mode_density.valid
        mode_loss = self._masked_mean(
            -mode_target * torch.log(torch.clamp(mode_density.forward, min=1e-12)),
            mode_valid,
        )

        sampled_target = self._proposal_target(sampled.f, sampled.wi)
        throughput_tail = sampled_target / torch.clamp(
            sampled.forward_pdf, min=1e-12
        )
        tail_loss = self._masked_mean(
            torch.log1p(throughput_tail.square()), sampled.valid
        )
        independent = model.pdf_prepared(
            sampler_prepared, wo, sampled.wi
        )
        pdf_error = torch.abs(sampled.forward_pdf - independent.forward)
        expected_weight = (
            sampled.f
            * torch.clamp(sampled.wi[..., 2:3], min=0.0)
            / torch.clamp(independent.forward[..., None], min=1e-12)
        )
        weight_error = torch.abs(sampled.weight - expected_weight).mean(dim=-1)
        identity_error = self._masked_mean(pdf_error + weight_error, sampled.valid)
        loss = (
            0.25 * score_loss
            + 0.5 * density_loss
            + 0.5 * mode_loss
            + 0.02 * tail_loss
            + identity_error
        )
        trace = sampler_prepared.trace
        component_histogram = F.one_hot(
            sampled.component, num_classes=11
        ).to(sampled.f.dtype)
        return loss, {
            "proposal_forward_kl_loss": score_loss.detach(),
            "proposal_density_fit_loss": density_loss.detach(),
            "proposal_mode_coverage_loss": mode_loss.detach(),
            "proposal_weight_tail_loss": tail_loss.detach(),
            "proposal_identity_error": identity_error.detach(),
            "proposal_valid_fraction": sample_valid_fraction.detach(),
            "proposal_state_trace": self._trace_metric(trace, "proposal_state"),
            "proposal_component_pdf_trace": (
                0.5
                * (
                    evaluator_density.component_pdfs.square().mean()
                    + mode_density.component_pdfs.square().mean()
                )
            ).detach(),
            "proposal_component_sample_trace": component_histogram.square().mean().detach(),
            "proposal_support_trace": mode_density.forward[:, 1].mean().detach(),
            "proposal_fallback_trace": sampler_prepared.proposal_state[:, -1, 0].mean().detach(),
            "proposal_sample_pdf_trace": sampled.forward_pdf.mean().detach(),
            "proposal_weight_identity_trace": (1.0 / (1.0 + identity_error.detach())),
        }

    def _qat_refine_objective(
        self,
        model: MetalFusedNeuralMaterialModel,
        batches: Mapping[str, OnlineTrainingBatch],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        runtime_names = set(metal_runtime_parameter_names(model))
        execution = _MetalQatRefineExecution(model, self)
        functional_state: dict[str, torch.Tensor] = {}
        available_runtime_names: set[str] = set()
        quantization_error = []
        for name, value in (
            *execution.named_parameters(),
            *execution.named_buffers(),
        ):
            model_name = name.removeprefix("model.")
            if name.startswith("model.") and model_name in runtime_names:
                available_runtime_names.add(model_name)
                functional_state[name] = fake_quantize_fp16_ste(value)
                quantization_error.append(
                    torch.mean(
                        torch.abs(
                            value.detach()
                            - value.detach().to(torch.float16).to(value.dtype)
                        )
                    )
                )
            else:
                functional_state[name] = value
        missing = runtime_names - available_runtime_names
        if missing:
            raise RuntimeError(
                f"Metal QAT runtime state is not functionally reachable: {sorted(missing)}"
            )
        loss, raw_metrics = torch.func.functional_call(
            execution,
            functional_state,
            (batches,),
            strict=True,
        )
        metrics = dict(raw_metrics)
        metrics["runtime_fp16_quantization_trace"] = torch.stack(
            quantization_error
        ).mean()
        return loss, metrics

    def training_objective(
        self,
        model: nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        if not isinstance(model, MetalFusedNeuralMaterialModel):
            raise TypeError("Metal fused method requires MetalFusedNeuralMaterialModel")
        phase_name = str(phase.get("name"))
        if phase_name == "proposal-fit":
            return self._proposal_objective(model, batches)
        if phase_name == "qat-refine":
            return self._qat_refine_objective(model, batches)
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
        state = checkpoint.get("model_state")
        training_config = checkpoint.get("training_config")
        if not isinstance(state, Mapping) or not isinstance(training_config, Mapping):
            raise ValueError("Metal runtime compilation requires checkpoint model_state")
        context = training_config.get("model_context")
        if not isinstance(context, Mapping):
            raise ValueError("Metal runtime compilation requires model_context")
        model = self.create_trainable(context)
        self.restore_training_state(model, state)
        packed = pack_metal_program(model)
        module = "ncls/backends/metal_fused/metal_fused.slang"
        closure = _module_closure(PROJECT_ROOT / "shaders" / module)
        return RuntimePayload(
            module,
            closure,
            {"shared-weights": packed.payload},
            {
                "shared-weights": {
                    "kind": "structured-buffer",
                    "dtype": "packed-float16x2-uint32@1",
                    "shape": [len(packed.payload) // 4],
                    "stride": 4,
                    "alignment": 16,
                    "usage": "gNclsRuntimeWeights",
                }
            },
            self.descriptor.capabilities,
            packed.defines,
        )

    def compile_asset(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload:
        deployment = self._deployment(snapshot, checkpoint)
        packed = deployment["packed_asset"]
        return MaterialPayload(
            snapshot.snapshot_id,
            packed.blobs,
            packed.descriptors,
            sampler_descriptors={
                "metal-grid-sampler": {
                    "kind": "sampler",
                    "usage": "gNclsMetalGridSampler",
                    "filter": "linear",
                    "address_mode": "wrap",
                }
            },
        )

    def compile_instance(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> InstancePayload:
        deployment = self._deployment(snapshot, checkpoint)
        tensors = deployment["tensors"]
        packed_asset = deployment["packed_asset"]
        compiled = pack_metal_compiled_material(
            deployment["program_state"],
            tensors,
            deployment["cooked"],
            domain_count=packed_asset.domain_count,
            maximum_extent=packed_asset.maximum_extent,
            maximum_mip=packed_asset.maximum_mip,
        )
        raw = pack_metal_raw_parameters(tensors)
        return InstancePayload(
            {"compiled_material_index": 0},
            {"metal-raw-parameters": raw, "compiled-material": compiled},
            {
                "metal-raw-parameters": {
                    "kind": "mutable-structured-buffer",
                    "dtype": "ncls-metal-raw-typed-parameters@1",
                    "shape": [METAL_RAW_WORD_COUNT],
                    "stride": 4,
                    "alignment": 16,
                    "usage": "gNclsMetalRawParameters",
                },
                "compiled-material": {
                    "kind": "mutable-structured-buffer",
                    "dtype": "ncls-metal-compiled-material@1",
                    "shape": [METAL_COMPILED_WORD_COUNT],
                    "stride": 4,
                    "alignment": 16,
                    "usage": "gNclsCompiledMaterials",
                },
            },
            {
                "schema": "ncls.typed-material-editor@1",
                "parameter_view": self._editor_view(snapshot, deployment["adapter"]),
                "raw_usage": "gNclsMetalRawParameters",
                "compiled_usage": "gNclsCompiledMaterials",
            },
            {"entry_point": "nclsCompileMaterial", "thread_group_size": [32, 1, 1]},
        )

    def package_validation(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        deployment = self._deployment(snapshot, checkpoint)
        with torch.no_grad():
            expected = evaluate_metal_cooked_asset(
                deployment["model"],
                deployment["cooked"],
                deployment["tensors"],
                uv=(0.371, 0.619),
                mip_level=0.375,
                wo=_PARITY_VIEW,
                wi=_PARITY_LIGHTS,
                address_modes=deployment["address_modes"],
            )
        return {
            "status": "gpu-parity-required",
            "parity": {
                "oracle": "metal-fused-full-fp16-int8-python@1",
                "uv": [0.371, 0.619],
                "mip_level": 0.375,
                "view": list(_PARITY_VIEW),
                "lights": [list(value) for value in _PARITY_LIGHTS],
                "expected_f": expected.tolist(),
                "relative_tolerance": 5e-2,
                "absolute_tolerance": 5e-4,
            },
            "storage": {
                "B_shared": len(pack_metal_program(deployment["model"]).payload),
                "B_asset": sum(len(value) for value in deployment["packed_asset"].blobs.values()),
                "B_instance": 4 * (METAL_RAW_WORD_COUNT + METAL_COMPILED_WORD_COUNT),
            },
        }

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
