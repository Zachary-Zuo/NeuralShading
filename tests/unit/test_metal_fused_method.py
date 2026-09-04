from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ncls.core.identity import sha256_file
from ncls.core.scattering import BackendCapability, validate_typed_parameter_view
from ncls.core.source import SourceSnapshot
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.source_adapters import _normalized_components
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalModel,
    MetalPreparedModel,
)
from ncls.learning.models.metal_fused_profile import (
    METAL_FUSED_FULL_PROFILE,
    load_metal_fused_layout,
)
from ncls.source_materials.mdl import (
    MDL_FAMILY_ID,
    MDL_NATIVE_SCHEMA,
    MdlMaterialSource,
)


def test_metal_parameter_normalization_accepts_vector_ranges() -> None:
    values = _normalized_components(
        {
            "value": [0.5, 0.25],
            "minimum": [0.0, 0.0],
            "maximum": [1.0, 0.5],
        },
        [0.25, 0.125],
    )
    assert values.tolist() == pytest.approx([-0.5, -0.5, 0.0, 0.0])


def test_metal_editor_normalizes_enum_default_and_shared_vector_range() -> None:
    module_root = Path("tests/fixtures/mdl").resolve()
    module = module_root / "constant_diffuse.mdl"
    source = MdlMaterialSource(
        module_root,
        "project.fixtures",
        "1",
        "::constant_diffuse",
        "::constant_diffuse::constant_diffuse(color)",
        {
            "mode": {
                "mdl_type": "enum",
                "value": {"name": "MediumPits", "value": 1},
                "editable": True,
                "choices": [
                    {"name": "LightPits", "value": 0},
                    {"name": "MediumPits", "value": 1},
                    {"name": "StrongPits", "value": 2},
                ],
            },
            "texture_scale": {
                "mdl_type": "float2",
                "value": [1.0, 1.0],
                "editable": True,
                "soft_minimum": [0.0, 0.0],
                "soft_maximum": [2.0, 2.0],
            },
        },
        "1.7",
    )
    snapshot = SourceSnapshot(
        MDL_FAMILY_ID,
        1,
        MDL_NATIVE_SCHEMA,
        sha256_file(module),
        source.to_payload(),
        {"constant_diffuse.mdl": sha256_file(module)},
        {"module_root": str(module_root)},
        source,
    )
    record = SimpleNamespace(
        parameters=(
            {
                "name": "mode",
                "type": "enum",
                "value": {"name": "MediumPits", "value": 1},
            },
            {
                "name": "texture_scale",
                "type": "float2",
                "value": [1.0, 1.0],
                "soft_minimum": [0.0, 0.0],
                "soft_maximum": [2.0, 2.0],
            },
        )
    )
    registry = SimpleNamespace(
        resolve_exact_locator=lambda _module, _export: record
    )
    view = METHOD_DEFINITION._editor_view(
        snapshot,
        SimpleNamespace(registry=registry),
    )

    validate_typed_parameter_view(view)
    nodes = view["root"]["children"][0]["children"]
    enum = next(node for node in nodes if node["path"] == "/arguments/mode")
    texture_scale = next(
        node for node in nodes if node["path"] == "/arguments/texture_scale"
    )
    assert enum["value"] == "MediumPits"
    assert enum["metadata"]["runtime"]["normalization"]["default"] == "MediumPits"
    assert texture_scale["metadata"]["runtime"]["normalization"] == {
        "default": [1.0, 1.0],
        "minimum": 0.0,
        "maximum": 2.0,
    }


def test_metal_full_profile_freezes_all_quality_first_shape_bounds() -> None:
    profile = METAL_FUSED_FULL_PROFILE
    assert profile.maximum_texture_slots == 9
    assert profile.maximum_typed_tokens == 32
    assert profile.encoder_widths == (64, 128, 192, 256)
    assert (profile.grid_high_channels, profile.grid_low_channels) == (8, 8)
    assert profile.learned_frame_count == 3
    assert (profile.core_lobe_count, profile.residual_lobe_count) == (6, 4)
    assert profile.maximum_reads == 9 * 2 * 5 + 4 * 4
    assert profile.maximum_state_bytes <= 4096
    assert profile.maximum_sample_random_values == 2
    assert profile.maximum_sample_evaluator_calls == 1
    layout = load_metal_fused_layout()
    assert layout["asset_packing"]["mip"].startswith("independent-source")
    assert len(layout["proposal_reservation"]["components"]) == 11


def test_metal_descriptor_registers_complete_evaluator_and_matched_sampler() -> None:
    descriptor = METHOD_DEFINITION.descriptor
    required = {
        component.component_id
        for component in descriptor.components
        if component.required
    }
    assert required == {
        "role-aware-texture-stems",
        "bundle-set-shared-unet-encoder",
        "independent-high-low-qat-grids",
        "shared-structured-decoder",
        "training-semantic-heads",
        "bounded-rank8-asset-adapter",
        "pure-typed-set-compiler",
        "target-visible-optimized-state-control",
        "deterministic-spatial-access-two-mip-prepare",
        "learned-lobe-frames-and-view-prepare",
        "raw-cartesian-direction",
        "stable-half-difference-direction",
        "shared-warped-angular-bank",
        "six-slot-source-aware-analytic-core",
        "bounded-multiplicative-correction",
        "four-positive-residual-lobes",
        "free-positive-rgb-tail",
        "eleven-component-matched-proposal-mixture",
        "folded-full-hemisphere-support",
        "sample-pdf-throughput-identity",
    }
    assert descriptor.capabilities & int(BackendCapability.PREPARE)
    assert descriptor.capabilities & int(BackendCapability.EVALUATE)
    assert descriptor.capabilities & int(BackendCapability.SAMPLE)
    assert descriptor.capabilities & int(BackendCapability.PDF)
    assert set(descriptor.parameter_groups) == {
        "codec_role_stems",
        "codec_encoder",
        "codec_decoder",
        "codec_semantic_heads",
        "asset_adapter",
        "quantization",
        "typed_compiler",
        "optimized_state_teacher",
        "prepared_model",
        "angular_bank",
        "analytic_core",
        "hybrid_evaluator",
        "proposal_sampler",
    }


def test_metal_model_rejects_a_tiny_or_partial_context() -> None:
    partial = dict(METAL_FUSED_REQUIRED_CONTEXT)
    partial["asset_count"] = 2
    with pytest.raises(ValueError, match="exact quality-first full profile"):
        MetalModel.from_context(partial)


def test_metal_runtime_compiler_requires_a_validated_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint model_state"):
        METHOD_DEFINITION.compile_program({})


def test_metal_tensor_schema_and_parameter_registry_are_exact() -> None:
    with torch.device("meta"):
        model = MetalModel.from_context(
            METAL_FUSED_REQUIRED_CONTEXT
        )
    registry = METHOD_DEFINITION.parameter_registry(model)
    assert set(registry) == set(METHOD_DEFINITION.descriptor.parameter_groups)
    assert sum(len(values) for values in registry.values()) == len(
        tuple(model.named_parameters())
    )
    assert {
        field.name for field in METHOD_DEFINITION.descriptor.tensor_state_schema
    } == set(model.state_dict())


def test_metal_spatial_access_transforms_footprint_and_rejects_nonfinite_state() -> None:
    uv = torch.tensor([[0.75, 0.75], [0.25, 0.5]])
    uv_dx = torch.tensor([[0.125, 0.0], [0.125, 0.0]])
    uv_dy = torch.tensor([[0.0, 0.25], [0.0, 0.25]])
    state = torch.zeros((2, 16))
    state[:, 0:2] = torch.tensor((2.0, 0.5))
    state[:, 4] = 1.0
    state[:, 6] = 1.0
    state[1, 0] = torch.nan
    transformed, transformed_dx, transformed_dy, valid = (
        MetalPreparedModel.execute_spatial_access(uv, uv_dx, uv_dy, state)
    )
    torch.testing.assert_close(transformed[0], torch.tensor((0.5, 0.375)))
    torch.testing.assert_close(transformed_dx[0], torch.tensor((0.25, 0.0)))
    torch.testing.assert_close(transformed_dy[0], torch.tensor((0.0, 0.125)))
    assert valid.tolist() == [True, False]
