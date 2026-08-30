from __future__ import annotations

import pytest
import torch

from ncls.core.scattering import BackendCapability
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.models.metal_fused import (
    METAL_FUSED_REQUIRED_CONTEXT,
    MetalFusedNeuralMaterialModel,
    MetalPreparedModel,
)
from ncls.learning.models.metal_fused_profile import (
    METAL_FUSED_FULL_PROFILE,
    load_metal_fused_layout,
)


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
        MetalFusedNeuralMaterialModel.from_context(partial)


def test_metal_full_method_still_fails_closed_at_runtime_package_boundary() -> None:
    with pytest.raises(RuntimeError, match="runtime deployment"):
        METHOD_DEFINITION.compile_program({})


def test_metal_tensor_schema_and_parameter_registry_are_exact() -> None:
    with torch.device("meta"):
        model = MetalFusedNeuralMaterialModel.from_context(
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
