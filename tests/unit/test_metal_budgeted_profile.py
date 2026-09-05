from __future__ import annotations

from dataclasses import replace

import pytest

from ncls.learning.methods.metal.profile import (
    METAL_BUDGETED_CENTER_DETAIL_PROFILE,
    METAL_BUDGETED_DIRECT_PROFILE,
    METAL_BUDGETED_DUAL_LOCAL_PROFILE,
    METAL_BUDGETED_HYBRID_PROFILE,
    METAL_BUDGETED_ROLE_DETAIL_PROFILE,
    load_metal_budgeted_layout,
)
from tools.learning.generate_metal_budgeted_layout import render


def test_metal_budgeted_profiles_close_the_nvidia_class_hard_budget() -> None:
    layout = load_metal_budgeted_layout()
    hybrid = METAL_BUDGETED_HYBRID_PROFILE
    direct = METAL_BUDGETED_DIRECT_PROFILE

    assert hybrid.evaluator_mode == "hybrid"
    assert direct.evaluator_mode == "direct"
    assert hybrid.prepare_dense_macs == 2_560
    assert hybrid.evaluate_dense_macs == 11_392
    assert hybrid.evaluate_dense_macs <= 20_000
    assert hybrid.prepared_state_bytes == 160
    assert hybrid.prepared_state_bytes <= 192
    assert hybrid.maximum_texture_reads == 2
    assert hybrid.evaluator_layers == direct.evaluator_layers
    assert METAL_BUDGETED_ROLE_DETAIL_PROFILE.asset_detail_aggregation == (
        "role-separated-slot-softmax@1"
    )
    assert METAL_BUDGETED_ROLE_DETAIL_PROFILE.evaluate_dense_macs == 11_392
    assert METAL_BUDGETED_ROLE_DETAIL_PROFILE.prepared_state_bytes == 160
    assert METAL_BUDGETED_ROLE_DETAIL_PROFILE.maximum_texture_reads == 2
    assert METAL_BUDGETED_CENTER_DETAIL_PROFILE.asset_detail_center == (
        "request-texel@1"
    )
    assert METAL_BUDGETED_CENTER_DETAIL_PROFILE.asset_detail_aggregation == (
        "shared-slot-softmax@1"
    )
    assert METAL_BUDGETED_CENTER_DETAIL_PROFILE.evaluate_dense_macs == 11_392
    assert METAL_BUDGETED_CENTER_DETAIL_PROFILE.prepared_state_bytes == 160
    assert METAL_BUDGETED_CENTER_DETAIL_PROFILE.maximum_texture_reads == 2
    assert METAL_BUDGETED_DUAL_LOCAL_PROFILE.asset_spatial_features == (
        "signed-cross-summary@1"
    )
    assert METAL_BUDGETED_DUAL_LOCAL_PROFILE.asset_context_resolution_divisor == 1
    assert METAL_BUDGETED_DUAL_LOCAL_PROFILE.evaluate_dense_macs == 11_392
    assert METAL_BUDGETED_DUAL_LOCAL_PROFILE.prepared_state_bytes == 160
    assert METAL_BUDGETED_DUAL_LOCAL_PROFILE.maximum_texture_reads == 2
    assert layout["identity"] in render()


def test_metal_budgeted_profile_rejects_hard_budget_overrun() -> None:
    profile = METAL_BUDGETED_HYBRID_PROFILE
    with pytest.raises(ValueError, match="dense-MAC hard bound"):
        replace(profile, maximum_evaluate_dense_macs=10_000)
    with pytest.raises(ValueError, match="192-byte hard bound"):
        replace(profile, maximum_prepared_state_bytes=128)
    with pytest.raises(ValueError, match="texture-read bound"):
        replace(profile, maximum_texture_reads=3)


def test_metal_budgeted_packed_fields_are_nonoverlapping_and_complete() -> None:
    layout = load_metal_budgeted_layout()
    assert layout["material_program_state"]["total_bytes"] == 160
    assert layout["prepared_state"]["total_bytes"] == 176
    assert layout["shape"]["asset_plane_count"] == 2
    assert layout["proposal"]["components"] == [
        "primary-specular",
        "secondary-specular",
        "full-hemisphere-fallback",
    ]
