from __future__ import annotations

import pytest

from ncls.core.material import (
    HomogeneousMedium,
    MaterialProgram,
    RoughConductorInterface,
    RoughDielectricInterface,
)
from ncls.data import CollectionConfig
from ncls.data.priors import (
    E0_LAYER_STACK_BOUNDARY_CASE_IDS,
    E0_LAYER_STACK_BOUNDARY_PROFILE_ID,
    e0_layer_stack_boundary_cases,
)
from ncls.data.providers import LayerStackProvider, LayerStackProviderConfig


def test_e0_boundary_profile_has_frozen_physical_cases() -> None:
    cases = e0_layer_stack_boundary_cases()
    assert tuple(case_id for case_id, _ in cases) == E0_LAYER_STACK_BOUNDARY_CASE_IDS

    narrow_dielectric = cases[0][1].interfaces[0]
    assert isinstance(narrow_dielectric, RoughDielectricInterface)
    assert (narrow_dielectric.alpha_x, narrow_dielectric.alpha_y) == (0.002, 0.002)

    narrow_conductor = cases[1][1].interfaces[0]
    assert isinstance(narrow_conductor, RoughConductorInterface)
    assert narrow_conductor.alpha_x == 0.002
    assert narrow_conductor.alpha_y == 0.08

    scattering_medium = cases[4][1].media[0]
    assert isinstance(scattering_medium, HomogeneousMedium)
    assert scattering_medium.sigma_s == (0.2, 0.5, 0.8)
    assert tuple(a + s for a, s in zip(scattering_medium.sigma_a, scattering_medium.sigma_s, strict=True)) == pytest.approx((1.0, 1.0, 1.0))
    assert len(cases[5][1].interfaces) == 3


def test_e0_boundary_profile_is_traceable_in_native_payload_and_provider_metadata() -> None:
    config = LayerStackProviderConfig(
        family_count=len(E0_LAYER_STACK_BOUNDARY_CASE_IDS),
        local_state_count=1,
        state_profile_id=E0_LAYER_STACK_BOUNDARY_PROFILE_ID,
    )
    provider = LayerStackProvider(CollectionConfig(view_count=1, light_count=4, seed=20260824), config)
    states = provider.source_states()
    assert len(states) == len(E0_LAYER_STACK_BOUNDARY_CASE_IDS)
    assert {state.split for state in states} == {0, 1, 2}
    assert provider.metadata()["provider_config"]["state_profile_id"] == E0_LAYER_STACK_BOUNDARY_PROFILE_ID

    for state, expected_case_id in zip(states, E0_LAYER_STACK_BOUNDARY_CASE_IDS, strict=True):
        program = MaterialProgram.from_json(state.native_payload.decode("utf-8"))
        assert program.metadata["state_profile_id"] == E0_LAYER_STACK_BOUNDARY_PROFILE_ID
        assert program.metadata["state_profile_case_id"] == expected_case_id


def test_e0_boundary_profile_rejects_unversioned_shape_changes() -> None:
    with pytest.raises(ValueError, match="family_count=6"):
        LayerStackProviderConfig(
            family_count=5,
            local_state_count=1,
            state_profile_id=E0_LAYER_STACK_BOUNDARY_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="local_state_count=1"):
        LayerStackProviderConfig(
            family_count=len(E0_LAYER_STACK_BOUNDARY_CASE_IDS),
            local_state_count=2,
            state_profile_id=E0_LAYER_STACK_BOUNDARY_PROFILE_ID,
        )
