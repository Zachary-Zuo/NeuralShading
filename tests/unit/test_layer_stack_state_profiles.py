from __future__ import annotations

import numpy as np
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
    E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID,
    E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID,
    E2_LAYER_STACK_SHARED_DECODER_FAMILY_COUNT,
    E2_LAYER_STACK_SHARED_DECODER_LOCAL_STATE_COUNT,
    E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
    e0_layer_stack_boundary_cases,
    e1_layer_stack_multi_interface_cases,
    e1_layer_stack_narrow_conductor_cases,
    e2_layer_stack_shared_decoder_families,
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


def test_e1_narrow_conductor_profile_is_one_explicit_capacity_state() -> None:
    cases = e1_layer_stack_narrow_conductor_cases()
    assert tuple(case_id for case_id, _ in cases) == ("narrow-anisotropic-conductor",)
    config = CollectionConfig(
        view_count=8,
        validation_view_count=2,
        test_view_count=2,
        adversarial_view_count=2,
        light_count=32,
        query_profile_id="ncls.e1-independent-peak-grazing-mixture@1",
    )
    provider = LayerStackProvider(
        config,
        LayerStackProviderConfig(
            family_count=1,
            local_state_count=1,
            state_profile_id=E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID,
        ),
        evaluator=object(),
    )
    assert len(provider.source_states()) == 1
    state = provider.source_states()[0]
    assert state.split == 0
    plan = provider.query_plan(state)
    assert plan.query_roles.tolist() == [0] * 8 + [1] * 2 + [2] * 2 + [3] * 2
    assert all("peak" in proposal for proposal in plan.proposal_id)
    assert np.min(plan.view_directions[:, 2]) < np.sin(np.deg2rad(5.0))


def test_e1_multi_interface_profile_is_one_explicit_residual_state() -> None:
    cases = e1_layer_stack_multi_interface_cases()
    assert tuple(case_id for case_id, _ in cases) == ("multi-interface-moving-peaks",)
    stack = cases[0][1]
    assert len(stack.interfaces) == 3
    assert len(stack.media) == 2

    provider = LayerStackProvider(
        CollectionConfig(view_count=1, light_count=4, seed=20260824),
        LayerStackProviderConfig(
            family_count=1,
            local_state_count=1,
            state_profile_id=E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID,
        ),
        evaluator=object(),
    )
    state = provider.source_states()[0]
    program = MaterialProgram.from_json(state.native_payload.decode("utf-8"))
    assert program.metadata["state_profile_id"] == E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID
    assert program.metadata["state_profile_case_id"] == "multi-interface-moving-peaks"


def test_e1_multi_interface_profile_rejects_shape_changes() -> None:
    with pytest.raises(ValueError, match="family_count=1"):
        LayerStackProviderConfig(
            family_count=2,
            local_state_count=1,
            state_profile_id=E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="local_state_count=1"):
        LayerStackProviderConfig(
            family_count=1,
            local_state_count=2,
            state_profile_id=E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID,
        )


def test_e2_shared_decoder_profile_freezes_topology_coverage() -> None:
    families = e2_layer_stack_shared_decoder_families(20260824)
    assert len(families) == E2_LAYER_STACK_SHARED_DECODER_FAMILY_COUNT
    assert all(len(family) == E2_LAYER_STACK_SHARED_DECODER_LOCAL_STATE_COUNT for family in families)
    assert [len(family[0].interfaces) for family in families[:8]] == list(range(1, 9))
    assert {
        type(family[0].interfaces[-1]).__name__ for family in families
    } == {"DiffuseInterface", "RoughConductorInterface", "SheenInterface"}
    for family in families:
        signature = tuple(type(item) for item in family[0].interfaces)
        assert all(tuple(type(item) for item in state.interfaces) == signature for state in family)


def test_e2_shared_decoder_profile_keeps_families_in_source_splits() -> None:
    provider = LayerStackProvider(
        CollectionConfig(view_count=1, light_count=4, seed=20260824),
        LayerStackProviderConfig(
            family_count=E2_LAYER_STACK_SHARED_DECODER_FAMILY_COUNT,
            local_state_count=E2_LAYER_STACK_SHARED_DECODER_LOCAL_STATE_COUNT,
            state_profile_id=E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
        ),
        evaluator=object(),
    )
    states = provider.source_states()
    assert len(states) == 24
    assert np.bincount([state.split for state in states], minlength=3).tolist() == [20, 2, 2]
    for start in range(0, len(states), E2_LAYER_STACK_SHARED_DECODER_LOCAL_STATE_COUNT):
        family = states[start : start + E2_LAYER_STACK_SHARED_DECODER_LOCAL_STATE_COUNT]
        assert len({state.split_group_id for state in family}) == 1
        assert len({state.split for state in family}) == 1
        for state in family:
            program = MaterialProgram.from_json(state.native_payload.decode("utf-8"))
            assert program.metadata["state_profile_id"] == E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID


def test_e2_shared_decoder_profile_rejects_shape_changes() -> None:
    with pytest.raises(ValueError, match="family_count=12"):
        LayerStackProviderConfig(
            family_count=11,
            local_state_count=2,
            state_profile_id=E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="local_state_count=2"):
        LayerStackProviderConfig(
            family_count=12,
            local_state_count=1,
            state_profile_id=E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
        )


def test_e2_query_profile_anchors_wo_and_tracks_single_sheen_peak() -> None:
    provider = LayerStackProvider(
        CollectionConfig(
            view_count=4,
            validation_view_count=4,
            test_view_count=4,
            adversarial_view_count=4,
            light_count=128,
            seed=20260824,
            query_profile_id="ncls.e2-layer-stack-independent-peak-grazing-mixture@2",
        ),
        LayerStackProviderConfig(
            family_count=12,
            local_state_count=2,
            state_profile_id=E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
        ),
        evaluator=object(),
    )
    sheen_state = provider.source_states()[0]
    assert type(sheen_state.runtime_state.interfaces[0]).__name__ == "SheenInterface"
    plan = provider.query_plan(sheen_state)
    assert np.bincount(plan.query_roles, minlength=4).tolist() == [4, 4, 4, 4]
    for role in range(4):
        selected = plan.query_roles == role
        assert np.sum(plan.view_directions[selected, 2] < np.sin(np.deg2rad(5.0))) == 1
    assert all("layer-stack-sheen-peak" in value for value in plan.proposal_id)
    centers = provider._sheen_peak_centers(
        plan.view_directions,
        sheen_state.runtime_state.interfaces[0].roughness,
    )
    legacy_centers = provider._sheen_peak_centers(
        plan.view_directions,
        sheen_state.runtime_state.interfaces[0].roughness,
        legacy_v1_semantics=True,
    )
    non_grazing = plan.view_directions[:, 2] > 0.1
    assert np.min(centers[non_grazing, 2]) > 0.1
    assert np.max(centers[:, 2]) < 0.5
    assert np.max(legacy_centers[:, 2]) < 0.02
    nearest = np.degrees(np.arccos(np.clip(
        np.max(np.sum(plan.light_directions * centers[:, None, :], axis=-1), axis=1),
        -1.0,
        1.0,
    )))
    assert np.max(nearest) < 0.5


def test_e2_adaptive_override_is_split_group_scoped_and_traceable() -> None:
    config = LayerStackProviderConfig(
        family_count=12,
        local_state_count=2,
        adaptive=True,
        batch_samples=2048,
        min_samples=8192,
        max_samples=131072,
        adaptive_max_samples_by_split_group=(("layer-stack-e2-family-0003", 262144),),
        state_profile_id=E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
    )
    provider = LayerStackProvider(
        CollectionConfig(view_count=1, light_count=16, seed=20260824),
        config,
        evaluator=object(),
    )
    states = provider.source_states()
    assert provider._adaptive_max_samples(states[6]) == 262144
    assert provider._adaptive_max_samples(states[7]) == 262144
    assert provider._adaptive_max_samples(states[5]) == 131072
    assert provider.metadata()["provider_config"]["adaptive_max_samples_by_split_group"] == (
        ("layer-stack-e2-family-0003", 262144),
    )
    with pytest.raises(ValueError, match="unknown split groups"):
        LayerStackProvider(
            CollectionConfig(view_count=1, light_count=16),
            LayerStackProviderConfig(
                family_count=12,
                local_state_count=2,
                adaptive_max_samples_by_split_group=(("typo", 262144),),
                state_profile_id=E2_LAYER_STACK_SHARED_DECODER_PROFILE_ID,
            ),
            evaluator=object(),
        )


def test_layer_stack_provider_honors_query_group_batch_for_dense_view_profiles() -> None:
    class RecordingEvaluator:
        light_count = 32

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def evaluate_query_groups(
            self,
            materials,
            view_directions,
            *,
            sample_count_per_replica,
            query_group_seeds,
            light_directions,
            sample_offset=0,
        ):
            del view_directions, sample_count_per_replica, query_group_seeds, light_directions, sample_offset
            self.batch_sizes.append(len(materials))
            mean = np.full((len(materials), self.light_count, 3), 0.2, dtype=np.float32)
            second = mean * mean + 0.01
            return mean, second, mean, second

    collection = CollectionConfig(
        view_count=8,
        validation_view_count=2,
        test_view_count=2,
        adversarial_view_count=2,
        light_count=32,
        query_profile_id="ncls.e1-independent-peak-grazing-mixture@1",
    )
    evaluator = RecordingEvaluator()
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=1,
            local_state_count=1,
            samples_per_replica=2,
            query_group_batch=3,
            state_profile_id=E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID,
        ),
        evaluator=evaluator,
    )
    state = provider.source_states()[0]
    surfaces = provider.surface_samples(state)
    plan = provider.query_plan(state, surfaces)
    result = provider.evaluate(state, surfaces, plan)
    assert evaluator.batch_sizes == [3, 3, 3, 3, 2]
    assert result.mean.shape == (1, 14, 32, 3)
