import numpy as np

from datagen.directions import equal_area_hemisphere, stratified_view_directions
from datagen.priors import assign_family_splits, sample_stack_families, sample_stacks
from schema import LayerType, pack_stack


def test_equal_area_hemisphere_grid() -> None:
    directions, weights = equal_area_hemisphere(128)
    np.testing.assert_allclose(np.linalg.norm(directions[:, :3], axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert np.all(directions[:, 2] > 0.0)
    np.testing.assert_allclose(np.sum(weights), 2.0 * np.pi, rtol=1e-6)


def test_view_grid_is_normalized_and_grazing_biased() -> None:
    views = stratified_view_directions(16)
    np.testing.assert_allclose(np.linalg.norm(views[:, :3], axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert np.all(views[:, 2] > 0.0)
    assert np.min(views[:, 2]) < 0.2


def test_stack_prior_is_deterministic_and_matches_supported_v0_family() -> None:
    first = sample_stacks(32, seed=1234)
    second = sample_stacks(32, seed=1234)
    assert [pack_stack(stack) for stack in first] == [pack_stack(stack) for stack in second]

    for stack in first:
        for layer in stack.layers[:-1]:
            assert layer.layer_type == LayerType.ROUGH_DIELECTRIC
        assert stack.layers[-1].layer_type in {
            LayerType.DIFFUSE,
            LayerType.ROUGH_CONDUCTOR,
            LayerType.SHEEN,
        }
        for medium in stack.media:
            if any(value > 0.0 for value in medium.sigma_s):
                sigma_t = np.asarray(medium.sigma_a) + np.asarray(medium.sigma_s)
                np.testing.assert_allclose(sigma_t, sigma_t[0], rtol=1e-6, atol=1e-7)


def test_local_states_preserve_family_topology_and_scattering_contract() -> None:
    first = sample_stack_families(6, 4, seed=77)
    second = sample_stack_families(6, 4, seed=77)
    assert [[pack_stack(stack) for stack in family] for family in first] == [
        [pack_stack(stack) for stack in family] for family in second
    ]
    for family in first:
        topology = tuple(layer.layer_type for layer in family[0].layers)
        assert all(tuple(layer.layer_type for layer in state.layers) == topology for state in family)
        for state in family:
            for medium in state.media:
                if any(value > 0.0 for value in medium.sigma_s):
                    sigma_t = np.asarray(medium.sigma_a) + np.asarray(medium.sigma_s)
                    np.testing.assert_allclose(sigma_t, sigma_t[0], rtol=1e-6, atol=1e-7)


def test_family_splits_are_deterministic_disjoint_and_nonempty() -> None:
    first = assign_family_splits(20, seed=91)
    second = assign_family_splits(20, seed=91)
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == {0, 1, 2}
    assert len(first) == 20
