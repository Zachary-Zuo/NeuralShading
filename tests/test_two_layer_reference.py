import numpy as np
import pytest


falcor = pytest.importorskip("falcor")

from datagen.two_layer_slice import diffuse_stack, evaluate_slice, gray_diffuse_stack
from schema import LayerMedium, LayerStack


pytestmark = pytest.mark.falcor


@pytest.mark.parametrize("shader_entry", ["evaluateTwoLayerReference", "evaluateTwoLayerTeacher"])
def test_same_seed_is_deterministic(shader_entry: str) -> None:
    angles = np.array([-45.0, 0.0, 45.0], dtype=np.float32)
    first = evaluate_slice(
        diffuse_stack(), shader_entry=shader_entry, sample_count=64, seed=17, angles_degrees=angles
    )
    second = evaluate_slice(
        diffuse_stack(), shader_entry=shader_entry, sample_count=64, seed=17, angles_degrees=angles
    )
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.variance, second.variance)


def test_response_is_finite_nonnegative_and_nonzero() -> None:
    result = evaluate_slice(
        diffuse_stack(),
        shader_entry="evaluateTwoLayerTeacher",
        sample_count=64,
        seed=3,
        angles_degrees=np.array([-70.0, -20.0, 0.0, 35.0, 70.0], dtype=np.float32),
    )
    assert np.all(np.isfinite(result.mean))
    assert np.all(result.mean >= 0.0)
    assert np.max(result.mean) > 0.0


def test_direct_coat_is_reciprocal_after_removing_light_cosine() -> None:
    stack = diffuse_stack()
    view_angle = 20.0
    light_angle = 50.0
    forward = evaluate_slice(
        stack,
        shader_entry="evaluateDirectCoat",
        view_angle_degrees=view_angle,
        angles_degrees=np.array([light_angle], dtype=np.float32),
    ).mean[0]
    reverse = evaluate_slice(
        stack,
        shader_entry="evaluateDirectCoat",
        view_angle_degrees=light_angle,
        angles_degrees=np.array([view_angle], dtype=np.float32),
    ).mean[0]

    forward_brdf = forward / np.cos(np.deg2rad(light_angle))
    reverse_brdf = reverse / np.cos(np.deg2rad(view_angle))
    np.testing.assert_allclose(forward_brdf, reverse_brdf, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("shader_entry", ["evaluateTwoLayerReference", "evaluateTwoLayerTeacher"])
def test_reciprocity_after_removing_light_cosine(shader_entry: str) -> None:
    stack = diffuse_stack()
    view_angle = 20.0
    light_angle = 50.0
    sample_count = 16384
    forward = evaluate_slice(
        stack,
        shader_entry=shader_entry,
        view_angle_degrees=view_angle,
        sample_count=sample_count,
        seed=11,
        angles_degrees=np.array([light_angle], dtype=np.float32),
    )
    reverse = evaluate_slice(
        stack,
        shader_entry=shader_entry,
        view_angle_degrees=light_angle,
        sample_count=sample_count,
        seed=29,
        angles_degrees=np.array([view_angle], dtype=np.float32),
    )

    forward_cosine = np.cos(np.deg2rad(light_angle))
    reverse_cosine = np.cos(np.deg2rad(view_angle))
    forward_brdf = forward.mean[0] / forward_cosine
    reverse_brdf = reverse.mean[0] / reverse_cosine
    standard_error = np.sqrt(
        forward.variance[0] / (sample_count * forward_cosine**2)
        + reverse.variance[0] / (sample_count * reverse_cosine**2)
    )
    assert np.all(np.abs(forward_brdf - reverse_brdf) <= 6.0 * standard_error + 2e-3)


def test_teacher_matches_falcor_before_russian_roulette() -> None:
    stack = gray_diffuse_stack()
    sample_count = 16384
    angles = np.array([-55.0, -20.0, 0.0, 35.0, 60.0], dtype=np.float32)
    reference = evaluate_slice(
        stack,
        shader_entry="evaluateTwoLayerReference",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=sample_count,
        max_depth=4,
        seed=11,
    )
    teacher = evaluate_slice(
        stack,
        shader_entry="evaluateTwoLayerTeacher",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=sample_count,
        max_depth=4,
        seed=53,
    )

    standard_error = np.sqrt((reference.variance + teacher.variance) / sample_count)
    assert np.all(np.abs(reference.mean - teacher.mean) <= 6.0 * standard_error + 2e-3)


def _with_medium(stack: LayerStack, medium: LayerMedium) -> LayerStack:
    return LayerStack(stack.layers, (medium,))


def test_zero_thickness_absorbing_medium_degenerates_to_clear_slab() -> None:
    clear = diffuse_stack()
    zero_thickness = _with_medium(
        clear,
        LayerMedium(sigma_a=(2.0, 1.0, 0.5), thickness=0.0),
    )
    angles = np.array([-40.0, 0.0, 45.0], dtype=np.float32)
    clear_result = evaluate_slice(
        clear,
        shader_entry="evaluateTwoLayerTeacher",
        sample_count=256,
        seed=71,
        angles_degrees=angles,
    )
    zero_result = evaluate_slice(
        zero_thickness,
        shader_entry="evaluateTwoLayerTeacher",
        sample_count=256,
        seed=71,
        angles_degrees=angles,
    )
    np.testing.assert_array_equal(clear_result.mean, zero_result.mean)


def test_absorption_reduces_indirect_response() -> None:
    clear = diffuse_stack()
    absorbing = _with_medium(
        clear,
        LayerMedium(sigma_a=(1.5, 1.0, 0.5), thickness=0.4),
    )
    angles = np.array([-35.0, 0.0, 40.0], dtype=np.float32)
    clear_result = evaluate_slice(
        clear,
        shader_entry="evaluateTwoLayerTeacher",
        sample_count=1024,
        seed=73,
        angles_degrees=angles,
    )
    absorbing_result = evaluate_slice(
        absorbing,
        shader_entry="evaluateTwoLayerTeacher",
        sample_count=1024,
        seed=73,
        angles_degrees=angles,
    )
    assert np.all(absorbing_result.mean <= clear_result.mean + 1e-7)
    assert np.any(absorbing_result.mean < 0.9 * clear_result.mean)


def test_scattering_medium_is_reciprocal_within_mc_error() -> None:
    stack = _with_medium(
        diffuse_stack(),
        LayerMedium(
            sigma_a=(0.8, 0.6, 0.4),
            sigma_s=(0.2, 0.4, 0.6),
            g=0.3,
            thickness=0.25,
        ),
    )
    view_angle = 20.0
    light_angle = 50.0
    sample_count = 32768
    forward = evaluate_slice(
        stack,
        shader_entry="evaluateTwoLayerTeacher",
        view_angle_degrees=view_angle,
        sample_count=sample_count,
        seed=79,
        angles_degrees=np.array([light_angle], dtype=np.float32),
    )
    reverse = evaluate_slice(
        stack,
        shader_entry="evaluateTwoLayerTeacher",
        view_angle_degrees=light_angle,
        sample_count=sample_count,
        seed=97,
        angles_degrees=np.array([view_angle], dtype=np.float32),
    )

    forward_cosine = np.cos(np.deg2rad(light_angle))
    reverse_cosine = np.cos(np.deg2rad(view_angle))
    forward_brdf = forward.mean[0] / forward_cosine
    reverse_brdf = reverse.mean[0] / reverse_cosine
    standard_error = np.sqrt(
        forward.variance[0] / (sample_count * forward_cosine**2)
        + reverse.variance[0] / (sample_count * reverse_cosine**2)
    )
    assert np.all(np.abs(forward_brdf - reverse_brdf) <= 6.0 * standard_error + 3e-3)


def test_chromatic_extinction_with_scattering_is_rejected_in_v0() -> None:
    invalid = _with_medium(
        diffuse_stack(),
        LayerMedium(sigma_a=(0.1, 0.2, 0.3), sigma_s=(0.2, 0.2, 0.2), thickness=0.5),
    )
    with pytest.raises(ValueError, match="achromatic"):
        evaluate_slice(
            invalid,
            shader_entry="evaluateTwoLayerTeacher",
            angles_degrees=np.array([0.0], dtype=np.float32),
        )
