import numpy as np
import pytest


falcor = pytest.importorskip("falcor")

from datagen.two_layer_slice import direction, evaluate_slice, gray_diffuse_stack
from schema import LayerInterface, LayerMedium, LayerStack, LayerType


pytestmark = pytest.mark.falcor


def test_single_layer_degenerates_to_analytic_lambert() -> None:
    albedo = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    stack = LayerStack(
        (LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=tuple(albedo)),),
        (),
    )
    angles = np.array([-50.0, 0.0, 55.0], dtype=np.float32)
    result = evaluate_slice(
        stack,
        shader_entry="evaluateLayerStackTeacher",
        view_angle_degrees=25.0,
        angles_degrees=angles,
        sample_count=8,
    )
    expected = albedo[None, :] * np.cos(np.deg2rad(angles))[:, None] / np.pi
    np.testing.assert_allclose(result.mean, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.variance, 0.0, atol=1e-8)


def test_general_n2_matches_two_layer_teacher_in_expectation() -> None:
    stack = gray_diffuse_stack()
    sample_count = 65536
    angles = np.array([-40.0, 0.0, 50.0], dtype=np.float32)
    specialized = evaluate_slice(
        stack,
        shader_entry="evaluateTwoLayerTeacher",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=sample_count,
        seed=101,
    )
    general = evaluate_slice(
        stack,
        shader_entry="evaluateLayerStackTeacher",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=sample_count,
        seed=131,
    )
    standard_error = np.sqrt((specialized.variance + general.variance) / sample_count)
    assert np.all(np.abs(specialized.mean - general.mean) <= 6.0 * standard_error + 3e-3)


def _three_layer_with_null_interface() -> LayerStack:
    original = gray_diffuse_stack()
    null_interface = LayerInterface(
        LayerType.ROUGH_DIELECTRIC,
        roughness_x=0.4,
        roughness_y=0.2,
        eta=(1.0, 1.0, 1.0),
    )
    return LayerStack(
        (original.layers[0], null_interface, original.layers[1]),
        (LayerMedium(thickness=0.4), LayerMedium(thickness=0.6)),
    )


def _nontrivial_three_layer_stack() -> LayerStack:
    return LayerStack(
        (
            LayerInterface(
                LayerType.ROUGH_DIELECTRIC,
                roughness_x=0.12,
                roughness_y=0.08,
                eta=(1.4, 1.4, 1.4),
            ),
            LayerInterface(
                LayerType.ROUGH_DIELECTRIC,
                roughness_x=0.24,
                roughness_y=0.16,
                eta=(1.15, 1.15, 1.15),
                tangent_rotation=0.35,
            ),
            LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.5, 0.25, 0.1)),
        ),
        (
            LayerMedium(sigma_a=(0.1, 0.05, 0.02), thickness=0.25),
            LayerMedium(thickness=0.35),
        ),
    )
def test_inserting_eta_one_interface_preserves_response_in_expectation() -> None:
    original = gray_diffuse_stack()
    expanded = _three_layer_with_null_interface()
    sample_count = 65536
    angles = np.array([-35.0, 0.0, 45.0], dtype=np.float32)
    original_result = evaluate_slice(
        original,
        shader_entry="evaluateLayerStackTeacher",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=sample_count,
        seed=151,
    )
    expanded_result = evaluate_slice(
        expanded,
        shader_entry="evaluateLayerStackTeacher",
        view_angle_degrees=20.0,
        angles_degrees=angles,
        sample_count=sample_count,
        seed=181,
    )
    standard_error = np.sqrt((original_result.variance + expanded_result.variance) / sample_count)
    assert np.all(np.abs(original_result.mean - expanded_result.mean) <= 6.0 * standard_error + 3e-3)


def test_three_layer_stack_is_reciprocal_within_mc_error() -> None:
    stack = _nontrivial_three_layer_stack()
    sample_count = 131072
    view_angle = 20.0
    light_angle = 50.0
    forward = evaluate_slice(
        stack,
        shader_entry="evaluateLayerStackTeacher",
        view_angle_degrees=view_angle,
        angles_degrees=np.array([light_angle], dtype=np.float32),
        sample_count=sample_count,
        seed=191,
    )
    reverse = evaluate_slice(
        stack,
        shader_entry="evaluateLayerStackTeacher",
        view_angle_degrees=light_angle,
        angles_degrees=np.array([view_angle], dtype=np.float32),
        sample_count=sample_count,
        seed=211,
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


def test_eight_interface_stack_executes() -> None:
    dielectric_layers = [
        LayerInterface(
            LayerType.ROUGH_DIELECTRIC,
            roughness_x=0.08 + 0.02 * index,
            roughness_y=0.1 + 0.015 * index,
            eta=(1.5 if index == 0 else 1.0,) * 3,
        )
        for index in range(7)
    ]
    base = LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.4, 0.2, 0.1))
    stack = LayerStack(tuple(dielectric_layers + [base]), tuple(LayerMedium(thickness=0.1) for _ in range(7)))
    result = evaluate_slice(
        stack,
        shader_entry="evaluateLayerStackTeacher",
        sample_count=1024,
        angles_degrees=np.array([-30.0, 0.0, 40.0], dtype=np.float32),
    )
    assert np.all(np.isfinite(result.mean))
    assert np.all(result.mean >= 0.0)
    assert np.max(result.mean) > 0.0
