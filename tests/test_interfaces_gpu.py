import numpy as np
import pytest


falcor = pytest.importorskip("falcor")

from datagen.two_layer_slice import diffuse_stack, direction, evaluate_slice
from datagen.validate_interfaces import SAMPLE_ALL, SAMPLE_REFLECTION, SAMPLE_TRANSMISSION, validate_interfaces
from schema import LayerInterface, LayerType


pytestmark = pytest.mark.falcor


def test_diffuse_eval_sampling_and_pdf_agree() -> None:
    albedo = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    layer = LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=tuple(albedo))
    result = validate_interfaces(
        [layer],
        direction(30.0),
        direction(-20.0),
        sample_modes=SAMPLE_REFLECTION,
        sample_count=256,
    )

    np.testing.assert_allclose(result.eval_f[0], albedo / np.pi, rtol=1e-6)
    np.testing.assert_allclose(result.sample_mean[0], albedo, rtol=1e-6)
    assert result.max_pdf_mismatch[0] == 0.0
    assert result.valid_count[0] == 256


def test_self_contained_dielectric_matches_falcor_direct_coat() -> None:
    stack = diffuse_stack()
    coat = stack.layers[0]
    angles = np.array([-50.0, -15.0, 0.0, 35.0, 60.0], dtype=np.float32)
    incident = np.repeat(direction(20.0), len(angles), axis=0)
    outgoing = direction(angles)
    ours = validate_interfaces(
        [coat] * len(angles),
        incident,
        outgoing,
        eta_i=1.0,
        eta_t=coat.eta[0],
        sample_modes=SAMPLE_ALL,
        sample_count=16,
    )
    falcor_direct = evaluate_slice(
        stack,
        shader_entry="evaluateDirectCoat",
        view_angle_degrees=20.0,
        angles_degrees=angles,
    ).mean

    ours_with_cosine = ours.eval_f * np.cos(np.deg2rad(angles))[:, None]
    np.testing.assert_allclose(ours_with_cosine, falcor_direct, rtol=2e-5, atol=1e-7)
    assert np.max(ours.max_pdf_mismatch) == 0.0


def test_self_contained_dielectric_transmission_matches_falcor() -> None:
    stack = diffuse_stack()
    coat = stack.layers[0]
    angles = np.array([105.0, 125.0, 150.0], dtype=np.float32)
    incident = np.repeat(direction(20.0), len(angles), axis=0)
    outgoing = direction(angles)
    ours = validate_interfaces(
        [coat] * len(angles),
        incident,
        outgoing,
        eta_i=1.0,
        eta_t=coat.eta[0],
        sample_modes=SAMPLE_ALL,
        sample_count=16,
    )
    falcor_direct = evaluate_slice(
        stack,
        shader_entry="evaluateDirectCoat",
        view_angle_degrees=20.0,
        angles_degrees=angles,
    ).mean

    ours_with_cosine = ours.eval_f * np.abs(np.cos(np.deg2rad(angles)))[:, None]
    np.testing.assert_allclose(ours_with_cosine, falcor_direct, rtol=2e-5, atol=1e-7)


def test_self_contained_dielectric_internal_transmission_matches_falcor() -> None:
    stack = diffuse_stack()
    coat = stack.layers[0]
    angles = np.array([-35.0, 0.0, 35.0], dtype=np.float32)
    incident = np.repeat(direction(160.0), len(angles), axis=0)
    outgoing = direction(angles)
    ours = validate_interfaces(
        [coat] * len(angles),
        incident,
        outgoing,
        eta_i=coat.eta[0],
        eta_t=1.0,
        sample_modes=SAMPLE_ALL,
        sample_count=16,
    )
    falcor_direct = evaluate_slice(
        stack,
        shader_entry="evaluateDirectCoat",
        view_angle_degrees=160.0,
        angles_degrees=angles,
    ).mean

    ours_with_cosine = ours.eval_f * np.cos(np.deg2rad(angles))[:, None]
    np.testing.assert_allclose(ours_with_cosine, falcor_direct, rtol=2e-5, atol=1e-7)


def test_forced_transmission_sampling_matches_falcor_expectation() -> None:
    stack = diffuse_stack()
    coat = stack.layers[0]
    sample_count = 65536
    ours = validate_interfaces(
        [coat],
        direction(20.0),
        direction(0.0),
        eta_i=1.0,
        eta_t=coat.eta[0],
        sample_modes=SAMPLE_TRANSMISSION,
        sample_count=sample_count,
        seed=7,
    )
    falcor_transmission = evaluate_slice(
        stack,
        shader_entry="sampleDirectCoatTransmission",
        view_angle_degrees=20.0,
        angles_degrees=np.array([0.0], dtype=np.float32),
        sample_count=sample_count,
        seed=31,
    )

    standard_error = np.sqrt(
        ours.sample_variance[0] / sample_count + falcor_transmission.variance[0] / sample_count
    )
    assert np.all(np.abs(ours.sample_mean[0] - falcor_transmission.mean[0]) <= 6.0 * standard_error + 2e-3)


def test_forced_internal_reflection_sampling_matches_falcor_expectation() -> None:
    stack = diffuse_stack()
    coat = stack.layers[0]
    sample_count = 65536
    ours = validate_interfaces(
        [coat],
        direction(160.0),
        direction(160.0),
        eta_i=coat.eta[0],
        eta_t=1.0,
        sample_modes=SAMPLE_REFLECTION,
        sample_count=sample_count,
        seed=13,
    )
    falcor_reflection = evaluate_slice(
        stack,
        shader_entry="sampleDirectCoatReflection",
        view_angle_degrees=160.0,
        angles_degrees=np.array([160.0], dtype=np.float32),
        sample_count=sample_count,
        seed=37,
    )

    standard_error = np.sqrt(
        ours.sample_variance[0] / sample_count + falcor_reflection.variance[0] / sample_count
    )
    assert np.all(np.abs(ours.sample_mean[0] - falcor_reflection.mean[0]) <= 6.0 * standard_error + 2e-3)


def test_atomic_interfaces_pass_white_furnace_bound() -> None:
    atomic_layers = [
        LayerInterface(LayerType.ROUGH_DIELECTRIC, 0.18, 0.1, eta=(1.5, 1.5, 1.5)),
        LayerInterface(
            LayerType.ROUGH_CONDUCTOR,
            0.25,
            0.12,
            eta=(0.2, 0.9, 1.1),
            k=(3.9, 2.5, 2.1),
        ),
        LayerInterface(LayerType.DIFFUSE, 1.0, 1.0, albedo=(0.7, 0.3, 0.1)),
        LayerInterface(LayerType.SHEEN, 0.45, 0.45, albedo=(0.8, 0.2, 0.1)),
    ]
    angles = [0.0, 45.0, 75.0]
    layers = [layer for layer in atomic_layers for _ in angles]
    incident = np.concatenate([direction(np.asarray(angles, dtype=np.float32)) for _ in atomic_layers])
    result = validate_interfaces(
        layers,
        incident,
        direction(np.zeros(len(layers), dtype=np.float32)),
        sample_modes=SAMPLE_ALL,
        sample_count=32768,
        seed=43,
    )

    assert np.all(result.sample_mean >= 0.0)
    assert np.all(result.sample_mean <= 1.02)
    assert np.max(result.max_pdf_mismatch) == 0.0
