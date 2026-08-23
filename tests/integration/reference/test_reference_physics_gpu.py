from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ncls.core.material import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughDielectricInterface,
)
from ncls.data.reference import FalcorReferenceEvaluator, evaluate_reference_fixed
from ncls.data import CollectionConfig, ReferenceDataset, collect_reference_dataset
from ncls.data.directions import equal_area_hemisphere
from ncls.data.providers import LayerStackProvider, LayerStackProviderConfig


pytest.importorskip("falcor")
pytestmark = pytest.mark.falcor


def _direction(theta_degrees: float) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    return np.asarray([np.sin(theta), 0.0, np.cos(theta), 0.0], dtype=np.float32)


def _evaluate(stack: LayerStackIR, view_angle: float, light_angles: list[float], samples: int, seed: int = 1):
    lights = np.stack([_direction(angle) for angle in light_angles])
    evaluator = FalcorReferenceEvaluator(lights, max_depth=64, max_query_group_batch=1)
    return evaluate_reference_fixed(
        evaluator,
        [stack],
        _direction(view_angle)[None, :],
        query_group_seeds=np.asarray([seed], dtype=np.uint32),
        samples_per_replica=samples,
    )


def test_single_interface_diffuse_matches_analytic_response_cos() -> None:
    color = np.asarray([0.6, 0.3, 0.1], dtype=np.float32)
    stack = LayerStackIR((DiffuseInterface(tuple(color)),), ())
    angles = [-50.0, 0.0, 55.0]
    result = _evaluate(stack, 25.0, angles, 4)
    expected = color[None, :] * np.cos(np.deg2rad(angles))[:, None] / np.pi
    np.testing.assert_allclose(result.mean[0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.variance[0], 0.0, atol=1e-8)


def _three_interface_stack() -> LayerStackIR:
    return LayerStackIR(
        (
            RoughDielectricInterface(0.12, 0.08, 1.4, 0.1),
            RoughDielectricInterface(0.24, 0.16, 1.15, -0.35),
            DiffuseInterface((0.5, 0.25, 0.1)),
        ),
        (
            HomogeneousMedium((0.1, 0.05, 0.02), thickness=0.25),
            HomogeneousMedium(thickness=0.35),
        ),
    )


def test_three_interface_reference_is_reciprocal_within_monte_carlo_error() -> None:
    stack = _three_interface_stack()
    samples = 32768
    view_angle = 20.0
    light_angle = 50.0
    forward = _evaluate(stack, view_angle, [light_angle], samples, seed=191)
    reverse = _evaluate(stack, light_angle, [view_angle], samples, seed=211)
    total_samples = 2 * samples
    forward_cosine = np.cos(np.deg2rad(light_angle))
    reverse_cosine = np.cos(np.deg2rad(view_angle))
    forward_f = forward.mean[0, 0] / forward_cosine
    reverse_f = reverse.mean[0, 0] / reverse_cosine
    standard_error = np.sqrt(
        forward.variance[0, 0] / (total_samples * forward_cosine**2)
        + reverse.variance[0, 0] / (total_samples * reverse_cosine**2)
    )
    assert np.all(np.abs(forward_f - reverse_f) <= 6.0 * standard_error + 3e-3)


def test_eight_interface_reference_executes_with_anisotropic_frames() -> None:
    interfaces = tuple(
        RoughDielectricInterface(
            0.08 + 0.02 * index,
            0.1 + 0.015 * index,
            1.5 if index == 0 else 1.0,
            0.1 * index,
        )
        for index in range(7)
    ) + (DiffuseInterface((0.4, 0.2, 0.1)),)
    stack = LayerStackIR(interfaces, tuple(HomogeneousMedium(thickness=0.1) for _ in range(7)))
    result = _evaluate(stack, 20.0, [-30.0, 0.0, 40.0], 64)
    assert np.all(np.isfinite(result.mean))
    assert np.all(result.mean >= 0.0)
    assert np.max(result.mean) > 0.0


def test_unsupported_chromatic_extinction_with_scattering_is_rejected() -> None:
    stack = LayerStackIR(
        (RoughDielectricInterface(0.1, 0.2, 1.5), DiffuseInterface((0.5, 0.5, 0.5))),
        (HomogeneousMedium((0.1, 0.2, 0.3), (0.2, 0.2, 0.2), 0.0, 0.2),),
    )
    with pytest.raises(RuntimeError, match="unsupported material state"):
        _evaluate(stack, 20.0, [0.0], 4)


def test_reference_generator_smoke(tmp_path: Path) -> None:
    collection = CollectionConfig(
        view_count=1,
        light_count=4,
        seed=23,
    )
    config = LayerStackProviderConfig(
        family_count=1,
        local_state_count=1,
        samples_per_replica=4,
        query_group_batch=1,
        max_depth=8,
    )
    lights, _ = equal_area_hemisphere(4)
    evaluator = FalcorReferenceEvaluator(lights, max_depth=8, max_query_group_batch=1)
    provider = LayerStackProvider(collection, config, evaluator=evaluator)
    manifest = collect_reference_dataset(
        tmp_path / "reference.h5",
        [provider],
        collection,
        created_at="2026-08-23T00:00:00+00:00",
        generator_git_commit="test",
    )
    with ReferenceDataset.open(tmp_path / "reference.h5") as dataset:
        statistics = dataset.statistics(0)
        assert manifest.provider_metadata[0]["reference_id"] == "ncls.layer-stack-random-walk@1"
        assert np.all(np.isfinite(statistics.mean))
        assert np.all(statistics.mean >= 0.0)
        assert statistics.sample_count.tolist() == [8, 8, 8, 8]


def test_reference_generator_supports_per_view_peak_grazing_queries(tmp_path: Path) -> None:
    collection = CollectionConfig(
        view_count=2,
        validation_view_count=1,
        test_view_count=1,
        adversarial_view_count=1,
        light_count=32,
        seed=31,
        query_profile_id="ncls.e0-peak-grazing-mixture@1",
    )
    config = LayerStackProviderConfig(
        family_count=1,
        local_state_count=1,
        samples_per_replica=2,
        query_group_batch=2,
        max_depth=8,
    )
    provider = LayerStackProvider(collection, config)
    collect_reference_dataset(
        tmp_path / "mixture-reference.h5",
        [provider],
        collection,
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )
    with ReferenceDataset.open(tmp_path / "mixture-reference.h5") as dataset:
        batch = dataset.group_batch((0, 1, 2, 3, 4))
        assert batch["wi"].shape == (5, 32, 3)
        np.testing.assert_array_equal(batch["query_role"], [0, 0, 1, 2, 3])
        assert not np.array_equal(batch["wi"][0], batch["wi"][1])
        np.testing.assert_allclose(
            batch["proposal_pdf"] * batch["solid_angle_weight"],
            1.0 / 32.0,
            rtol=1e-6,
            atol=1e-7,
        )
