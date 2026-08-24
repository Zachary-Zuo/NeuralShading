from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ncls.core.material import DiffuseInterface, HomogeneousMedium, LayerStackIR, RoughDielectricInterface
from ncls.data import CollectionConfig, ReferenceDataset, collect_reference_dataset
from ncls.data.providers import LayerStackProvider, LayerStackProviderConfig
from ncls.data.reference import FalcorReferenceEvaluator, evaluate_reference_fixed


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


def test_single_interface_diffuse_matches_reference_measure() -> None:
    color = np.asarray([0.6, 0.3, 0.1], dtype=np.float32)
    stack = LayerStackIR((DiffuseInterface(tuple(color)),), ())
    angles = [-50.0, 0.0, 55.0]
    result = _evaluate(stack, 25.0, angles, 4)
    expected = color[None, :] * np.cos(np.deg2rad(angles))[:, None] / np.pi
    np.testing.assert_allclose(result.mean[0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.variance[0], 0.0, atol=1e-8)


def test_multilayer_reference_is_reciprocal_within_monte_carlo_error() -> None:
    stack = LayerStackIR(
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
    samples = 32768
    forward = _evaluate(stack, 20.0, [50.0], samples, seed=191)
    reverse = _evaluate(stack, 50.0, [20.0], samples, seed=211)
    forward_cosine = np.cos(np.deg2rad(50.0))
    reverse_cosine = np.cos(np.deg2rad(20.0))
    forward_f = forward.mean[0, 0] / forward_cosine
    reverse_f = reverse.mean[0, 0] / reverse_cosine
    standard_error = np.sqrt(
        forward.variance[0, 0] / (2 * samples * forward_cosine**2)
        + reverse.variance[0, 0] / (2 * samples * reverse_cosine**2)
    )
    assert np.all(np.abs(forward_f - reverse_f) <= 6.0 * standard_error + 3e-3)


def test_layer_stack_v1_reference_shard_smoke(tmp_path: Path) -> None:
    collection = CollectionConfig(
        name="uniform-v1",
        query_role="test",
        view_count=1,
        light_count=4,
        seed=23,
    )
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=4,
            states_per_family=3,
            heldout_family_count=1,
            fixed_samples_per_replica=1,
            max_dispatch_queries=4,
            max_depth=8,
        ),
    )
    path = tmp_path / "reference.h5"
    collect_reference_dataset(
        path,
        (provider,),
        collection,
        created_at="2026-08-24T00:00:00+00:00",
        generator_git_commit="test",
    )
    with ReferenceDataset.open(path) as dataset:
        assert dataset.manifest.format_name == "reference-shard"
        assert dataset.manifest.sampling_name == "uniform-v1"
        assert dataset.state_count == 12
        assert dataset.direction_count == 4
        assert np.all(np.isfinite(dataset.stream["responses/mean"][...]))
        assert np.all(dataset.stream["responses/sample_count"][...] == 2)
