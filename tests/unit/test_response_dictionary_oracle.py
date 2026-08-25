from __future__ import annotations

import numpy as np

from ncls.learning.direct_fit.response_dictionary import (
    _distortion_rows,
    _fit_kmeans,
    _matched_pca,
    _paired_bootstrap,
    _top2_reconstruct,
)


def test_top2_closed_form_reconstructs_points_on_codeword_segment() -> None:
    centers = np.asarray([[0.0, 0.0], [2.0, 4.0], [8.0, 1.0]], dtype=np.float32)
    values = np.asarray([[0.5, 1.0], [1.5, 3.0]], dtype=np.float32)
    reconstruction, indices, weights = _top2_reconstruct(values, centers)

    np.testing.assert_allclose(reconstruction, values, atol=1e-3)
    np.testing.assert_array_equal(indices, np.asarray([[0, 1], [1, 0]], dtype=np.uint16))
    np.testing.assert_allclose(weights, np.asarray([0.25, 0.25]), atol=1e-3)


def test_kmeans_and_matched_pca_are_deterministic_and_respect_byte_budget() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(12, 18)).astype(np.float32)
    first, labels_a, iterations_a = _fit_kmeans(
        values, 4, seed=19, maximum_iterations=20
    )
    second, labels_b, iterations_b = _fit_kmeans(
        values, 4, seed=19, maximum_iterations=20
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(labels_a, labels_b)
    assert iterations_a == iterations_b

    byte_budget = 4 * first.size + 6 * len(values)
    reconstruction, rank, stored_bytes = _matched_pca(values, byte_budget)
    assert reconstruction.shape == values.shape
    assert 0 < rank < len(values)
    assert stored_bytes <= byte_budget


def test_m3_paired_bootstrap_uses_matched_units() -> None:
    baseline = np.linspace(0.2, 0.4, 30)
    candidate = baseline - 0.05
    result = _paired_bootstrap(baseline, candidate, seed=7)
    assert result["iterations"] == 1000
    assert result["dictionary_significantly_better"]
    assert result["interval"][1] < 0.0


def test_relative_l1_uses_a_representative_floor_for_zero_signal_units() -> None:
    reference = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    reconstruction = np.asarray([[0.1, 0.1], [1.1, 0.9]], dtype=np.float32)
    transformed, linear, normalization = _distortion_rows(
        reference,
        reconstruction,
        np.ones(2, dtype=np.float32),
    )

    assert np.isclose(normalization["transformed_l1_floor"], 0.02)
    assert np.all(np.isfinite(transformed)) and np.all(np.isfinite(linear))
    assert np.isclose(transformed[0], 10.0)
