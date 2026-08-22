import numpy as np

from baselines.ibl_eval import _integrate, analytic_probe_bank, sample_equirectangular
from datagen.directions import equal_area_hemisphere


def test_analytic_probe_bank_is_finite_nonnegative_and_integrates_constant_response() -> None:
    lights, weights = equal_area_hemisphere(32)
    names, probes = analytic_probe_bank(lights[:, :3])
    assert len(names) == len(probes) == 26
    assert np.all(np.isfinite(probes))
    assert np.all(probes >= 0.0)
    response = np.ones((2, 32, 3), dtype=np.float32)
    integrated = _integrate(response, probes[:1], weights)
    np.testing.assert_allclose(integrated[:, 0], 2.0 * np.pi, rtol=1e-6)


def test_equirectangular_sampling_preserves_constant_radiance() -> None:
    lights, _ = equal_area_hemisphere(32)
    image = np.ones((8, 16, 3), dtype=np.float32) * np.asarray([2.0, 0.5, 0.1])
    for azimuth in (0.0, 0.7, np.pi):
        sampled = sample_equirectangular(image, lights[:, :3], azimuth)
        np.testing.assert_allclose(
            sampled,
            np.repeat(image[0:1, 0], len(lights), axis=0),
            rtol=1e-6,
            atol=1e-7,
        )
