import numpy as np
import pytest
import torch

from baselines.closure_families import (
    OracleClosureModule,
    _inverse_softplus,
    eval_ggx,
    eval_ltc,
    eval_sg,
    fit_oracle_batch,
)
from datagen.directions import equal_area_hemisphere


def test_closure_evaluators_are_finite_nonnegative_and_have_expected_shape() -> None:
    lights, _ = equal_area_hemisphere(16)
    light_tensor = torch.from_numpy(lights[:, :3])
    views = torch.tensor([[0.2, 0.0, np.sqrt(0.96)]], dtype=torch.float32)
    raw_axis = torch.tensor([[[0.0, 0.0, 0.5], [0.2, -0.1, 0.3]]], dtype=torch.float32)
    raw_amplitude = torch.zeros((1, 2, 3), dtype=torch.float32)

    sg = eval_sg(light_tensor, raw_axis, torch.zeros((1, 2)), raw_amplitude)
    ggx = eval_ggx(views, light_tensor, raw_axis, torch.full((1, 2), -1.0), raw_amplitude)
    ltc = eval_ltc(
        light_tensor,
        torch.zeros((1, 2, 2)),
        torch.zeros((1, 2, 3)),
        torch.zeros((1, 2)),
        raw_amplitude,
    )
    for values in (sg, ggx, ltc):
        assert values.shape == (1, 16, 3)
        assert torch.all(torch.isfinite(values))
        assert torch.all(values >= 0.0)


def test_sg_oracle_recovers_single_synthetic_lobe() -> None:
    lights, _ = equal_area_hemisphere(48)
    axis = np.asarray([0.35, -0.2, np.sqrt(1.0 - 0.35**2 - 0.2**2)], dtype=np.float32)
    amplitude = np.asarray([0.8, 0.35, 0.1], dtype=np.float32)
    target = np.exp(24.0 * (lights[:, :3] @ axis - 1.0))[:, None] * amplitude[None, :]
    result = fit_oracle_batch(
        target[None],
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        lights[:, :3],
        family="sg",
        lobe_count=1,
        steps=180,
        restarts=1,
        learning_rate=0.04,
        device="cpu",
        seed=7,
    )
    assert result.smape[0] < 2e-3
    assert result.relative_l1[0] < 2e-3
    assert result.parameters["sharpness"][0, 0] == pytest.approx(24.0, rel=2e-2)


def test_ggx_oracle_recovers_single_synthetic_lobe() -> None:
    lights, _ = equal_area_hemisphere(48)
    views = np.asarray([[0.2, 0.0, np.sqrt(0.96)]], dtype=np.float32)
    target = eval_ggx(
        torch.from_numpy(views),
        torch.from_numpy(lights[:, :3]),
        torch.tensor([[[0.0, 0.0, _inverse_softplus(torch.tensor(1.0)).item()]]]),
        torch.tensor([[np.log(0.22)]], dtype=torch.float32),
        _inverse_softplus(torch.tensor([[[0.5, 0.25, 0.1]]], dtype=torch.float32)),
    ).numpy()
    result = fit_oracle_batch(
        target,
        views,
        lights[:, :3],
        family="ggx",
        lobe_count=1,
        steps=500,
        restarts=2,
        learning_rate=0.03,
        device="cpu",
        seed=11,
    )
    assert result.smape[0] < 1e-3
    assert result.relative_l1[0] < 1e-3


def test_identity_ltc_is_exact_lambert_response() -> None:
    lights, _ = equal_area_hemisphere(32)
    albedo = torch.tensor([[[0.6, 0.3, 0.1]]], dtype=torch.float32)
    response = eval_ltc(
        torch.from_numpy(lights[:, :3]),
        torch.zeros((1, 1, 2)),
        torch.zeros((1, 1, 3)),
        torch.zeros((1, 1)),
        _inverse_softplus(albedo),
    )[0]
    expected = torch.from_numpy(lights[:, 2:3]) * albedo[0] / np.pi
    torch.testing.assert_close(response, expected, rtol=1e-6, atol=1e-7)


def test_exported_parameters_include_evaluator_shape_bounds() -> None:
    target = torch.ones((1, 8, 3), dtype=torch.float32)
    views = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    ltc = OracleClosureModule("ltc", 2, views, target, seed=3)
    ltc.log_scale.data[0, 0] = torch.tensor([4.0, -4.0])
    exported_ltc = ltc.export_parameters()
    np.testing.assert_allclose(
        exported_ltc["inverse_scale"][0, 0],
        np.exp(np.asarray([3.0, -3.0], dtype=np.float32)),
        rtol=1e-6,
    )

    ggx = OracleClosureModule("ggx", 1, views, target, seed=5)
    ggx.log_shape.data[0, 0] = 2.0
    assert ggx.export_parameters()["alpha"][0, 0] == pytest.approx(1.0)
