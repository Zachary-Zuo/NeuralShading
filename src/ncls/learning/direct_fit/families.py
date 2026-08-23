from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def _hemisphere_axis(raw_axis: torch.Tensor) -> torch.Tensor:
    axis = torch.cat((raw_axis[..., :2], F.softplus(raw_axis[..., 2:3]) + 1e-4), dim=-1)
    return F.normalize(axis, dim=-1)


def _positive_amplitude(raw_amplitude: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw_amplitude)


def eval_sg(
    light_directions: torch.Tensor,
    raw_axis: torch.Tensor,
    log_sharpness: torch.Tensor,
    raw_amplitude: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a positive spherical-Gaussian mixture over the incident hemisphere."""
    axis = _hemisphere_axis(raw_axis)
    sharpness = torch.exp(torch.clamp(log_sharpness, math.log(0.05), math.log(1024.0)))
    amplitude = _positive_amplitude(raw_amplitude)
    cosine = torch.einsum("tkc,bc->tkb", axis, light_directions)
    basis = torch.exp(sharpness[..., None] * (cosine - 1.0))
    return torch.einsum("tkb,tkc->tbc", basis, amplitude)


def _smith_g1(cosine: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    cosine = torch.clamp(cosine, min=1e-6)
    alpha2 = alpha * alpha
    return 2.0 * cosine / (
        cosine + torch.sqrt(torch.clamp(alpha2 + (1.0 - alpha2) * cosine * cosine, min=1e-12))
    )


def eval_ggx(
    view_directions: torch.Tensor,
    light_directions: torch.Tensor,
    raw_normal: torch.Tensor,
    log_alpha: torch.Tensor,
    raw_amplitude: torch.Tensor,
) -> torch.Tensor:
    """Evaluate constant-Fresnel isotropic GGX closures as f*cos(theta_l)."""
    normal = _hemisphere_axis(raw_normal)
    alpha = torch.exp(torch.clamp(log_alpha, math.log(0.02), math.log(1.0)))
    amplitude = _positive_amplitude(raw_amplitude)
    ndv = torch.einsum("tkc,tc->tk", normal, view_directions)
    ndl = torch.einsum("tkc,bc->tkb", normal, light_directions)
    half_vector = F.normalize(view_directions[:, None, :] + light_directions[None, :, :], dim=-1)
    ndh = torch.einsum("tkc,tbc->tkb", normal, half_vector)
    alpha2 = alpha * alpha
    denominator = math.pi * torch.square(1.0 + (alpha2[..., None] - 1.0) * ndh * ndh)
    distribution = alpha2[..., None] / torch.clamp(denominator, min=1e-10)
    geometry = _smith_g1(ndv, alpha)[..., None] * _smith_g1(ndl, alpha[..., None])
    shape = distribution * geometry / torch.clamp(4.0 * ndv[..., None], min=1e-5)
    shape = torch.where((ndv[..., None] > 0.0) & (ndl > 0.0), shape, torch.zeros_like(shape))
    return torch.einsum("tkb,tkc->tbc", shape, amplitude)


def eval_ltc(
    light_directions: torch.Tensor,
    log_scale: torch.Tensor,
    shear: torch.Tensor,
    angle: torch.Tensor,
    raw_amplitude: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a compact anisotropic linearly transformed cosine mixture.

    Each inverse transform is upper triangular. Two positive scales, three
    shears and one tangent-frame angle define the lobe; RGB amplitude is
    separate. The density includes the exact solid-angle Jacobian.
    """
    scale = torch.exp(torch.clamp(log_scale, -3.0, 3.0))
    bounded_shear = 3.0 * torch.tanh(shear)
    cosine = torch.cos(angle)[..., None]
    sine = torch.sin(angle)[..., None]
    light_x = light_directions[None, None, :, 0]
    light_y = light_directions[None, None, :, 1]
    light_z = light_directions[None, None, :, 2]
    rotated_x = cosine * light_x + sine * light_y
    rotated_y = -sine * light_x + cosine * light_y
    qx = (
        scale[..., 0:1] * rotated_x
        + bounded_shear[..., 0:1] * rotated_y
        + bounded_shear[..., 1:2] * light_z
    )
    qy = scale[..., 1:2] * rotated_y + bounded_shear[..., 2:3] * light_z
    qz = light_z.expand_as(qx)
    norm2 = torch.clamp(qx * qx + qy * qy + qz * qz, min=1e-10)
    determinant = (scale[..., 0] * scale[..., 1])[..., None]
    basis = determinant * torch.clamp(qz, min=0.0) / (math.pi * norm2 * norm2)
    amplitude = _positive_amplitude(raw_amplitude)
    return torch.einsum("tkb,tkc->tbc", basis, amplitude)


class DirectFitClosureModule(nn.Module):
    def __init__(
        self,
        family: str,
        lobe_count: int,
        views: torch.Tensor,
        target: torch.Tensor,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        if family not in {"ggx", "ltc", "sg"}:
            raise ValueError(f"unsupported closure family: {family}")
        if lobe_count < 1:
            raise ValueError("lobe_count must be positive")
        self.family = family
        self.lobe_count = lobe_count
        query_group_count = len(target)
        generator = torch.Generator(device=target.device)
        generator.manual_seed(seed)

        peak = torch.amax(target, dim=1).clamp_min(1e-3)
        mean = torch.mean(target, dim=1).clamp_min(1e-4)
        if family == "sg":
            initial_amplitude = (0.02 * mean[:, None, :]).repeat(1, lobe_count, 1)
            initial_amplitude[:, 0, :] = peak
        elif family == "ltc":
            initial_amplitude = (0.02 * mean[:, None, :]).repeat(1, lobe_count, 1)
            initial_amplitude[:, 0, :] = math.pi * mean
        else:
            initial_amplitude = (peak[:, None, :] / lobe_count).repeat(1, lobe_count, 1)
        self.raw_amplitude = nn.Parameter(_inverse_softplus(initial_amplitude.clamp_min(1e-4)))

        if family in {"ggx", "sg"}:
            raw_axis = torch.zeros((query_group_count, lobe_count, 3), device=target.device, dtype=target.dtype)
            if family == "sg":
                specular = torch.stack((-views[:, 0], -views[:, 1], views[:, 2]), dim=-1)
                raw_axis[:, 0, :2] = specular[:, :2]
                raw_axis[:, 0, 2] = _inverse_softplus(specular[:, 2].clamp_min(1e-3))
                if lobe_count > 1:
                    raw_axis[:, 1:, :2] = 0.7 * torch.randn(
                        (query_group_count, lobe_count - 1, 2), generator=generator, device=target.device
                    )
                    raw_axis[:, 1:, 2] = _inverse_softplus(
                        torch.rand(
                            (query_group_count, lobe_count - 1), generator=generator, device=target.device
                        )
                        * 0.8
                        + 0.15
                    )
            else:
                raw_axis[..., :2] = 0.08 * torch.randn(
                    (query_group_count, lobe_count, 2), generator=generator, device=target.device
                )
                raw_axis[..., 2] = _inverse_softplus(torch.ones_like(raw_axis[..., 2]))
            self.raw_axis = nn.Parameter(raw_axis)

        if family == "ggx":
            alpha = torch.linspace(0.08, 0.8, lobe_count, device=target.device, dtype=target.dtype)
            self.log_shape = nn.Parameter(torch.log(alpha)[None, :].repeat(query_group_count, 1))
        elif family == "sg":
            if lobe_count == 1:
                sharpness = torch.full((1,), 32.0, device=target.device, dtype=target.dtype)
            else:
                sharpness = torch.logspace(
                    math.log10(0.5),
                    math.log10(128.0),
                    lobe_count,
                    device=target.device,
                    dtype=target.dtype,
                )
                closest = int(torch.argmin(torch.abs(sharpness - 32.0)))
                sharpness[0], sharpness[closest] = sharpness[closest].clone(), sharpness[0].clone()
                self.raw_amplitude.data[:, closest, :] = _inverse_softplus(mean)
            self.log_shape = nn.Parameter(torch.log(sharpness)[None, :].repeat(query_group_count, 1))
        else:
            self.log_scale = nn.Parameter(
                0.03
                * torch.randn((query_group_count, lobe_count, 2), generator=generator, device=target.device)
            )
            self.shear = nn.Parameter(
                0.03
                * torch.randn((query_group_count, lobe_count, 3), generator=generator, device=target.device)
            )
            initial_angles = torch.linspace(0.0, math.pi, lobe_count + 1, device=target.device)[:-1]
            self.angle = nn.Parameter(initial_angles[None, :].repeat(query_group_count, 1))

    def forward(self, views: torch.Tensor, lights: torch.Tensor) -> torch.Tensor:
        if self.family == "ggx":
            return eval_ggx(views, lights, self.raw_axis, self.log_shape, self.raw_amplitude)
        if self.family == "sg":
            return eval_sg(lights, self.raw_axis, self.log_shape, self.raw_amplitude)
        return eval_ltc(lights, self.log_scale, self.shear, self.angle, self.raw_amplitude)

    @torch.no_grad()
    def export_parameters(self) -> dict[str, np.ndarray]:
        result = {"amplitude": _positive_amplitude(self.raw_amplitude).cpu().numpy()}
        if self.family in {"ggx", "sg"}:
            result["axis"] = _hemisphere_axis(self.raw_axis).cpu().numpy()
            lower, upper = (
                (math.log(0.02), math.log(1.0))
                if self.family == "ggx"
                else (math.log(0.05), math.log(1024.0))
            )
            result["alpha" if self.family == "ggx" else "sharpness"] = torch.exp(
                torch.clamp(self.log_shape, lower, upper)
            ).cpu().numpy()
        else:
            result["inverse_scale"] = torch.exp(torch.clamp(self.log_scale, -3.0, 3.0)).cpu().numpy()
            result["shear"] = (3.0 * torch.tanh(self.shear)).cpu().numpy()
            result["angle"] = self.angle.cpu().numpy()
        return result


def directional_smape(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    peak = torch.amax(target, dim=(1, 2), keepdim=True)
    floor = 1e-3 * peak + 1e-5
    return torch.mean(
        2.0 * torch.abs(prediction - target) / (torch.abs(prediction) + torch.abs(target) + floor),
        dim=(1, 2),
    )


@dataclass(frozen=True)
class DirectFitResult:
    prediction: np.ndarray
    smape: np.ndarray
    relative_l1: np.ndarray
    parameters: dict[str, np.ndarray]


def fit_direct_batch(
    target: np.ndarray,
    views: np.ndarray,
    lights: np.ndarray,
    *,
    family: str,
    lobe_count: int,
    steps: int = 400,
    restarts: int = 2,
    learning_rate: float = 0.03,
    device: str | torch.device | None = None,
    seed: int = 1,
) -> DirectFitResult:
    if steps < 1 or restarts < 1:
        raise ValueError("steps and restarts must be positive")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=torch_device)
    view_tensor = torch.as_tensor(views, dtype=torch.float32, device=torch_device)
    light_tensor = torch.as_tensor(lights, dtype=torch.float32, device=torch_device)
    scale = torch.amax(target_tensor, dim=(1, 2), keepdim=True).clamp_min(1e-4)
    normalized_target = target_tensor / scale
    floor = 1e-3

    best_smape = torch.full((len(target_tensor),), torch.inf, device=torch_device)
    best_prediction = torch.zeros_like(target_tensor)
    best_parameters: dict[str, np.ndarray] = {}
    for restart in range(restarts):
        module = DirectFitClosureModule(
            family,
            lobe_count,
            view_tensor,
            normalized_target,
            seed=seed + restart * 1009,
        ).to(torch_device)
        optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = module(view_tensor, light_tensor)
            log_delta = torch.log(prediction + floor) - torch.log(normalized_target + floor)
            loss = torch.mean(log_delta * log_delta) + 0.05 * torch.mean(
                torch.abs(prediction - normalized_target)
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = module(view_tensor, light_tensor) * scale
            smape = directional_smape(prediction, target_tensor)
            improved = smape < best_smape
            best_smape = torch.where(improved, smape, best_smape)
            best_prediction[improved] = prediction[improved]
        exported = module.export_parameters()
        if not best_parameters:
            best_parameters = {name: np.empty_like(values) for name, values in exported.items()}
        improved_cpu = improved.cpu().numpy()
        for name, values in exported.items():
            scaled_values = values.copy()
            if name == "amplitude":
                scaled_values *= scale.cpu().numpy()
            best_parameters[name][improved_cpu] = scaled_values[improved_cpu]

    absolute_error = torch.sum(torch.abs(best_prediction - target_tensor), dim=(1, 2))
    relative_l1 = absolute_error / torch.sum(torch.abs(target_tensor), dim=(1, 2)).clamp_min(1e-8)
    return DirectFitResult(
        prediction=best_prediction.cpu().numpy(),
        smape=best_smape.cpu().numpy(),
        relative_l1=relative_l1.cpu().numpy(),
        parameters=best_parameters,
    )


@torch.no_grad()
def evaluate_exported_parameters(
    family: str,
    parameters: dict[str, np.ndarray],
    views: np.ndarray,
    lights: np.ndarray,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Evaluate the meaningful parameter arrays exported by the direct fitter."""
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    view_tensor = torch.as_tensor(views, dtype=torch.float32, device=torch_device)
    light_tensor = torch.as_tensor(lights, dtype=torch.float32, device=torch_device)
    amplitude = torch.as_tensor(parameters["amplitude"], dtype=torch.float32, device=torch_device)
    raw_amplitude = _inverse_softplus(amplitude.clamp_min(1e-8))
    if family in {"ggx", "sg"}:
        axis = torch.as_tensor(parameters["axis"], dtype=torch.float32, device=torch_device)
        raw_axis = torch.cat(
            (axis[..., :2], _inverse_softplus(axis[..., 2:3].clamp_min(1e-5))), dim=-1
        )
        if family == "ggx":
            alpha = torch.as_tensor(parameters["alpha"], dtype=torch.float32, device=torch_device)
            prediction = eval_ggx(view_tensor, light_tensor, raw_axis, torch.log(alpha), raw_amplitude)
        else:
            sharpness = torch.as_tensor(
                parameters["sharpness"], dtype=torch.float32, device=torch_device
            )
            prediction = eval_sg(light_tensor, raw_axis, torch.log(sharpness), raw_amplitude)
    elif family == "ltc":
        inverse_scale = torch.as_tensor(
            parameters["inverse_scale"], dtype=torch.float32, device=torch_device
        )
        shear = torch.as_tensor(parameters["shear"], dtype=torch.float32, device=torch_device)
        angle = torch.as_tensor(parameters["angle"], dtype=torch.float32, device=torch_device)
        raw_shear = torch.atanh(torch.clamp(shear / 3.0, -0.999999, 0.999999))
        prediction = eval_ltc(
            light_tensor,
            torch.log(inverse_scale),
            raw_shear,
            angle,
            raw_amplitude,
        )
    else:
        raise ValueError(f"unsupported closure family: {family}")
    return prediction.cpu().numpy()
