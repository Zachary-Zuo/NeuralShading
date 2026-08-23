from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from ncls.core.material import (
    DiffuseInterface,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
)

from .state import LegacyLtcK2State


RAW_RESIDUAL_DIMENSION = 18


@dataclass(frozen=True)
class LegacyLtcK2Tensors:
    interface_kind: torch.Tensor
    alpha: torch.Tensor
    relative_ior: torch.Tensor
    eta: torch.Tensor
    k: torch.Tensor
    color: torch.Tensor
    tangent_rotation: torch.Tensor
    amplitude: torch.Tensor
    inverse_scale: torch.Tensor
    shear: torch.Tensor
    angle: torch.Tensor


def decode_ltc_residual(
    raw_parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map 18 unconstrained network outputs to two meaningful LTC lobes."""
    if raw_parameters.shape[-1] != RAW_RESIDUAL_DIMENSION:
        raise ValueError(f"raw_parameters last dimension must be {RAW_RESIDUAL_DIMENSION}")
    raw = raw_parameters.reshape(*raw_parameters.shape[:-1], 2, 9)
    amplitude = torch.nn.functional.softplus(raw[..., 0:3])
    inverse_scale = torch.exp(torch.clamp(raw[..., 3:5], -3.0, 3.0))
    shear = 3.0 * torch.tanh(raw[..., 5:8])
    angle = math.pi * torch.tanh(raw[..., 8])
    return amplitude, inverse_scale, shear, angle


def states_to_tensors(
    states: Sequence[LegacyLtcK2State],
    *,
    device: str | torch.device | None = None,
) -> LegacyLtcK2Tensors:
    torch_device = torch.device(device or "cpu")
    layers = [state.direct_top for state in states]

    def alpha(layer) -> tuple[float, float]:
        if isinstance(layer, (RoughDielectricInterface, RoughConductorInterface)):
            return layer.alpha_x, layer.alpha_y
        if isinstance(layer, SheenInterface):
            return layer.roughness, layer.roughness
        return 1.0, 1.0

    def relative_ior(layer) -> float:
        return layer.relative_ior if isinstance(layer, RoughDielectricInterface) else 1.0

    def eta(layer) -> tuple[float, float, float]:
        return layer.eta if isinstance(layer, RoughConductorInterface) else (0.0, 0.0, 0.0)

    def k(layer) -> tuple[float, float, float]:
        return layer.k if isinstance(layer, RoughConductorInterface) else (0.0, 0.0, 0.0)

    def color(layer) -> tuple[float, float, float]:
        return layer.color if isinstance(layer, (DiffuseInterface, SheenInterface)) else (0.0, 0.0, 0.0)

    def rotation(layer) -> float:
        return layer.tangent_rotation if isinstance(layer, (RoughDielectricInterface, RoughConductorInterface)) else 0.0

    return LegacyLtcK2Tensors(
        interface_kind=torch.tensor([int(layer.kind) for layer in layers], device=torch_device),
        alpha=torch.tensor([alpha(layer) for layer in layers], dtype=torch.float32, device=torch_device),
        relative_ior=torch.tensor(
            [relative_ior(layer) for layer in layers], dtype=torch.float32, device=torch_device
        ),
        eta=torch.tensor([eta(layer) for layer in layers], dtype=torch.float32, device=torch_device),
        k=torch.tensor([k(layer) for layer in layers], dtype=torch.float32, device=torch_device),
        color=torch.tensor([color(layer) for layer in layers], dtype=torch.float32, device=torch_device),
        tangent_rotation=torch.tensor(
            [rotation(layer) for layer in layers], dtype=torch.float32, device=torch_device
        ),
        amplitude=torch.tensor(
            [[lobe.amplitude for lobe in state.residual_lobes] for state in states],
            dtype=torch.float32,
            device=torch_device,
        ),
        inverse_scale=torch.tensor(
            [[lobe.inverse_scale for lobe in state.residual_lobes] for state in states],
            dtype=torch.float32,
            device=torch_device,
        ),
        shear=torch.tensor(
            [[lobe.shear for lobe in state.residual_lobes] for state in states],
            dtype=torch.float32,
            device=torch_device,
        ),
        angle=torch.tensor(
            [[lobe.angle for lobe in state.residual_lobes] for state in states],
            dtype=torch.float32,
            device=torch_device,
        ),
    )


def eval_ltc_residual(
    light_directions: torch.Tensor,
    amplitude: torch.Tensor,
    inverse_scale: torch.Tensor,
    shear: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the two exported LTC residual lobes as f*cos(theta_l)."""
    cosine = torch.cos(angle)[..., None]
    sine = torch.sin(angle)[..., None]
    light_x = light_directions[None, None, :, 0]
    light_y = light_directions[None, None, :, 1]
    light_z = light_directions[None, None, :, 2]
    rotated_x = cosine * light_x + sine * light_y
    rotated_y = -sine * light_x + cosine * light_y
    qx = (
        inverse_scale[..., 0:1] * rotated_x
        + shear[..., 0:1] * rotated_y
        + shear[..., 1:2] * light_z
    )
    qy = inverse_scale[..., 1:2] * rotated_y + shear[..., 2:3] * light_z
    qz = light_z.expand_as(qx)
    norm2 = torch.clamp(qx * qx + qy * qy + qz * qz, min=1e-10)
    determinant = (inverse_scale[..., 0] * inverse_scale[..., 1])[..., None]
    basis = determinant * torch.clamp(qz, min=0.0) / (math.pi * norm2 * norm2)
    return torch.einsum("tkb,tkc->tbc", basis, amplitude)


def _to_layer_frame(directions: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(rotation)
    sine = torch.sin(rotation)
    return torch.stack(
        (
            cosine * directions[..., 0] + sine * directions[..., 1],
            -sine * directions[..., 0] + cosine * directions[..., 1],
            directions[..., 2],
        ),
        dim=-1,
    )


def _safe_normalize(value: torch.Tensor) -> torch.Tensor:
    length_squared = torch.sum(value * value, dim=-1, keepdim=True)
    fallback = torch.zeros_like(value)
    fallback[..., 2] = 1.0
    return torch.where(length_squared > 1e-20, value * torch.rsqrt(length_squared), fallback)


def _ggx_lambda(direction: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    cosine_squared = direction[..., 2] ** 2
    projected = (
        alpha[..., 0] ** 2 * direction[..., 0] ** 2
        + alpha[..., 1] ** 2 * direction[..., 1] ** 2
    )
    regular = 0.5 * (torch.sqrt(1.0 + projected / torch.clamp(cosine_squared, min=1e-20)) - 1.0)
    return torch.where(cosine_squared <= 1e-20, torch.full_like(regular, 1e20), regular)


def _ggx_d(half_vector: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    denominator = (
        (half_vector[..., 0] / alpha[..., 0]) ** 2
        + (half_vector[..., 1] / alpha[..., 1]) ** 2
        + half_vector[..., 2] ** 2
    )
    value = 1.0 / (math.pi * alpha[..., 0] * alpha[..., 1] * denominator * denominator)
    return torch.where(half_vector[..., 2] > 0.0, value, torch.zeros_like(value))


def _fresnel_dielectric(eta_i_over_eta_t: torch.Tensor, cosine_i: torch.Tensor) -> torch.Tensor:
    cosine_i = torch.clamp(torch.abs(cosine_i), 0.0, 1.0)
    sine_t_squared = eta_i_over_eta_t * eta_i_over_eta_t * torch.clamp(
        1.0 - cosine_i * cosine_i, min=0.0
    )
    cosine_t = torch.sqrt(torch.clamp(1.0 - sine_t_squared, min=0.0))
    rs = (eta_i_over_eta_t * cosine_i - cosine_t) / torch.clamp(
        eta_i_over_eta_t * cosine_i + cosine_t, min=1e-20
    )
    rp = (eta_i_over_eta_t * cosine_t - cosine_i) / torch.clamp(
        eta_i_over_eta_t * cosine_t + cosine_i, min=1e-20
    )
    value = 0.5 * (rs * rs + rp * rp)
    return torch.where(sine_t_squared >= 1.0, torch.ones_like(value), value)


def _fresnel_conductor(
    eta: torch.Tensor, k: torch.Tensor, cosine_i: torch.Tensor
) -> torch.Tensor:
    cosine_i = torch.clamp(torch.abs(cosine_i), 0.0, 1.0)
    cosine_squared = cosine_i * cosine_i
    sine_squared = torch.clamp(1.0 - cosine_squared, min=0.0)
    sine_fourth = sine_squared * sine_squared
    inner = eta * eta - k * k - sine_squared
    a_squared_plus_b_squared = torch.sqrt(
        torch.clamp(inner * inner + 4.0 * eta * eta * k * k, min=0.0)
    )
    a = torch.sqrt(torch.clamp(0.5 * (a_squared_plus_b_squared + inner), min=0.0))
    rs = (a_squared_plus_b_squared + cosine_squared - 2.0 * a * cosine_i) / torch.clamp(
        a_squared_plus_b_squared + cosine_squared + 2.0 * a * cosine_i, min=1e-20
    )
    rp = (
        cosine_squared * a_squared_plus_b_squared
        + sine_fourth
        - 2.0 * a * cosine_i * sine_squared
    ) / torch.clamp(
        cosine_squared * a_squared_plus_b_squared
        + sine_fourth
        + 2.0 * a * cosine_i * sine_squared,
        min=1e-20,
    )
    return 0.5 * (rs + rs * rp)


def _sheen_lambda(cosine: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    cosine = torch.clamp(cosine, 0.0, 1.0)
    r = (1.0 - roughness) ** 2
    one_minus_r = 1.0 - r
    a = 25.3245 * r + 21.5473 * one_minus_r
    b = 3.32435 * r + 3.82987 * one_minus_r
    c = 0.16801 * r + 0.19823 * one_minus_r
    d = -1.27393 * r - 1.97760 * one_minus_r
    e = -4.85967 * r - 4.32054 * one_minus_r

    def sheen_l(value: torch.Tensor) -> torch.Tensor:
        return a / (1.0 + b * torch.pow(value, c)) + d * value + e

    return torch.where(
        cosine < 0.5,
        torch.exp(sheen_l(cosine)),
        torch.exp(2.0 * sheen_l(torch.full_like(cosine, 0.5)) - sheen_l(1.0 - cosine)),
    )


def eval_direct_top(
    state: LegacyLtcK2Tensors,
    view_directions: torch.Tensor,
    light_directions: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact top-interface reflection as f*cos(theta_l)."""
    tile_count = len(view_directions)
    view = _to_layer_frame(view_directions, state.tangent_rotation)
    lights = light_directions[None, :, :].expand(tile_count, -1, -1)
    light = _to_layer_frame(lights, state.tangent_rotation[:, None])
    wi = view[:, None, :].expand_as(light)
    positive = (wi[..., 2] > 1e-6) & (light[..., 2] > 1e-6)
    alpha = torch.clamp(state.alpha, min=1e-3)[:, None, :]
    half_vector = _safe_normalize(wi + light)
    distribution = _ggx_d(half_vector, alpha)
    geometry = 1.0 / (
        1.0 + _ggx_lambda(wi, alpha) + _ggx_lambda(light, alpha)
    )
    denominator = torch.clamp(4.0 * wi[..., 2] * light[..., 2], min=1e-20)

    eta_ratio = state.relative_ior[:, None]
    fresnel_dielectric = _fresnel_dielectric(
        1.0 / torch.clamp(eta_ratio, min=1e-20),
        torch.sum(wi * half_vector, dim=-1),
    )
    dielectric = (distribution * geometry * fresnel_dielectric / denominator)[..., None].expand(
        -1, -1, 3
    )

    fresnel_conductor = _fresnel_conductor(
        state.eta[:, None, :],
        state.k[:, None, :],
        torch.sum(wi * half_vector, dim=-1, keepdim=True),
    )
    conductor = fresnel_conductor * (distribution * geometry / denominator)[..., None]

    diffuse = state.color[:, None, :] / math.pi

    roughness = torch.clamp(state.alpha[:, 0:1], min=1e-3)
    inverse_alpha = 1.0 / roughness
    sine_squared = torch.clamp(1.0 - half_vector[..., 2] ** 2, min=0.0078125)
    charlie_d = (2.0 + inverse_alpha) * torch.pow(
        sine_squared, 0.5 * inverse_alpha
    ) / (2.0 * math.pi)
    wi_cosine = torch.clamp(wi[..., 2], 0.0, 1.0)
    wo_cosine = torch.clamp(light[..., 2], 0.0, 1.0)
    softened = torch.pow(
        _sheen_lambda(wo_cosine, roughness),
        1.0 + 2.0 * torch.pow(1.0 - wo_cosine, 8.0),
    )
    sheen_g = 1.0 / (1.0 + softened + _sheen_lambda(wi_cosine, roughness))
    sheen = state.color[:, None, :] * (
        charlie_d * sheen_g / denominator
    )[..., None]

    layer_type = state.interface_kind[:, None, None]
    result = torch.where(
        layer_type == 0,
        dielectric,
        torch.where(layer_type == 1, conductor, torch.where(layer_type == 2, diffuse, sheen)),
    )
    result = torch.where(positive[..., None], result, torch.zeros_like(result))
    return result * torch.clamp(light_directions[None, :, 2:3], min=0.0)


def evaluate_state_response_cos(
    state: LegacyLtcK2Tensors,
    view_directions: torch.Tensor,
    light_directions: torch.Tensor,
) -> torch.Tensor:
    return eval_direct_top(state, view_directions, light_directions) + eval_ltc_residual(
        light_directions,
        state.amplitude,
        state.inverse_scale,
        state.shear,
        state.angle,
    )


def evaluate_state_bsdf(
    state: LegacyLtcK2Tensors,
    view_directions: torch.Tensor,
    light_directions: torch.Tensor,
) -> torch.Tensor:
    response = evaluate_state_response_cos(state, view_directions, light_directions)
    cosine = torch.clamp(light_directions[None, :, 2:3], min=1e-6)
    return torch.where(
        light_directions[None, :, 2:3] > 0.0,
        response / cosine,
        torch.zeros_like(response),
    )
