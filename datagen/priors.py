from __future__ import annotations

import numpy as np

from schema import LayerInterface, LayerMedium, LayerStack, LayerType


PRIOR_VERSION = "v0.2"


def assign_family_splits(family_count: int, seed: int) -> np.ndarray:
    """Return uint8 train/validation/test labels at material-family granularity."""
    if family_count < 1:
        raise ValueError("family_count must be positive")
    labels = np.zeros(family_count, dtype=np.uint8)
    if family_count < 3:
        return labels
    validation_count = max(1, int(round(0.1 * family_count)))
    test_count = max(1, int(round(0.1 * family_count)))
    if validation_count + test_count >= family_count:
        validation_count = 1
        test_count = 1
    order = np.random.default_rng(seed ^ 0x4E434C53).permutation(family_count)
    labels[order[:validation_count]] = 1
    labels[order[validation_count : validation_count + test_count]] = 2
    return labels


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _roughness(rng: np.random.Generator) -> tuple[float, float]:
    base = _log_uniform(rng, 0.025, 0.8)
    aspect = _log_uniform(rng, 0.5, 2.0)
    return min(base * aspect, 1.0), min(base / aspect, 1.0)


def _dielectric(rng: np.random.Generator, *, top: bool) -> LayerInterface:
    roughness_x, roughness_y = _roughness(rng)
    eta_ratio = float(rng.uniform(1.1, 1.8) if top else rng.uniform(0.8, 1.25))
    return LayerInterface(
        LayerType.ROUGH_DIELECTRIC,
        roughness_x,
        roughness_y,
        eta=(eta_ratio,) * 3,
        tangent_rotation=float(rng.uniform(-np.pi, np.pi)),
    )


def _base(rng: np.random.Generator) -> LayerInterface:
    base_type = rng.choice([LayerType.DIFFUSE, LayerType.ROUGH_CONDUCTOR, LayerType.SHEEN], p=[0.55, 0.3, 0.15])
    roughness_x, roughness_y = _roughness(rng)
    if base_type == LayerType.ROUGH_CONDUCTOR:
        return LayerInterface(
            base_type,
            roughness_x,
            roughness_y,
            eta=tuple(rng.uniform(0.1, 1.5, size=3)),
            k=tuple(rng.uniform(1.0, 5.0, size=3)),
            tangent_rotation=float(rng.uniform(-np.pi, np.pi)),
        )
    return LayerInterface(
        base_type,
        roughness_x,
        roughness_y,
        albedo=tuple(rng.uniform(0.03, 0.9, size=3)),
        tangent_rotation=float(rng.uniform(-np.pi, np.pi)),
    )


def _medium(rng: np.random.Generator) -> LayerMedium:
    thickness = _log_uniform(rng, 1e-3, 1.0)
    sigma_t = _log_uniform(rng, 1e-3, 3.0)
    if rng.random() < 0.35:
        scattering_albedo = rng.uniform(0.0, 0.85, size=3)
        sigma_s = sigma_t * scattering_albedo
        sigma_a = sigma_t - sigma_s
        return LayerMedium(
            sigma_a=tuple(sigma_a),
            sigma_s=tuple(sigma_s),
            g=float(rng.uniform(-0.5, 0.8)),
            thickness=thickness,
        )
    sigma_a = sigma_t * rng.uniform(0.5, 1.5, size=3)
    return LayerMedium(sigma_a=tuple(sigma_a), thickness=thickness)


def sample_stack(rng: np.random.Generator, *, min_layers: int = 1, max_layers: int = 8) -> LayerStack:
    if not 1 <= min_layers <= max_layers <= 8:
        raise ValueError("layer bounds must satisfy 1 <= min_layers <= max_layers <= 8")
    layer_count = int(rng.integers(min_layers, max_layers + 1))
    if layer_count == 1:
        return LayerStack((_base(rng),), ())
    layers = [_dielectric(rng, top=True)]
    layers.extend(_dielectric(rng, top=False) for _ in range(layer_count - 2))
    layers.append(_base(rng))
    media = tuple(_medium(rng) for _ in range(layer_count - 1))
    return LayerStack(tuple(layers), media)


def sample_stacks(count: int, seed: int) -> list[LayerStack]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return [sample_stack(rng) for _ in range(count)]


def _scaled_rgb(
    rng: np.random.Generator,
    values: tuple[float, float, float],
    *,
    low: float,
    high: float,
    sigma: float = 0.18,
) -> tuple[float, float, float]:
    result = np.asarray(values) * np.exp(rng.normal(0.0, sigma, size=3))
    return tuple(np.clip(result, low, high))


def _perturb_interface(rng: np.random.Generator, layer: LayerInterface) -> LayerInterface:
    roughness_x = float(np.clip(layer.roughness_x * np.exp(rng.normal(0.0, 0.22)), 0.02, 1.0))
    roughness_y = float(np.clip(layer.roughness_y * np.exp(rng.normal(0.0, 0.22)), 0.02, 1.0))
    tangent_rotation = float(layer.tangent_rotation + rng.normal(0.0, 0.2))
    eta = layer.eta
    k = layer.k
    albedo = layer.albedo
    if layer.layer_type == LayerType.ROUGH_DIELECTRIC:
        eta_ratio = float(np.clip(layer.eta[0] * np.exp(rng.normal(0.0, 0.04)), 0.7, 2.0))
        eta = (eta_ratio,) * 3
    elif layer.layer_type == LayerType.ROUGH_CONDUCTOR:
        eta = _scaled_rgb(rng, layer.eta, low=0.02, high=3.0)
        k = _scaled_rgb(rng, layer.k, low=0.1, high=8.0)
    else:
        albedo = _scaled_rgb(rng, layer.albedo, low=0.01, high=0.98)
    return LayerInterface(
        layer.layer_type,
        roughness_x,
        roughness_y,
        eta=eta,
        k=k,
        albedo=albedo,
        tangent_rotation=tangent_rotation,
        flags=layer.flags,
    )


def _perturb_medium(rng: np.random.Generator, medium: LayerMedium) -> LayerMedium:
    thickness = float(np.clip(medium.thickness * np.exp(rng.normal(0.0, 0.2)), 1e-4, 2.0))
    sigma_a = np.asarray(medium.sigma_a, dtype=np.float64)
    sigma_s = np.asarray(medium.sigma_s, dtype=np.float64)
    if np.any(sigma_s > 0.0):
        sigma_t = float(np.mean(sigma_a + sigma_s) * np.exp(rng.normal(0.0, 0.18)))
        albedo = np.clip(sigma_s / np.maximum(sigma_a + sigma_s, 1e-12), 0.0, 0.95)
        albedo = np.clip(albedo + rng.normal(0.0, 0.035, size=3), 0.0, 0.95)
        perturbed_sigma_s = sigma_t * albedo
        perturbed_sigma_a = sigma_t - perturbed_sigma_s
        return LayerMedium(
            sigma_a=tuple(perturbed_sigma_a),
            sigma_s=tuple(perturbed_sigma_s),
            g=float(np.clip(medium.g + rng.normal(0.0, 0.06), -0.8, 0.9)),
            thickness=thickness,
        )
    return LayerMedium(
        sigma_a=_scaled_rgb(rng, medium.sigma_a, low=1e-5, high=6.0),
        thickness=thickness,
    )


def sample_stack_families(
    family_count: int,
    local_state_count: int,
    seed: int,
) -> list[list[LayerStack]]:
    """Sample material families, keeping all local states of a family in one split.

    A family fixes layer count/types and a central material state. Its local states
    are bounded parameter perturbations, analogous to texels in one material map.
    """
    if family_count < 1 or local_state_count < 1:
        raise ValueError("family_count and local_state_count must be positive")
    rng = np.random.default_rng(seed)
    families: list[list[LayerStack]] = []
    for _ in range(family_count):
        template = sample_stack(rng)
        states = [
            LayerStack(
                tuple(_perturb_interface(rng, layer) for layer in template.layers),
                tuple(_perturb_medium(rng, medium) for medium in template.media),
            )
            for _ in range(local_state_count)
        ]
        families.append(states)
    return families
