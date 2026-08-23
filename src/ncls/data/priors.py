from __future__ import annotations

import numpy as np

from ncls.core.material import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
)


LAYER_STACK_RESEARCH_PRIOR_ID = "ncls.layer-stack-research-prior@1"
E0_LAYER_STACK_BOUNDARY_PROFILE_ID = "ncls.e0-layer-stack-boundary@1"
E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID = "ncls.e1-layer-stack-narrow-conductor@1"
E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID = "ncls.e1-layer-stack-multi-interface@1"
LAYER_STACK_STATE_PROFILE_IDS = (
    LAYER_STACK_RESEARCH_PRIOR_ID,
    E0_LAYER_STACK_BOUNDARY_PROFILE_ID,
    E1_LAYER_STACK_NARROW_CONDUCTOR_PROFILE_ID,
    E1_LAYER_STACK_MULTI_INTERFACE_PROFILE_ID,
)
E0_LAYER_STACK_BOUNDARY_CASE_IDS = (
    "narrow-dielectric-over-diffuse",
    "narrow-anisotropic-conductor",
    "rotated-anisotropic-dielectric",
    "chromatic-absorption-slab",
    "chromatic-scattering-slab",
    "multi-interface-moving-peaks",
)


def assign_family_splits(family_count: int, seed: int) -> np.ndarray:
    """在材质族粒度返回 train/validation/test 的 uint8 标签。"""

    if family_count < 1:
        raise ValueError("family_count must be positive")
    labels = np.zeros(family_count, dtype=np.uint8)
    if family_count < 3:
        return labels
    validation_count = max(1, int(round(0.1 * family_count)))
    test_count = max(1, int(round(0.1 * family_count)))
    if validation_count + test_count >= family_count:
        validation_count = test_count = 1
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


def _dielectric(rng: np.random.Generator, *, top: bool) -> RoughDielectricInterface:
    alpha_x, alpha_y = _roughness(rng)
    relative_ior = float(rng.uniform(1.1, 1.8) if top else rng.uniform(0.8, 1.25))
    return RoughDielectricInterface(alpha_x, alpha_y, relative_ior, float(rng.uniform(-np.pi, np.pi)))


def _base(rng: np.random.Generator):
    choice = int(rng.choice(3, p=[0.55, 0.3, 0.15]))
    alpha_x, alpha_y = _roughness(rng)
    if choice == 1:
        return RoughConductorInterface(
            alpha_x,
            alpha_y,
            tuple(rng.uniform(0.1, 1.5, size=3)),
            tuple(rng.uniform(1.0, 5.0, size=3)),
            float(rng.uniform(-np.pi, np.pi)),
        )
    color = tuple(rng.uniform(0.03, 0.9, size=3))
    return DiffuseInterface(color) if choice == 0 else SheenInterface(color, alpha_x)


def _medium(rng: np.random.Generator) -> HomogeneousMedium:
    thickness = _log_uniform(rng, 1e-3, 1.0)
    sigma_t = _log_uniform(rng, 1e-3, 3.0)
    if rng.random() < 0.35:
        scattering_albedo = rng.uniform(0.0, 0.85, size=3)
        sigma_s = sigma_t * scattering_albedo
        sigma_a = sigma_t - sigma_s
        return HomogeneousMedium(
            tuple(sigma_a),
            tuple(sigma_s),
            float(rng.uniform(-0.5, 0.8)),
            thickness,
        )
    sigma_a = sigma_t * rng.uniform(0.5, 1.5, size=3)
    return HomogeneousMedium(tuple(sigma_a), thickness=thickness)


def sample_stack(rng: np.random.Generator, *, min_interfaces: int = 1, max_interfaces: int = 8) -> LayerStackIR:
    if not 1 <= min_interfaces <= max_interfaces <= 8:
        raise ValueError("interface bounds must satisfy 1 <= min_interfaces <= max_interfaces <= 8")
    interface_count = int(rng.integers(min_interfaces, max_interfaces + 1))
    if interface_count == 1:
        return LayerStackIR((_base(rng),), ())
    interfaces = [_dielectric(rng, top=True)]
    interfaces.extend(_dielectric(rng, top=False) for _ in range(interface_count - 2))
    interfaces.append(_base(rng))
    return LayerStackIR(tuple(interfaces), tuple(_medium(rng) for _ in range(interface_count - 1)))


def sample_stacks(count: int, seed: int) -> list[LayerStackIR]:
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


def _perturb_interface(rng: np.random.Generator, interface):
    if isinstance(interface, RoughDielectricInterface):
        return RoughDielectricInterface(
            float(np.clip(interface.alpha_x * np.exp(rng.normal(0.0, 0.22)), 0.02, 1.0)),
            float(np.clip(interface.alpha_y * np.exp(rng.normal(0.0, 0.22)), 0.02, 1.0)),
            float(np.clip(interface.relative_ior * np.exp(rng.normal(0.0, 0.04)), 0.7, 2.0)),
            float(interface.tangent_rotation + rng.normal(0.0, 0.2)),
        )
    if isinstance(interface, RoughConductorInterface):
        return RoughConductorInterface(
            float(np.clip(interface.alpha_x * np.exp(rng.normal(0.0, 0.22)), 0.02, 1.0)),
            float(np.clip(interface.alpha_y * np.exp(rng.normal(0.0, 0.22)), 0.02, 1.0)),
            _scaled_rgb(rng, interface.eta, low=0.02, high=3.0),
            _scaled_rgb(rng, interface.k, low=0.1, high=8.0),
            float(interface.tangent_rotation + rng.normal(0.0, 0.2)),
        )
    if isinstance(interface, DiffuseInterface):
        return DiffuseInterface(_scaled_rgb(rng, interface.color, low=0.01, high=0.98))
    if isinstance(interface, SheenInterface):
        return SheenInterface(
            _scaled_rgb(rng, interface.color, low=0.01, high=0.98),
            float(np.clip(interface.roughness * np.exp(rng.normal(0.0, 0.22)), 0.02, 1.0)),
        )
    raise TypeError(f"unsupported interface {type(interface)!r}")


def _perturb_medium(rng: np.random.Generator, medium: HomogeneousMedium) -> HomogeneousMedium:
    thickness = float(np.clip(medium.thickness * np.exp(rng.normal(0.0, 0.2)), 1e-4, 2.0))
    sigma_a = np.asarray(medium.sigma_a, dtype=np.float64)
    sigma_s = np.asarray(medium.sigma_s, dtype=np.float64)
    if np.any(sigma_s > 0.0):
        sigma_t = float(np.mean(sigma_a + sigma_s) * np.exp(rng.normal(0.0, 0.18)))
        albedo = np.clip(sigma_s / np.maximum(sigma_a + sigma_s, 1e-12), 0.0, 0.95)
        albedo = np.clip(albedo + rng.normal(0.0, 0.035, size=3), 0.0, 0.95)
        perturbed_sigma_s = sigma_t * albedo
        perturbed_sigma_a = sigma_t - perturbed_sigma_s
        return HomogeneousMedium(
            tuple(perturbed_sigma_a),
            tuple(perturbed_sigma_s),
            float(np.clip(medium.g + rng.normal(0.0, 0.06), -0.8, 0.9)),
            thickness,
        )
    return HomogeneousMedium(
        _scaled_rgb(rng, medium.sigma_a, low=1e-5, high=6.0),
        thickness=thickness,
    )


def sample_stack_families(family_count: int, local_state_count: int, seed: int) -> list[list[LayerStackIR]]:
    """采样材质族；同族局部状态共享结构且必须进入同一数据划分。"""

    if family_count < 1 or local_state_count < 1:
        raise ValueError("family_count and local_state_count must be positive")
    rng = np.random.default_rng(seed)
    families: list[list[LayerStackIR]] = []
    for _ in range(family_count):
        template = sample_stack(rng)
        families.append(
            [
                LayerStackIR(
                    tuple(_perturb_interface(rng, item) for item in template.interfaces),
                    tuple(_perturb_medium(rng, item) for item in template.media),
                )
                for _ in range(local_state_count)
            ]
        )
    return families


def e0_layer_stack_boundary_cases() -> tuple[tuple[str, LayerStackIR], ...]:
    """返回 E0 固定边界案例；它是 coverage probe，不是训练 prior。"""

    return (
        (
            "narrow-dielectric-over-diffuse",
            LayerStackIR(
                (
                    RoughDielectricInterface(0.002, 0.002, 1.5),
                    DiffuseInterface((0.42, 0.19, 0.06)),
                ),
                (HomogeneousMedium(thickness=0.08),),
            ),
        ),
        (
            "narrow-anisotropic-conductor",
            LayerStackIR(
                (
                    RoughConductorInterface(
                        0.002,
                        0.08,
                        (0.17, 0.35, 1.5),
                        (3.1, 2.7, 1.9),
                        0.61,
                    ),
                ),
                (),
            ),
        ),
        (
            "rotated-anisotropic-dielectric",
            LayerStackIR(
                (
                    RoughDielectricInterface(0.004, 0.12, 1.62, -0.73),
                    DiffuseInterface((0.12, 0.48, 0.2)),
                ),
                (HomogeneousMedium(thickness=0.18),),
            ),
        ),
        (
            "chromatic-absorption-slab",
            LayerStackIR(
                (
                    RoughDielectricInterface(0.018, 0.055, 1.45, 0.37),
                    DiffuseInterface((0.72, 0.58, 0.34)),
                ),
                (HomogeneousMedium((0.25, 1.2, 3.0), thickness=0.6),),
            ),
        ),
        (
            "chromatic-scattering-slab",
            LayerStackIR(
                (
                    RoughDielectricInterface(0.05, 0.05, 1.3),
                    DiffuseInterface((0.55, 0.45, 0.35)),
                ),
                (
                    HomogeneousMedium(
                        (0.8, 0.5, 0.2),
                        (0.2, 0.5, 0.8),
                        0.7,
                        0.75,
                    ),
                ),
            ),
        ),
        (
            "multi-interface-moving-peaks",
            LayerStackIR(
                (
                    RoughDielectricInterface(0.012, 0.05, 1.48, 0.1),
                    RoughDielectricInterface(0.045, 0.01, 0.82, 1.03),
                    RoughConductorInterface(
                        0.07,
                        0.025,
                        (0.22, 0.9, 1.4),
                        (3.8, 2.6, 1.7),
                        -0.42,
                    ),
                ),
                (
                    HomogeneousMedium((0.03, 0.08, 0.16), thickness=0.12),
                    HomogeneousMedium(thickness=0.22),
                ),
            ),
        ),
    )


def e1_layer_stack_narrow_conductor_cases() -> tuple[tuple[str, LayerStackIR], ...]:
    """返回 E1 首个容量压力材质；独立 profile 防止把 E0 六案例外形当训练合同。"""

    selected = tuple(
        (case_id, stack)
        for case_id, stack in e0_layer_stack_boundary_cases()
        if case_id == "narrow-anisotropic-conductor"
    )
    if len(selected) != 1:
        raise RuntimeError("E1 narrow conductor case is absent from the frozen E0 boundary set")
    return selected


def e1_layer_stack_multi_interface_cases() -> tuple[tuple[str, LayerStackIR], ...]:
    """返回 E1 多界面残差压力材质；新 profile 固定其训练用途与 provenance。"""

    selected = tuple(
        (case_id, stack)
        for case_id, stack in e0_layer_stack_boundary_cases()
        if case_id == "multi-interface-moving-peaks"
    )
    if len(selected) != 1:
        raise RuntimeError("E1 multi-interface case is absent from the frozen E0 boundary set")
    return selected
