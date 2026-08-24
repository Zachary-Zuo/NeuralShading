from __future__ import annotations

import hashlib

import numpy as np

from ncls.core.material import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
)


LAYER_STACK_PROFILE = "layer-stack-v1"


def _lhs(sample_count: int, dimension_count: int, rng: np.random.Generator) -> np.ndarray:
    """生成确定性的 Latin hypercube；每个参数维度的每个分层恰好命中一次。"""

    if sample_count < 1 or dimension_count < 1:
        raise ValueError("Latin hypercube dimensions must be positive")
    result = np.empty((sample_count, dimension_count), dtype=np.float64)
    for dimension in range(dimension_count):
        order = rng.permutation(sample_count)
        jitter = rng.random(sample_count)
        result[:, dimension] = (order + jitter) / sample_count
    return result


def _linear(u: float, low: float, high: float) -> float:
    return float(low + (high - low) * u)


def _log(u: float, low: float, high: float) -> float:
    return float(np.exp(np.log(low) + (np.log(high) - np.log(low)) * u))


def _rgb(values: np.ndarray, low: float, high: float) -> tuple[float, float, float]:
    return tuple(float(low + (high - low) * item) for item in values)


def _stack_from_lhs(family_index: int, values: np.ndarray) -> LayerStackIR:
    cursor = 0

    def take() -> float:
        nonlocal cursor
        value = float(values[cursor])
        cursor += 1
        return value

    def take3() -> np.ndarray:
        return np.asarray((take(), take(), take()), dtype=np.float64)

    layer_count = 1 + family_index % 4
    base_kind = (family_index // 4) % 3
    interfaces: list[object] = []
    for interface_index in range(max(0, layer_count - 1)):
        base = _log(take(), 0.008, 0.6)
        aspect = _log(take(), 0.45, 2.2)
        interfaces.append(RoughDielectricInterface(
            min(base * aspect, 1.0),
            min(base / aspect, 1.0),
            _linear(take(), 0.82 if interface_index else 1.1, 1.9),
            _linear(take(), -np.pi, np.pi),
        ))
    if base_kind == 0:
        interfaces.append(DiffuseInterface(_rgb(take3(), 0.02, 0.95)))
    elif base_kind == 1:
        base = _log(take(), 0.008, 0.7)
        aspect = _log(take(), 0.4, 2.5)
        interfaces.append(RoughConductorInterface(
            min(base * aspect, 1.0),
            min(base / aspect, 1.0),
            _rgb(take3(), 0.05, 2.5),
            _rgb(take3(), 0.2, 6.0),
            _linear(take(), -np.pi, np.pi),
        ))
    else:
        interfaces.append(SheenInterface(
            _rgb(take3(), 0.02, 0.95),
            _log(take(), 0.008, 0.8),
        ))

    media: list[HomogeneousMedium] = []
    medium_variant = (family_index // 12) % 2
    for gap_index in range(layer_count - 1):
        sigma_t = _log(take(), 1e-3, 3.0)
        thickness = _log(take(), 1e-3, 1.0)
        scattering = (family_index + gap_index + medium_variant) % 3 == 0
        if scattering:
            albedo = np.asarray(_rgb(take3(), 0.02, 0.85), dtype=np.float64)
            sigma_s = sigma_t * albedo
            media.append(HomogeneousMedium(
                tuple(sigma_t - sigma_s),
                tuple(sigma_s),
                _linear(take(), -0.5, 0.8),
                thickness,
            ))
        else:
            tint = np.asarray(_rgb(take3(), 0.5, 1.5), dtype=np.float64)
            media.append(HomogeneousMedium(tuple(sigma_t * tint), thickness=thickness))
    return LayerStackIR(tuple(interfaces), tuple(media))


def layer_stack_v1_families(
    family_count: int,
    states_per_family: int,
    seed: int,
) -> tuple[tuple[str, tuple[LayerStackIR, ...]], ...]:
    """生成结构固定、连续参数用 LHS 覆盖的 LayerStack v1 state corpus。"""

    if family_count < 4 or states_per_family < 3 or seed < 0:
        raise ValueError("layer-stack-v1 requires >=4 families, >=3 states/family and a nonnegative seed")
    result = []
    for family_index in range(family_count):
        family_seed = seed ^ int.from_bytes(
            hashlib.sha256(f"layer-stack-v1/{family_index}".encode("utf-8")).digest()[:8],
            "little",
        )
        lhs = _lhs(states_per_family, 64, np.random.default_rng(family_seed))
        layer_count = 1 + family_index % 4
        base_name = ("diffuse", "conductor", "sheen")[(family_index // 4) % 3]
        structure_id = f"layers-{layer_count:02d}-{base_name}-variant-{family_index:02d}"
        result.append((
            structure_id,
            tuple(_stack_from_lhs(family_index, row) for row in lhs),
        ))
    return tuple(result)


def layer_stack_difficulty(stack: LayerStackIR) -> tuple[str, tuple[str, ...]]:
    roughness: list[float] = []
    for interface in stack.interfaces:
        if isinstance(interface, (RoughDielectricInterface, RoughConductorInterface)):
            roughness.extend((float(interface.alpha_x), float(interface.alpha_y)))
        elif isinstance(interface, SheenInterface):
            roughness.append(float(interface.roughness))
    narrowest = min(roughness, default=1.0)
    difficulty = "S" if narrowest < 0.05 else "G" if narrowest < 0.2 else "W"
    tags = ("M",) if len(stack.interfaces) >= 2 else ()
    return difficulty, tags


def layer_stack_v1_splits(
    family_count: int,
    states_per_family: int,
    heldout_family_count: int,
    seed: int,
) -> dict[tuple[int, int], tuple[int, str]]:
    """返回 source split 与 G2/G2s cohort；父子编辑链由独立 split_group 约束。"""

    if not 1 <= heldout_family_count < family_count:
        raise ValueError("heldout family count must leave at least one fitted family")
    order = sorted(
        range(family_count),
        key=lambda index: hashlib.sha256(f"{seed}\0family\0{index}".encode("utf-8")).digest(),
    )
    heldout = set(order[:heldout_family_count])
    result: dict[tuple[int, int], tuple[int, str]] = {}
    for family_index in range(family_count):
        if family_index in heldout:
            for state_index in range(states_per_family):
                result[(family_index, state_index)] = (2, "g2s")
            continue
        state_order = sorted(
            range(states_per_family),
            key=lambda index: hashlib.sha256(
                f"{seed}\0state\0{family_index}\0{index}".encode("utf-8")
            ).digest(),
        )
        validation_index, test_index = state_order[:2]
        for state_index in range(states_per_family):
            if state_index == validation_index:
                result[(family_index, state_index)] = (1, "validation")
            elif state_index == test_index:
                result[(family_index, state_index)] = (2, "g2")
            else:
                result[(family_index, state_index)] = (0, "train")
    return result
