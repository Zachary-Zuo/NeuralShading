from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ncls.core.material import (
    MAX_INTERFACES,
    DiffuseInterface,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
)
from ncls.data import ReferenceDataset


FEATURE_CONTRACT_ID = "ncls.layer-stack-token-features@2"
CONTINUOUS_FEATURE_COUNT = 25
FEATURE_CONTRACT = {
    "feature_contract_id": FEATURE_CONTRACT_ID,
    "source_ir": "ncls.layer-stack-ir@1",
    "token_order": "top-interface-to-opaque-base",
    "padded_interface_count": MAX_INTERFACES,
    "continuous_feature_count": CONTINUOUS_FEATURE_COUNT,
    "interface_kind_encoding": "NclsInterfaceKind integer",
    "continuous_layout": [
        {"range": [0, 2], "name": "alpha_xy", "transform": "1 + log(max(x, 1e-3)) / log(1000); zero when not applicable"},
        {"range": [2, 5], "name": "relative_ior_rgb", "transform": "x / 3; zero except rough dielectric"},
        {"range": [5, 8], "name": "conductor_eta_rgb", "transform": "x / 3; zero except rough conductor"},
        {"range": [8, 11], "name": "conductor_k_rgb", "transform": "x / 8; zero except rough conductor"},
        {"range": [11, 14], "name": "base_color_rgb", "transform": "identity; zero except diffuse/sheen"},
        {"range": [14, 16], "name": "tangent_rotation", "transform": "sin/cos; zero when not applicable"},
        {"range": [16, 19], "name": "sigma_a_rgb", "transform": "log1p(x) / log(7)"},
        {"range": [19, 22], "name": "sigma_s_rgb", "transform": "log1p(x) / log(7)"},
        {"range": [22, 23], "name": "phase_g", "transform": "identity"},
        {"range": [23, 24], "name": "thickness", "transform": "log1p(x) / log(3)"},
        {"range": [24, 25], "name": "has_medium", "transform": "0 or 1"},
    ],
    "direction_encoding": "local_xyz-outward",
    "unused_typed_fields": "zero",
    "color_model": "linear-srgb",
}


@dataclass(frozen=True)
class StackFeatureTable:
    interface_kinds: np.ndarray
    continuous: np.ndarray
    interface_counts: np.ndarray
    top_kind: np.ndarray
    top_alpha: np.ndarray
    top_relative_ior: np.ndarray
    top_eta: np.ndarray
    top_k: np.ndarray
    top_color: np.ndarray
    top_rotation: np.ndarray


def _roughness_feature(value: float) -> float:
    return float(1.0 + np.log(max(value, 1e-3)) / np.log(1000.0))


def _interface_fields(interface) -> tuple[
    tuple[float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]:
    alpha = (0.0, 0.0)
    relative_ior = (0.0, 0.0, 0.0)
    eta = (0.0, 0.0, 0.0)
    k = (0.0, 0.0, 0.0)
    color = (0.0, 0.0, 0.0)
    rotation = 0.0
    if isinstance(interface, RoughDielectricInterface):
        alpha = (interface.alpha_x, interface.alpha_y)
        relative_ior = (interface.relative_ior,) * 3
        rotation = interface.tangent_rotation
    elif isinstance(interface, RoughConductorInterface):
        alpha = (interface.alpha_x, interface.alpha_y)
        eta = interface.eta
        k = interface.k
        rotation = interface.tangent_rotation
    elif isinstance(interface, DiffuseInterface):
        color = interface.color
    elif isinstance(interface, SheenInterface):
        alpha = (interface.roughness, interface.roughness)
        color = interface.color
    else:
        raise TypeError(f"unsupported interface {type(interface)!r}")
    return alpha, relative_ior, eta, k, color, rotation  # type: ignore[return-value]


def encode_layer_stack(stack: LayerStackIR) -> tuple[np.ndarray, np.ndarray, int]:
    kinds = np.zeros(MAX_INTERFACES, dtype=np.int64)
    continuous = np.zeros((MAX_INTERFACES, CONTINUOUS_FEATURE_COUNT), dtype=np.float32)
    for index, interface in enumerate(stack.interfaces):
        kinds[index] = int(interface.kind)
        alpha, relative_ior, eta, k, color, rotation = _interface_fields(interface)
        has_rotated_frame = isinstance(interface, (RoughDielectricInterface, RoughConductorInterface))
        values = [
            _roughness_feature(alpha[0]) if alpha[0] > 0.0 else 0.0,
            _roughness_feature(alpha[1]) if alpha[1] > 0.0 else 0.0,
            *(np.asarray(relative_ior, dtype=np.float32) / 3.0),
            *(np.asarray(eta, dtype=np.float32) / 3.0),
            *(np.asarray(k, dtype=np.float32) / 8.0),
            *color,
            np.sin(rotation) if has_rotated_frame else 0.0,
            np.cos(rotation) if has_rotated_frame else 0.0,
        ]
        if index < len(stack.media):
            medium = stack.media[index]
            values.extend(
                [
                    *(np.log1p(np.asarray(medium.sigma_a, dtype=np.float32)) / np.log(7.0)),
                    *(np.log1p(np.asarray(medium.sigma_s, dtype=np.float32)) / np.log(7.0)),
                    medium.g,
                    np.log1p(medium.thickness) / np.log(3.0),
                    1.0,
                ]
            )
        else:
            values.extend([0.0] * 9)
        continuous[index] = np.asarray(values, dtype=np.float32)
    return kinds, continuous, len(stack.interfaces)


def load_feature_table(dataset: ReferenceDataset) -> StackFeatureTable:
    state_count = int(dataset.manifest.counts["material_state_count"])
    interface_kinds = np.zeros((state_count, MAX_INTERFACES), dtype=np.int64)
    continuous = np.zeros((state_count, MAX_INTERFACES, CONTINUOUS_FEATURE_COUNT), dtype=np.float32)
    interface_counts = np.zeros(state_count, dtype=np.int64)
    top_kind = np.zeros(state_count, dtype=np.int64)
    top_alpha = np.zeros((state_count, 2), dtype=np.float32)
    top_relative_ior = np.ones(state_count, dtype=np.float32)
    top_eta = np.zeros((state_count, 3), dtype=np.float32)
    top_k = np.zeros((state_count, 3), dtype=np.float32)
    top_color = np.zeros((state_count, 3), dtype=np.float32)
    top_rotation = np.zeros(state_count, dtype=np.float32)
    for state_index, state in enumerate(dataset.material_states):
        stack = dataset.canonical_material_ir(int(state["canonical_ir_index"]))
        kinds, values, count = encode_layer_stack(stack)
        interface_kinds[state_index] = kinds
        continuous[state_index] = values
        interface_counts[state_index] = count
        top = stack.interfaces[0]
        alpha, relative, eta, k, color, rotation = _interface_fields(top)
        top_kind[state_index] = int(top.kind)
        top_alpha[state_index] = alpha
        top_relative_ior[state_index] = relative[0] if isinstance(top, RoughDielectricInterface) else 1.0
        top_eta[state_index] = eta
        top_k[state_index] = k
        top_color[state_index] = color
        top_rotation[state_index] = rotation
    return StackFeatureTable(
        interface_kinds,
        continuous,
        interface_counts,
        top_kind,
        top_alpha,
        top_relative_ior,
        top_eta,
        top_k,
        top_color,
        top_rotation,
    )
