from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from schema import BINARY_SIZE, MAX_LAYERS, SCHEMA_MAGIC, SCHEMA_VERSION, LayerStack


FEATURE_VERSION = "stack-token-v1"
CONTINUOUS_FEATURE_COUNT = 22

_INTERFACE_DTYPE = np.dtype(
    [
        ("type", "<u4"),
        ("flags", "<u4"),
        ("roughness", "<f4", (2,)),
        ("eta", "<f4", (3,)),
        ("k", "<f4", (3,)),
        ("albedo", "<f4", (3,)),
        ("tangent_rotation", "<f4"),
        ("reserved", "<f4", (2,)),
    ],
    align=False,
)
_MEDIUM_DTYPE = np.dtype(
    [
        ("sigma_a", "<f4", (3,)),
        ("sigma_s", "<f4", (3,)),
        ("g", "<f4"),
        ("thickness", "<f4"),
    ],
    align=False,
)
_STACK_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u4"),
        ("layer_count", "<u4"),
        ("medium_count", "<u4"),
        ("layers", _INTERFACE_DTYPE, (MAX_LAYERS,)),
        ("media", _MEDIUM_DTYPE, (MAX_LAYERS - 1,)),
    ],
    align=False,
)
assert _STACK_DTYPE.itemsize == BINARY_SIZE


@dataclass(frozen=True)
class StackFeatureTable:
    layer_types: np.ndarray
    continuous: np.ndarray
    layer_counts: np.ndarray
    top_type: np.ndarray
    top_roughness: np.ndarray
    top_eta: np.ndarray
    top_k: np.ndarray
    top_albedo: np.ndarray
    top_rotation: np.ndarray


def _roughness_feature(value: float) -> float:
    return float(1.0 + np.log(max(value, 1e-3)) / np.log(1000.0))


def encode_stack(stack: LayerStack) -> tuple[np.ndarray, np.ndarray, int]:
    layer_types = np.zeros(MAX_LAYERS, dtype=np.int64)
    continuous = np.zeros((MAX_LAYERS, CONTINUOUS_FEATURE_COUNT), dtype=np.float32)
    for layer_index, layer in enumerate(stack.layers):
        layer_types[layer_index] = int(layer.layer_type)
        values = [
            _roughness_feature(layer.roughness_x),
            _roughness_feature(layer.roughness_y),
            *(np.asarray(layer.eta, dtype=np.float32) / 3.0),
            *(np.asarray(layer.k, dtype=np.float32) / 8.0),
            *layer.albedo,
            np.sin(layer.tangent_rotation),
            np.cos(layer.tangent_rotation),
        ]
        if layer_index < len(stack.media):
            medium = stack.media[layer_index]
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
        continuous[layer_index] = np.asarray(values, dtype=np.float32)
    return layer_types, continuous, len(stack.layers)


def load_stack_feature_table(dataset_dir: Path) -> StackFeatureTable:
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    state_count = int(metadata["state_count"])
    stack_path = dataset_dir / "stacks.bin"
    if stack_path.stat().st_size != state_count * BINARY_SIZE:
        raise ValueError("stacks.bin size does not match dataset state_count")
    records = np.memmap(stack_path, mode="r", dtype=_STACK_DTYPE, shape=(state_count,))
    if np.any(records["magic"] != SCHEMA_MAGIC) or np.any(records["version"] != SCHEMA_VERSION):
        raise ValueError("stacks.bin contains an unsupported schema")
    layers = records["layers"]
    media = records["media"]
    layer_counts = np.asarray(records["layer_count"], dtype=np.int64)
    layer_types = np.asarray(layers["type"], dtype=np.int64)
    continuous = np.zeros((state_count, MAX_LAYERS, CONTINUOUS_FEATURE_COUNT), dtype=np.float32)
    roughness = np.asarray(layers["roughness"], dtype=np.float32)
    continuous[..., 0:2] = 1.0 + np.log(np.maximum(roughness, 1e-3)) / np.log(1000.0)
    continuous[..., 2:5] = layers["eta"] / 3.0
    continuous[..., 5:8] = layers["k"] / 8.0
    continuous[..., 8:11] = layers["albedo"]
    continuous[..., 11] = np.sin(layers["tangent_rotation"])
    continuous[..., 12] = np.cos(layers["tangent_rotation"])
    continuous[:, : MAX_LAYERS - 1, 13:16] = np.log1p(media["sigma_a"]) / np.log(7.0)
    continuous[:, : MAX_LAYERS - 1, 16:19] = np.log1p(media["sigma_s"]) / np.log(7.0)
    continuous[:, : MAX_LAYERS - 1, 19] = media["g"]
    continuous[:, : MAX_LAYERS - 1, 20] = np.log1p(media["thickness"]) / np.log(3.0)
    medium_mask = np.arange(MAX_LAYERS - 1)[None, :] < np.asarray(
        records["medium_count"], dtype=np.int64
    )[:, None]
    continuous[:, : MAX_LAYERS - 1, 21] = medium_mask
    layer_mask = np.arange(MAX_LAYERS)[None, :] < layer_counts[:, None]
    continuous *= layer_mask[..., None]
    top = layers[:, 0]
    return StackFeatureTable(
        layer_types,
        continuous,
        layer_counts,
        np.asarray(top["type"], dtype=np.int64),
        np.asarray(top["roughness"], dtype=np.float32),
        np.asarray(top["eta"], dtype=np.float32),
        np.asarray(top["k"], dtype=np.float32),
        np.asarray(top["albedo"], dtype=np.float32),
        np.asarray(top["tangent_rotation"], dtype=np.float32),
    )
