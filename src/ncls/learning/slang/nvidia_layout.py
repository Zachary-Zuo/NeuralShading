from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_LAYOUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "abi"
    / "nvidia_neural_appearance_layout_v1.json"
)


def _load_layout() -> dict[str, Any]:
    value = json.loads(_LAYOUT_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != {
        "name": "nvidia-neural-appearance-layout",
        "version": 1,
    }:
        raise ValueError("unsupported NVIDIA neural appearance layout")
    return value


NVIDIA_NEURAL_APPEARANCE_LAYOUT = _load_layout()


def render_nvidia_neural_appearance_layout_slang() -> str:
    """从唯一 JSON ABI 生成 NVIDIA baseline 的 Slang 常量。"""

    layout = NVIDIA_NEURAL_APPEARANCE_LAYOUT
    evaluator = layout["evaluator"]
    sampler = layout["sampler"]
    values = (
        ("NCLS_NVIDIA_NEURAL_LATENT_COUNT", layout["compiled_material"]["latent_count"]),
        ("NCLS_NVIDIA_NEURAL_FRAME_RAW_COUNT", evaluator["frame_output"]),
        ("NCLS_NVIDIA_NEURAL_EVALUATE_INPUT", evaluator["input"]),
        ("NCLS_NVIDIA_NEURAL_EVALUATE_WIDTH", evaluator["width"]),
        ("NCLS_NVIDIA_NEURAL_RGB_COUNT", evaluator["output"]),
        ("NCLS_NVIDIA_NEURAL_SAMPLER_INPUT", sampler["input"]),
        ("NCLS_NVIDIA_NEURAL_SAMPLER_WIDTH", sampler["width"]),
        ("NCLS_NVIDIA_NEURAL_SAMPLER_RAW_COUNT", sampler["output"]),
    )
    constants = "\n".join(
        f"static const uint {name} = {int(value)}u;" for name, value in values
    )
    minimum_cosine = format(
        float(layout["response_adapter"]["minimum_cosine"]), ".8g"
    )
    return (
        "// Generated from src/ncls/learning/abi/"
        "nvidia_neural_appearance_layout_v1.json; do not edit.\n"
        "#ifndef NCLS_NVIDIA_NEURAL_APPEARANCE_LAYOUT_SLANG\n"
        "#define NCLS_NVIDIA_NEURAL_APPEARANCE_LAYOUT_SLANG\n\n"
        f"{constants}\n"
        "static const float NCLS_NVIDIA_NEURAL_MIN_COS = "
        f"{minimum_cosine}f;\n\n"
        "#endif\n"
    )


def nvidia_neural_appearance_layout_sha256() -> str:
    payload = json.dumps(
        NVIDIA_NEURAL_APPEARANCE_LAYOUT,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
