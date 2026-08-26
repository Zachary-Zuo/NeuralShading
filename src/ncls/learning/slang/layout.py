from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "abi" / "unified_neural_layout_v1.json"


def _load_layout() -> dict[str, Any]:
    value = json.loads(_LAYOUT_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != {"name": "unified-neural-layout", "version": 1}:
        raise ValueError("unsupported unified neural layout")
    return value


UNIFIED_LAYOUT = _load_layout()


def render_unified_layout_slang() -> str:
    """从唯一JSON ABI生成Slang常量，避免Python/Slang各维护一份offset。"""
    realtime = UNIFIED_LAYOUT["realtime"]
    paper = UNIFIED_LAYOUT["paper"]
    sampler = UNIFIED_LAYOUT["sampler"]
    values = (
        ("NCLS_UNIFIED_LATENT_COUNT", UNIFIED_LAYOUT["compiled_material"]["latent_count"]),
        ("NCLS_UNIFIED_PREPARE_INPUT", realtime["prepare_input"]),
        ("NCLS_UNIFIED_PREPARE_WIDTH", realtime["prepare_width"]),
        ("NCLS_UNIFIED_PREPARE_OUTPUT", realtime["prepare_output"]),
        ("NCLS_UNIFIED_EVALUATOR_STATE", realtime["prepare_output"] - sampler["ltc_raw_count"]),
        ("NCLS_UNIFIED_EVALUATE_INPUT", realtime["evaluate_input"]),
        ("NCLS_UNIFIED_EVALUATE_WIDTH", realtime["evaluate_width"]),
        ("NCLS_UNIFIED_PAPER_WIDTH", paper["evaluate_width"]),
        ("NCLS_UNIFIED_RGB_COUNT", realtime["evaluate_output"]),
        ("NCLS_UNIFIED_NVIDIA_RAW_COUNT", sampler["nvidia_raw_count"]),
        ("NCLS_UNIFIED_LTC_RAW_COUNT", sampler["ltc_raw_count"]),
    )
    constants = "\n".join(f"static const uint {name} = {int(value)}u;" for name, value in values)
    safety = format(float(sampler["safety_weight"]), ".8g")
    return (
        "// Generated from src/ncls/learning/abi/unified_neural_layout_v1.json; do not edit.\n"
        "#ifndef NCLS_UNIFIED_NEURAL_LAYOUT_SLANG\n"
        "#define NCLS_UNIFIED_NEURAL_LAYOUT_SLANG\n\n"
        f"{constants}\n"
        f"static const float NCLS_UNIFIED_SAFETY_WEIGHT = {safety}f;\n\n"
        "#endif\n"
    )


def unified_layout_sha256() -> str:
    payload = json.dumps(
        UNIFIED_LAYOUT,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
