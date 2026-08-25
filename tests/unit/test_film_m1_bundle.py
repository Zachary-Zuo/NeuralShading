from __future__ import annotations

from pathlib import Path
import re

from ncls.bundle.film_m1 import _serialize_runtime_weights
from ncls.learning.models import ConditionedSharedEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_film_m1_serialized_layout_matches_fixed_slang_offsets() -> None:
    model = ConditionedSharedEvaluator(
        state_count=1,
        output_scale=((1.0, 1.0, 1.0),),
        width=256,
        latent_dim=64,
        prepare_blocks=3,
        evaluate_blocks=6,
        fourier_bands=5,
        initial_output_ratio=0.05,
    )
    weights, layout = _serialize_runtime_weights(model, 0)
    assert weights.dtype.str == "<f4"
    assert weights.size == layout["total_floats"] == 1_338_118
    assert layout["tensors"]["compiled_material.condition"]["offset"] == 1_333_251
    assert layout["tensors"]["compiled_material.output_scale"]["offset"] == 1_338_115

    shader = (
        PROJECT_ROOT / "shaders" / "ncls" / "backends" / "film_m1" / "film_m1.slang"
    ).read_text(encoding="utf-8")
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"static const uint (NCLS_FILM_M1_[A-Z0-9_]+) = (\d+)u;",
            shader,
        )
    }
    assert constants["NCLS_FILM_M1_PREPARE_INPUT_WEIGHT"] == 0
    assert constants["NCLS_FILM_M1_EVALUATE_INPUT_WEIGHT"] == 463_872
    assert constants["NCLS_FILM_M1_HEAD_NORM_WEIGHT"] == 1_331_968
    assert constants["NCLS_FILM_M1_CONDITION"] == 1_333_251
    assert constants["NCLS_FILM_M1_OUTPUT_SCALE"] == 1_338_115
    assert constants["NCLS_FILM_M1_TOTAL_FLOATS"] == weights.size
