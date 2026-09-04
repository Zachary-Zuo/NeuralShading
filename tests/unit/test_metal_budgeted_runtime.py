from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from ncls.learning.metal_budgeted_asset_cook import (
    MetalBudgetedAssetCompiler,
    MetalBudgetedCompiledAsset,
)
from ncls.learning.metal_budgeted_runtime import (
    METAL_BUDGETED_COMPILED_WORD_COUNT,
    _sample_level,
    evaluate_metal_budgeted_cooked_asset,
    pack_metal_budgeted_compiled_material,
    pack_metal_budgeted_program,
    quantize_metal_budgeted_program_state,
    quantize_metal_budgeted_runtime_model,
)
from ncls.learning.models.metal_budgeted import MetalBudgetedModel


def _conditioning(*, profile_asset_index: int = 0) -> dict[str, torch.Tensor]:
    presence = torch.zeros((1, 32), dtype=torch.int64)
    presence[:, :8] = 1
    return {
        "metal_graph_index": torch.tensor([1]),
        "metal_schema_index": torch.tensor([2]),
        "metal_recipe_index": torch.tensor([3]),
        "metal_identity_index": torch.tensor([4]),
        "metal_finish_index": torch.tensor([5]),
        "metal_asset_index": torch.tensor([profile_asset_index]),
        "metal_typed_semantic_id": torch.arange(32)[None],
        "metal_typed_type_id": torch.remainder(torch.arange(32), 8)[None],
        "metal_typed_responsibility_id": torch.remainder(
            torch.arange(32), 6
        )[None],
        "metal_typed_discrete": torch.remainder(torch.arange(32), 11)[None],
        "metal_typed_continuous": torch.linspace(-1.0, 1.0, 128).reshape(
            1, 32, 4
        ),
        "metal_typed_presence": presence,
        "metal_canonical_optical": torch.linspace(0.1, 0.9, 16)[None],
        "metal_access_state": torch.tensor(
            [[1.0, 1.0, 0.125, 0.25, 1.0, 0.0, 1.0, 0.0] + [0.0] * 8]
        ),
        "metal_frame_state": torch.zeros((1, 8)),
        "metal_distribution_id": torch.tensor([1]),
    }


def _compiled_asset(profile_id: str) -> MetalBudgetedCompiledAsset:
    generator = np.random.default_rng(17)

    def levels(base: int) -> tuple[np.ndarray, ...]:
        result = []
        extent = base
        while True:
            result.append(
                generator.integers(
                    -100, 101, size=(extent, extent, 4), dtype=np.int8
                )
            )
            if extent == 1:
                return tuple(result)
            extent = max(1, extent // 2)

    return MetalBudgetedCompiledAsset(
        profile_id,
        "encoder-only@1",
        "fixture-collection",
        "fixture-asset",
        "fixture-schema",
        "wrap",
        levels(8),
        levels(2),
    )


def test_budgeted_runtime_texture_sampling_wraps_bilinear_neighbors() -> None:
    level = np.asarray(
        [
            [[1, 1, 1, 1], [3, 3, 3, 3]],
            [[5, 5, 5, 5], [7, 7, 7, 7]],
        ],
        dtype=np.int8,
    )

    wrapped = _sample_level(level, torch.tensor([0.0, 0.0]), address_mode="wrap")
    clamped = _sample_level(level, torch.tensor([0.0, 0.0]), address_mode="clamp")
    centered = _sample_level(level, torch.tensor([0.25, 0.25]), address_mode="wrap")

    torch.testing.assert_close(wrapped, torch.full((1, 4), 4.0 / 127.0))
    torch.testing.assert_close(clamped, torch.full((1, 4), 1.0 / 127.0))
    torch.testing.assert_close(centered, torch.full((1, 4), 1.0 / 127.0))


def test_budgeted_runtime_pack_has_exact_offsets_flags_and_profile_mode() -> None:
    torch.manual_seed(23)
    for profile_id, expected_mode in (
        ("metal_budgeted_hybrid_v3", 0),
        ("metal_budgeted_direct_control_v3", 1),
        ("metal_budgeted_hybrid_role_detail_v4", 0),
        ("metal_budgeted_hybrid_center_detail_v5", 0),
    ):
        model = quantize_metal_budgeted_runtime_model(
            MetalBudgetedModel.from_context(
                {
                    "profile_id": profile_id,
                    "asset_variant_count": 52,
                    "maximum_texture_slots": 9,
                    "maximum_typed_tokens": 32,
                    "runtime_asset_reads": 2,
                }
            )
        )
        program = quantize_metal_budgeted_program_state(
            model.compile_program_state(_conditioning())
        )
        packed_program = pack_metal_budgeted_program(model)
        compiled = pack_metal_budgeted_compiled_material(
            program, _compiled_asset(profile_id)
        )
        words = np.frombuffer(compiled, dtype="<u4")
        assert len(compiled) == 4 * METAL_BUDGETED_COMPILED_WORD_COUNT
        assert words[32:40].tolist() == program.resource_and_flags[0].tolist()
        assert words[40:48].tolist() == [1, 8, 8, 4, 2, 2, 2, expected_mode]
        assert len(packed_program.payload) % 4 == 0
        assert set(packed_program.defines) == {
            "NCLS_METAL_BUDGETED_W_" + name.replace(".", "_").upper()
            for name in packed_program.layout
        }


def test_budgeted_cooked_asset_evaluator_is_finite_and_selects_distribution() -> None:
    torch.manual_seed(31)
    model = quantize_metal_budgeted_runtime_model(MetalBudgetedModel().eval())
    tensors = _conditioning()
    result = evaluate_metal_budgeted_cooked_asset(
        model,
        _compiled_asset(model.profile.profile_id),
        tensors,
        uv=(0.371, 0.619),
        mip_level=0.375,
        filter_random=0.2,
        wo=(0.17364818, -0.33682409, 0.92541658),
        wi=((0.0, 0.0, 1.0), (0.34202015, 0.16317591, 0.92541658)),
    )
    assert result.shape == (2, 3)
    assert torch.isfinite(result).all() and torch.all(result >= 0.0)
    program = model.compile_program_state(tensors)
    sampled = model.sample_asset(
        {
            **tensors,
            "uv": torch.zeros((1, 2)),
            "metal_mip_fraction": torch.zeros(1),
            "metal_budgeted_detail": torch.zeros((1, 4)),
            "metal_budgeted_context": torch.zeros((1, 4)),
        },
        program,
    )
    prepared = model.prepare_from_components(
        program, sampled, torch.tensor([[0.0, 0.0, 1.0]])
    )
    assert prepared.identity_and_flags[0, 2].item() == 1


class _FixtureAssets:
    def __init__(self) -> None:
        domain = SimpleNamespace(
            level_shapes=((8, 8), (4, 4), (2, 2), (1, 1)),
            address_mode="clamp",
        )
        self.descriptors = (
            SimpleNamespace(
                asset_id="fixture-asset",
                schema_id="fixture-schema",
                domains=(domain,),
            ),
        )
        self.collection_id = "fixture-collection"

    def sample_local_patches(
        self,
        asset_indices: torch.Tensor,
        uv: torch.Tensor,
        mip: torch.Tensor,
        *,
        patch_size: int,
        active_asset_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del asset_indices, mip, active_asset_indices
        batch = uv.shape[0]
        seed = torch.cat((uv, uv.square()), dim=1)
        patches = seed[:, None, None, :, None, None].expand(
            batch, 1, 2, 4, patch_size, patch_size
        )
        return (
            patches,
            torch.ones((batch, 1), dtype=torch.bool),
            torch.zeros((batch, 1), dtype=torch.int64),
        )


def test_budgeted_asset_compiler_emits_detail_and_quarter_context_mips() -> None:
    torch.manual_seed(37)
    model = MetalBudgetedModel().eval()
    result = MetalBudgetedAssetCompiler(
        model, _FixtureAssets(), batch_size=17
    ).compile(0)
    assert [level.shape for level in result.detail_levels] == [
        (8, 8, 4),
        (4, 4, 4),
        (2, 2, 4),
        (1, 1, 4),
    ]
    assert [level.shape for level in result.context_levels] == [
        (2, 2, 4),
        (1, 1, 4),
    ]
    assert all(level.dtype == np.int8 for level in result.detail_levels)
    assert len(result.identity) == 64
