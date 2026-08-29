from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ncls.core.material import DiffuseInterface, LayerStackIR
from ncls.references.programs import get_reference_program_for_source
from ncls.references.backend import create_reference_backend
from ncls.references.mdl import resolve_mdl_program_toolchain
from ncls.references.plan import compile_single_program_plan
from ncls.references.query import ScatteringQuery
from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack
from ncls.source_materials.families.mdl import MdlFamilyDefinition
from ncls.source_materials.mdl_metal import (
    MdlMetalRegistry,
    MdlMetalStatePool,
    MdlMetalTypedStateRecipe,
)
from ncls.core.source import create_source_family


pytest.importorskip("falcor")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.falcor
def test_generic_backend_session_evaluate_sample_pdf_share_one_backend() -> None:
    snapshot = snapshot_from_layer_stack(
        LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ())
    )
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    plan = compile_single_program_plan(
        definition, (snapshot,), query_recipe={"recipe_id": "gpu-test@1"}
    )
    session = create_reference_backend().open(
        plan, query_capacity=256, device="cuda:0"
    )
    count = 256
    query = ScatteringQuery(
        torch.zeros(count, dtype=torch.int64, device="cuda:0"),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda:0").expand(count, 3),
        plan.groups[0].group_id,
    )
    wi = torch.tensor([[[0.3, 0.0, math.sqrt(0.91)]]], device="cuda:0").expand(
        count, 1, 3
    )
    seeds = torch.arange(count, dtype=torch.int64, device="cuda:0")[:, None]

    evaluated = session.evaluate(query, wi, seeds)
    torch.testing.assert_close(
        evaluated.f,
        (
            torch.tensor([0.6, 0.3, 0.1], device="cuda:0") / math.pi
        ).expand(count, 1, 3),
        rtol=3e-5,
        atol=3e-6,
    )
    assert bool(evaluated.valid.all())
    evaluated.lease.release()
    session.end_iteration()

    sampled = session.sample(query, seeds)
    assert bool(sampled.valid.all())
    sampled_wi = sampled.wi[:, None, :].clone()
    sampled_weight = sampled.weight.clone()
    sampled_pdf = sampled.pdf_forward.clone()
    sampled.lease.release()
    session.end_iteration()

    density = session.pdf(query, sampled_wi, seeds)
    assert bool(density.valid.all())
    torch.testing.assert_close(density.forward[:, 0], sampled_pdf, rtol=2e-6, atol=2e-7)
    density.lease.release()
    session.end_iteration()

    evaluated_sample = session.evaluate(query, sampled_wi, seeds)
    expected_weight = (
        evaluated_sample.f[:, 0]
        * torch.abs(sampled_wi[:, 0, 2:3])
        / sampled_pdf[:, None]
    )
    torch.testing.assert_close(sampled_weight, expected_weight, rtol=3e-5, atol=3e-6)
    evaluated_sample.lease.release()
    session.end_iteration()
    session.close()


@pytest.mark.falcor
@pytest.mark.parametrize(
    ("family_id", "locator"),
    (
        (
            "materialx.document@1.39.4",
            {"kind": "catalog-asset", "asset_id": "american_walnut_veneer"},
        ),
        (
            "openpbr.material@1.1.1",
            {
                "kind": "materialx-document",
                "path": str(
                    PROJECT_ROOT
                    / "external/OpenPBR/examples/open_pbr_brass.mtlx"
                ),
            },
        ),
        (
            "merl.measured-brdf@1",
            {"kind": "catalog-asset", "material_id": "alum-bronze"},
        ),
        (
            "mdl.program@1",
            {
                "kind": "mdl-export",
                "module_root": str(PROJECT_ROOT / "tests/fixtures/mdl"),
                "module": "::constant_diffuse",
                "export": "constant_diffuse",
                "arguments": {"tint": [0.8, 0.2, 0.1]},
            },
        ),
    ),
)
def test_generic_backend_session_supports_registered_source_families(
    family_id: str, locator: dict[str, str]
) -> None:
    if family_id == "merl.measured-brdf@1" and not (
        PROJECT_ROOT / "assets/source-materials/merl-brdf/v1/complete.json"
    ).is_file():
        pytest.skip("MERL source material asset is not downloaded")
    if (
        family_id == "mdl.program@1"
        and not resolve_mdl_program_toolchain().bridge_executable.is_file()
    ):
        pytest.skip("MDL SDK bridge is not built")
    family = create_source_family(family_id)
    snapshot = family.load_snapshot(locator)
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    plan = compile_single_program_plan(
        definition, (snapshot,), query_recipe={"recipe_id": "gpu-test@1"}
    )
    session = create_reference_backend().open(
        plan, query_capacity=128, device="cuda:0"
    )
    count = 128
    query = ScatteringQuery(
        torch.zeros(count, dtype=torch.int64, device="cuda:0"),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda:0").expand(count, 3),
        plan.groups[0].group_id,
        uv=torch.tensor([[0.37, 0.63]], device="cuda:0").expand(count, 2),
    )
    wi = torch.tensor([[[0.3, 0.2, math.sqrt(0.87)]]], device="cuda:0").expand(
        count, 1, 3
    )
    seeds = torch.arange(count, dtype=torch.int64, device="cuda:0")[:, None]

    evaluated = session.evaluate(query, wi, seeds)
    assert bool(evaluated.valid.all())
    assert bool(torch.isfinite(evaluated.f).all())
    evaluated.lease.release()
    session.end_iteration()

    sampled = session.sample(query, seeds)
    valid = sampled.valid.clone()
    assert float(valid.to(torch.float32).mean()) > 0.5
    sampled_wi = sampled.wi[:, None, :].clone()
    sampled_pdf = sampled.pdf_forward.clone()
    sampled.lease.release()
    session.end_iteration()

    density = session.pdf(query, sampled_wi, seeds)
    assert bool(density.valid.all())
    continuous = valid & (sampled_pdf > 0.0)
    torch.testing.assert_close(
        density.forward[:, 0][continuous],
        sampled_pdf[continuous],
        rtol=2e-4,
        atol=2e-6,
    )
    density.lease.release()
    session.end_iteration()
    session.close()


@pytest.mark.falcor
def test_mdl_reference_footprint_integrates_full_response_separately_from_stochastic_samples() -> None:
    if not resolve_mdl_program_toolchain().bridge_executable.is_file():
        pytest.skip("MDL SDK bridge is not built")
    family = create_source_family("mdl.program@1")
    snapshot = family.load_snapshot(
        {
            "kind": "mdl-export",
            "module_root": str(PROJECT_ROOT / "tests/fixtures/mdl"),
            "module": "::textured_diffuse",
            "export": "textured_diffuse",
        }
    )
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    plan = compile_single_program_plan(
        definition,
        (snapshot,),
        query_recipe={
            "recipe_id": "gpu-footprint-test@1",
            "evaluation_samples": 1,
            "footprint_samples": 64,
        },
    )
    session = create_reference_backend().open(plan, query_capacity=64, device="cuda:0")
    count = 64
    source_index = torch.zeros(count, dtype=torch.int64, device="cuda:0")
    wo = torch.tensor([[0.0, 0.0, 1.0]], device="cuda:0").expand(count, 3)
    uv = torch.tensor([[0.125, 0.125]], device="cuda:0").expand(count, 2)
    zero = torch.zeros((count, 2), device="cuda:0")
    wide_dx = torch.tensor([[1.0, 0.0]], device="cuda:0").expand(count, 2)
    wide_dy = torch.tensor([[0.0, 1.0]], device="cuda:0").expand(count, 2)
    wi = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda:0").expand(count, 1, 3)
    seeds = torch.arange(count, dtype=torch.int64, device="cuda:0")[:, None]

    def evaluate(uv_dx: torch.Tensor, uv_dy: torch.Tensor, footprint_samples: int):
        result = session.evaluate(
            ScatteringQuery(
                source_index,
                wo,
                plan.groups[0].group_id,
                uv=uv,
                uv_dx=uv_dx,
                uv_dy=uv_dy,
            ),
            wi,
            seeds,
            evaluation_samples=1,
            footprint_samples=footprint_samples,
        )
        try:
            assert bool(result.valid.all())
            return result.f.clone(), result.pdf_forward.clone()
        finally:
            result.lease.release()
            session.end_iteration()

    center_f, center_pdf = evaluate(zero, zero, 1)
    degenerate_f, degenerate_pdf = evaluate(zero, zero, 64)
    filtered_f, filtered_pdf = evaluate(wide_dx, wide_dy, 64)
    torch.testing.assert_close(degenerate_f, center_f, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(degenerate_pdf, center_pdf, rtol=2e-5, atol=2e-6)
    assert float(torch.linalg.vector_norm(filtered_f - center_f, dim=-1).mean()) > 0.02
    assert bool(torch.isfinite(filtered_f).all())
    assert bool(torch.isfinite(filtered_pdf).all())
    session.close()


@pytest.mark.falcor
def test_mdl_metal_typed_states_share_one_group_with_explicit_argument_offsets() -> None:
    if not resolve_mdl_program_toolchain().bridge_executable.is_file():
        pytest.skip("MDL SDK bridge is not built")
    module_root = PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials"
    if not module_root.is_dir():
        pytest.skip("vMaterials 2 assets are not installed")
    registry = MdlMetalRegistry.load(
        PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"
    )
    selected = next(
        record
        for record in registry.exports
        if record.exact_locator["module"] == "::vMaterials_2::Metal::Aging_Copper"
        and record.exact_locator["export"].split("::")[-1].startswith("Aging_Copper(")
    )
    family = MdlFamilyDefinition()
    pool = MdlMetalStatePool.generate(
        registry,
        family,
        MdlMetalTypedStateRecipe(
            "gpu-typed-state@1",
            "train",
            41,
            4,
            default_weight=0.0,
            boundary_weight=1.0,
        ),
        module_root=module_root,
        exports=(selected,),
    )
    assert len(pool.snapshots) >= 2
    validation_pool = MdlMetalStatePool.generate(
        registry,
        family,
        MdlMetalTypedStateRecipe(
            "gpu-typed-state@1",
            "validation",
            41,
            4,
            default_weight=0.0,
            boundary_weight=0.0,
        ),
        module_root=module_root,
        exports=(selected,),
    )
    assert pool.identity != validation_pool.identity
    assert {
        snapshot.snapshot_id for snapshot in pool.snapshots[1:]
    }.isdisjoint(
        snapshot.snapshot_id for snapshot in validation_pool.snapshots[1:]
    )
    definition = get_reference_program_for_source("mdl.program@1", 1)
    plan = compile_single_program_plan(
        definition,
        pool.snapshots,
        query_recipe={
            "recipe_id": "gpu-metal-typed-state@1",
            "state_pool_identity": pool.identity,
        },
    )
    assert len(plan.groups) == 1
    records = plan.groups[0].records
    assert records[0].argument_block_offset == 0
    assert all(record.argument_block_offset % 16 == 0 for record in records)
    assert len({record.argument_block_offset for record in records}) == len(records)
    session = create_reference_backend().open(
        plan, query_capacity=len(records), device="cuda:0"
    )
    count = len(records)
    query = ScatteringQuery(
        torch.arange(count, dtype=torch.int64, device="cuda:0"),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda:0").expand(count, 3),
        plan.groups[0].group_id,
        uv=torch.tensor([[0.37, 0.63]], device="cuda:0").expand(count, 2),
    )
    wi = torch.tensor([[[0.2, 0.1, math.sqrt(0.95)]]], device="cuda:0").expand(
        count, 1, 3
    )
    seeds = torch.arange(count, dtype=torch.int64, device="cuda:0")[:, None]
    result = session.evaluate(query, wi, seeds)
    try:
        assert bool(result.valid.all())
        assert bool(torch.isfinite(result.f).all())
    finally:
        result.lease.release()
        session.end_iteration()
    sampled = session.sample(query, seeds)
    try:
        assert bool(sampled.valid.all())
        sampled_wi = sampled.wi[:, None, :].clone()
        sampled_pdf = sampled.pdf_forward.clone()
    finally:
        sampled.lease.release()
        session.end_iteration()
    density = session.pdf(query, sampled_wi, seeds)
    try:
        assert bool(density.valid.all())
        continuous = sampled_pdf > 0.0
        torch.testing.assert_close(
            density.forward[:, 0][continuous],
            sampled_pdf[continuous],
            rtol=2e-4,
            atol=2e-6,
        )
    finally:
        density.lease.release()
        session.end_iteration()
        session.close()
