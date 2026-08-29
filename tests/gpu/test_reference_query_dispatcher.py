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
