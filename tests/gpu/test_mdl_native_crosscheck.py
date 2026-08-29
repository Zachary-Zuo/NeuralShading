from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ncls.core.source import create_source_family
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import create_mdl_program_provider
from ncls.references.programs import get_reference_program_for_source
from ncls.references.backend import create_reference_backend
from ncls.references.query import ScatteringQuery
from tools.reference.mdl_native_protocol import (
    read_native_result_packet,
    write_native_query_packet,
)


@pytest.mark.falcor
def test_mdl_sdk_native_and_current_falcor_match_on_disjoint_diffuse_queries(
    tmp_path: Path,
) -> None:
    module_root = PROJECT_ROOT / "tests/fixtures/mdl"
    try:
        bridge = create_mdl_program_provider(module_root)
    except FileNotFoundError as error:
        pytest.skip(str(error))

    wo = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.31, -0.21, 0.927),
            (-0.58, 0.17, 0.797),
            (0.11, 0.73, 0.675),
            (-0.42, -0.51, 0.751),
        ),
        dtype=np.float32,
    )
    wi = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (-0.25, 0.41, 0.877),
            (0.61, -0.09, 0.787),
            (-0.12, -0.82, 0.56),
            (0.52, 0.47, 0.713),
        ),
        dtype=np.float32,
    )
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    position = np.asarray(
        tuple((0.13 * index, -0.07 * index, 0.0) for index in range(len(wo))),
        dtype=np.float32,
    )
    uv = np.asarray(
        tuple((0.17 + 0.11 * index, 0.23 + 0.09 * index) for index in range(len(wo))),
        dtype=np.float32,
    )
    tint = np.asarray((0.17, 0.53, 0.81), dtype=np.float32)
    query_path = tmp_path / "native-query.bin"
    result_path = tmp_path / "native-result.bin"
    write_native_query_packet(query_path, wo, wi, position, uv)
    artifact = bridge.native_evaluate(
        "::constant_diffuse",
        "constant_diffuse",
        {"tint": tint.tolist()},
        queries=query_path,
        output=tmp_path / "compiled",
        result=result_path,
    )
    native_response, native_pdf = read_native_result_packet(result_path)

    family = create_source_family("mdl.program@1")
    snapshot = family.load_snapshot(
        {
            "kind": "mdl-export",
            "module_root": str(module_root),
            "module": "::constant_diffuse",
            "export": "constant_diffuse",
            "arguments": {"tint": tint.tolist()},
        }
    )
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    runtime = create_reference_backend().open(
        definition, (snapshot,), query_capacity=len(wo), device="cuda:0"
    )
    try:
        device = torch.device("cuda:0")
        result = runtime.evaluate(
            ScatteringQuery(
                torch.zeros(len(wo), dtype=torch.int64, device=device),
                torch.as_tensor(wo, device=device),
                position=torch.as_tensor(position, device=device),
                uv=torch.as_tensor(uv, device=device),
            ),
            torch.as_tensor(wi, device=device)[:, None, :],
            torch.arange(len(wo), dtype=torch.int64, device=device)[:, None],
        )
        current_falcor_response = (
            result.f[:, 0]
            * torch.abs(torch.as_tensor(wi[:, 2:3], device=device))
        ).cpu().numpy().copy()
        current_falcor_pdf = result.pdf_forward[:, 0].cpu().numpy().copy()
        result.lease.release()
        runtime.end_iteration()
    finally:
        runtime.close()

    expected = tint[None, :] * wi[:, 2:3] / np.pi
    np.testing.assert_allclose(native_response, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(current_falcor_response, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(current_falcor_response, native_response, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(current_falcor_pdf, native_pdf, rtol=3e-6, atol=3e-7)


@pytest.mark.falcor
def test_mdl_public_f_preserves_geometry_normal_transport_response(
    tmp_path: Path,
) -> None:
    module_root = PROJECT_ROOT / "tests/fixtures/mdl"
    try:
        bridge = create_mdl_program_provider(module_root)
    except FileNotFoundError as error:
        pytest.skip(str(error))

    wo = np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32)
    wi = np.asarray(((0.6, 0.0, 0.8),), dtype=np.float32)
    position = np.zeros((1, 3), dtype=np.float32)
    uv = np.zeros((1, 2), dtype=np.float32)
    tint = np.asarray((0.17, 0.53, 0.81), dtype=np.float32)
    query_path = tmp_path / "native-query.bin"
    result_path = tmp_path / "native-result.bin"
    write_native_query_packet(query_path, wo, wi, position, uv)
    bridge.native_evaluate(
        "::tilted_normal_diffuse",
        "tilted_normal_diffuse",
        {"tint": tint.tolist()},
        queries=query_path,
        output=tmp_path / "compiled",
        result=result_path,
    )
    native_response, native_pdf = read_native_result_packet(result_path)

    snapshot = create_source_family("mdl.program@1").load_snapshot(
        {
            "kind": "mdl-export",
            "module_root": str(module_root),
            "module": "::tilted_normal_diffuse",
            "export": "tilted_normal_diffuse",
            "arguments": {"tint": tint.tolist()},
        }
    )
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    runtime = create_reference_backend().open(
        definition, (snapshot,), query_capacity=1, device="cuda:0"
    )
    try:
        device = torch.device("cuda:0")
        result = runtime.evaluate(
            ScatteringQuery(
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.as_tensor(wo, device=device),
                position=torch.as_tensor(position, device=device),
                uv=torch.as_tensor(uv, device=device),
            ),
            torch.as_tensor(wi, device=device)[:, None, :],
            torch.zeros((1, 1), dtype=torch.int64, device=device),
        )
        public_response = (
            result.f[:, 0] * torch.abs(torch.as_tensor(wi[:, 2:3], device=device))
        ).cpu().numpy().copy()
        public_pdf = result.pdf_forward[:, 0].cpu().numpy().copy()
        result.lease.release()
        runtime.end_iteration()
    finally:
        runtime.close()

    # The SDK owns geometry.normal and its shading-normal adjustment.  The
    # public f is relative to the input +Z frame and must reconstruct the SDK's
    # transport response after multiplying by wi.z.
    np.testing.assert_allclose(public_response, native_response, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(public_pdf, native_pdf, rtol=3e-6, atol=3e-7)
    base_normal_response = tint[None, :] * wi[:, 2:3] / np.pi
    assert float(np.max(np.abs(native_response - base_normal_response))) > 1e-3
