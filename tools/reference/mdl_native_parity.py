from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ncls.data.providers.mdl import MdlGpuQueryRuntime
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import (
    MDL_SDK_BUILD,
    STB_COMMIT,
    STB_IMAGE_SHA256,
    MdlSdkCompilerBridge,
)
from mdl_native_protocol import (
    read_native_result_packet,
    write_native_query_packet,
)


ABSOLUTE_TOLERANCE = 3e-7
RELATIVE_TOLERANCE = 3e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _queries() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wo = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.31, -0.21, 0.927),
            (-0.58, 0.17, 0.797),
            (0.11, 0.73, 0.675),
            (-0.42, -0.51, 0.751),
            (0.83, 0.19, 0.524),
            (-0.67, 0.59, 0.451),
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
            (-0.79, 0.22, 0.572),
            (0.28, -0.91, 0.304),
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
    return wo, wi, position, uv


def main() -> int:
    parser = argparse.ArgumentParser(description="MDL SDK native 与正式 current-Falcor fixture parity")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/reference-parity/mdl/native-fixtures-v1",
    )
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    allowed = (PROJECT_ROOT / "artifacts/reference-parity/mdl").resolve()
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("native parity output must be a new directory below artifacts/reference-parity/mdl")
    output.mkdir(parents=True)

    module_root = PROJECT_ROOT / "tests/fixtures/mdl"
    bridge = MdlSdkCompilerBridge(module_root)
    wo, wi, position, uv = _queries()
    tint = np.asarray((0.17, 0.53, 0.81), dtype=np.float32)
    query_path = output / "query.bin"
    native_packet = output / "native-result.bin"
    write_native_query_packet(query_path, wo, wi, position, uv)
    artifact_output = PROJECT_ROOT / "build/mdl-reference/native-parity" / output.name
    artifact = bridge.native_evaluate(
        "::constant_diffuse",
        "constant_diffuse",
        {"tint": tint.tolist()},
        queries=query_path,
        output=artifact_output,
        result=native_packet,
    )
    native_response, native_pdf = read_native_result_packet(native_packet)

    runtime = MdlGpuQueryRuntime(artifact, sdk_root=bridge.sdk_root, query_capacity=len(wo))
    try:
        response, pdf = runtime.evaluate_torch(
            0,
            wo,
            wi,
            uv,
            np.zeros((len(wo), 4), dtype=np.float32),
            position,
        )
        formal_response = response.detach().cpu().numpy().copy()
        formal_pdf = pdf.detach().cpu().numpy().copy()
    finally:
        runtime.close()
    expected = tint[None, :] * wi[:, 2:3] / np.pi
    np.savez(output / "formal.npz", value=formal_response, pdf=formal_pdf)
    np.savez(output / "native.npz", value=native_response, pdf=native_pdf)

    response_absolute = float(np.max(np.abs(formal_response - native_response)))
    pdf_absolute = float(np.max(np.abs(formal_pdf - native_pdf)))
    analytic_absolute = float(
        max(np.max(np.abs(formal_response - expected)), np.max(np.abs(native_response - expected)))
    )
    response_relative = float(
        np.max(np.abs(formal_response - native_response) / np.maximum(np.abs(native_response), 1e-6))
    )
    pdf_relative = float(
        np.max(np.abs(formal_pdf - native_pdf) / np.maximum(np.abs(native_pdf), 1e-6))
    )
    passed = bool(
        response_absolute <= ABSOLUTE_TOLERANCE
        and pdf_absolute <= ABSOLUTE_TOLERANCE
        and analytic_absolute <= ABSOLUTE_TOLERANCE
        and response_relative <= RELATIVE_TOLERANCE
        and pdf_relative <= RELATIVE_TOLERANCE
    )
    report = {
        "schema_name": "ncls.mdl-native-parity-report",
        "schema_version": 1,
        "fixture": "::constant_diffuse::constant_diffuse(color)",
        "query_count": len(wo),
        "query_packet_sha256": _sha256(query_path),
        "formal_result_sha256": _sha256(output / "formal.npz"),
        "native_result_sha256": _sha256(output / "native.npz"),
        "thresholds": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE,
            "source": "float32 analytic Lambertian fixture; frozen before this formal run",
            "failure_action": "treat as renderer/native ABI integration defect; do not widen",
        },
        "metrics": {
            "response_max_absolute": response_absolute,
            "response_max_relative": response_relative,
            "pdf_max_absolute": pdf_absolute,
            "pdf_max_relative": pdf_relative,
            "analytic_max_absolute": analytic_absolute,
        },
        "provenance": {
            "formal_executor": "Falcor 8.0 @ 9dc819c162b2070335c65060436041690b7937f8",
            "crosscheck_executor": "MDL SDK MB_NATIVE",
            "mdl_sdk": MDL_SDK_BUILD,
            "compiled_artifact_sha256": artifact.artifact_sha256,
            "bridge_executable_sha256": artifact.manifest["compiler_identity"][
                "bridge_executable_sha256"
            ],
            "stb_commit": STB_COMMIT,
            "stb_image_sha256": STB_IMAGE_SHA256,
        },
        "passed": passed,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
