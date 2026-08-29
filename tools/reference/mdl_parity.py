from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from ncls.core.source import SourceSnapshot, create_source_family
from ncls.paths import PROJECT_ROOT
from ncls.references.acceptance import (
    DeterministicDirectionalGate,
    deterministic_directional_metrics,
)
from ncls.references.mdl import (
    MDL_SDK_BUILD,
    STB_COMMIT,
    STB_IMAGE_SHA256,
    create_mdl_program_provider,
)
from ncls.references.programs import get_reference_program_for_source
from ncls.references.backend import create_reference_backend
from ncls.references.query import ScatteringQuery

from mdl_oracle.protocol import canonical_json


FORMAL_ASSET_IDS = (
    "carpaint-shifting-flakes",
    "copper-antique-brushed-patinated",
)
SMOKE_ASSET_IDS = (
    "aluminum-scratched",
    "ceramic-tiles-glazed-versailles",
    "velvet",
    "wood-tiles-pine-mosaic",
)
SEEDS = {"calibration": 0x4D444C31, "formal": 0x4D444C32}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hemisphere_directions(rng: np.random.Generator, count: int) -> np.ndarray:
    z = rng.uniform(0.08, 1.0, count)
    phi = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.asarray(
        np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1),
        dtype=np.float32,
    )


def _query(mode: str) -> dict[str, Any]:
    seed = SEEDS[mode]
    rng = np.random.default_rng(seed)
    view_count, light_count, surface_count = ((2, 5, 2) if mode == "calibration" else (4, 11, 3))
    views = _hemisphere_directions(rng, view_count)
    lights = _hemisphere_directions(rng, light_count)
    views[0] = (0.0, 0.0, 1.0)
    lights[0] = (0.0, 0.0, 1.0)
    surfaces = []
    for _ in range(surface_count):
        uv = rng.uniform(0.07, 0.93, 2)
        surfaces.append(
            {
                "position": [float(2.0 * uv[0] - 1.0), float(2.0 * uv[1] - 1.0), 0.0],
                "uv": [float(uv[0]), float(uv[1])],
                "uv_dx": [0.0, 0.0],
                "uv_dy": [0.0, 0.0],
            }
        )
    return {
        "query_set": mode,
        "seed": seed,
        "frame": {"normal": [0.0, 0.0, 1.0], "tangent": [1.0, 0.0, 0.0]},
        "state": {
            "animation_time": 0.0,
            "meters_per_scene_unit": 1.0,
            "object_id": 0,
            "exterior_ior": 1.0,
            "learnable": False,
            "texture_lod": 0.0,
        },
        "response_measure": "rgb-bsdf-times-absolute-shading-normal-light-cosine",
        "surfaces": surfaces,
        "view_directions": views.tolist(),
        "light_directions": lights.tolist(),
    }


def _source_locator(asset_id: str) -> dict[str, Any]:
    if asset_id == "constant-diffuse":
        return {
            "kind": "mdl-export",
            "module_root": str(PROJECT_ROOT / "tests/fixtures/mdl"),
            "module": "::constant_diffuse",
            "export": "constant_diffuse",
        }
    if asset_id == "textured-diffuse":
        return {
            "kind": "mdl-export",
            "module_root": str(PROJECT_ROOT / "tests/fixtures/mdl"),
            "module": "::textured_diffuse",
            "export": "textured_diffuse",
        }
    manifest = json.loads(
        (PROJECT_ROOT / "references/mdl-vmaterials2-v1/assets.json").read_text(
            encoding="utf-8"
        )
    )
    records = {str(item["asset_id"]): item for item in manifest["assets"]}
    record = records[asset_id]
    module_root = (
        PROJECT_ROOT
        / "assets/source-materials/mdl-vmaterials2/2.4.0"
        / str(manifest["module_root"])
    )
    return {
        "kind": "mdl-export",
        "module_root": str(module_root),
        "module": str(record["module"]),
        "export": str(record["export"]),
        "pack_id": "nvidia.vmaterials",
        "pack_version": "2.4.0",
    }


def _material_name(exact_export: str) -> str:
    return exact_export.rsplit("::", 1)[-1].split("(", 1)[0]


def _make_request(
    snapshot: SourceSnapshot, locator: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    payload = json.loads(snapshot.native_payload.decode("utf-8"))
    module_root = Path(str(locator["module_root"])).resolve().relative_to(
        PROJECT_ROOT.resolve()
    ).as_posix()
    return {
        "schema_name": "ncls.mdl-oracle-request",
        "schema_version": 1,
        "source": {
            "asset_id": payload["module"],
            "module_root": module_root,
            "module": payload["module"],
            "material": _material_name(payload["export"]),
            "exact_export": payload["export"],
            "arguments": payload.get("arguments", {}),
            "source_snapshot_id": snapshot.snapshot_id,
            "mdl_sdk": payload["mdl_sdk"],
            "compilation_mode": payload["compilation_mode"],
        },
        "query": _query(mode),
    }


def _formal_result(
    snapshot: SourceSnapshot, request: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    query = request["query"]
    surfaces = tuple(query["surfaces"])
    views = np.asarray(query["view_directions"], dtype=np.float32)
    lights = np.asarray(query["light_directions"], dtype=np.float32)
    batch = len(surfaces) * len(views)
    positions = np.repeat(
        np.asarray([item["position"] for item in surfaces], dtype=np.float32),
        len(views),
        axis=0,
    )
    uv = np.repeat(
        np.asarray([item["uv"] for item in surfaces], dtype=np.float32),
        len(views),
        axis=0,
    )
    uv_dx = np.repeat(
        np.asarray([item["uv_dx"] for item in surfaces], dtype=np.float32),
        len(views),
        axis=0,
    )
    uv_dy = np.repeat(
        np.asarray([item["uv_dy"] for item in surfaces], dtype=np.float32),
        len(views),
        axis=0,
    )
    wo = np.tile(views, (len(surfaces), 1))
    definition = get_reference_program_for_source(
        snapshot.family_id, snapshot.source_contract_version
    )
    session = create_reference_backend().open(
        definition,
        (snapshot,),
        query_capacity=batch * len(lights),
        device="cuda:0",
    )
    device = torch.device("cuda:0")
    wi = torch.as_tensor(lights, device=device)[None, :, :].expand(
        batch, -1, -1
    )
    try:
        result = session.evaluate(
            ScatteringQuery(
                torch.zeros(batch, dtype=torch.int64, device=device),
                torch.as_tensor(wo, device=device),
                position=torch.as_tensor(positions, device=device),
                uv=torch.as_tensor(uv, device=device),
                uv_dx=torch.as_tensor(uv_dx, device=device),
                uv_dy=torch.as_tensor(uv_dy, device=device),
            ),
            wi,
            torch.arange(
                int(query["seed"]),
                int(query["seed"]) + batch * len(lights),
                dtype=torch.int64,
                device=device,
            ).reshape(batch, len(lights)),
        )
        response = (
            result.f * torch.abs(wi[..., 2:3])
        ).cpu().numpy().copy().reshape(len(surfaces), len(views), len(lights), 3)
        pdf = result.pdf_forward.cpu().numpy().copy().reshape(
            len(surfaces), len(views), len(lights)
        )
        result.lease.release()
        session.end_iteration()
        return response, pdf
    finally:
        session.close()


def _run_oracle(request_path: Path, output_dir: Path) -> None:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "scripts/run_falcor2_mdl_oracle.ps1"),
        "-Request",
        str(request_path),
        "-Output",
        str(output_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, timeout=900)


def _metric_dict(native: np.ndarray, candidate: np.ndarray, gate: DeterministicDirectionalGate):
    return asdict(
        deterministic_directional_metrics(
            np.asarray(native, dtype=np.float32).reshape(-1, 3),
            np.asarray(candidate, dtype=np.float32).reshape(-1, 3),
            gate,
        )
    )


def _gate(path: Path) -> tuple[DeterministicDirectionalGate, Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_name") != "ncls.mdl-oracle-parity-gate" or value.get("schema_version") != 1:
        raise ValueError("unsupported MDL oracle parity gate")
    return DeterministicDirectionalGate.from_dict(value["thresholds"]), value


def _asset_parity(asset_id: str, mode: str, output_root: Path, gate_path: Path) -> dict[str, Any]:
    locator = _source_locator(asset_id)
    family = create_source_family("mdl.program@1")
    snapshot = family.load_snapshot(locator)
    artifact = create_mdl_program_provider(Path(str(locator["module_root"]))).compile_snapshot(snapshot)
    asset_dir = output_root / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    request = _make_request(snapshot, locator, mode)
    request_path = asset_dir / "request.json"
    request_path.write_bytes(canonical_json(request))
    formal_value, formal_pdf = _formal_result(snapshot, request)
    np.savez(asset_dir / "formal.npz", value=formal_value, pdf=formal_pdf)
    oracle_dir = asset_dir / "oracle"
    _run_oracle(request_path, oracle_dir)
    oracle_metadata = json.loads(
        (oracle_dir / "result.json").read_text(encoding="utf-8")
    )
    if oracle_metadata["source_snapshot_id"] != request["source"]["source_snapshot_id"]:
        raise RuntimeError("oracle response source identity differs from the request")
    oracle = np.load(oracle_dir / "result.npz")
    oracle_value = np.asarray(oracle["value"], dtype=np.float32)
    oracle_pdf = np.asarray(oracle["pdf"], dtype=np.float32)
    if formal_value.shape != oracle_value.shape or formal_pdf.shape != oracle_pdf.shape:
        raise RuntimeError("formal/oracle result shape mismatch")

    metric_gate, gate_metadata = _gate(gate_path)
    response_metrics = _metric_dict(oracle_value, formal_value, metric_gate)
    pdf_metrics = _metric_dict(
        np.repeat(oracle_pdf[..., None], 3, axis=-1),
        np.repeat(formal_pdf[..., None], 3, axis=-1),
        metric_gate,
    )
    passed = None if mode == "calibration" else bool(
        response_metrics["passed"] and pdf_metrics["passed"]
    )
    return {
        "asset_id": asset_id,
        "source_snapshot_id": request["source"]["source_snapshot_id"],
        "request": request_path.relative_to(output_root).as_posix(),
        "request_sha256": _sha256(request_path),
        "formal_result": "formal.npz",
        "formal_result_sha256": _sha256(asset_dir / "formal.npz"),
        "oracle_result": "oracle/result.npz",
        "oracle_result_sha256": _sha256(oracle_dir / "result.npz"),
        "response_metrics": response_metrics,
        "pdf_metrics": pdf_metrics,
        "gate_identity": gate_metadata["gate_id"],
        "passed": passed,
        "formal_provenance": {
            "executor": "Falcor 8.0 current project runtime",
            "falcor_commit": "9dc819c162b2070335c65060436041690b7937f8",
            "mdl_sdk": MDL_SDK_BUILD,
            "compiled_artifact_sha256": artifact.artifact_sha256,
            "bridge_executable_sha256": artifact.manifest["compiler_identity"][
                "bridge_executable_sha256"
            ],
            "stb_commit": STB_COMMIT,
            "stb_image_sha256": STB_IMAGE_SHA256,
            "formal_provider_imported_falcor2": False,
        },
        "oracle_provenance": oracle_metadata["provenance"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-check the formal MDL reference with falcor2")
    parser.add_argument("--mode", choices=("calibration", "formal"), default="formal")
    parser.add_argument("--asset-id", action="append")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/reference-parity/mdl/formal-v1",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=PROJECT_ROOT / "references/mdl-vmaterials2-v1/parity-gate.json",
    )
    arguments = parser.parse_args()
    assets = tuple(arguments.asset_id or FORMAL_ASSET_IDS)
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("MDL parity asset IDs must be nonempty and unique")
    known = {"constant-diffuse", "textured-diffuse", *FORMAL_ASSET_IDS, *SMOKE_ASSET_IDS}
    unknown = sorted(set(assets) - known)
    if unknown:
        raise ValueError(f"unknown MDL parity assets: {unknown}")
    output_root = arguments.output_dir.resolve()
    allowed = (PROJECT_ROOT / "artifacts/reference-parity/mdl").resolve()
    if not output_root.is_relative_to(allowed):
        raise ValueError(f"parity output must stay under {allowed}")
    output_root.mkdir(parents=True, exist_ok=True)

    records = [
        _asset_parity(asset_id, arguments.mode, output_root, arguments.gate.resolve())
        for asset_id in assets
    ]
    passed = None if arguments.mode == "calibration" else all(record["passed"] for record in records)
    report = {
        "schema_name": "ncls.mdl-falcor2-parity-report",
        "schema_version": 1,
        "mode": arguments.mode,
        "reference_id": "ncls.mdl-vmaterials2@1",
        "formal_executor": "project MDL SDK bridge / Falcor 8 @ 9dc819c1",
        "validation_oracle": "falcor2.MDLMaterial @ d629c967 / same MDL SDK",
        "query_set_seed": SEEDS[arguments.mode],
        "query_scope": "upper hemisphere, canonical +Z/+X frame, ExplicitLod(0), learnable=false",
        "assets": records,
        "passed": passed,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
