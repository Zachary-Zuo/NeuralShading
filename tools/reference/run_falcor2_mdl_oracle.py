from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import slangpy as spy
import falcor2 as f2
from falcor2.editor import SceneShaderHelper, create_device

from mdl_oracle.protocol import load_request, query_arrays, sha256_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FALCOR2_COMMIT = "d629c967fa800af81cf5c916bfb2a825b012f473"
MDL_SDK_ID = "2025.0.0-387700.1252"


def _contained_path(root: Path, relative: str, *, label: str) -> Path:
    source = Path(relative)
    if source.is_absolute() or ".." in source.parts:
        raise ValueError(f"{label} must be a project-relative contained path")
    result = (root / source).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the project root")
    return result


def _argument_value(name: str, descriptor: Any, module_root: Path) -> Any:
    if not isinstance(descriptor, dict) or "mdl_type" not in descriptor or "value" not in descriptor:
        raise ValueError(f"argument {name!r} must contain mdl_type and value")
    mdl_type = str(descriptor["mdl_type"])
    value = descriptor["value"]
    if mdl_type == "bool":
        return bool(value)
    if mdl_type in {"int", "enum"}:
        return int(value)
    if mdl_type in {"float", "double"}:
        return float(value)
    if mdl_type == "float2":
        return spy.float2(*map(float, value))
    if mdl_type in {"float3", "color"}:
        return spy.float3(*map(float, value))
    if mdl_type == "float4":
        return spy.float4(*map(float, value))
    if mdl_type == "texture_2d":
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError(f"texture argument {name!r} has no contained path")
        texture = _contained_path(module_root, value["path"], label=f"argument {name}")
        if not texture.is_file():
            raise FileNotFoundError(texture)
        return str(texture)
    raise ValueError(f"falcor2 oracle does not support argument type {mdl_type!r}")


def _read_tensor_float4(tensor: spy.Tensor, count: int) -> np.ndarray:
    cursor = tensor.cursor()
    cursor.load()
    result = np.empty((count, 4), dtype=np.float32)
    for index in range(count):
        value = cursor[index].read()
        result[index] = (float(value.x), float(value.y), float(value.z), float(value.w))
    return result


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run(request_path: Path, output_dir: Path) -> None:
    request, request_bytes = load_request(request_path)
    allowed_output = (PROJECT_ROOT / "artifacts/reference-parity/mdl").resolve()
    output = output_dir.resolve()
    if not output.is_relative_to(allowed_output):
        raise ValueError(f"oracle output must stay under {allowed_output}")
    output.mkdir(parents=True, exist_ok=True)

    source = request["source"]
    module_root = _contained_path(PROJECT_ROOT, source["module_root"], label="source.module_root")
    allowed_sources = (
        (PROJECT_ROOT / "tests/fixtures/mdl").resolve(),
        (PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials").resolve(),
    )
    if not any(module_root == root or module_root.is_relative_to(root) for root in allowed_sources):
        raise ValueError("oracle source root is outside the registered MDL roots")
    if not module_root.is_dir():
        raise FileNotFoundError(module_root)

    falcor2_root = (PROJECT_ROOT / "external/falcor2").resolve()
    actual_commit = _git_commit(falcor2_root)
    if actual_commit != FALCOR2_COMMIT:
        raise RuntimeError(f"falcor2 commit mismatch: {actual_commit}")

    device = create_device(spy.DeviceType.d3d12, enable_debug_layers=False)
    try:
        scene = f2.Scene.create(device, f2.UVOrigin.lower_left)
        properties = f2.Properties()
        properties["mdl_library_path"] = str(module_root)
        module = str(source["module"])[2:]
        properties["mdl_material_name"] = f"{module}::{source['material']}"
        properties["mdl_class_compilation"] = True
        properties["learnable"] = False
        properties["debug_write_shader_path"] = str(output / "falcor2-${NAME}.slang")
        for name, descriptor in source.get("arguments", {}).items():
            properties[str(name)] = _argument_value(str(name), descriptor, module_root)
            if descriptor["mdl_type"] == "texture_2d":
                gamma = float(descriptor["value"].get("effective_gamma", 1.0))
                properties[f"{name}:srgb"] = abs(gamma - 2.2) < 0.2
        material = scene.create_material("MDLMaterial", properties)
        scene.update()
        generated_shaders = tuple(output.glob("falcor2-*.slang"))
        if len(generated_shaders) != 1:
            raise RuntimeError("falcor2 did not emit exactly one generated MDL shader")
        generated_shader = generated_shaders[0]

        query_source = (PROJECT_ROOT / "tools/reference/mdl_oracle/query.slang").read_text(
            encoding="utf-8"
        )
        query_module = spy.Module(
            device.load_module_from_source("ncls_falcor2_mdl_oracle_query", query_source)
        )
        helper = SceneShaderHelper(device)
        module = helper.get_module(scene, query_module)
        evaluate = (
            module["ncls_mdl_evaluate<TinyUniformSampleGenerator>"]
            .as_func()
            .write(helper.bind_scene)
        )

        positions, uv, views, lights = query_arrays(request)
        count = len(views)
        seeds = np.zeros((count, 3), dtype=np.uint32)
        seeds[:, 0] = np.arange(count, dtype=np.uint32) + np.uint32(int(request["query"]["seed"]))
        tensor = evaluate(seeds, material, positions, uv, views, lights)
        flat = _read_tensor_float4(tensor, count)
        shape = (
            len(request["query"]["surfaces"]),
            len(request["query"]["view_directions"]),
            len(request["query"]["light_directions"]),
        )
        value = flat[:, :3].reshape(*shape, 3)
        pdf = flat[:, 3].reshape(shape)
        if not np.isfinite(value).all() or not np.isfinite(pdf).all() or np.any(pdf < 0.0):
            raise RuntimeError("falcor2 oracle returned an invalid evaluate/PDF result")
        np.savez(output / "result.npz", value=value, pdf=pdf)
        result_hash = hashlib.sha256((output / "result.npz").read_bytes()).hexdigest()
        metadata = {
            "schema_name": "ncls.mdl-oracle-result",
            "schema_version": 1,
            "request_sha256": sha256_bytes(request_bytes),
            "source_snapshot_id": source["source_snapshot_id"],
            "asset_id": source["asset_id"],
            "query_shape": list(shape),
            "response_measure": request["query"]["response_measure"],
            "result_npz": "result.npz",
            "result_npz_sha256": result_hash,
            "generated_shader": generated_shader.name,
            "generated_shader_sha256": hashlib.sha256(generated_shader.read_bytes()).hexdigest(),
            "provenance": {
                "executor": "falcor2.MDLMaterial",
                "falcor2_commit": actual_commit,
                "mdl_sdk": MDL_SDK_ID,
                "mdl_class_compilation": True,
                "learnable": False,
                "uv_origin": "lower_left",
                "texture_filtering": "ExplicitLodSampler(0)",
                "formal_provider_imported": False,
            },
        }
        (output / "result.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        device.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated falcor2 MDL validation oracle")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.request.resolve(), arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
