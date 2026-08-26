from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

from ncls.core.material import MaterialProgram, canonicalize_layer_stack, pack_layer_stack
from ncls.core.scattering import BackendDescriptor

from .loader import sha256_file
from .manifest import MethodBundleManifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def export_compiled_set_bundle(
    compiled_set_path: Path | str,
    preview_material_path: Path | str,
    parity_path: Path | str,
    output_path: Path | str,
    *,
    display_name: str,
    state_id: str,
) -> MethodBundleManifest:
    """把带 runtime adapter 的冻结 compiled set 封装成通用 MethodBundle。"""

    compiled_root = Path(compiled_set_path).resolve()
    preview_source = Path(preview_material_path).resolve()
    parity_source = Path(parity_path).resolve()
    output = Path(output_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("MethodBundle output directory must be new or empty")
    compiled = _read_json(compiled_root / "manifest.json")
    adapter = compiled.get("runtime_adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("compiled set has no standard runtime_adapter")
    descriptor = BackendDescriptor.from_dict(adapter["backend_descriptor"])
    state_ids = tuple(map(str, compiled["state_ids"]))
    if state_id not in state_ids:
        raise ValueError("preview state is absent from the compiled material table")
    material_index = state_ids.index(state_id)
    record_stride = int(adapter["compiled_material_stride"])
    state_stride = int(adapter["packed_state_stride"])
    if record_stride != int(compiled["record_stride"]):
        raise ValueError("runtime adapter compiled-material stride disagrees with asset")
    if descriptor.state_stride != state_stride:
        raise ValueError("runtime adapter state stride disagrees with backend descriptor")
    if adapter.get("shared_weight_storage") != "float16-little-endian":
        raise ValueError("viewer runtime requires packed float16 shared weights")

    compiled_files = compiled.get("files")
    if not isinstance(compiled_files, Mapping):
        raise ValueError("compiled set files must be an object")
    for name in ("compiled_materials.bin", "shared_weights_fp16.bin"):
        source = compiled_root / name
        if not source.is_file() or sha256_file(source) != str(compiled_files.get(name, "")):
            raise ValueError(f"compiled set content hash mismatch: {name}")
    materials_size = (compiled_root / "compiled_materials.bin").stat().st_size
    if materials_size != record_stride * len(state_ids):
        raise ValueError("compiled material table length disagrees with record stride/count")

    material = MaterialProgram.from_json(preview_source.read_text(encoding="utf-8"))
    material_ir_sha256 = hashlib.sha256(
        pack_layer_stack(canonicalize_layer_stack(material))
    ).hexdigest()
    parity = _read_json(parity_source)
    if parity.get("format_name") != "ncls.backend-parity-probe" or int(parity.get("format_version", 0)) != 1:
        raise ValueError("unsupported backend parity probe")
    if parity.get("compiled_state_id") != state_id:
        raise ValueError("parity probe and preview state disagree")
    if parity.get("compiled_set_id") != compiled.get("compiled_set_id"):
        raise ValueError("parity probe and compiled set identity disagree")

    shader_module = str(adapter["shader_module"])
    shader_source = PROJECT_ROOT / "shaders" / shader_module
    if not shader_source.is_file():
        raise ValueError(f"runtime adapter shader module is missing: {shader_module}")

    output.mkdir(parents=True, exist_ok=True)
    relative_paths = {
        "compiled_materials": "assets/compiled-materials.bin",
        "shared_weights": "assets/shared-weights-fp16.bin",
        "backend_shader": f"shaders/{shader_module}",
        "compiled_set_manifest": "provenance/compiled-set.json",
        "preview_material": "resources/preview-material.json",
        "parity": "validation/parity.json",
    }
    sources = {
        "compiled_materials": compiled_root / "compiled_materials.bin",
        "shared_weights": compiled_root / "shared_weights_fp16.bin",
        "backend_shader": shader_source,
        "compiled_set_manifest": compiled_root / "manifest.json",
        "preview_material": preview_source,
        "parity": parity_source,
    }
    for logical_name, source in sources.items():
        target = output / relative_paths[logical_name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    content_hashes = {
        uri: sha256_file(output / uri) for uri in relative_paths.values()
    }
    identity = {
        "compiled_set_id": compiled["compiled_set_id"],
        "compiled_state_id": state_id,
        "backend_descriptor": descriptor.to_dict(),
        "content_hashes": content_hashes,
    }
    method_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    runtime = {
        "platform": "windows-x86_64",
        "graphics_api": "d3d12",
        "shader_model": "6.5",
        "slang_version": "2024.1.34",
        "shader_specialization": {
            "module": shader_module,
            "defines": dict(adapter["shader_defines"]),
            "compiled_material_stride": record_stride,
            "packed_state_stride": state_stride,
            "compiled_material_index": material_index,
            "shared_weight_storage": adapter["shared_weight_storage"],
        },
        "environment_query_budget": 1,
        "rectangle_query_budget": 1,
    }
    cost = dict(compiled["cost"])
    manifest = MethodBundleManifest(
        method_id=method_id,
        display_name=display_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_git_commit=_git_commit(),
        material_program_schema_versions=(1,),
        supported_ir_ids=tuple(descriptor.supported_ir_ids),
        scattering_contract_version=descriptor.scattering_contract_version,
        backend_id=descriptor.backend_id,
        backend_version=descriptor.backend_version,
        backend_descriptor=descriptor.to_dict(),
        runtime_class=str(compiled["runtime_class"]),
        compiler={
            "kind": "latent",
            "runtime_implementation": "slang",
            "architecture_id": str(adapter["architecture_id"]),
            "compile_mode": "compiled-corpus-state-table",
            "compiled_set_id": str(compiled["compiled_set_id"]),
            "compiled_state_id": state_id,
            "compiled_material_index": material_index,
            "compiled_material_ir_sha256": material_ir_sha256,
            "precision": "float16-packed",
        },
        runtime=runtime,
        capabilities={
            "directional_evaluation": True,
            "scattering_sampling": True,
            "path_tracing_compatible": True,
            "reverse_pdf": True,
            "transmission": False,
            "nonlocal_transport": False,
        },
        cost_claims={
            "compiled_material_bytes": record_stride,
            "shared_weight_bytes": int(cost["B_shared"]),
            "state_bytes_per_pixel": state_stride,
            "prepare_macs": int(cost["C_prepare_macs"]),
            "evaluate_macs": int(cost["C_eval_macs"]),
            "precision": "float16-packed",
            "bounded_execution": True,
        },
        training_provenance={
            "pipeline": str(compiled["pipeline"]),
            "data_id": str(compiled["data_id"]),
            "compiled_set_id": str(compiled["compiled_set_id"]),
            "checkpoint_sha256": str(
                compiled.get("checkpoint_sha256", compiled.get("evaluator_checkpoint_sha256", ""))
            ),
            "sampler_checkpoint_sha256": str(compiled.get("sampler_checkpoint_sha256", "")),
        },
        validation_provenance={
            "gpu_parity": "viewer-load-time-required",
            "parity_source_sha256": sha256_file(parity_source),
        },
        files=relative_paths,
        content_hashes=content_hashes,
    )
    (output / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return manifest


__all__ = ["export_compiled_set_bundle"]
