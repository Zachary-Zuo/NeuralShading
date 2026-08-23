from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch

from ncls.core.material import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    RoughDielectricInterface,
    pack_layer_stack,
)
from ncls.core.representations.legacy_ltc_k2 import (
    p1_backend_descriptor,
)
from ncls.learning.export import flatten_p1_weights
from ncls.learning.features import FEATURE_CONTRACT, encode_layer_stack
from ncls.learning.models import create_model
from ncls.learning.prediction import predict_legacy_ltc_k2_response
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file as checkpoint_sha256

from .loader import MethodBundle, sha256_file
from .manifest import MethodBundleManifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
    )


def _copy(path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)


def _method_id(semantic: dict[str, Any], content_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ncls.method-bundle\0v1\0")
    digest.update(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for uri, value in sorted(content_hashes.items()):
        digest.update(uri.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def export_legacy_ltc_k2_checkpoint(
    checkpoint_path: Path | str,
    output_dir: Path | str,
    *,
    display_name: str = "Legacy LTC K2 P1",
    source_run_manifest: Path | str | None = None,
    created_at: str | None = None,
) -> MethodBundle:
    """从不可变 checkpoint 导出可由 Slang 独立运行的 realtime bundle。"""

    checkpoint_file = Path(checkpoint_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("MethodBundle output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
    if checkpoint["representation_id"] != "legacy-ltc-k2@1":
        raise ValueError("checkpoint does not target legacy-ltc-k2")

    width = int(checkpoint["training_config"]["width"])
    model = create_model(checkpoint["architecture_id"], width=width)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    flattened_weights, weight_layout = flatten_p1_weights(
        model.state_dict(),
        width=width,
    )
    _write_bytes(output / "weights" / "model.bin", flattened_weights.tobytes(order="C"))
    _write_bytes(output / "weights" / "layout.json", weight_layout.to_json().encode("utf-8"))
    _write_json(output / "schemas" / "feature_contract.json", FEATURE_CONTRACT)
    _write_json(
        output / "schemas" / "p1_runtime.json",
        {
            "format_name": "ncls.legacy-ltc-k2-p1-runtime",
            "format_version": 1,
            "architecture_id": checkpoint["architecture_id"],
            "representation_id": checkpoint["representation_id"],
            "source_ir": "ncls.layer-stack-ir@1",
            "maximum_interface_count": 8,
            "width": width,
            "type_width": weight_layout.type_width,
            "raw_output": {
                "lobe_count": 2,
                "values_per_lobe": 9,
                "layout": ["amplitude_rgb", "log_inverse_scale_xy", "shear_xyz", "angle"],
                "decoder": {
                    "amplitude": "softplus(x)",
                    "inverse_scale": "exp(clamp(x,-3,3))",
                    "shear": "3*tanh(x)",
                    "angle": "pi*tanh(x)",
                },
            },
        },
    )

    shader_files = {
        "backend_shader": "shaders/ncls/backends/legacy_ltc_k2/legacy_ltc_k2.slang",
        "compiler_shader": "shaders/ncls/backends/legacy_ltc_k2/p1_compiler.slang",
        "scattering_backend_contract": "shaders/ncls/contracts/scattering_backend.slang",
        "scattering_contract": "shaders/ncls/contracts/scattering_contract.slang",
        "layer_stack_ir_shader": "shaders/ncls/contracts/layer_stack_ir.slang",
        "reference_interfaces": "shaders/ncls/reference/interfaces.slang",
        "reference_sampling": "shaders/ncls/reference/sampling.slang",
    }
    for uri in shader_files.values():
        _copy(PROJECT_ROOT / uri, output / uri)

    parity_material = LayerStackIR(
        (
            RoughDielectricInterface(0.12, 0.08, 1.5, 0.25),
            DiffuseInterface((0.5, 0.25, 0.1)),
        ),
        (HomogeneousMedium((0.04, 0.04, 0.04), (0.1, 0.1, 0.1), 0.2, 0.35),),
    )
    kinds, continuous, interface_count = encode_layer_stack(parity_material)
    view = np.asarray([[0.2, 0.1, np.sqrt(0.95)]], dtype=np.float32)
    lights = np.asarray([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]], dtype=np.float32)
    top = parity_material.interfaces[0]
    assert isinstance(top, RoughDielectricInterface)
    batch = {
        "interface_kinds": torch.from_numpy(kinds[None]),
        "continuous": torch.from_numpy(continuous[None]),
        "interface_counts": torch.tensor([interface_count]),
        "view": torch.from_numpy(view),
        "top_kind": torch.tensor([int(top.kind)]),
        "top_alpha": torch.tensor([[top.alpha_x, top.alpha_y]], dtype=torch.float32),
        "top_relative_ior": torch.tensor([top.relative_ior], dtype=torch.float32),
        "top_eta": torch.zeros((1, 3), dtype=torch.float32),
        "top_k": torch.zeros((1, 3), dtype=torch.float32),
        "top_color": torch.zeros((1, 3), dtype=torch.float32),
        "top_rotation": torch.tensor([top.tangent_rotation], dtype=torch.float32),
    }
    with torch.no_grad():
        expected = predict_legacy_ltc_k2_response(
            model,
            batch,
            torch.from_numpy(lights),
        )[0].numpy()
    _write_bytes(
        output / "validation" / "parity_material.bin",
        pack_layer_stack(parity_material),
    )
    parity = {
        "format_name": "ncls.backend-parity-probe",
        "format_version": 1,
        "backend_id": "legacy-ltc-k2",
        "architecture_id": checkpoint["architecture_id"],
        "weight_width": width,
        "material_ir": "validation/parity_material.bin",
        "view_direction_local": view[0].tolist(),
        "light_directions_local": lights.tolist(),
        "expected_response_cos": expected.tolist(),
        "tolerance": {"rtol": 4e-5, "atol": 4e-6},
        "execution": "compile material with P1 Slang prepare, then evaluate response_cos",
    }
    _write_json(output / "validation" / "parity.json", parity)
    _write_json(
        output / "validation" / "metrics.json",
        {
            "checkpoint_step": checkpoint["step"],
            "validation_metrics": checkpoint.get("validation_metrics"),
        },
    )

    files = {
        "weights": "weights/model.bin",
        "weight_layout": "weights/layout.json",
        "feature_contract": "schemas/feature_contract.json",
        "compiler_contract": "schemas/p1_runtime.json",
        **shader_files,
        "parity_material": "validation/parity_material.bin",
        "parity": "validation/parity.json",
        "validation_metrics": "validation/metrics.json",
    }
    content_hashes = {uri: sha256_file(output / uri) for uri in files.values()}
    descriptor = p1_backend_descriptor(parameter_count=weight_layout.total_floats)
    run_provenance: dict[str, Any] = {}
    if source_run_manifest is not None:
        run_path = Path(source_run_manifest)
        run_provenance = json.loads(run_path.read_text(encoding="utf-8"))
    semantic = {
        "backend_id": descriptor.backend_id,
        "backend_version": descriptor.backend_version,
        "architecture_id": checkpoint["architecture_id"],
        "feature_contract_id": checkpoint["feature_contract_id"],
        "checkpoint_sha256": checkpoint_sha256(checkpoint_file),
        "runtime_class": "realtime",
    }
    manifest = MethodBundleManifest(
        method_id=_method_id(semantic, content_hashes),
        display_name=display_name,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        source_git_commit=_git_commit(),
        material_program_schema_versions=(1,),
        supported_ir_ids=descriptor.supported_ir_ids,
        scattering_contract_version=descriptor.scattering_contract_version,
        backend_id=descriptor.backend_id,
        backend_version=descriptor.backend_version,
        backend_descriptor=descriptor.to_dict(),
        runtime_class="realtime",
        compiler={
            "kind": "parameter-network",
            "feature_contract": checkpoint["feature_contract_id"],
            "normalization_contract": "schemas/feature_contract.json",
            "architecture_id": checkpoint["architecture_id"],
            "weight_files": ["weights/model.bin"],
            "weight_layout": "weights/layout.json",
            "runtime_contract": "schemas/p1_runtime.json",
            "precision": "float32",
            "runtime_implementation": "slang",
        },
        runtime={
            "platform": "windows-x86_64",
            "graphics_api": "d3d12",
            "shader_model": "6.7",
            "slang_version": "2024.1.34",
            "entry_points": dict(descriptor.shader_entry_points),
        },
        capabilities={"bitmask": int(descriptor.capabilities)},
        cost_claims=descriptor.cost_model.to_dict(),
        training_provenance={
            "dataset_id": checkpoint["dataset_id"],
            "checkpoint_step": checkpoint["step"],
            "checkpoint_sha256": checkpoint_sha256(checkpoint_file),
            "run_id": run_provenance.get("run_id"),
            "selection_split": "validation",
        },
        validation_provenance={
            "metrics": "validation/metrics.json",
            "parity_probe": "validation/parity.json",
            "held_out_test": "not-included-unless-evaluated-separately",
            "known_baseline": {"relative_l1_median": 0.0673, "relative_l1_p90": 0.3120},
        },
        files=files,
        content_hashes=content_hashes,
    )
    _write_bytes(output / "manifest.json", manifest.to_json().encode("utf-8"))
    return MethodBundle.open(output)
