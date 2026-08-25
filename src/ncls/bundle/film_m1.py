from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch

from ncls.core.material import (
    MaterialProgram,
    canonicalize_layer_stack,
    pack_layer_stack,
)
from ncls.core.scattering import (
    BackendCapability,
    BackendCostModel,
    BackendDescriptor,
    StateStorage,
)
from ncls.learning.models import ConditionedSharedEvaluator
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig

from .loader import sha256_file as bundle_sha256_file
from .manifest import MethodBundleManifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ID = "film-m1-direct-neural"
BACKEND_VERSION = 1
ARCHITECTURE_ID = "film-prepare-evaluate-calibrated-softplus-v2@m1-m"
WEIGHT_FORMAT = "ncls.film-m1-weights"
DEFAULT_PREVIEW_STATE_ID = (
    "6324e3b293866fb9ac02d9b373ce260cf988b3af31babf3cd9d6ff87e9579df1"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def _normalized(values: tuple[float, float, float]) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return tuple(float(value) for value in vector)


def _serialize_runtime_weights(
    model: ConditionedSharedEvaluator,
    state_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    parts: list[np.ndarray] = []
    tensors: dict[str, Any] = {}
    offset = 0

    def append(name: str, value: torch.Tensor) -> None:
        nonlocal offset
        array = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        flat = array.reshape(-1)
        tensors[name] = {
            "offset": offset,
            "count": int(flat.size),
            "shape": list(array.shape),
        }
        parts.append(flat)
        offset += int(flat.size)

    append("prepare_input.weight", model.prepare_input.weight)
    append("prepare_input.bias", model.prepare_input.bias)
    for index, layer in enumerate(model.prepare_layers):
        append(f"prepare_layers.{index}.norm.weight", layer.norm.weight)
        append(f"prepare_layers.{index}.norm.bias", layer.norm.bias)
        append(f"prepare_layers.{index}.linear_1.weight", layer.linear_1.weight)
        append(f"prepare_layers.{index}.linear_1.bias", layer.linear_1.bias)
        append(f"prepare_layers.{index}.linear_2.weight", layer.linear_2.weight)
        append(f"prepare_layers.{index}.linear_2.bias", layer.linear_2.bias)
    append("evaluate_input.weight", model.evaluate_input.weight)
    append("evaluate_input.bias", model.evaluate_input.bias)
    for index, layer in enumerate(model.evaluate_layers):
        append(f"evaluate_layers.{index}.norm.weight", layer.norm.weight)
        append(f"evaluate_layers.{index}.norm.bias", layer.norm.bias)
        append(f"evaluate_layers.{index}.linear_1.weight", layer.linear_1.weight)
        append(f"evaluate_layers.{index}.linear_1.bias", layer.linear_1.bias)
        append(f"evaluate_layers.{index}.linear_2.weight", layer.linear_2.weight)
        append(f"evaluate_layers.{index}.linear_2.bias", layer.linear_2.bias)
    append("head.norm.weight", model.head[0].weight)
    append("head.norm.bias", model.head[0].bias)
    append("head.linear.weight", model.head[1].weight)
    append("head.linear.bias", model.head[1].bias)

    with torch.no_grad():
        state = torch.tensor([state_index], dtype=torch.long)
        condition = model.condition(model.latent(state))[0]
    append("compiled_material.condition", condition)
    append("compiled_material.output_scale", model.output_scale[state_index])

    payload = np.concatenate(parts).astype("<f4", copy=False)
    layout = {
        "format_name": WEIGHT_FORMAT,
        "format_version": 1,
        "dtype": "float32-little-endian",
        "architecture_id": ARCHITECTURE_ID,
        "width": model.width,
        "prepare_blocks": len(model.prepare_layers),
        "evaluate_blocks": len(model.evaluate_layers),
        "fourier_bands": model.fourier_bands,
        "prepare_feature_count": 7,
        "direction_feature_count": model.evaluate_input.in_features - model.width,
        "condition_count": int(condition.numel()),
        "state_float_count": model.width,
        "total_floats": int(payload.size),
        "tensors": tensors,
    }
    return payload, layout


def export_film_m1_bundle(
    data_path: Path | str,
    checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    state_id: str = DEFAULT_PREVIEW_STATE_ID,
    quality_report_path: Path | str | None = None,
) -> MethodBundleManifest:
    """把 P1 M1-M 的一个真实语料状态导出为可视化用 diagnostic MethodBundle。"""

    data = Path(data_path)
    checkpoint_file = Path(checkpoint_path)
    output = Path(output_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("MethodBundle output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
    if checkpoint.get("pipeline") != "film-evaluator-m-v1":
        raise ValueError("Film M1 exporter only accepts the P1 M1-M checkpoint")
    config = TrainingConfig.from_dict(checkpoint["training_config"])
    expected_model = {
        "state_count": 30,
        "width": 256,
        "latent_dim": 64,
        "prepare_blocks": 3,
        "evaluate_blocks": 6,
        "fourier_bands": 5,
    }
    if config.model != expected_model:
        raise ValueError("Film M1 exporter requires the frozen M1-M architecture")

    pipeline = create_pipeline(config.pipeline)
    store = pipeline.open_store(str(data))
    try:
        if checkpoint.get("data_id") != store.data_id:
            raise ValueError("checkpoint and corpus identities disagree")
        state_ids = list(map(str, store.state_strings("state_id").tolist()))
        if state_id not in state_ids:
            raise ValueError(f"preview state is not present in the checkpoint corpus: {state_id}")
        state_index = state_ids.index(state_id)
        fitted_state = checkpoint.get("fitted_training_state")
        if not isinstance(fitted_state, dict):
            raise ValueError("checkpoint fitted training state is missing")
        pipeline.load_training_state(fitted_state)
        model = pipeline.create_model(config.model)
        if not isinstance(model, ConditionedSharedEvaluator):
            raise ValueError("checkpoint did not create the Film M1 evaluator")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        material_payload = store.state_payload(state_index)
        material = MaterialProgram.from_json(material_payload.decode("utf-8"))
        material_ir = pack_layer_stack(canonicalize_layer_stack(material))
        material_ir_sha256 = hashlib.sha256(material_ir).hexdigest()

        weights, layout = _serialize_runtime_weights(model, state_index)
        weights_path = output / "weights" / "evaluator-f32.bin"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights.tofile(weights_path)
        layout_path = output / "schemas" / "weight-layout.json"
        _write_json(layout_path, layout)

        shader_source = PROJECT_ROOT / "shaders" / "ncls" / "backends" / "film_m1" / "film_m1.slang"
        if not shader_source.is_file():
            raise ValueError(f"Film M1 Slang backend is missing: {shader_source}")
        shader_path = output / "shaders" / "film_m1.slang"
        shader_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shader_source, shader_path)

        preview_path = output / "resources" / "preview-material.json"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(material_payload)

        quality_source = Path(quality_report_path) if quality_report_path else checkpoint_file.parents[1] / "quality-test.json"
        if not quality_source.is_file():
            raise ValueError(f"quality report is missing: {quality_source}")
        quality = json.loads(quality_source.read_text(encoding="utf-8"))
        if quality.get("data_id") != store.data_id or state_id not in quality.get("states", {}):
            raise ValueError("quality report does not cover the exported state and corpus")
        quality_path = output / "validation" / "quality-test.json"
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(quality_source, quality_path)

        view = _normalized((0.37, -0.21, 0.905))
        lights = tuple(
            _normalized(value)
            for value in (
                (0.0, 0.0, 1.0),
                (0.42, 0.11, 0.9),
                (-0.53, 0.24, 0.81),
                (0.18, -0.72, 0.67),
                (-0.76, -0.29, 0.58),
                (0.91, 0.12, 0.39),
                (-0.22, 0.94, 0.26),
                (0.68, -0.72, 0.14),
            )
        )
        with torch.no_grad():
            prediction_f = model(
                torch.tensor([state_index], dtype=torch.long),
                torch.tensor([view], dtype=torch.float32),
                torch.tensor([lights], dtype=torch.float32),
            )[0]
            cosine = torch.clamp(torch.tensor(lights, dtype=torch.float32)[:, 2:3], min=0.0)
            response_cos = (prediction_f * cosine).cpu().numpy()
        parity = {
            "format_name": "ncls.backend-parity-probe",
            "format_version": 1,
            "architecture_id": ARCHITECTURE_ID,
            "compiled_state_id": state_id,
            "weight_total_floats": int(weights.size),
            "view_direction_local": list(view),
            "light_directions_local": [list(value) for value in lights],
            "expected_response_cos": response_cos.tolist(),
            "tolerance": {"rtol": 8e-4, "atol": 8e-6},
            "precision": "python-float32-vs-slang-float32",
        }
        parity_path = output / "validation" / "parity.json"
        _write_json(parity_path, parity)

        relative_paths = {
            "backend_shader": "shaders/film_m1.slang",
            "weights": "weights/evaluator-f32.bin",
            "weight_layout": "schemas/weight-layout.json",
            "parity": "validation/parity.json",
            "preview_material": "resources/preview-material.json",
            "quality_metrics": "validation/quality-test.json",
        }
        content_hashes = {
            uri: bundle_sha256_file(output / uri) for uri in relative_paths.values()
        }
        condition_count = int(layout["condition_count"]) + 3
        asset_bytes = condition_count * 4
        state_bytes = int(layout["state_float_count"]) * 4
        descriptor = BackendDescriptor(
            backend_id=BACKEND_ID,
            backend_version=BACKEND_VERSION,
            supported_ir_ids=("ncls.layer-stack-ir@1",),
            capabilities=(
                BackendCapability.PREPARE
                | BackendCapability.EVALUATE
                | BackendCapability.ANISOTROPIC_FRAME
            ),
            state_storage=StateStorage.STRUCTURED,
            state_stride=state_bytes,
            state_alignment=16,
            deterministic_eval=True,
            bounded_execution=True,
            shader_entry_points={
                "prepare": "nclsFilmM1Prepare",
                "evaluate": "nclsFilmM1EvaluateF",
            },
            cost_model=BackendCostModel(
                compiled_material_bytes=asset_bytes,
                state_bytes_per_pixel=state_bytes,
                prepare_parameter_count=sum(
                    int(item["count"])
                    for name, item in layout["tensors"].items()
                    if name.startswith("prepare_") or name.startswith("prepare_layers")
                ),
                data_dependent_loops=False,
            ),
        )
        compiler = {
            "kind": "direct-neural",
            "runtime_implementation": "slang",
            "architecture_id": ARCHITECTURE_ID,
            "compile_mode": "frozen-corpus-autodecoder-state",
            "compiled_state_id": state_id,
            "compiled_material_ir_sha256": material_ir_sha256,
            "feature_contract": "p1-direction-features-v1",
            "normalization_contract": "p1-output-scale@3",
            "precision": "float32",
        }
        runtime = {
            "platform": "windows-x86_64",
            "graphics_api": "d3d12",
            "shader_model": "6.5",
            "slang_version": "2024.1.34",
            "entry_points": {"prepare": "main", "lighting": "main", "parity": "main"},
            "environment_query_budget": 1,
            "rectangle_query_budget": 1,
        }
        checkpoint_sha256 = sha256_file(checkpoint_file)
        identity = {
            "backend_id": BACKEND_ID,
            "backend_version": BACKEND_VERSION,
            "architecture_id": ARCHITECTURE_ID,
            "checkpoint_sha256": checkpoint_sha256,
            "compiled_state_id": state_id,
            "content_hashes": content_hashes,
        }
        method_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        manifest = MethodBundleManifest(
            method_id=method_id,
            display_name="P1 Film M1-M（frozen state 6324e3b2）",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_git_commit=_git_commit(),
            material_program_schema_versions=(1,),
            supported_ir_ids=("ncls.layer-stack-ir@1",),
            scattering_contract_version=1,
            backend_id=BACKEND_ID,
            backend_version=BACKEND_VERSION,
            backend_descriptor=descriptor.to_dict(),
            runtime_class="diagnostic",
            compiler=compiler,
            runtime=runtime,
            capabilities={
                "directional_evaluation": True,
                "scattering_sampling": False,
                "path_tracing_compatible": False,
                "environment_integration": "bounded-uniform-quadrature",
                "polygon_integration": "bounded-point-quadrature",
                "transmission": False,
                "nonlocal_transport": False,
            },
            cost_claims={
                "compiled_material_bytes": asset_bytes,
                "shared_weight_bytes": int(weights.nbytes - asset_bytes),
                "state_bytes_per_pixel": state_bytes,
                "prepare_blocks": 3,
                "evaluate_blocks": 6,
                "width": 256,
                "precision": "float32",
                "bounded_execution": True,
                "runtime_class_reason": "缺少与 evaluator 匹配的 sample/pdf，仅用于 deferred 外观验证",
            },
            training_provenance={
                "pipeline": checkpoint["pipeline"],
                "pipeline_sha256": checkpoint["pipeline_sha256"],
                "data_id": checkpoint["data_id"],
                "checkpoint_uri": checkpoint_file.as_posix(),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": int(checkpoint["step"]),
            },
            validation_provenance={
                "quality_report_sha256": quality["report_sha256"],
                "evaluation_role": quality["evaluation_role"],
                "state_directional_l1": quality["states"][state_id]["directional_l1"],
                "gpu_parity": "viewer-load-time-required",
            },
            files=relative_paths,
            content_hashes=content_hashes,
        )
        (output / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        return manifest
    finally:
        store.close()
