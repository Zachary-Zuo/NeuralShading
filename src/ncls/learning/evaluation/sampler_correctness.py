from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from ncls.learning.models import (
    NvidiaNeuralAppearanceLtcAdaptationModel,
    NvidiaNeuralAppearanceModel,
    UnifiedNeuralModel,
    adapt_nvidia_model_for_sampler,
)
from ncls.learning.pipelines import create_pipeline
from ncls.learning.training.checkpoint import load_checkpoint, sha256_file
from ncls.learning.training.config import TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_PATH = PROJECT_ROOT / "configs/evaluation/unified-sampler-correctness-v1.json"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_unified_sampler_protocol(path: Path | str = PROTOCOL_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema", "name", "seed", "coverage", "normalization",
        "sample_pdf", "histogram", "mc_unbiasedness",
    }
    if set(value) != expected:
        raise ValueError("unified sampler protocol fields are not frozen v1")
    if value["schema"] != {"name": "unified-sampler-correctness", "version": 1}:
        raise ValueError("unsupported unified sampler protocol")
    if value["name"] != "unified-sampler-correctness-v1" or value["seed"] != 20260824:
        raise ValueError("unified sampler protocol name/seed drifted")
    if value["coverage"] != {
        "state_count": 30,
        "views_per_state": 4,
        "include_grazing": True,
        "source_role": "validation",
        "selection": "sorted-wo-z-then-azimuth-ordinals-v1",
        "ordinals": [0, 5, 10, 15],
    }:
        raise ValueError("unified sampler coverage differs from the frozen protocol")
    if value["sample_pdf"] != {
        "samples_per_state_view": 262144, "rtol": 0.00002, "atol": 1e-7
    }:
        raise ValueError("unified sample/pdf gate differs from the frozen protocol")
    if value["normalization"] != {
        "quadrature": "gauss-legendre-cosine-64xazimuth-256",
        "absolute_tolerance": 0.005,
        "nvidia_includes_null_mass": True,
    }:
        raise ValueError("unified sampler normalization gate differs from the frozen protocol")
    if value["histogram"] != {
        "equal_solid_angle_bins": 128, "maximum_total_variation": 0.03
    }:
        raise ValueError("unified sampler histogram gate differs from the frozen protocol")
    mc = value["mc_unbiasedness"]
    if mc != {
        "dense_direction_count": 8192,
        "replicas": 64,
        "samples_per_replica": 16384,
        "maximum_standardized_error": 3.5,
        "family_bootstrap_confidence": 0.99,
    }:
        raise ValueError("unified sampler MC gate differs from the frozen protocol")
    return value


def unified_sampler_protocol_sha256(path: Path | str = PROTOCOL_PATH) -> str:
    value = load_unified_sampler_protocol(path)
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tanh_approx(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return np.where(
        magnitude <= 1.0,
        value / np.sqrt(1.0 + value * value),
        np.sign(value) / np.sqrt(1.0 + 1.0 / np.maximum(magnitude * magnitude, 1e-30)),
    )


def _ltc_pdf(transform_raw: np.ndarray, directions: np.ndarray) -> np.ndarray:
    inverse_scale = np.exp(np.clip(transform_raw[:2], -4.0, 4.0))
    shear = 2.0 * _tanh_approx(transform_raw[2:5])
    angle = np.pi * float(_tanh_approx(transform_raw[5:6])[0])
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotated_x = cosine * directions[:, 0] + sine * directions[:, 1]
    rotated_y = -sine * directions[:, 0] + cosine * directions[:, 1]
    query = np.column_stack((
        inverse_scale[0] * rotated_x + shear[0] * rotated_y + shear[1] * directions[:, 2],
        inverse_scale[1] * rotated_y + shear[2] * directions[:, 2],
        directions[:, 2],
    ))
    norm_squared = np.maximum(np.sum(query * query, axis=1), 1e-10)
    result = (
        inverse_scale[0] * inverse_scale[1] * np.maximum(query[:, 2], 0.0)
        / (np.pi * norm_squared * norm_squared)
    )
    return np.where(directions[:, 2] > 1e-6, result, 0.0)


def independent_unified_sampler_pdf(
    prepared: np.ndarray,
    wo: np.ndarray,
    wi: np.ndarray,
    sampler: str,
) -> np.ndarray:
    """独立NumPy公式oracle；不调用生产Slang的sample/pdf路径。"""

    state = np.asarray(prepared, dtype=np.float64)
    view = np.asarray(wo, dtype=np.float64)
    directions = np.asarray(wi, dtype=np.float64)
    if state.shape != (27,) or view.shape != (3,) or directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("independent sampler oracle input shapes are invalid")
    cosine_pdf = np.where(directions[:, 2] > 1e-6, directions[:, 2] / np.pi, 0.0)
    raw = state[14:]
    if sampler == "ltc-k2":
        if len(raw) != 13:
            raise ValueError("LTC oracle requires 13 raw parameters")
        learned_weight = 0.5 + 0.5 * float(_tanh_approx(raw[12:13])[0])
        learned = learned_weight * _ltc_pdf(raw[:6], directions)
        learned += (1.0 - learned_weight) * _ltc_pdf(raw[6:12], directions)
        return (1.0 / 32.0) * cosine_pdf + (31.0 / 32.0) * learned
    if sampler != "nvidia-diffuse-ggx9":
        raise ValueError("unsupported independent unified sampler oracle")
    raw = raw[:9]
    alpha = 1e-4 + 0.5 * (1.0 + _tanh_approx(raw[:2]))
    rho = float(np.clip(_tanh_approx(raw[2:3])[0], -0.999999, 0.999999))
    specular_slope = raw[3:5] * np.sqrt(1.0 + raw[3:5] * raw[3:5])
    diffuse_slope = raw[5:7] * np.sqrt(1.0 + raw[5:7] * raw[5:7])
    logits = raw[7:9] - np.max(raw[7:9])
    learned_weights = np.exp(logits)
    learned_weights /= np.sum(learned_weights)

    tilted_normal = np.asarray(
        [-diffuse_slope[0], -diffuse_slope[1], 1.0], dtype=np.float64
    )
    tilted_normal /= np.sqrt(np.sum(tilted_normal * tilted_normal))
    tilted = np.maximum(np.sum(directions * tilted_normal[None, :], axis=1), 0.0) / np.pi
    tilted = np.where(directions[:, 2] > 1e-6, tilted, 0.0)

    half_vector = directions + view[None, :]
    half_length = np.sqrt(np.sum(half_vector * half_vector, axis=1))
    valid = (
        (directions[:, 2] > 1e-6)
        & (view[2] > 1e-6)
        & (half_length > 1e-10)
    )
    half_vector = half_vector / np.maximum(half_length[:, None], 1e-30)
    half_vector *= np.where(half_vector[:, 2:3] >= 0.0, 1.0, -1.0)
    wo_dot_half = np.sum(half_vector * view[None, :], axis=1)
    valid &= (np.abs(half_vector[:, 2]) > 1e-4) & (np.abs(wo_dot_half) > 1e-8)
    slope_x = -half_vector[:, 0] / np.where(valid, half_vector[:, 2], 1.0) - specular_slope[0]
    slope_y = -half_vector[:, 1] / np.where(valid, half_vector[:, 2], 1.0) - specular_slope[1]
    sqrt_one_minus_rho = np.sqrt(max(1.0 - rho * rho, 1e-12))
    normalization = 1.0 / (alpha[0] * alpha[1] * sqrt_one_minus_rho)
    standard_x = slope_x / alpha[0]
    standard_y = (alpha[0] * slope_y - rho * alpha[1] * slope_x) * normalization
    p22 = normalization / (np.pi * (1.0 + standard_x * standard_x + standard_y * standard_y) ** 2)
    half_pdf = p22 / np.where(valid, half_vector[:, 2] ** 3, 1.0)
    specular = half_pdf / np.where(valid, 4.0 * np.abs(wo_dot_half), 1.0)
    specular = np.where(valid & np.isfinite(specular) & (specular >= 0.0), specular, 0.0)
    learned = learned_weights[0] * specular + learned_weights[1] * tilted
    return (1.0 / 32.0) * cosine_pdf + (31.0 / 32.0) * learned


def _legendre_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.cos(
        np.pi * (np.arange(1, order + 1, dtype=np.float64) - 0.25) / (order + 0.5)
    )
    derivative = np.zeros_like(nodes)
    for _ in range(16):
        previous = np.ones_like(nodes)
        current = nodes.copy()
        for degree in range(2, order + 1):
            following = (
                (2.0 * degree - 1.0) * nodes * current - (degree - 1.0) * previous
            ) / degree
            previous, current = current, following
        derivative = order * (nodes * current - previous) / (nodes * nodes - 1.0)
        update = current / derivative
        nodes -= update
        if float(np.max(np.abs(update))) < 2e-15:
            break
    weights = 2.0 / ((1.0 - nodes * nodes) * derivative * derivative)
    indices = np.argsort(nodes)
    return nodes[indices], weights[indices]


def _quadrature() -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _legendre_nodes_weights(64)
    z = 0.5 * (nodes + 1.0)
    z_weights = 0.5 * weights
    phi = (np.arange(256, dtype=np.float64) + 0.5) * (2.0 * np.pi / 256.0)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    radial = np.sqrt(np.maximum(1.0 - zz * zz, 0.0))
    directions = np.stack(
        (radial * np.cos(pp), radial * np.sin(pp), zz), axis=-1
    ).reshape(-1, 3)
    solid_angle = np.broadcast_to(
        z_weights[:, None] * (2.0 * np.pi / 256.0), (64, 256)
    ).reshape(-1)
    return directions.astype(np.float32), solid_angle


def _dense_uniform_hemisphere(count: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(count, dtype=np.float64)
    z = (index + 0.5) / count
    phi = np.mod(index * (np.pi * (3.0 - np.sqrt(5.0))), 2.0 * np.pi)
    radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    directions = np.column_stack((radial * np.cos(phi), radial * np.sin(phi), z))
    return directions.astype(np.float32), np.full(count, 2.0 * np.pi / count)


def _solid_angle_bin_mass(directions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    z = np.clip(directions[:, 2], 0.0, np.nextafter(1.0, 0.0))
    phi = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi)
    z_index = np.minimum((z * 8).astype(np.int64), 7)
    phi_index = np.minimum((phi * (16.0 / (2.0 * np.pi))).astype(np.int64), 15)
    return np.bincount(
        z_index * 16 + phi_index, weights=masses, minlength=128
    )


def _write_json_atomic(path: Path | str, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _load_sampler_model(
    data_path: Path | str,
    evaluator_checkpoint_path: Path | str,
    sampler_checkpoint_path: Path | str,
    device: torch.device,
):
    evaluator_path = Path(evaluator_checkpoint_path).resolve()
    sampler_path = Path(sampler_checkpoint_path).resolve()
    evaluator = load_checkpoint(evaluator_path, map_location=device)
    sampler_checkpoint = load_checkpoint(sampler_path, map_location=device)
    if sampler_checkpoint.get("checkpoint_role") != "unified-sampler-head":
        raise ValueError("sampler correctness requires a sampler-head checkpoint")
    if sampler_checkpoint.get("source_evaluator_checkpoint_sha256") != sha256_file(evaluator_path):
        raise ValueError("sampler correctness evaluator/checkpoint identity mismatch")
    config = TrainingConfig.from_dict(evaluator["training_config"])
    pipeline = create_pipeline(str(evaluator["pipeline"]))
    store = pipeline.open_store(str(data_path))
    if evaluator.get("data_id") != store.data_id or sampler_checkpoint.get("data_id") != store.data_id:
        store.close()
        raise ValueError("sampler correctness data identity mismatch")
    pipeline.load_training_state(evaluator["fitted_training_state"])
    source_model = pipeline.create_model(config.model).to(device)
    source_model.load_state_dict(evaluator["model_state"])
    model = (
        adapt_nvidia_model_for_sampler(source_model, str(sampler_checkpoint["sampler"]))
        if isinstance(source_model, NvidiaNeuralAppearanceModel)
        else source_model
    ).to(device)
    model.load_state_dict(sampler_checkpoint["model_state"])
    if not isinstance(
        model,
        (
            NvidiaNeuralAppearanceModel,
            NvidiaNeuralAppearanceLtcAdaptationModel,
            UnifiedNeuralModel,
        ),
    ):
        store.close()
        raise TypeError("sampler correctness requires a registered neural evaluator")
    model.eval()
    return evaluator, sampler_checkpoint, config, pipeline, store, model


def _sampler_audit_payload(
    model: torch.nn.Module,
    state_indices: torch.Tensor,
    views: torch.Tensor,
    sampler: str,
) -> torch.Tensor:
    """把方法私有 sampler raw 放进审计专用27-float容器，不改变运行时ABI。"""

    if isinstance(model, UnifiedNeuralModel):
        prepared, _ = model._prepared_with_head(
            state_indices,
            views,
            sampler,
            detach_shared=True,
        )
        return prepared
    if isinstance(model, NvidiaNeuralAppearanceModel):
        if sampler != "nvidia-diffuse-ggx9":
            raise ValueError("exact NVIDIA model requires its native sampler")
        raw = model.sampler_raw(state_indices, views, detach_latent=True)
    elif isinstance(model, NvidiaNeuralAppearanceLtcAdaptationModel):
        if sampler != "ltc-k2":
            raise ValueError("NVIDIA adaptation requires the LTC-K2 sampler")
        raw = model.sampler_raw(state_indices, views)
    else:
        raise TypeError("sampler audit payload requires a registered neural evaluator")
    payload = torch.zeros((raw.shape[0], 27), dtype=raw.dtype, device=raw.device)
    payload[:, 14 : 14 + raw.shape[1]] = raw
    return payload


def _select_cases(
    store: Any,
    pipeline: Any,
    model: torch.nn.Module,
    sampler: str,
    device: torch.device,
):
    indices = store.partition_indices(pipeline.descriptor.partition_policy_id, "validation")
    state_parts: list[np.ndarray] = []
    wo_parts: list[np.ndarray] = []
    for raw in store.iter_batches(indices, 64, fields=("state_index", "wo")):
        state_parts.append(np.asarray(raw["state_index"], dtype=np.int64))
        wo_parts.append(np.asarray(raw["wo"], dtype=np.float32))
    state_indices = np.concatenate(state_parts)
    views = np.concatenate(wo_parts)
    selected_rows = []
    for state_index in range(store.state_count):
        available = np.flatnonzero(state_indices == state_index)
        azimuth = np.mod(np.arctan2(views[available, 1], views[available, 0]), 2.0 * np.pi)
        order = np.lexsort((azimuth, views[available, 2]))
        if len(order) != 16:
            raise ValueError("sampler correctness requires 16 validation views per state")
        selected_rows.extend(available[order[[0, 5, 10, 15]]].tolist())
    selected = np.asarray(selected_rows, dtype=np.int64)
    case_states = state_indices[selected]
    case_views = views[selected]
    with torch.no_grad():
        prepared = _sampler_audit_payload(
            model,
            torch.as_tensor(case_states, dtype=torch.long, device=device),
            torch.as_tensor(case_views, dtype=torch.float32, device=device),
            sampler,
        )
    return case_states, case_views, prepared.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def _evaluate_case(
    model: torch.nn.Module,
    pipeline: Any,
    store: Any,
    state_index: int,
    wo: np.ndarray,
    directions: np.ndarray,
    device: torch.device,
    *,
    chunk_size: int = 65_536,
) -> np.ndarray:
    parts = []
    state = torch.as_tensor([state_index], dtype=torch.long, device=device)
    view = torch.as_tensor(wo[None, :], dtype=torch.float32, device=device)
    for start in range(0, len(directions), chunk_size):
        wi = torch.as_tensor(
            directions[None, start : start + chunk_size],
            dtype=torch.float32,
            device=device,
        )
        prediction = pipeline.predict_f(
            model, {"state_index": state, "wo": view, "wi": wi}, store, device
        )
        parts.append(prediction[0].detach().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)


def _family_bootstrap_rgb(
    replicas: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    iterations: int = 10_000,
) -> dict[str, Any]:
    values = np.asarray(replicas, dtype=np.float64)
    center = np.mean(values, axis=0)
    standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    safe_se = np.maximum(standard_error, 1e-15)
    rng = np.random.default_rng(seed)
    maxima = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        selected = rng.integers(0, len(values), size=(count, len(values)))
        bootstrap_mean = np.mean(values[selected], axis=1)
        maxima[start : start + count] = np.max(
            np.abs(bootstrap_mean - center[None, :]) / safe_se[None, :], axis=1
        )
    critical = float(np.quantile(maxima, 0.99))
    lower = center - critical * safe_se
    upper = center + critical * safe_se
    return {
        "confidence": 0.99,
        "iterations": iterations,
        "critical_max_t": critical,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "covers_reference": bool(np.all((reference >= lower) & (reference <= upper))),
    }


def _mc_case_metrics(
    model: UnifiedNeuralModel,
    pipeline: Any,
    store: Any,
    state_index: int,
    wo: np.ndarray,
    sampled: np.ndarray,
    metadata: np.ndarray,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    dense_directions, dense_weights = _dense_uniform_hemisphere(8192)
    dense_f = _evaluate_case(
        model, pipeline, store, state_index, wo, dense_directions, device
    )
    reference = np.sum(
        dense_f * dense_directions[:, 2:3] * dense_weights[:, None], axis=0
    )
    continuous = (metadata[:, 1] == 1.0) & (metadata[:, 2] == 0.0)
    sampled_f = _evaluate_case(
        model, pipeline, store, state_index, wo, sampled[:, :3], device
    )
    estimator = np.zeros_like(sampled_f)
    pdf = sampled[:, 3]
    estimator[continuous] = (
        sampled_f[continuous]
        * np.maximum(sampled[continuous, 2:3], 0.0)
        / np.maximum(pdf[continuous, None], 1e-30)
    )
    replicas = estimator.reshape(64, 16_384, 3).mean(axis=1)
    mean = np.mean(replicas, axis=0)
    se = np.std(replicas, axis=0, ddof=1) / np.sqrt(64.0)
    standardized = np.abs(mean - reference) / np.maximum(se, 1e-15)

    cosine_random = np.random.default_rng(seed ^ 0xC051).random(
        (64 * 16_384, 2), dtype=np.float32
    )
    radius = np.sqrt(cosine_random[:, 0])
    phi = 2.0 * np.pi * cosine_random[:, 1]
    cosine_directions = np.column_stack((
        radius * np.cos(phi),
        radius * np.sin(phi),
        np.sqrt(np.maximum(1.0 - cosine_random[:, 0], 0.0)),
    )).astype(np.float32)
    cosine_f = _evaluate_case(
        model, pipeline, store, state_index, wo, cosine_directions, device
    )
    cosine_replicas = (np.pi * cosine_f).reshape(64, 16_384, 3).mean(axis=1)
    luminance = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    proposal_scalar = np.sum(replicas * luminance[None, :], axis=1)
    cosine_scalar = np.sum(cosine_replicas * luminance[None, :], axis=1)
    proposal_variance = float(np.var(proposal_scalar, ddof=1))
    cosine_variance = float(np.var(cosine_scalar, ddof=1))
    family = _family_bootstrap_rgb(replicas, reference, seed=seed ^ 0xB007)
    return {
        "reference_rgb": reference.tolist(),
        "replica_mean_rgb": mean.tolist(),
        "standard_error_rgb": se.tolist(),
        "standardized_error_rgb": standardized.tolist(),
        "maximum_standardized_error": float(np.max(standardized)),
        "family_bootstrap": family,
        "luminance_replica_variance": proposal_variance,
        "cosine_luminance_replica_variance": cosine_variance,
        "cosine_relative_variance": proposal_variance / max(cosine_variance, 1e-30),
    }


def run_unified_sampler_correctness(
    data_path: Path | str,
    evaluator_checkpoint_path: Path | str,
    sampler_checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    device_name: str = "cuda",
    progress: Any | None = None,
) -> dict[str, Any]:
    """按冻结120-case协议审计learned sampler；Falcor在隔离子进程中执行。"""
    from ncls.learning.slang import (
        nvidia_matched_ltc_implementation_sha256,
        nvidia_neural_appearance_implementation_sha256,
        nvidia_neural_appearance_layout_sha256,
        unified_layout_sha256,
        unified_slang_implementation_sha256,
    )

    protocol = load_unified_sampler_protocol()
    torch_device = torch.device(device_name)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("formal unified sampler correctness requires CUDA")
    (
        evaluator_checkpoint,
        sampler_checkpoint,
        _config,
        pipeline,
        store,
        model,
    ) = _load_sampler_model(
        data_path,
        evaluator_checkpoint_path,
        sampler_checkpoint_path,
        torch_device,
    )
    try:
        sampler = str(sampler_checkpoint["sampler"])
        nvidia_method = isinstance(
            model,
            (NvidiaNeuralAppearanceModel, NvidiaNeuralAppearanceLtcAdaptationModel),
        )
        method = "nvidia" if nvidia_method else "unified"
        slang_implementation_sha256 = (
            nvidia_neural_appearance_implementation_sha256()
            if nvidia_method
            else unified_slang_implementation_sha256()
        )
        layout_sha256 = (
            nvidia_neural_appearance_layout_sha256()
            if nvidia_method
            else unified_layout_sha256()
        )
        case_states, case_views, prepared = _select_cases(
            store, pipeline, model, sampler, torch_device
        )
        if len(case_states) != 120:
            raise ValueError("sampler correctness did not select 30x4 cases")
        quadrature_directions, quadrature_weights = _quadrature()
        sample_count = int(protocol["sample_pdf"]["samples_per_state_view"])
        mc = protocol["mc_unbiasedness"]
        state_ids = list(map(str, store.state_strings("state_id").tolist()))
        report: dict[str, Any] = {
            "format_name": "unified-sampler-correctness-report",
            "format_version": 1,
            "status": "running",
            "passed": False,
            "protocol": protocol,
            "protocol_sha256": unified_sampler_protocol_sha256(),
            "data_id": store.data_id,
            "pipeline": evaluator_checkpoint["pipeline"],
            "sampler": sampler,
            "evaluator_checkpoint": {
                "uri": str(Path(evaluator_checkpoint_path)),
                "sha256": sha256_file(evaluator_checkpoint_path),
                "step": int(evaluator_checkpoint["step"]),
                "implementation_identity": evaluator_checkpoint.get(
                    "implementation_identity"
                ),
            },
            "sampler_checkpoint": {
                "uri": str(Path(sampler_checkpoint_path)),
                "sha256": sha256_file(sampler_checkpoint_path),
                "step": int(sampler_checkpoint["step"]),
                "implementation_identity": sampler_checkpoint.get(
                    "implementation_identity"
                ),
            },
            "method_implementation": method,
            "slang_implementation_sha256": slang_implementation_sha256,
            "layout_sha256": layout_sha256,
            "cases": [],
        }
        if isinstance(model, NvidiaNeuralAppearanceLtcAdaptationModel):
            report["sampler_adaptation"] = "nvidia-frozen-evaluator-ltc-k2-v1"
            report["sampler_adaptation_implementation_sha256"] = (
                nvidia_matched_ltc_implementation_sha256()
            )
        _write_json_atomic(output_path, report)
        output_target = Path(output_path).resolve()
        work_root = output_target.with_name(output_target.stem + "-work")
        if work_root.exists():
            raise ValueError("sampler correctness work directory must be new")
        work_root.mkdir(parents=True)
        cases_path = work_root / "cases.npz"
        falcor_output = work_root / "falcor"
        np.savez(
            cases_path,
            state_indices=case_states,
            views=case_views,
            prepared=prepared,
        )
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "scripts/run_falcor_python.ps1"),
            "-m",
            "ncls.learning.evaluation.sampler_falcor_worker",
            "--cases",
            str(cases_path),
            "--output",
            str(falcor_output),
            "--method",
            method,
            "--sampler",
            sampler,
            "--seed",
            str(protocol["seed"]),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        for case_index, (state_index, wo, state) in enumerate(
            zip(case_states, case_views, prepared, strict=True)
        ):
            if progress is not None:
                progress(
                    f"sampler={sampler} case={case_index + 1}/120 "
                    f"state={state_ids[int(state_index)][:12]}"
                )
            case_path = falcor_output / f"case-{case_index:03d}.npz"
            falcor_case = np.load(case_path, allow_pickle=False)
            pdf = np.asarray(falcor_case["queried_pdf"], dtype=np.float64)
            sampled = np.asarray(falcor_case["sampled"], dtype=np.float32)
            metadata = np.asarray(falcor_case["metadata"], dtype=np.float32)
            oracle = independent_unified_sampler_pdf(
                state, wo, quadrature_directions, sampler
            )
            oracle_close = np.isclose(
                pdf,
                oracle,
                rtol=float(protocol["sample_pdf"]["rtol"]),
                atol=float(protocol["sample_pdf"]["atol"]),
            )
            continuous_mass = float(np.sum(pdf * quadrature_weights))
            subset = slice(0, sample_count)
            valid = metadata[subset, 1] == 1.0
            null = metadata[subset, 2] != 0.0
            continuous = valid & ~null
            pdf_close = np.isclose(
                sampled[subset, 3][continuous],
                metadata[subset, 0][continuous],
                rtol=float(protocol["sample_pdf"]["rtol"]),
                atol=float(protocol["sample_pdf"]["atol"]),
            )
            null_frequency = float(np.mean(null))
            normalization_error = abs(continuous_mass + null_frequency - 1.0)
            expected_bins = _solid_angle_bin_mass(
                quadrature_directions, pdf * quadrature_weights
            )
            empirical_bins = _solid_angle_bin_mass(
                sampled[subset, :3][continuous],
                np.full(np.count_nonzero(continuous), 1.0 / sample_count),
            )
            total_variation = 0.5 * (
                float(np.sum(np.abs(empirical_bins - expected_bins)))
                + abs(null_frequency - (1.0 - continuous_mass))
            )
            mc_metrics = _mc_case_metrics(
                model,
                pipeline,
                store,
                int(state_index),
                wo,
                sampled,
                metadata,
                torch_device,
                seed=int(protocol["seed"]) + case_index * 7919,
            )
            case = {
                "case_index": case_index,
                "state_id": state_ids[int(state_index)],
                "view_ordinal": protocol["coverage"]["ordinals"][case_index % 4],
                "wo": list(map(float, wo)),
                "continuous_pdf_integral": continuous_mass,
                "null_frequency": null_frequency,
                "normalization_error": normalization_error,
                "invalid_sample_count": int(np.count_nonzero(~valid)),
                "sample_pdf_maximum_absolute_error": float(
                    np.max(np.abs(sampled[subset, 3][continuous] - metadata[subset, 0][continuous]))
                    if np.any(continuous)
                    else math.inf
                ),
                "sample_pdf_passed": bool(np.all(pdf_close) and np.any(continuous)),
                "independent_oracle_maximum_absolute_error": float(
                    np.max(np.abs(pdf - oracle))
                ),
                "independent_oracle_passed": bool(np.all(oracle_close)),
                "histogram_total_variation": total_variation,
                "mc_unbiasedness": mc_metrics,
            }
            case["passed"] = bool(
                case["invalid_sample_count"] == 0
                and normalization_error <= float(protocol["normalization"]["absolute_tolerance"])
                and case["sample_pdf_passed"]
                and case["independent_oracle_passed"]
                and total_variation <= float(protocol["histogram"]["maximum_total_variation"])
                and mc_metrics["maximum_standardized_error"]
                <= float(mc["maximum_standardized_error"])
                and mc_metrics["family_bootstrap"]["covers_reference"]
            )
            report["cases"].append(case)
            _write_json_atomic(output_path, report)
            del falcor_case, sampled, metadata
            case_path.unlink()

        cases = report["cases"]
        report["summary"] = {
            "case_count": len(cases),
            "maximum_normalization_error": max(
                item["normalization_error"] for item in cases
            ),
            "maximum_histogram_total_variation": max(
                item["histogram_total_variation"] for item in cases
            ),
            "maximum_mc_standardized_error": max(
                item["mc_unbiasedness"]["maximum_standardized_error"] for item in cases
            ),
            "maximum_cosine_relative_variance": max(
                item["mc_unbiasedness"]["cosine_relative_variance"] for item in cases
            ),
            "median_cosine_relative_variance": float(np.median([
                item["mc_unbiasedness"]["cosine_relative_variance"] for item in cases
            ])),
            "failed_case_indices": [
                item["case_index"] for item in cases if not item["passed"]
            ],
        }
        report["status"] = "complete"
        report["passed"] = all(item["passed"] for item in cases)
        report["report_sha256"] = _sha256_json(report)
        _write_json_atomic(output_path, report)
        cases_path.unlink()
        falcor_output.rmdir()
        work_root.rmdir()
        return report
    finally:
        store.close()
