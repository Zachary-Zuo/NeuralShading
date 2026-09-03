from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_json, write_json_atomic
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.training import TrainingConfig
from ncls.source_materials.mdl_metal import MdlMetalRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_SMOKE_PATH = PROJECT_ROOT / "configs/learning/metal-fused-full-windows-smoke.json"
LINUX_SMOKE_PATH = PROJECT_ROOT / "configs/learning/metal-fused-full-linux-smoke.json"
LINUX_LONG_PATH = PROJECT_ROOT / "configs/learning/metal-fused-full-linux-long.json"
REGISTRY_PATH = PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"

_LONG_PHASE_STEPS = {
    "joint-coarse-to-fine": 105_000,
    "qat-refine": 15_000,
}
_LINUX_ROUTE_GEOMETRY = {
    "asset": (12, 1),
    "evaluator": (64, 1),
    "sampler": (64, 1),
}

_END_TO_END_GROUPS = [
    "codec_role_stems",
    "codec_encoder",
    "codec_decoder",
    "codec_semantic_heads",
    "asset_adapter",
    "quantization",
    "typed_compiler",
    "optimized_state_teacher",
    "prepared_model",
    "angular_bank",
    "analytic_core",
    "hybrid_evaluator",
    "proposal_sampler",
]
_QAT_GROUPS = [
    name for name in _END_TO_END_GROUPS if name != "optimized_state_teacher"
]
_END_TO_END_LOSSES = [
    "codec-full",
    "response-robust",
    "linear-energy",
    "peak-support",
    "reciprocity",
    "analytic-core-preservation",
    "teacher-response",
    "compiler-functional-distillation",
    "proposal-forward-kl",
    "proposal-density-fit",
    "proposal-mode-coverage",
    "proposal-weight-tail",
    "sample-pdf-identity",
]


def build_windows_smoke(seed: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize either the historical or current smoke into the shared recipe."""

    result = deepcopy(dict(seed))
    routes: dict[str, dict[str, Any]] = {}
    recipes: dict[str, Any] = {}
    for phase in result["phases"]:
        recipes.update(phase["recipes"])
        for route in phase["routes"]:
            routes.setdefault(str(route["name"]), deepcopy(route))
    if set(routes) != {"asset", "evaluator", "sampler"}:
        raise ValueError("Metal smoke seed does not contain all three typed routes")
    recipes.pop("runtime_quantization", None)
    recipes["proposal_weight"] = {
        "schema": "linear-nonzero-ramp@1",
        "start": 0.05,
        "end": 1.0,
        "ramp_steps": 5000,
    }
    optimizer = {
        "kind": "adam",
        "betas": [0.9, 0.99],
        "epsilon": 1e-8,
        "weight_decay": 0.000001,
    }
    joint = {
        "name": "joint-coarse-to-fine",
        "steps": 8,
        "routes": [routes["asset"], routes["evaluator"], routes["sampler"]],
        "parameter_groups": list(_END_TO_END_GROUPS),
        "loss_terms": list(_END_TO_END_LOSSES),
        "recipes": recipes,
        "optimizer": optimizer,
        "optimizer_state_policy": "reset",
        "schedule": {
            "kind": "cosine",
            "start": 0.0003,
            "end": 0.00005,
            "total_steps": 16,
            "offset": 0,
        },
        "precision": {"autocast": "bfloat16", "gradient_scaler": False},
        "checkpoint_boundary": True,
        "transition": None,
        "log_interval": 1,
        "gradient_audit_interval": 4,
        "prefetch_depth": 1,
    }
    qat_recipes = {
        **recipes,
        "proposal_weight": {
            "schema": "linear-nonzero-ramp@1",
            "start": 1.0,
            "end": 1.0,
            "ramp_steps": 1,
        },
        "runtime_quantization": "fp16-runtime-ste-int8-grid-qat-sensitive-fp32@1",
    }
    qat = {
        **deepcopy(joint),
        "name": "qat-refine",
        "parameter_groups": list(_QAT_GROUPS),
        "loss_terms": [*_END_TO_END_LOSSES, "runtime-fp16-qat"],
        "recipes": qat_recipes,
        "optimizer_state_policy": "carry-overlap",
        "schedule": {
            "kind": "cosine",
            "start": 0.0001,
            "end": 0.00002,
            "total_steps": 16,
            "offset": 8,
        },
        "precision": {"autocast": "fp32", "gradient_scaler": False},
    }
    result["phases"] = [joint, qat]
    result["online_query"]["group_schedule"] = {
        "recipe": "group-block-balanced@1",
        "weight": "record-count",
        "block_steps": 64,
        "validation_offset_blocks": 104729,
    }
    return result


def _full_cohort_materials(
    registry: MdlMetalRegistry, module_root: str
) -> list[dict[str, Any]]:
    result = []
    for record in registry.exports:
        exact = record.exact_locator
        result.append(
            {
                "locator": {
                    "kind": exact["kind"],
                    "module_root": module_root,
                    "module": exact["module"],
                    "export": exact["export"],
                    "pack_id": exact["pack_id"],
                    "pack_version": exact["pack_version"],
                }
            }
        )
    if len(result) != 692:
        raise RuntimeError("Metal Linux training config requires all 692 opaque exports")
    return result


def build_linux_configs(
    windows_smoke: Mapping[str, Any], registry: MdlMetalRegistry
) -> tuple[dict[str, Any], dict[str, Any]]:
    smoke = deepcopy(dict(windows_smoke))
    module_root = str(smoke["source"]["materials"][0]["locator"]["module_root"])
    smoke["recipe_id"] = "metal-fused-full-cohort-linux-smoke-16step@1"
    smoke["source"]["materials"] = _full_cohort_materials(registry, module_root)
    typed_recipe = smoke["online_query"]["typed_state_recipe"]
    typed_recipe["recipe_id"] = "metal-fused-full-cohort-states@1"
    typed_recipe["states_per_export"] = 4
    for phase in smoke["phases"]:
        for route in phase["routes"]:
            batch_size, direction_count = _LINUX_ROUTE_GEOMETRY[route["name"]]
            route["batch_size"] = batch_size
            route["direction_count"] = direction_count
            if route["name"] == "asset":
                route["options"]["asset_indices"] = list(range(52))

    long_run = deepcopy(smoke)
    long_run["run_class"] = "formal"
    long_run["recipe_id"] = "metal-fused-full-cohort-linux-long-120k@1"
    total_steps = sum(_LONG_PHASE_STEPS.values())
    offset = 0
    for phase in long_run["phases"]:
        steps = _LONG_PHASE_STEPS[phase["name"]]
        phase["steps"] = steps
        phase["schedule"]["total_steps"] = total_steps
        phase["schedule"]["offset"] = offset
        phase["log_interval"] = 10
        phase["gradient_audit_interval"] = 100
        offset += steps
    long_run["validation"] = {"interval": 5000, "batches": 4}
    return smoke, long_run


def semantic_training_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash every Linux smoke/long field except budget and cadence."""

    value = deepcopy(dict(config))
    value.pop("run_class")
    value.pop("recipe_id")
    value["validation"] = {"interval": "<cadence>", "batches": "<cadence>"}
    for phase in value["phases"]:
        phase["steps"] = "<budget>"
        phase["schedule"]["total_steps"] = "<budget>"
        phase["schedule"]["offset"] = "<budget>"
        phase["log_interval"] = "<cadence>"
        phase["gradient_audit_interval"] = "<cadence>"
    return sha256_json(value)


def validate_linux_config_pair(
    smoke_payload: Mapping[str, Any], long_payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    smoke = TrainingConfig.from_dict(smoke_payload)
    long_run = TrainingConfig.from_dict(long_payload)
    METHOD_DEFINITION.validate_training_config(smoke.to_dict())
    METHOD_DEFINITION.validate_training_config(long_run.to_dict())
    smoke_fingerprint = semantic_training_fingerprint(smoke.to_dict())
    long_fingerprint = semantic_training_fingerprint(long_run.to_dict())
    if smoke_fingerprint != long_fingerprint:
        raise ValueError("Metal Linux smoke/long semantic configs drifted")
    if len(smoke.source["materials"]) != 692 or len(long_run.source["materials"]) != 692:
        raise ValueError("Metal Linux smoke/long configs must both cover the full cohort")
    return {
        "schema": "ncls.metal-linux-training-config-pair@1",
        "semantic_fingerprint": smoke_fingerprint,
        "smoke_config_sha256": smoke.sha256,
        "long_config_sha256": long_run.sha256,
        "smoke_steps": smoke.total_steps,
        "long_steps": long_run.total_steps,
        "source_count": len(smoke.source["materials"]),
        "phase_names": [phase.name for phase in smoke.phases],
        "distributed": False,
        "visible_gpu_count": 1,
    }


def _load_payload(path: Path) -> Mapping[str, Any]:
    return TrainingConfig.load(path).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="机械生成并验证Metal full-cohort Linux单GPU训练配置"
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    registry = MdlMetalRegistry.load(REGISTRY_PATH)
    windows_seed = _load_payload(WINDOWS_SMOKE_PATH)
    windows = build_windows_smoke(windows_seed)
    expected_smoke, expected_long = build_linux_configs(windows, registry)
    if arguments.write:
        write_json_atomic(WINDOWS_SMOKE_PATH, windows)
        write_json_atomic(LINUX_SMOKE_PATH, expected_smoke)
        write_json_atomic(LINUX_LONG_PATH, expected_long)
    actual_windows = _load_payload(WINDOWS_SMOKE_PATH)
    if actual_windows != windows:
        raise ValueError("checked-in Metal Windows smoke is not the canonical generated form")
    actual_smoke = _load_payload(LINUX_SMOKE_PATH)
    actual_long = _load_payload(LINUX_LONG_PATH)
    if actual_smoke != expected_smoke or actual_long != expected_long:
        raise ValueError("checked-in Metal Linux configs are not the canonical generated form")
    report = validate_linux_config_pair(actual_smoke, actual_long)
    if arguments.report is not None:
        write_json_atomic(arguments.report, report)
    print(report["semantic_fingerprint"])
    print(
        f"sources={report['source_count']} smoke_steps={report['smoke_steps']} "
        f"long_steps={report['long_steps']} single_gpu=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
