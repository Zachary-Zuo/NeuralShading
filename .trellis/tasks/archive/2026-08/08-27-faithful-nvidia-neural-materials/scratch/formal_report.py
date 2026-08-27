from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ncls.core.identity import sha256_file
from ncls.learning.training import TrainingConfig


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_metrics(path: Path, *, recorded_step: int) -> dict[str, Any]:
    first_training: dict[str, Any] | None = None
    final_training: dict[str, Any] | None = None
    validations: list[dict[str, Any]] = []
    observed_record_count = 0
    included_record_count = 0
    excluded_tail_record_count = 0
    observed_max_step = 0
    peak_memory_bytes = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            observed_record_count += 1
            step = int(record.get("step", -1))
            if step < 0:
                raise ValueError(f"{path}:{line_number} has no valid step")
            observed_max_step = max(observed_max_step, step)
            if step > recorded_step:
                excluded_tail_record_count += 1
                continue
            included_record_count += 1
            kind = record.get("record_kind")
            if kind == "training":
                if first_training is None:
                    first_training = record
                final_training = record
                peak_memory_bytes = max(
                    peak_memory_bytes, int(record.get("peak_memory_bytes", 0))
                )
            elif kind == "validation":
                validations.append(record)
    if first_training is None or final_training is None:
        raise ValueError(f"{path} has no training records")
    return {
        "observed_record_count": observed_record_count,
        "included_record_count": included_record_count,
        "excluded_tail_record_count": excluded_tail_record_count,
        "observed_max_step": observed_max_step,
        "first_training": first_training,
        "final_training": final_training,
        "validations": validations,
        "peak_memory_bytes": peak_memory_bytes,
    }


def relative_change(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final) or initial == 0.0:
        raise ValueError("convergence endpoints must be finite and initial must be nonzero")
    return (final - initial) / abs(initial)


def package_costs(package_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    files = [path for path in package_root.rglob("*") if path.is_file()]
    runtime_files = [path for path in files if path.relative_to(package_root).parts[0] == "runtime"]
    material_files = [path for path in files if path.relative_to(package_root).parts[0] == "materials"]
    descriptor = manifest["program"]["blobs"]["runtime/blob/shared-weights"]
    weight_count = math.prod(int(value) for value in descriptor["shape"])
    dtype_bytes = {"float16": 2, "float32": 4}[str(descriptor["dtype"])]
    return {
        "package_total_bytes": sum(path.stat().st_size for path in files),
        "runtime_tree_bytes": sum(path.stat().st_size for path in runtime_files),
        "material_tree_bytes": sum(path.stat().st_size for path in material_files),
        "shared_weight_bytes": weight_count * dtype_bytes,
        "file_count": len(files),
    }


def validate_capture(
    capture: dict[str, Any],
    *,
    expected_modes: tuple[str, str],
    expected_package_ids: tuple[str, str],
) -> None:
    if capture.get("format_name") != "ncls.viewer-capture" or capture.get("format_version") != 4:
        raise ValueError("viewer evidence must use ncls.viewer-capture@4")
    slots = capture.get("slots")
    if not isinstance(slots, list) or len(slots) != 2:
        raise ValueError("viewer evidence must contain exactly two symmetric slots")
    for index, (slot, expected_mode, expected_package_id) in enumerate(
        zip(slots, expected_modes, expected_package_ids, strict=True)
    ):
        if slot.get("slot_index") != index or slot.get("mode") != expected_mode:
            raise ValueError(f"viewer slot {index} has the wrong index or mode")
        if slot.get("status") != "ready":
            raise ValueError(f"viewer slot {index} is not ready: {slot.get('diagnostic', '')}")
        if slot.get("package_id") != expected_package_id:
            raise ValueError(f"viewer slot {index} has the wrong package request")


parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("metrics", type=Path)
parser.add_argument("package", type=Path)
parser.add_argument("directional_evaluation", type=Path)
parser.add_argument("neural_dual_capture", type=Path)
parser.add_argument("reference_neural_capture", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--recorded-step", type=int, required=True)
args = parser.parse_args()

config = TrainingConfig.load(args.config)
if config.run_class != "formal" or config.total_steps != 300_000:
    raise ValueError("formal report only accepts the frozen 300k formal recipe")
if args.recorded_step != 200_000:
    raise ValueError("this report records the user-frozen step 200k checkpoint only")

metrics = read_metrics(args.metrics, recorded_step=args.recorded_step)
manifest = read_json(args.package / "manifest.json")
directional = read_json(args.directional_evaluation)
neural_dual = read_json(args.neural_dual_capture)
reference_neural = read_json(args.reference_neural_capture)

checkpoint_sha256 = sha256_file(args.checkpoint)
package_id = str(manifest["package_id"])
if int(metrics["final_training"]["step"]) != args.recorded_step:
    raise ValueError("metrics do not contain the explicitly recorded step")
if metrics["final_training"].get("training_config_sha256") != config.sha256:
    raise ValueError("metrics have a stale training config identity")
if manifest.get("provenance", {}).get("checkpoint_sha256") != checkpoint_sha256:
    raise ValueError("package provenance does not identify the final checkpoint")
if int(manifest.get("validation", {}).get("checkpoint_step", -1)) != args.recorded_step:
    raise ValueError("package validation does not identify the recorded step")
if directional.get("training_config_sha256") != config.sha256:
    raise ValueError("directional report has a stale training config identity")
if directional.get("checkpoint_sha256") != checkpoint_sha256:
    raise ValueError("directional report has a stale checkpoint identity")
if int(directional.get("checkpoint_step", -1)) != args.recorded_step:
    raise ValueError("directional report did not execute the recorded step")
expected_phase = "complete" if args.recorded_step == config.total_steps else "finetune"
if directional.get("checkpoint_phase") != expected_phase:
    raise ValueError("directional report has the wrong checkpoint phase")
if directional.get("package_id") != package_id:
    raise ValueError("directional report did not execute the final package")

validate_capture(
    neural_dual,
    expected_modes=("path-tracing", "deferred"),
    expected_package_ids=(package_id, package_id),
)
validate_capture(
    reference_neural,
    expected_modes=("path-tracing", "path-tracing"),
    expected_package_ids=("source-reference", package_id),
)

first = metrics["first_training"]
final = metrics["final_training"]
report = {
    "schema_name": "ncls.nvidia-functional-reproduction-record",
    "schema_version": 1,
    "claim": "NVIDIA RTA 2024公开方法实现的MaterialX source-domain functional reproduction；本次观测冻结在step 200k",
    "claim_boundary": (
        "只对登记的MaterialX snapshot、公开regular FP16 functional path和本机观测负责；"
        "不声称复现作者未公开资产、tensor-core intrinsic或论文图像数值；"
        "本次运行按用户决定在慢收敛区间以200k checkpoint登记，不声称完成原配置的300k训练协议。"
    ),
    "run_disposition": {
        "status": "user-frozen-at-slow-convergence",
        "recorded_step": args.recorded_step,
        "configured_total_steps": config.total_steps,
        "protocol_completion_fraction": args.recorded_step / config.total_steps,
        "observed_unrecorded_tail_max_step": metrics["observed_max_step"],
        "excluded_tail_record_count": metrics["excluded_tail_record_count"],
        "decision": "用户明确要求停止等待并按200k记录；所有正式数值均截断到step 200k。",
    },
    "identities": {
        "training_config_sha256": config.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": args.recorded_step,
        "package_id": package_id,
        "program_runtime_id": manifest["program_runtime_id"],
        "material_asset_id": manifest["material_asset_id"],
        "source_snapshot_id": manifest["source_snapshot_id"],
        "correspondence_id": config.correspondence_id,
        "recipe_id": config.recipe_id,
        "source_adaptation_id": config.source_adaptation_id,
    },
    "training": {
        "elapsed_seconds": float(final["elapsed_seconds"]),
        "logical_steps": args.recorded_step,
        "queries_per_step": sum(route.batch_size for route in config.routes),
        "total_work_units": int(final["work_units"]),
        "average_steps_per_second": float(final["steps_per_second"]),
        "peak_memory_bytes": metrics["peak_memory_bytes"],
        "metric_record_count": metrics["included_record_count"],
        "validation_records": metrics["validations"],
        "initial": first,
        "final": final,
        "relative_change": {
            "evaluator_log1p_l1": relative_change(
                float(first["evaluator_log1p_l1"]),
                float(final["evaluator_log1p_l1"]),
            ),
            "sampler_forward_kl": relative_change(
                float(first["sampler_forward_kl"]),
                float(final["sampler_forward_kl"]),
            ),
        },
    },
    "quality": directional,
    "deployment": {
        **package_costs(args.package, manifest),
        "compiled_material_bytes": 32,
        "scattering_state_bytes_per_pixel": 96,
        "static_network_macs": {
            "prepare_frames": 96,
            "prepare_sampler_head": 2688,
            "evaluate_per_direction": 9664,
        },
        "runtime_path": "regular-packed-fp16",
        "viewer_neural_pt_deferred": {
            "capture": str(args.neural_dual_capture),
            "slots": neural_dual["slots"],
            "gpu_ms": neural_dual["gpu_ms"],
        },
        "viewer_reference_neural_pt": {
            "capture": str(args.reference_neural_capture),
            "slots": reference_neural["slots"],
            "gpu_ms": reference_neural["gpu_ms"],
        },
    },
}

args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix(args.output.suffix + ".tmp")
temporary.write_text(
    json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    encoding="utf-8",
)
temporary.replace(args.output)
print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
