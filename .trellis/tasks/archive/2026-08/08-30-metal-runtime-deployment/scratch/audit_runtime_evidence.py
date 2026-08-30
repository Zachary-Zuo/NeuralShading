from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyexr
import torch

from ncls.learning.metal_runtime import metal_runtime_parameter_names
from ncls.learning.models.metal_fused import MetalFusedNeuralMaterialModel
from ncls.learning.models.metal_fused_profile import METAL_FUSED_FULL_PROFILE


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _mac_accounting() -> dict[str, int]:
    profile = METAL_FUSED_FULL_PROFILE
    token_count = profile.maximum_typed_tokens
    width = profile.typed_token_width
    feed_forward = 4 * width

    compiler_token_projection = token_count * (4 * width + width * width)
    compiler_optical_projection = 16 * width + width * width
    compiler_attention_block = (
        token_count * width * (3 * width)
        + token_count * width * width
        + 2 * profile.typed_attention_heads * token_count * token_count
        * (width // profile.typed_attention_heads)
        + token_count * (width * feed_forward + feed_forward * width)
    )
    compiler_heads = width * (
        profile.structured_width
        + profile.core_lobe_count * 9
        + profile.residual_lobe_count * 7
        + profile.evaluator_blocks * profile.asset_adapter_rank
        + (profile.core_lobe_count + profile.residual_lobe_count + 1) * 4
        + 3
    )
    compiler = (
        compiler_token_projection
        + compiler_optical_projection
        + profile.typed_attention_blocks * compiler_attention_block
        + compiler_heads
    )

    decoder_input_width = (
        profile.grid_high_channels
        + profile.grid_low_channels
        + 2 * profile.encoder_role_width
    )
    decoder_block = (
        2 * profile.decoder_width * profile.decoder_width
        + 4 * profile.decoder_width * profile.asset_adapter_rank
    )
    decoder_per_domain_mip = (
        decoder_input_width * profile.decoder_width
        + profile.decoder_blocks * decoder_block
        + profile.decoder_width * profile.structured_width
    )
    # The deployed asset has seven semantic domains. prepare() always decodes
    # the two adjacent independently supervised mip levels for every domain.
    decoder_per_surface = 7 * 2 * decoder_per_domain_mip

    condition_width = 2 * profile.structured_width
    prepared = (
        (condition_width + 8) * 128
        + 128 * profile.learned_frame_count * 6
        + condition_width * 128
        + 128 * profile.core_lobe_count * 9
        + condition_width * 128
        + 128 * profile.residual_lobe_count * 7
        + (3 + condition_width) * 128
        + 128 * profile.structured_width
        + condition_width
        * (profile.core_lobe_count + profile.residual_lobe_count + 1)
        * 4
    )

    directional_width = (
        10
        + 8
        + profile.learned_frame_count * 6
        + profile.angular_levels * profile.angular_channels
        + profile.angular_difference_rank
        + 3 * profile.structured_width
    )
    evaluator_block = (
        2 * profile.evaluator_width * profile.evaluator_width
        + 4 * profile.evaluator_width * profile.asset_adapter_rank
    )
    evaluator = (
        directional_width * profile.evaluator_width
        + profile.evaluator_blocks * evaluator_block
        + profile.evaluator_width
        * (3 + profile.residual_lobe_count * 3 + 3)
    )
    return {
        "compiler_once_per_edit": compiler,
        "compiler_attention_block": compiler_attention_block,
        "decoder_per_domain_mip": decoder_per_domain_mip,
        "decoder_per_surface_seven_domains_two_mips": decoder_per_surface,
        "prepare_heads_per_surface": prepared,
        "prepare_total_per_surface": decoder_per_surface + prepared,
        "evaluate_per_direction": evaluator,
        "prepare_plus_one_evaluate": decoder_per_surface + prepared + evaluator,
        "sample_after_prepare": evaluator,
        "pdf_after_prepare": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metrics = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    capture = json.loads(args.capture.read_text(encoding="utf-8"))

    with torch.device("meta"):
        model = MetalFusedNeuralMaterialModel()
    state = model.state_dict()
    runtime_names = metal_runtime_parameter_names(model)
    runtime_elements = sum(state[name].numel() for name in runtime_names)
    shared_blob = args.package / manifest["files"]["program/blob/shared-weights"]
    packed_runtime_bytes = ((runtime_elements * 2 + 3) // 4) * 4
    if packed_runtime_bytes != shared_blob.stat().st_size:
        raise RuntimeError("FP16 shared pack byte count disagrees with runtime parameters")

    gradient_coverage = checkpoint["gradient_coverage"]
    required_gradient_flags = (
        "finite_observed",
        "nonzero_gradient_observed",
        "parameter_update_observed",
    )
    if not gradient_coverage or not all(
        all(bool(group.get(flag)) for flag in required_gradient_flags)
        for group in gradient_coverage.values()
    ):
        raise RuntimeError("one or more required training groups lack gradient/update evidence")
    if not _all_finite(metrics):
        raise RuntimeError("training metrics contain a non-finite numeric value")
    if any(slot["status"] != "ready" for slot in capture["slots"]):
        raise RuntimeError("viewer capture contains a non-ready package slot")
    if any(
        slot["package_id"] != manifest["package_id"]
        or slot["program_id"] != manifest["program_id"]
        or slot["asset_id"] != manifest["asset_id"]
        or slot["instance_id"] != manifest["instance_id"]
        for slot in capture["slots"]
    ):
        raise RuntimeError("viewer capture identity disagrees with the audited package")
    capture_root = args.capture.parent
    images = {
        name: pyexr.read(str(capture_root / capture["files"][name]))
        for name in (
            "slot_0_linear",
            "slot_1_linear",
            "difference_linear",
        )
    }
    if not all(np.isfinite(image).all() for image in images.values()):
        raise RuntimeError("viewer capture contains non-finite linear pixels")
    expected_shape = (
        int(capture["view_resolution"][1]),
        int(capture["view_resolution"][0]),
        3,
    )
    if any(image.shape != expected_shape for image in images.values()):
        raise RuntimeError("viewer capture linear image extent is inconsistent")
    slot_difference = np.abs(images["slot_0_linear"] - images["slot_1_linear"])

    profile = METAL_FUSED_FULL_PROFILE
    report = {
        "package": {
            "package_id": manifest["package_id"],
            "program_id": manifest["program_id"],
            "asset_id": manifest["asset_id"],
            "instance_id": manifest["instance_id"],
            "source_snapshot_id": manifest["source_snapshot_id"],
            "storage_bytes": manifest["validation"]["storage"],
            "runtime_parameter_elements": runtime_elements,
            "shared_fp16_bytes": shared_blob.stat().st_size,
            "shared_alignment_padding_bytes": packed_runtime_bytes
            - runtime_elements * 2,
        },
        "bounded_runtime": {
            "prepared_state_bytes": profile.maximum_state_bytes,
            "random_access_reads": profile.maximum_reads,
            "read_accounting": "9 bounded texture slots * 2 mips * (4 high-grid gathers + 1 low-grid filtered read) + 4 angular levels * 4 gathers",
            "macs": _mac_accounting(),
            "scope": "MAC counts exclude scalar normalization/transcendental/analytic-BSDF work and count compiler separately because it runs only after typed edits",
        },
        "training": {
            "global_step": checkpoint["global_step"],
            "phase_name": checkpoint["phase_name"],
            "metrics_records": len(metrics),
            "metrics_all_finite": True,
            "peak_memory_bytes": max(
                int(row.get("peak_memory_bytes", 0)) for row in metrics
            ),
            "gradient_groups": {
                name: {
                    flag: bool(value[flag]) for flag in required_gradient_flags
                }
                | {"last_audit_step": int(value["last_audit_step"])}
                for name, value in gradient_coverage.items()
            },
        },
        "viewer": {
            "resolution": capture["resolution"],
            "view_resolution": capture["view_resolution"],
            "scene_bounce_cap": capture["reference_scene_max_bounces"],
            "lighting": {
                "environment": capture["lighting"]["use_environment"],
                "rectangle": capture["lighting"]["use_rectangle"],
                "sun": capture["lighting"]["use_sun"],
            },
            "slot_gpu_ms": [slot["gpu_ms"] for slot in capture["slots"]],
            "slot_modes": [slot["mode"] for slot in capture["slots"]],
            "linear_images": {
                name: {
                    "shape": list(image.shape),
                    "finite": True,
                    "minimum": float(image.min()),
                    "maximum": float(image.max()),
                }
                for name, image in images.items()
            },
            "mean_absolute_slot_difference": float(slot_difference.mean()),
            "maximum_absolute_slot_difference": float(slot_difference.max()),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
