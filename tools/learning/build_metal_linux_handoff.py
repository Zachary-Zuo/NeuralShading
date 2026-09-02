from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import subprocess
from typing import Any

import torch

from ncls.core.identity import sha256_file, sha256_json, write_json_atomic
from ncls.learning.methods.metal_fused import METHOD_DEFINITION
from ncls.learning.training import TrainingConfig
from ncls.source_materials.mdl_metal import MdlMetalRegistry
from tools.learning.build_metal_training_configs import (
    LINUX_LONG_PATH,
    LINUX_SMOKE_PATH,
    REGISTRY_PATH,
    validate_linux_config_pair,
)
from tools.learning.preflight_metal_fused import build_preflight_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def build_handoff_manifest(
    *, windows_checkpoint: Path | None = None
) -> dict[str, Any]:
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if not visible_gpu.isdecimal():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index"
        )
    smoke = TrainingConfig.load(LINUX_SMOKE_PATH)
    long_run = TrainingConfig.load(LINUX_LONG_PATH)
    pair = validate_linux_config_pair(smoke.to_dict(), long_run.to_dict())
    registry = MdlMetalRegistry.load(REGISTRY_PATH)
    preflight = build_preflight_report(registry)
    tracked_dirty = _git("status", "--short", "--untracked-files=no").splitlines()
    body: dict[str, Any] = {
        "schema": "ncls.metal-linux-training-handoff@1",
        "repository_commit": _git("rev-parse", "HEAD"),
        "tracked_worktree_dirty": tracked_dirty,
        "method_key": METHOD_DEFINITION.descriptor.method_key,
        "method_descriptor_sha256": METHOD_DEFINITION.descriptor.descriptor_sha256,
        "registry_identity": registry.identity,
        "full_cohort_preflight_identity": preflight["identity"],
        "config_pair": dict(pair),
        "files": {
            "environment_yml_sha256": sha256_file(PROJECT_ROOT / "environment.yml"),
            "torch_requirements_sha256": sha256_file(
                PROJECT_ROOT / "requirements-torch-cu128.txt"
            ),
            "reference_toolchains_sha256": sha256_file(
                PROJECT_ROOT / "references/reference-backend-toolchains.json"
            ),
            "linux_launcher_sha256": sha256_file(
                PROJECT_ROOT / "scripts/run_falcor_python.sh"
            ),
            "linux_deploy_sha256": sha256_file(
                PROJECT_ROOT / "scripts/deploy_reference_linux.sh"
            ),
            "linux_smoke_config_sha256": smoke.sha256,
            "linux_long_config_sha256": long_run.sha256,
        },
        "producer": {
            "processes": 1,
            "visible_gpus": 1,
            "distributed": False,
            "persistent_training_batches": False,
            "response_target_device": "cuda:0",
        },
        "generation_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "commands": {
            "deploy": f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/deploy_reference_linux.sh",
            "config_check": (
                "PYTHONPATH=src conda run --no-capture-output -n neural-shading python "
                "tools/learning/build_metal_training_configs.py"
            ),
            "full_cohort_preflight": (
                "PYTHONPATH=src conda run --no-capture-output -n neural-shading python "
                "tools/learning/preflight_metal_fused.py --output "
                "artifacts/metal-linux-training/full-cohort-preflight.json"
            ),
            "smoke": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls.cli "
                "learn train configs/learning/metal-fused-full-linux-smoke.json "
                "artifacts/metal-linux-training/smoke/checkpoint.pt"
            ),
            "long_start_recoverable": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls.cli "
                "learn train configs/learning/metal-fused-full-linux-long.json "
                "artifacts/metal-linux-training/long/checkpoint.pt --stop-at-step 16"
            ),
            "long_resume": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls.cli "
                "learn train configs/learning/metal-fused-full-linux-long.json "
                "artifacts/metal-linux-training/long/checkpoint.pt --resume "
                "artifacts/metal-linux-training/long/checkpoint.pt"
            ),
            "monitor_metrics": (
                "tail -f artifacts/metal-linux-training/long/checkpoint.metrics.jsonl"
            ),
            "monitor_gpu": "nvidia-smi dmon -s pucvmet -d 5",
            "recover_latest_periodic": (
                "ls -1 artifacts/metal-linux-training/long/checkpoint.step*.pt "
                "| sort | tail -n 1"
            ),
            "evaluate": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls.cli "
                "learn evaluate configs/learning/metal-fused-full-linux-long.json "
                "artifacts/metal-linux-training/long/checkpoint.pt --batches 8"
            ),
            "export": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls.cli "
                "learn export artifacts/metal-linux-training/long/checkpoint.pt "
                "artifacts/metal-linux-training/long/package --material-index 0"
            ),
        },
        "linux_execution_status": "pending-on-target-host",
        "automatic_followups": [],
        "review_artifact": "artifacts/metal-linux-training/long/checkpoint.review.json",
    }
    if windows_checkpoint is not None:
        body["windows_correctness_checkpoint"] = {
            "path": windows_checkpoint.as_posix(),
            "sha256": sha256_file(windows_checkpoint),
            "scope": "Windows correctness/deployment evidence; not a Linux long-run seed",
        }
    return {**body, "identity": sha256_json(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成Metal Linux单GPU训练交接manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-checkpoint", type=Path)
    arguments = parser.parse_args()
    manifest = build_handoff_manifest(
        windows_checkpoint=arguments.windows_checkpoint
    )
    write_json_atomic(arguments.output, manifest)
    print(manifest["identity"])
    print("linux_execution_status=pending-on-target-host distributed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
