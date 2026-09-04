from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import subprocess
from typing import Any

import torch

from ncls.core.identity import sha256_file, sha256_json, write_json_atomic
from ncls.learning.methods.metal_budgeted import METHOD_DEFINITION
from ncls.learning.training import TrainingPlanResolver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYBRID_PILOT_PATH = (
    PROJECT_ROOT / "configs/training/runs/metal-budgeted-hybrid-pilot.yaml"
)
DIRECT_PILOT_PATH = (
    PROJECT_ROOT / "configs/training/runs/metal-budgeted-direct-pilot.yaml"
)
REQUIRED_WORKTREE_PATHS = (
    "configs/training",
    "src/ncls/learning",
    "tests/unit",
    "tools/learning/build_metal_linux_handoff.py",
)


def _config_pair(hybrid, direct) -> dict[str, Any]:
    hybrid_phases = [phase.name for phase in hybrid.phases]
    direct_phases = [phase.name for phase in direct.phases]
    if hybrid_phases != direct_phases:
        raise ValueError("Metal budgeted direct/hybrid phase graphs disagree")
    if hybrid.source != direct.source or hybrid.online_query != direct.online_query:
        raise ValueError("Metal budgeted direct/hybrid source/query contracts disagree")
    if hybrid.total_steps != direct.total_steps:
        raise ValueError("Metal budgeted direct/hybrid pilot caps disagree")
    return {
        "source_count": len(hybrid.source["materials"]),
        "pilot_steps": hybrid.total_steps,
        "phase_names": hybrid_phases,
        "hybrid_profile": hybrid.model_context["profile_id"],
        "direct_profile": direct.model_context["profile_id"],
        "validation_interval": hybrid.validation["interval"],
        "validation_batches": hybrid.validation["batches"],
        "devices": [0],
        "distributed": False,
        "visible_gpu_count": 1,
    }


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
    resolver = TrainingPlanResolver(PROJECT_ROOT)
    hybrid_plan = resolver.resolve(HYBRID_PILOT_PATH)
    direct_plan = resolver.resolve(DIRECT_PILOT_PATH)
    hybrid = hybrid_plan.to_runtime_config()
    direct = direct_plan.to_runtime_config()
    pair = _config_pair(hybrid, direct)
    pair.update(
        {
            "hybrid_resolved_plan_sha256": hybrid_plan.sha256,
            "direct_resolved_plan_sha256": direct_plan.sha256,
            "hybrid_training_config_sha256": hybrid.sha256,
            "direct_training_config_sha256": direct.sha256,
        }
    )
    tracked_dirty = _git("status", "--short", "--untracked-files=no").splitlines()
    required_changes = _git(
        "status", "--short", "--untracked-files=all", "--", *REQUIRED_WORKTREE_PATHS
    ).splitlines()
    body: dict[str, Any] = {
        "schema": "ncls.metal-budgeted-linux-pilot-handoff@1",
        "repository_commit": _git("rev-parse", "HEAD"),
        "tracked_worktree_dirty": tracked_dirty,
        "required_worktree_changes": required_changes,
        "repository_state": (
            "clean-commit" if not required_changes else "working-tree-snapshot-required"
        ),
        "transfer_precondition": (
            "Linux must receive the exact required working-tree changes or a commit "
            "containing them before executing any command"
        ),
        "method_key": "metal",
        "method_implementation_key": METHOD_DEFINITION.descriptor.method_key,
        "method_descriptor_sha256": METHOD_DEFINITION.descriptor.descriptor_sha256,
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
            "hybrid_pilot_config_sha256": sha256_file(HYBRID_PILOT_PATH),
            "direct_pilot_config_sha256": sha256_file(DIRECT_PILOT_PATH),
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
            "hybrid_step0": (
                "CUDA_VISIBLE_DEVICES="
                f"{visible_gpu} bash scripts/run_falcor_python.sh -m ncls train "
                "configs/training/runs/metal-budgeted-hybrid-pilot.yaml --devices "
                f"{visible_gpu} --stop-at-step 0 --output "
                "artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt"
            ),
            "hybrid_step0_validation": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls "
                "validate artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt "
                f"--batches 256 --device {visible_gpu}"
            ),
            "hybrid_start_recoverable": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls "
                "train configs/training/runs/metal-budgeted-hybrid-pilot.yaml --devices "
                f"{visible_gpu} --output artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt "
                "--resume artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt --stop-at-step 128"
            ),
            "hybrid_resume": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls "
                "train configs/training/runs/metal-budgeted-hybrid-pilot.yaml --devices "
                f"{visible_gpu} --output artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt "
                "--resume artifacts/metal-budgeted-pilot/hybrid/checkpoint.pt"
            ),
            "direct_step0": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls "
                "train configs/training/runs/metal-budgeted-direct-pilot.yaml --devices "
                f"{visible_gpu} --stop-at-step 0 --output "
                "artifacts/metal-budgeted-pilot/direct/checkpoint.pt"
            ),
            "direct_step0_validation": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls "
                "validate artifacts/metal-budgeted-pilot/direct/checkpoint.pt "
                f"--batches 256 --device {visible_gpu}"
            ),
            "direct_start_recoverable": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls "
                "train configs/training/runs/metal-budgeted-direct-pilot.yaml --devices "
                f"{visible_gpu} --output artifacts/metal-budgeted-pilot/direct/checkpoint.pt "
                "--resume artifacts/metal-budgeted-pilot/direct/checkpoint.pt --stop-at-step 128"
            ),
            "direct_resume": (
                f"CUDA_VISIBLE_DEVICES={visible_gpu} bash scripts/run_falcor_python.sh -m ncls "
                "train configs/training/runs/metal-budgeted-direct-pilot.yaml --devices "
                f"{visible_gpu} --output artifacts/metal-budgeted-pilot/direct/checkpoint.pt "
                "--resume artifacts/metal-budgeted-pilot/direct/checkpoint.pt"
            ),
            "monitor_metrics": (
                "tail -f artifacts/metal-budgeted-pilot/hybrid/checkpoint.metrics.jsonl"
            ),
            "monitor_gpu": "nvidia-smi dmon -s pucvmet -d 5",
            "recover_latest_periodic": (
                "ls -1 artifacts/metal-budgeted-pilot/*/checkpoint.step*.pt "
                "| sort | tail -n 1"
            ),
        },
        "linux_execution_status": "pending-on-target-host",
        "automatic_followups": [],
        "selection_artifact": (
            "artifacts/metal-budgeted-pilot/single-material-selection.json"
        ),
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
