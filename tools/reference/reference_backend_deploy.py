from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Iterable
import urllib.request

from ncls.paths import PROJECT_ROOT
from ncls.references.backend import create_reference_backend
from ncls.references.backend_manifest import (
    BinaryArchive,
    GitSourceProvider,
    MdlSdkLayout,
    load_reference_backend_manifest,
)


@dataclass(frozen=True)
class DeploymentStep:
    step_id: str
    status: str
    detail: str


def _contained(root: Path, target: Path) -> Path:
    resolved_root = root.resolve()
    resolved = target.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"deployment target escapes {resolved_root}: {resolved}") from error
    if resolved == resolved_root:
        raise ValueError("deployment target must be below its managed root")
    return resolved


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {message}")
    return result.stdout.strip()


def _git_state(provider: GitSourceProvider, project_root: Path) -> DeploymentStep:
    target = _contained(project_root / "external", project_root / provider.path)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        return DeploymentStep(provider.provider_id, "invalid", f"partial target exists: {partial}")
    if not target.exists():
        return DeploymentStep(provider.provider_id, "fresh", str(target))
    if not (target / ".git").is_dir():
        return DeploymentStep(provider.provider_id, "invalid", f"not a Git checkout: {target}")
    head = _run(["git", "rev-parse", "HEAD"], cwd=target)
    dirty = _run(["git", "status", "--porcelain"], cwd=target)
    if head != provider.revision:
        return DeploymentStep(
            provider.provider_id,
            "invalid",
            f"revision mismatch: expected {provider.revision}, got {head}",
        )
    if dirty:
        return DeploymentStep(provider.provider_id, "invalid", f"checkout is dirty: {target}")
    return DeploymentStep(provider.provider_id, "reused", str(target))


def _ensure_git(provider: GitSourceProvider, project_root: Path) -> DeploymentStep:
    state = _git_state(provider, project_root)
    if state.status == "invalid":
        raise RuntimeError(state.detail)
    target = project_root / provider.path
    if state.status == "fresh":
        partial = target.with_name(target.name + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.mkdir()
        _run(["git", "init"], cwd=partial)
        _run(["git", "remote", "add", "origin", provider.url], cwd=partial)
        _run(["git", "fetch", "--depth", "1", "origin", provider.revision], cwd=partial)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=partial)
        if provider.submodules != "none":
            command = ["git", "submodule", "update", "--init"]
            if provider.submodules == "recursive-upstream-locked":
                command.append("--recursive")
            _run(command, cwd=partial)
        os.replace(partial, target)
    elif provider.submodules != "none":
        status = _run(["git", "submodule", "status", "--recursive"], cwd=target)
        if any(line[:1] in {"-", "+", "U"} for line in status.splitlines()):
            raise RuntimeError(f"submodule state is incomplete or drifted: {target}")
    final = _git_state(provider, project_root)
    if final.status == "invalid":
        raise RuntimeError(final.detail)
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive(path: Path, archive: BinaryArchive) -> None:
    if path.stat().st_size != archive.size:
        raise RuntimeError(
            f"archive size mismatch for {path}: expected {archive.size}, got {path.stat().st_size}"
        )
    actual = _sha256(path)
    if actual != archive.sha256:
        raise RuntimeError(
            f"archive SHA-256 mismatch for {path}: expected {archive.sha256}, got {actual}"
        )


def _safe_tar_members(
    members: Iterable[tarfile.TarInfo], *, expected_root: str
) -> tuple[tarfile.TarInfo, ...]:
    values = tuple(members)
    for member in values:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise RuntimeError(f"MDL SDK archive contains an unsafe member: {member.name}")
        if not path.parts or path.parts[0] != expected_root:
            raise RuntimeError(f"MDL SDK archive root mismatch: {member.name}")
    return values


def _sdk_required_paths(sdk: MdlSdkLayout, project_root: Path) -> tuple[Path, ...]:
    root = project_root / sdk.archive.root
    return (
        root / "include/mi/mdl_sdk.h",
        root / sdk.library,
        *(root / plugin for plugin in sdk.plugins),
        root / sdk.target_code_types,
    )


def _copy_target_code_types_source(
    sdk: MdlSdkLayout, *, project_root: Path, target_root: Path
) -> None:
    source_name = sdk.target_code_types_source
    source_sha256 = sdk.target_code_types_source_sha256
    if source_name is None or source_sha256 is None:
        return
    source = _contained(project_root, project_root / source_name)
    if not source.is_file():
        raise RuntimeError(f"MDL target-code source is missing: {source}")
    actual = _sha256(source)
    if actual != source_sha256:
        raise RuntimeError(
            "MDL target-code source SHA-256 mismatch: "
            f"expected {source_sha256}, got {actual}"
        )
    destination = _contained(target_root, target_root / sdk.target_code_types)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _sdk_state(sdk: MdlSdkLayout, project_root: Path) -> DeploymentStep:
    external = project_root / "external"
    target = _contained(external, project_root / sdk.archive.root)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        return DeploymentStep("mdl-sdk", "invalid", f"partial target exists: {partial}")
    if target.exists():
        if not target.is_dir():
            return DeploymentStep("mdl-sdk", "invalid", f"SDK target is not a directory: {target}")
        missing = tuple(path for path in _sdk_required_paths(sdk, project_root) if not path.is_file())
        if missing:
            return DeploymentStep(
                "mdl-sdk",
                "invalid",
                "incomplete SDK target: " + ", ".join(str(path) for path in missing),
            )
        return DeploymentStep("mdl-sdk", "reused", str(target))
    return DeploymentStep("mdl-sdk", "fresh", str(target))


def _ensure_sdk(sdk: MdlSdkLayout, project_root: Path) -> DeploymentStep:
    archive = sdk.archive
    state = _sdk_state(sdk, project_root)
    if state.status == "invalid":
        raise RuntimeError(state.detail)
    external = project_root / "external"
    target = project_root / archive.root
    downloads = external / ".downloads"
    archive_path = downloads / archive.name
    if state.status == "reused":
        return state
    downloads.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        _verify_archive(archive_path, archive)
    else:
        partial_archive = archive_path.with_name(archive_path.name + ".partial")
        if partial_archive.exists():
            raise RuntimeError(f"partial archive exists: {partial_archive}")
        with urllib.request.urlopen(archive.url) as response, partial_archive.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        _verify_archive(partial_archive, archive)
        os.replace(partial_archive, archive_path)
    partial_target = target.with_name(target.name + ".partial")
    partial_target.mkdir(parents=False)
    if archive.archive_type != "tar.gz":
        raise RuntimeError("Linux deployment requires the pinned tar.gz MDL SDK archive")
    expected_root = Path(archive.root).name
    with tarfile.open(archive_path, "r:gz") as stream:
        members = _safe_tar_members(stream.getmembers(), expected_root=expected_root)
        for member in members:
            relative = Path(*Path(member.name).parts[1:])
            if not relative.parts:
                continue
            destination = _contained(partial_target, partial_target / relative)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(destination, member.mode & 0o777)
            else:
                raise RuntimeError(f"unsupported archive member: {member.name}")
    _copy_target_code_types_source(
        sdk,
        project_root=project_root,
        target_root=partial_target,
    )
    os.replace(partial_target, target)
    final = _sdk_state(sdk, project_root)
    if final.status == "invalid":
        raise RuntimeError(final.detail)
    return state


def deployment_plan(platform_id: str, project_root: Path) -> tuple[DeploymentStep, ...]:
    manifest = load_reference_backend_manifest()
    platform = manifest.for_platform(platform_id)
    return (
        *(_git_state(provider, project_root) for provider in manifest.source_providers),
        _sdk_state(platform.mdl_sdk, project_root),
    )


def fetch(platform_id: str, project_root: Path) -> tuple[DeploymentStep, ...]:
    manifest = load_reference_backend_manifest()
    platform = manifest.for_platform(platform_id)
    planned = deployment_plan(platform_id, project_root)
    invalid = tuple(value for value in planned if value.status == "invalid")
    if invalid:
        raise RuntimeError("; ".join(value.detail for value in invalid))
    result = [
        _ensure_git(provider, project_root) for provider in manifest.source_providers
    ]
    result.append(_ensure_sdk(platform.mdl_sdk, project_root))
    return tuple(result)


def _optional_run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _first_line(value: str | None) -> str | None:
    return value.splitlines()[0] if value else None


def _os_release() -> dict[str, str] | None:
    path = Path("/etc/os-release")
    if not path.is_file():
        return None
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, raw = line.split("=", 1)
        result[key] = raw.strip().strip('"')
    return result


def _environment_report() -> dict[str, object]:
    vulkan = _optional_run(["vulkaninfo", "--summary"])
    return {
        "platform": sys.platform,
        "uname": list(os.uname()) if hasattr(os, "uname") else None,
        "os_release": _os_release(),
        "glibc": _first_line(_optional_run(["ldd", "--version"])),
        "gpu": _optional_run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "vulkan_summary": "\n".join(vulkan.splitlines()[:24]) if vulkan else None,
        "compiler": _first_line(_optional_run(["gcc", "--version"])),
        "git": _first_line(_optional_run(["git", "--version"])),
        "conda": _first_line(_optional_run(["conda", "--version"])),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "falcor_gpu_index": os.environ.get(
            "NCLS_FALCOR_GPU_INDEX", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        ),
        "python": sys.version,
    }


def write_report(
    output: Path,
    *,
    deployment_status: str,
    steps: tuple[DeploymentStep, ...],
) -> None:
    backend = create_reference_backend()
    doctor = backend.doctor()
    descriptor = backend.descriptor
    value = {
        "schema_name": "ncls.reference-backend-deployment-report",
        "schema_version": 1,
        "deployment_status": deployment_status,
        "platform_id": descriptor.platform_id,
        "backend_identity": descriptor.identity,
        "semantic_identity": descriptor.semantic_identity,
        "build_identity": descriptor.build_identity,
        "toolchains": {
            "falcor_revision": descriptor.falcor_revision,
            "slang_revision": descriptor.slang_revision,
            "mdl_sdk_build": backend.platform.mdl_sdk.build,
            "device_api": descriptor.device_api,
        },
        "assets": "not-managed",
        "summary_zh": (
            "统一 Reference Backend 部署已就绪；source assets 不归部署管理。"
            if deployment_status == "ready"
            else f"统一 Reference Backend 部署状态：{deployment_status}；source assets 未被修改。"
        ),
        "environment": _environment_report(),
        "steps": [asdict(value) for value in steps],
        "doctor": {
            "ready": doctor.ready,
            "statuses": [asdict(value) for value in doctor.statuses],
        },
        "next_command": "python -m ncls reference probe",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _parse_step(value: str) -> DeploymentStep:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("step must use id=status=detail")
    return DeploymentStep(*parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="manifest-driven reference backend deployment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("--platform-id", required=True)
        command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
        command.add_argument("--output", type=Path)
    report = commands.add_parser("report")
    report.add_argument("output", type=Path)
    report.add_argument("--deployment-status", required=True)
    report.add_argument("--step", action="append", default=[], type=_parse_step)
    report.add_argument("--steps-file", action="append", default=[], type=Path)
    layout = commands.add_parser("layout")
    layout.add_argument("--platform-id", required=True)
    layout.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    arguments = parser.parse_args(argv)
    if arguments.command == "report":
        recorded_steps = list(arguments.step)
        for path in arguments.steps_file:
            if not path.is_file():
                continue
            values = json.loads(path.read_text(encoding="utf-8"))
            recorded_steps.extend(DeploymentStep(**value) for value in values)
        write_report(
            arguments.output,
            deployment_status=arguments.deployment_status,
            steps=tuple(recorded_steps),
        )
        return 0
    if arguments.command == "layout":
        manifest = load_reference_backend_manifest()
        platform = manifest.for_platform(arguments.platform_id)
        root = arguments.project_root.resolve()
        source = next(
            value
            for value in manifest.source_providers
            if value.provider_id == manifest.execution_provider
        )
        print(
            json.dumps(
                {
                    "falcor_root": str(root / source.path),
                    "falcor_revision": manifest.falcor_revision,
                    "falcor_extension": str(
                        root
                        / platform.falcor.python_module_root
                        / platform.falcor.python_extension
                    ),
                    "falcor_runtime_library_root": str(
                        root / platform.falcor.runtime_library_root
                    ),
                    "falcor_python_module_root": str(
                        root / platform.falcor.python_module_root
                    ),
                    "falcor_python_extension": platform.falcor.python_extension,
                    "sdk_root": str(root / platform.mdl_sdk.archive.root),
                    "stb_root": str(root / "external/stb"),
                    "mdl_bridge_executable": str(
                        root / platform.mdl_bridge.executable
                    ),
                }
            )
        )
        return 0
    operation = deployment_plan if arguments.command == "plan" else fetch
    steps = operation(arguments.platform_id, arguments.project_root.resolve())
    payload = json.dumps([asdict(value) for value in steps], ensure_ascii=False, indent=2)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
