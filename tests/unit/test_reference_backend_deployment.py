from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pytest

from ncls.references.backend_manifest import load_reference_backend_manifest
from tools.reference.reference_backend_deploy import (
    _safe_tar_members,
    deployment_plan,
    write_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_asset_free_linux_deployment_plan_never_requires_assets(tmp_path: Path) -> None:
    steps = deployment_plan("linux-x86_64@1", tmp_path)
    assert {value.step_id for value in steps} == {
        "falcor",
        "materialx",
        "openpbr",
        "openpbr-bsdf",
        "glm",
        "stb",
        "mdl-sdk",
    }
    assert all(value.status == "fresh" for value in steps)
    assert not (tmp_path / "assets").exists()


def test_linux_deployment_scripts_forbid_privileged_and_asset_fetches() -> None:
    paths = (
        PROJECT_ROOT / "scripts/deploy_reference_linux.sh",
        PROJECT_ROOT / "scripts/build_falcor_python_linux.sh",
        PROJECT_ROOT / "scripts/build_mdl_program_provider.sh",
        PROJECT_ROOT / "tools/reference/reference_backend_deploy.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    for forbidden in (
        "sudo ",
        "fetch_mdl_assets",
        "fetch_source_materials",
        "assets/source-materials",
        "conda installer",
        "nvidia driver",
    ):
        assert forbidden not in text
    assert "asset-free-probe" in text
    assert "assets were not managed" in text
    assert "environment.sha256" not in text


def test_linux_deployment_preflight_precedes_environment_and_fetch() -> None:
    script = (PROJECT_ROOT / "scripts/deploy_reference_linux.sh").read_text(
        encoding="utf-8"
    )
    preflight = script.index("for command in")
    environment = script.index("conda env create")
    fetch = script.index('reference_backend_deploy.py" fetch')
    assert preflight < environment < fetch


def test_linux_falcor_build_repairs_windows_copy_metadata() -> None:
    script = (PROJECT_ROOT / "scripts/build_falcor_python_linux.sh").read_text(
        encoding="utf-8"
    )
    assert "git -C \"${falcor_root}\" ls-files -s -z" in script
    assert "chmod a+x \"${falcor_root}/${path}\"" in script
    assert "find \"${nvtt_root}\" -maxdepth 1 -type f -name 'libnvtt.so.*'" in script
    assert "ln -s \"${versioned_libraries[0]}\" \"${nvtt_link}\"" in script




def test_portable_mdl_provider_has_one_loader_boundary_and_explicit_plugins() -> None:
    source = (
        PROJECT_ROOT / "tools/reference/mdl_sdk_bridge/main.cpp"
    ).read_text(encoding="utf-8")
    cmake = (
        PROJECT_ROOT / "tools/reference/mdl_sdk_bridge/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    assert "class SharedLibrary" in source
    assert "LoadLibraryW" in source
    assert "dlopen" in source
    assert 'argument == "--sdk-library"' in source
    assert 'argument == "--plugin"' in source
    assert "${CMAKE_DL_LIBS}" in cmake


def test_safe_tar_validation_rejects_traversal_and_links() -> None:
    safe = tarfile.TarInfo("sdk-root/lib/libmdl_sdk.so")
    assert _safe_tar_members((safe,), expected_root="sdk-root") == (safe,)
    with pytest.raises(RuntimeError, match="unsafe"):
        _safe_tar_members(
            (tarfile.TarInfo("sdk-root/../escape"),), expected_root="sdk-root"
        )
    link = tarfile.TarInfo("sdk-root/lib/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/outside"
    with pytest.raises(RuntimeError, match="unsafe"):
        _safe_tar_members((link,), expected_root="sdk-root")


def test_deployment_report_records_not_managed_asset_policy(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_report(output, deployment_status="fixture", steps=())
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["schema_name"] == "ncls.reference-backend-deployment-report"
    assert value["assets"] == "not-managed"
    assert value["summary_zh"]
    assert value["toolchains"]["mdl_sdk_build"] == "2025.0.0-387700.1252"
    assert value["next_command"].endswith("reference probe")
    assert value["backend_identity"]
    assert set(value["environment"]) >= {
        "os_release",
        "glibc",
        "gpu",
        "vulkan_summary",
        "compiler",
        "git",
        "conda",
        "python",
    }


def test_existing_incomplete_sdk_is_invalid_instead_of_reused(tmp_path: Path) -> None:
    platform = load_reference_backend_manifest().for_platform("linux-x86_64@1")
    (tmp_path / platform.mdl_sdk.archive.root).mkdir(parents=True)
    steps = deployment_plan("linux-x86_64@1", tmp_path)
    sdk = next(value for value in steps if value.step_id == "mdl-sdk")
    assert sdk.status == "invalid"
    assert "incomplete SDK target" in sdk.detail


def test_manifest_drives_linux_build_layout() -> None:
    platform = load_reference_backend_manifest().for_platform("linux-x86_64@1")
    assert platform.falcor.device_api == "vulkan"
    assert platform.mdl_sdk.library.endswith(".so")
    assert platform.mdl_bridge.executable.endswith("ncls_mdl_sdk_bridge")
