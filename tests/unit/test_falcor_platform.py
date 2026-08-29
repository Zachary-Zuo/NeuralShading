"""锁定统一 reference backend 在 Windows/D3D12 与 Linux/Vulkan 间的平台合同。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ncls.references.backend import create_reference_backend
from ncls.references.programs import discover_reference_programs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeFalcor:
    class DeviceType:
        D3D12 = "d3d12"
        Vulkan = "vulkan"

    class Device:
        def __init__(self, *, type: str, gpu: int) -> None:
            self.type = type
            self.gpu = gpu


@pytest.mark.parametrize(
    ("platform", "expected"),
    (("win32", "d3d12"), ("linux", "vulkan"), ("linux2", "vulkan")),
)
def test_reference_backend_device_selects_platform_api(
    tmp_path,
    platform: str,
    expected: str,
) -> None:
    backend = create_reference_backend(
        platform_name=platform, machine="x86_64", project_root=tmp_path
    )
    assert backend._create_device(_FakeFalcor).type == expected  # noqa: SLF001


def test_reference_backend_reuses_process_device(tmp_path) -> None:
    backend = create_reference_backend(
        platform_name="linux", machine="x86_64", project_root=tmp_path
    )
    first = backend._create_device(_FakeFalcor)  # noqa: SLF001
    second = backend._create_device(_FakeFalcor)  # noqa: SLF001
    assert second is first


def test_reference_backend_selects_explicit_physical_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NCLS_FALCOR_GPU_INDEX", "3")
    backend = create_reference_backend(
        platform_name="linux", machine="x86_64", project_root=tmp_path
    )
    assert backend._create_device(_FakeFalcor).gpu == 3  # noqa: SLF001


@pytest.mark.parametrize("value", ("-1", "0,1", "GPU-deadbeef"))
def test_reference_backend_rejects_invalid_gpu_index(
    tmp_path, monkeypatch, value: str
) -> None:
    monkeypatch.setenv("NCLS_FALCOR_GPU_INDEX", value)
    backend = create_reference_backend(
        platform_name="linux", machine="x86_64", project_root=tmp_path
    )
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        backend._create_device(_FakeFalcor)  # noqa: SLF001


def test_reference_backend_rejects_software_adapter(tmp_path) -> None:
    class SoftwareFalcor(_FakeFalcor):
        class Device:
            def __init__(self, *, type: str, gpu: int) -> None:
                class Info:
                    adapter_name = "llvmpipe"

                self.type = type
                self.gpu = gpu
                self.info = Info()

    backend = create_reference_backend(
        platform_name="linux", machine="x86_64", project_root=tmp_path
    )
    with pytest.raises(RuntimeError, match="hardware graphics adapter"):
        backend._create_device(SoftwareFalcor)  # noqa: SLF001


def test_reference_backend_rejects_unsupported_platform(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not support platform"):
        create_reference_backend(
            platform_name="darwin", machine="x86_64", project_root=tmp_path
        )


@pytest.mark.parametrize(
    ("platform", "expected_id", "backend", "build_name"),
    (
        ("win32", "windows-x86_64@1", "d3d12", "windows-vs2022"),
        ("linux", "linux-x86_64@1", "vulkan", "linux-gcc"),
    ),
)
def test_falcor_execution_capability_owns_platform_layout(
    tmp_path,
    platform: str,
    expected_id: str,
    backend: str,
    build_name: str,
) -> None:
    capability = create_reference_backend(
        platform_name=platform,
        machine="x86_64",
        project_root=tmp_path,
    )
    descriptor = capability.descriptor
    assert descriptor.platform_id == expected_id
    assert descriptor.device_api == backend
    assert descriptor.build_root.name == build_name
    assert descriptor.python_module_root == descriptor.runtime_library_root / "python"


def test_falcor_execution_capability_augments_platform_environment(tmp_path) -> None:
    capability = create_reference_backend(
        platform_name="linux",
        machine="x86_64",
        project_root=tmp_path,
    )
    environment = capability.augment_environment(
        {"PATH": "existing", "PYTHONPATH": "python-existing"}
    )
    assert str(capability.descriptor.runtime_library_root) in environment["PATH"]
    assert str(capability.descriptor.python_module_root) in environment["PYTHONPATH"]
    assert environment["LD_LIBRARY_PATH"] == str(
        capability.descriptor.runtime_library_root
    )


def test_reference_backend_doctor_never_requires_source_assets(tmp_path) -> None:
    extension = (
        tmp_path
        / "external/Falcor/build/windows-vs2022/bin/Release/python/falcor"
        / "falcor_ext.cp310-win_amd64.pyd"
    )
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"fixture-extension")
    layer_stack = next(
        value
        for value in discover_reference_programs()
        if value.descriptor.program_key == "ncls.layer-stack-random-walk"
    )
    backend = create_reference_backend(
        platform_name="win32", machine="x86_64", project_root=tmp_path
    )
    report = backend.doctor((layer_stack,))
    assert report.ready
    assert not (tmp_path / "assets").exists()
    assert len(report.descriptor.semantic_identity) == 64
    assert len(report.descriptor.build_identity) == 64
    assert len(report.descriptor.identity) == 64


def test_reference_backend_doctor_reports_program_provider_separately(
    tmp_path,
) -> None:
    mdl = next(
        value
        for value in discover_reference_programs()
        if value.descriptor.program_key == "ncls.mdl-vmaterials2"
    )
    report = create_reference_backend(
        platform_name="linux", machine="x86_64", project_root=tmp_path
    ).doctor((mdl,))
    statuses = {value.requirement_id: value for value in report.statuses}
    assert statuses["mdl-sdk"].category == "program-provider"
    assert statuses["mdl-sdk"].status == "missing"
    assert "stb" in statuses


def test_upper_layers_do_not_own_platform_or_legacy_falcor_details() -> None:
    roots = (
        PROJECT_ROOT / "src/ncls/learning",
        PROJECT_ROOT / "src/ncls/cli.py",
        PROJECT_ROOT / "src/ncls/references/query.py",
    )
    files = tuple(
        path
        for root in roots
        for path in ((root,) if root.is_file() else root.rglob("*.py"))
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for forbidden in (
        "ReferenceQueryDispatcher",
        "create_falcor_device",
        "import_falcor",
        "DeviceType.D3D12",
        "DeviceType.Vulkan",
        "windows-vs2022",
        "linux-gcc",
        "sys.platform",
    ):
        assert forbidden not in source
