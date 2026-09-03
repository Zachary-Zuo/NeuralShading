from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from ncls.core.identity import sha256_file, sha256_json
from ncls.core.scattering import ReferenceProgramDefinition
from ncls.core.source import SourceSnapshot
from ncls.paths import PROJECT_ROOT
from ncls.references.backend_manifest import (
    ReferenceBackendManifest,
    ReferencePlatformToolchain,
    current_reference_platform_id,
    load_reference_backend_manifest,
)
from ncls.references.programs import discover_reference_programs


QUERY_SHADER = PROJECT_ROOT / "shaders/ncls/reference_query/reference_query.cs.slang"
_FALCOR_DEVICE_CACHE: dict[tuple[object, str, int], object] = {}
_SOFTWARE_ADAPTER_NAMES = ("llvmpipe", "lavapipe", "microsoft basic render", "warp")


@dataclass(frozen=True)
class ReferenceBackendDescriptor:
    backend_key: str
    version: int
    platform_id: str
    falcor_revision: str
    slang_revision: str
    device_api: str
    build_root: Path
    python_module_root: Path
    runtime_library_root: Path
    semantic_identity: str
    build_identity: str

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.reference-backend-identity@1",
                "semantic_identity": self.semantic_identity,
                "build_identity": self.build_identity,
            }
        )


@dataclass(frozen=True)
class ReferenceCapabilityStatus:
    requirement_id: str
    category: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.category not in {"execution", "program-provider", "environment"}:
            raise ValueError("unknown reference capability status category")
        if self.status not in {"ready", "missing", "invalid"}:
            raise ValueError("unknown reference capability status")


@dataclass(frozen=True)
class ReferenceBackendReport:
    descriptor: ReferenceBackendDescriptor
    statuses: tuple[ReferenceCapabilityStatus, ...]

    @property
    def ready(self) -> bool:
        return all(item.status == "ready" for item in self.statuses)

    def require_ready(self) -> None:
        failures = tuple(item for item in self.statuses if item.status != "ready")
        if failures:
            detail = "; ".join(
                f"{item.requirement_id}: {item.detail}" for item in failures
            )
            raise RuntimeError(f"reference backend is not ready: {detail}")


class ReferenceBackendCapability:
    """五种 canonical reference program 共用的唯一平台能力入口。"""

    def __init__(
        self,
        manifest: ReferenceBackendManifest,
        platform: ReferencePlatformToolchain,
        *,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.manifest = manifest
        self.platform = platform
        self.project_root = project_root.resolve()
        self.descriptor = self._create_descriptor()
        self._dll_directories: list[object] = []

    def _resolve(self, relative: str) -> Path:
        return self.project_root / Path(relative)

    def _python_extensions(self) -> tuple[Path, ...]:
        pattern = self.platform.falcor.python_extension.replace("/", os.sep)
        root = self._resolve(self.platform.falcor.python_module_root)
        return tuple(sorted(root.glob(pattern)))

    def _create_descriptor(self) -> ReferenceBackendDescriptor:
        falcor = self.platform.falcor
        extension_artifacts = [
            {
                "path": path.relative_to(self.project_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in self._python_extensions()
            if path.is_file()
        ]
        semantic_identity = sha256_json(
            {
                "schema": "ncls.reference-backend-semantics@1",
                "backend_key": self.manifest.backend_key,
                "backend_version": self.manifest.backend_version,
                "falcor_revision": self.manifest.falcor_revision,
                "slang_revision": self.manifest.slang_revision,
                "query_shader_sha256": sha256_file(QUERY_SHADER),
            }
        )
        build_identity = sha256_json(
            {
                "schema": "ncls.reference-backend-build@1",
                "manifest_sha256": self.manifest.sha256,
                "platform": self.platform.to_identity_dict(),
                "falcor_python_extensions": extension_artifacts,
            }
        )
        return ReferenceBackendDescriptor(
            self.manifest.backend_key,
            self.manifest.backend_version,
            self.platform.platform_id,
            self.manifest.falcor_revision,
            self.manifest.slang_revision,
            falcor.device_api,
            self._resolve(falcor.build_root),
            self._resolve(falcor.python_module_root),
            self._resolve(falcor.runtime_library_root),
            semantic_identity,
            build_identity,
        )

    @staticmethod
    def _path_status(
        requirement_id: str,
        category: str,
        path: Path,
        *,
        expected: str,
    ) -> ReferenceCapabilityStatus:
        exists = path.is_dir() if expected == "directory" else path.is_file()
        return ReferenceCapabilityStatus(
            requirement_id,
            category,
            "ready" if exists else "missing",
            str(path),
        )

    def _execution_statuses(self) -> tuple[ReferenceCapabilityStatus, ...]:
        extensions = self._python_extensions()
        extension_pattern = (
            self.descriptor.python_module_root
            / self.platform.falcor.python_extension
        )
        return (
            self._path_status(
                "falcor-build-root",
                "execution",
                self.descriptor.build_root,
                expected="directory",
            ),
            ReferenceCapabilityStatus(
                "falcor-python-extension",
                "execution",
                "ready" if extensions else "missing",
                ", ".join(str(path) for path in extensions)
                if extensions
                else str(extension_pattern),
            ),
            self._path_status(
                "falcor-runtime-library-root",
                "execution",
                self.descriptor.runtime_library_root,
                expected="directory",
            ),
        )

    def _provider_status(self, provider_id: str) -> ReferenceCapabilityStatus:
        if provider_id == "mdl-sdk":
            sdk = self.platform.mdl_sdk
            sdk_root = self._resolve(sdk.archive.root)
            required = (
                sdk_root / sdk.library,
                *(sdk_root / plugin for plugin in sdk.plugins),
                self._resolve(self.platform.mdl_bridge.executable),
            )
            missing = tuple(path for path in required if not path.is_file())
            return ReferenceCapabilityStatus(
                provider_id,
                "program-provider",
                "missing" if missing else "ready",
                ", ".join(str(path) for path in (missing or required)),
            )
        try:
            provider = next(
                item
                for item in self.manifest.source_providers
                if item.provider_id == provider_id
            )
        except StopIteration:
            return ReferenceCapabilityStatus(
                provider_id,
                "program-provider",
                "invalid",
                "provider is not declared by the backend manifest",
            )
        path = self._resolve(provider.path)
        return self._path_status(
            provider_id, "program-provider", path, expected="directory"
        )

    def doctor(
        self,
        programs: Sequence[ReferenceProgramDefinition] | None = None,
    ) -> ReferenceBackendReport:
        selected = tuple(programs) if programs is not None else discover_reference_programs()
        statuses = list(self._execution_statuses())
        provider_ids: set[str] = set()
        for definition in selected:
            requirements = self.manifest.for_program(
                definition.descriptor.program_key, definition.descriptor.version
            )
            provider_ids.update(requirements.providers)
        statuses.extend(self._provider_status(value) for value in sorted(provider_ids))
        statuses.extend(
            ReferenceCapabilityStatus(
                value.requirement_id,
                "program-provider",
                value.status,
                value.detail,
            )
            for definition in selected
            for value in definition.preflight_provider(
                platform_id=self.platform.platform_id,
                project_root=self.project_root,
            )
        )
        return ReferenceBackendReport(self.descriptor, tuple(statuses))

    def augment_environment(self, base: Mapping[str, str]) -> dict[str, str]:
        environment = dict(base)

        def prepend(name: str, value: Path) -> None:
            current = environment.get(name, "")
            environment[name] = str(value) + (os.pathsep + current if current else "")

        prepend("PATH", self.descriptor.runtime_library_root)
        prepend("PYTHONPATH", self.project_root / "src")
        prepend("PYTHONPATH", self.descriptor.python_module_root)
        if self.descriptor.device_api == "vulkan":
            prepend("LD_LIBRARY_PATH", self.descriptor.runtime_library_root)
        return environment

    def _load_falcor(self):
        module_root = str(self.descriptor.python_module_root)
        if module_root not in sys.path:
            sys.path.insert(0, module_root)
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            self._dll_directories.append(
                os.add_dll_directory(str(self.descriptor.runtime_library_root))
            )
        try:
            return importlib.import_module("falcor")
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Falcor Python runtime could not be loaded from the reference backend "
                f"descriptor: {self.descriptor.python_module_root}"
            ) from error

    def _create_device(self, falcor):
        device_type = {
            "d3d12": falcor.DeviceType.D3D12,
            "vulkan": falcor.DeviceType.Vulkan,
        }[self.descriptor.device_api]
        raw_gpu_index = os.environ.get("NCLS_FALCOR_GPU_INDEX", "0")
        try:
            gpu_index = int(raw_gpu_index)
        except ValueError as error:
            raise RuntimeError("NCLS_FALCOR_GPU_INDEX must be a nonnegative integer") from error
        if gpu_index < 0 or str(gpu_index) != raw_gpu_index:
            raise RuntimeError("NCLS_FALCOR_GPU_INDEX must be a nonnegative integer")
        cache_key = (falcor, self.descriptor.device_api, gpu_index)
        cached = _FALCOR_DEVICE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        device = falcor.Device(type=device_type, gpu=gpu_index)
        adapter_name = str(
            getattr(getattr(device, "info", None), "adapter_name", "")
        )
        if any(value in adapter_name.lower() for value in _SOFTWARE_ADAPTER_NAMES):
            raise RuntimeError(
                "reference backend requires a hardware graphics adapter; "
                f"Falcor selected {adapter_name!r}"
            )
        _FALCOR_DEVICE_CACHE[cache_key] = device
        return device

    def open(
        self,
        plan: "ReferenceExecutionPlan",
        *,
        query_capacity: int,
        device: object = "cuda:0",
        slot_count: int = 2,
        max_resident_groups: int = 8,
        requested_operations: Sequence[str] = ("evaluate", "sample", "pdf"),
    ):
        from ncls.references.plan import ReferenceExecutionPlan

        if not isinstance(plan, ReferenceExecutionPlan):
            raise TypeError("reference backend open() requires ReferenceExecutionPlan@1")
        definitions = tuple(
            {group.definition.descriptor.program_key: group.definition for group in plan.groups}.values()
        )
        self.doctor(definitions).require_ready()
        falcor = self._load_falcor()
        device_handle = self._create_device(falcor)
        from ncls.references.query import ReferenceBackendSession

        return ReferenceBackendSession(
            plan,
            backend_descriptor=self.descriptor,
            falcor=falcor,
            device_handle=device_handle,
            query_capacity=query_capacity,
            device=device,
            slot_count=slot_count,
            max_resident_groups=max_resident_groups,
            requested_operations=requested_operations,
        )


def create_reference_backend(
    *,
    manifest: ReferenceBackendManifest | None = None,
    platform_id: str | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> ReferenceBackendCapability:
    selected_manifest = manifest or load_reference_backend_manifest()
    selected_platform_id = platform_id or current_reference_platform_id(
        platform_name=platform_name, machine=machine
    )
    return ReferenceBackendCapability(
        selected_manifest,
        selected_manifest.for_platform(selected_platform_id),
        project_root=project_root,
    )


__all__ = [
    "ReferenceBackendCapability",
    "ReferenceBackendDescriptor",
    "ReferenceBackendReport",
    "ReferenceCapabilityStatus",
    "create_reference_backend",
]
