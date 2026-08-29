from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform as host_platform
import sys
from typing import Any, Mapping

from ncls.core.identity import require_sha256, safe_relative_uri, sha256_file, sha256_json
from ncls.paths import PROJECT_ROOT


REFERENCE_BACKEND_MANIFEST = PROJECT_ROOT / "references/reference-backend-toolchains.json"
REFERENCE_BACKEND_SCHEMA = "ncls.reference-backend-toolchains@1"
CANONICAL_PROGRAMS = {
    ("ncls.layer-stack-random-walk", 1),
    ("ncls.merl-brdf", 1),
    ("ncls.openpbr", 1),
    ("ncls.materialx-polyhaven", 1),
    ("ncls.mdl-vmaterials2", 1),
}


def _fields(name: str, value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields must be exactly {sorted(expected)}")


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _project_relative(name: str, value: object) -> str:
    try:
        result = safe_relative_uri(str(value))
    except ValueError as error:
        raise ValueError(f"unsafe reference backend {name}: {value!r}") from error
    if result == "assets" or result.startswith("assets/"):
        raise ValueError("reference backend build inputs must not address source assets")
    return result


def normalize_reference_architecture(machine: str) -> str:
    value = machine.strip().lower()
    if value in {"amd64", "x86_64"}:
        return "x86_64"
    raise RuntimeError(f"reference backend does not support architecture {machine!r}")


def current_reference_platform_id(
    *, platform_name: str | None = None, machine: str | None = None
) -> str:
    os_name = platform_name or sys.platform
    architecture = normalize_reference_architecture(machine or host_platform.machine())
    if os_name == "win32":
        return f"windows-{architecture}@1"
    if os_name.startswith("linux"):
        return f"linux-{architecture}@1"
    raise RuntimeError(f"reference backend does not support platform {os_name!r}")


@dataclass(frozen=True)
class GitSourceProvider:
    provider_id: str
    path: str
    url: str
    revision: str
    submodules: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GitSourceProvider":
        _fields(
            "reference source provider",
            value,
            {"provider_id", "path", "url", "revision", "submodules"},
        )
        provider_id = str(value["provider_id"])
        path = _project_relative("provider path", value["path"])
        if not path.startswith("external/"):
            raise ValueError("reference source providers must live below external/")
        url = str(value["url"])
        if not url.startswith("https://github.com/") or not url.endswith(".git"):
            raise ValueError("reference source provider URL must be an HTTPS GitHub clone URL")
        revision = str(value["revision"])
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("reference source provider revision must be a full lowercase Git hash")
        submodules = str(value["submodules"])
        if submodules not in {"none", "upstream-locked", "recursive-upstream-locked"}:
            raise ValueError("reference source provider has an unsupported submodule policy")
        if not provider_id:
            raise ValueError("reference source provider id is required")
        return cls(provider_id, path, url, revision, submodules)


@dataclass(frozen=True)
class BinaryArchive:
    name: str
    archive_type: str
    url: str
    size: int
    sha256: str
    root: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BinaryArchive":
        _fields(
            "reference binary archive",
            value,
            {"name", "type", "url", "size", "sha256", "root"},
        )
        archive_type = str(value["type"])
        if archive_type not in {"zip", "tar.gz"}:
            raise ValueError("reference binary archive type must be zip or tar.gz")
        url = str(value["url"])
        if not url.startswith("https://github.com/NVIDIA/MDL-SDK/releases/download/"):
            raise ValueError("MDL SDK archive must use the official HTTPS release URL")
        size = int(value["size"])
        if size < 1:
            raise ValueError("reference binary archive size must be positive")
        return cls(
            str(value["name"]),
            archive_type,
            url,
            size,
            require_sha256("reference binary archive", str(value["sha256"])),
            _project_relative("archive root", value["root"]),
        )


@dataclass(frozen=True)
class FalcorPlatformLayout:
    device_api: str
    build_root: str
    runtime_library_root: str
    python_module_root: str
    python_extension: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FalcorPlatformLayout":
        _fields(
            "Falcor platform layout",
            value,
            {
                "device_api",
                "build_root",
                "runtime_library_root",
                "python_module_root",
                "python_extension",
            },
        )
        device_api = str(value["device_api"])
        if device_api not in {"d3d12", "vulkan"}:
            raise ValueError("reference backend device API must be d3d12 or vulkan")
        return cls(
            device_api,
            _project_relative("Falcor build root", value["build_root"]),
            _project_relative("Falcor runtime root", value["runtime_library_root"]),
            _project_relative("Falcor Python root", value["python_module_root"]),
            _project_relative("Falcor Python extension", value["python_extension"]),
        )


@dataclass(frozen=True)
class MdlSdkLayout:
    build: str
    archive: BinaryArchive
    library: str
    plugins: tuple[str, ...]
    runtime_library_directory: str
    target_code_types: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MdlSdkLayout":
        _fields(
            "MDL SDK layout",
            value,
            {
                "build",
                "archive",
                "library",
                "plugins",
                "runtime_library_directory",
                "target_code_types",
            },
        )
        plugins_value = value["plugins"]
        if not isinstance(plugins_value, list) or not plugins_value:
            raise ValueError("MDL SDK layout requires plugins")
        plugins = tuple(_project_relative("MDL plugin", item) for item in plugins_value)
        if len(set(plugins)) != len(plugins):
            raise ValueError("MDL SDK plugins must be unique")
        return cls(
            str(value["build"]),
            BinaryArchive.from_dict(_mapping("MDL archive", value["archive"])),
            _project_relative("MDL library", value["library"]),
            plugins,
            _project_relative("MDL runtime directory", value["runtime_library_directory"]),
            _project_relative("MDL target-code types", value["target_code_types"]),
        )


@dataclass(frozen=True)
class MdlBridgeLayout:
    executable: str
    generator: str
    configuration: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MdlBridgeLayout":
        _fields(
            "MDL bridge layout",
            value,
            {"executable", "generator", "configuration"},
        )
        configuration = str(value["configuration"])
        if configuration != "Release":
            raise ValueError("MDL bridge manifest describes only the Release artifact")
        return cls(
            _project_relative("MDL bridge executable", value["executable"]),
            str(value["generator"]),
            configuration,
        )


@dataclass(frozen=True)
class ReferencePlatformToolchain:
    platform_id: str
    os: str
    architecture: str
    falcor: FalcorPlatformLayout
    mdl_sdk: MdlSdkLayout
    mdl_bridge: MdlBridgeLayout

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReferencePlatformToolchain":
        _fields(
            "reference platform toolchain",
            value,
            {"platform_id", "os", "architecture", "falcor", "mdl_sdk", "mdl_bridge"},
        )
        os_name = str(value["os"])
        architecture = str(value["architecture"])
        if os_name not in {"windows", "linux"} or architecture != "x86_64":
            raise ValueError("reference backend supports windows/linux x86_64 only")
        platform_id = str(value["platform_id"])
        if platform_id != f"{os_name}-{architecture}@1":
            raise ValueError("reference platform id disagrees with OS/architecture")
        falcor = FalcorPlatformLayout.from_dict(_mapping("Falcor layout", value["falcor"]))
        if falcor.device_api != ("d3d12" if os_name == "windows" else "vulkan"):
            raise ValueError("reference platform device API disagrees with OS")
        return cls(
            platform_id,
            os_name,
            architecture,
            falcor,
            MdlSdkLayout.from_dict(_mapping("MDL SDK", value["mdl_sdk"])),
            MdlBridgeLayout.from_dict(_mapping("MDL bridge", value["mdl_bridge"])),
        )

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "device_api": self.falcor.device_api,
            "falcor_build_root": self.falcor.build_root,
            "mdl_sdk_archive_sha256": self.mdl_sdk.archive.sha256,
            "mdl_bridge_executable": self.mdl_bridge.executable,
        }


@dataclass(frozen=True)
class ProgramBuildRequirements:
    program_key: str
    version: int
    providers: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProgramBuildRequirements":
        _fields("reference program requirements", value, {"program_key", "version", "providers"})
        providers_value = value["providers"]
        if not isinstance(providers_value, list):
            raise ValueError("reference program providers must be a list")
        providers = tuple(str(item) for item in providers_value)
        if len(set(providers)) != len(providers):
            raise ValueError("reference program providers must be unique")
        return cls(str(value["program_key"]), int(value["version"]), providers)


@dataclass(frozen=True)
class ReferenceBackendManifest:
    backend_key: str
    backend_version: int
    falcor_revision: str
    slang_revision: str
    execution_provider: str
    asset_policy: str
    source_providers: tuple[GitSourceProvider, ...]
    platforms: tuple[ReferencePlatformToolchain, ...]
    programs: tuple[ProgramBuildRequirements, ...]
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: Path = REFERENCE_BACKEND_MANIFEST) -> "ReferenceBackendManifest":
        resolved = path.resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("reference backend manifest must be an object")
        _fields(
            "reference backend manifest",
            value,
            {
                "schema_name",
                "schema_version",
                "asset_policy",
                "backend",
                "source_providers",
                "platforms",
                "programs",
            },
        )
        if (
            value["schema_name"] != "ncls.reference-backend-toolchains"
            or int(value["schema_version"]) != 1
        ):
            raise ValueError("unsupported reference backend manifest schema")
        if value["asset_policy"] != "external-only-no-source-assets":
            raise ValueError("reference backend manifest must not manage source assets")
        backend = _mapping("reference backend", value["backend"])
        _fields(
            "reference backend",
            backend,
            {"backend_key", "version", "falcor_revision", "slang_revision", "execution_provider"},
        )
        sources_value = value["source_providers"]
        platforms_value = value["platforms"]
        programs_value = value["programs"]
        if not all(isinstance(item, list) for item in (sources_value, platforms_value, programs_value)):
            raise ValueError("reference backend providers/platforms/programs must be lists")
        sources = tuple(GitSourceProvider.from_dict(_mapping("source provider", item)) for item in sources_value)
        platforms = tuple(ReferencePlatformToolchain.from_dict(_mapping("platform", item)) for item in platforms_value)
        programs = tuple(ProgramBuildRequirements.from_dict(_mapping("program", item)) for item in programs_value)
        cls._require_unique("source provider ids", [item.provider_id for item in sources])
        cls._require_unique("source provider paths", [item.path for item in sources])
        cls._require_unique("platform ids", [item.platform_id for item in platforms])
        cls._require_unique("program identities", [(item.program_key, item.version) for item in programs])
        if {item.platform_id for item in platforms} != {
            "windows-x86_64@1",
            "linux-x86_64@1",
        }:
            raise ValueError("reference backend manifest requires Windows and Linux x86_64")
        if {(item.program_key, item.version) for item in programs} != CANONICAL_PROGRAMS:
            raise ValueError("reference backend manifest must cover the five canonical programs")
        provider_ids = {item.provider_id for item in sources} | {"mdl-sdk"}
        if str(backend["execution_provider"]) not in provider_ids:
            raise ValueError("reference backend execution provider is unknown")
        unknown = {
            provider
            for program in programs
            for provider in program.providers
            if provider not in provider_ids
        }
        if unknown:
            raise ValueError(f"reference programs use unknown providers: {sorted(unknown)}")
        sdk_builds = {item.mdl_sdk.build for item in platforms}
        if len(sdk_builds) != 1:
            raise ValueError("Windows/Linux MDL SDK builds must agree")
        return cls(
            str(backend["backend_key"]),
            int(backend["version"]),
            str(backend["falcor_revision"]),
            str(backend["slang_revision"]),
            str(backend["execution_provider"]),
            str(value["asset_policy"]),
            sources,
            platforms,
            programs,
            resolved,
            sha256_file(resolved),
        )

    @staticmethod
    def _require_unique(name: str, values: list[object]) -> None:
        if len(set(values)) != len(values):
            raise ValueError(f"reference backend {name} must be unique")

    def for_platform(self, platform_id: str) -> ReferencePlatformToolchain:
        try:
            return next(item for item in self.platforms if item.platform_id == platform_id)
        except StopIteration as error:
            raise RuntimeError(f"no reference backend toolchain for {platform_id!r}") from error

    def for_program(self, program_key: str, version: int) -> ProgramBuildRequirements:
        try:
            return next(
                item
                for item in self.programs
                if item.program_key == program_key and item.version == version
            )
        except StopIteration as error:
            raise RuntimeError(f"no build requirements for {program_key}@{version}") from error

    @property
    def semantic_identity(self) -> str:
        return sha256_json(
            {
                "schema": REFERENCE_BACKEND_SCHEMA,
                "manifest_sha256": self.sha256,
                "backend_key": self.backend_key,
                "backend_version": self.backend_version,
            }
        )


def load_reference_backend_manifest(
    path: Path = REFERENCE_BACKEND_MANIFEST,
) -> ReferenceBackendManifest:
    return ReferenceBackendManifest.load(path)


__all__ = [
    "CANONICAL_PROGRAMS",
    "REFERENCE_BACKEND_MANIFEST",
    "REFERENCE_BACKEND_SCHEMA",
    "BinaryArchive",
    "GitSourceProvider",
    "MdlBridgeLayout",
    "MdlSdkLayout",
    "ProgramBuildRequirements",
    "ReferenceBackendManifest",
    "ReferencePlatformToolchain",
    "current_reference_platform_id",
    "load_reference_backend_manifest",
    "normalize_reference_architecture",
]
