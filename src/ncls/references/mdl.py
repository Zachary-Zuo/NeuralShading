from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Protocol
import uuid

from ncls.core.identity import canonical_json, require_sha256, sha256_file, sha256_json
from ncls.paths import PROJECT_ROOT
from ncls.references.backend_manifest import (
    REFERENCE_BACKEND_MANIFEST,
    ReferencePlatformToolchain,
    current_reference_platform_id,
    load_reference_backend_manifest,
)


_BACKEND_MANIFEST = load_reference_backend_manifest()
MDL_SDK_BUILD = _BACKEND_MANIFEST.platforms[0].mdl_sdk.build
ARTIFACT_SCHEMA = "ncls.mdl-compiled-artifact@1"
DISCOVERY_SCHEMA = "ncls.mdl-module-discovery@1"
STB_COMMIT = "013ac3beddff3dbffafd5177e7972067cd2b5083"
STB_IMAGE_SHA256 = "594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3"
CODEGEN_OPTIONS = {
    "compile_constants": True,
    "df_handle_slot_mode": "none",
    "enable_auxiliary": True,
    "fast_math": True,
    "fold_all_bool_parameters": False,
    "fold_all_enum_parameters": False,
    "fold_ternary_on_df": False,
    "ignore_noinline": True,
    "internal_space_request": "coordinate_world (rejected by pinned HLSL backend)",
    "num_texture_results": 16,
    "num_texture_spaces": 4,
    "opt_level": 2,
    "texture_runtime_with_derivs": False,
    "use_renderer_adapt_normal": True,
}


@dataclass(frozen=True)
class MdlProgramProviderOverrides:
    """测试或显式部署注入；上层 source/reference 不直接使用。"""

    platform_id: str | None = None
    sdk_root: Path | None = None
    executable: Path | None = None
    cache_root: Path | None = None
    manifest_path: Path = REFERENCE_BACKEND_MANIFEST


@dataclass(frozen=True)
class MdlProgramToolchainDescriptor:
    platform_id: str
    sdk_build: str
    sdk_root: Path
    sdk_library: Path
    plugin_libraries: tuple[Path, ...]
    runtime_library_root: Path
    target_code_types: Path
    bridge_executable: Path
    cache_root: Path
    archive_sha256: str
    backend_manifest_sha256: str
    semantic_identity: str
    build_identity: str
    bridge_executable_sha256: str | None


@dataclass(frozen=True)
class MdlCapabilityReport:
    platform_id: str
    ready: bool
    missing: tuple[str, ...]
    semantic_identity: str
    build_identity: str


class MdlProgramProvider(Protocol):
    descriptor: MdlProgramToolchainDescriptor
    module_root: Path
    cache_root: Path

    def discover_module(self, module: str, *, output: Path) -> "MdlModuleDiscovery": ...

    def inspect(
        self,
        module: str,
        material: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        output: Path,
    ) -> "MdlCompiledArtifact": ...

    def compile_snapshot(self, snapshot: "SourceSnapshot") -> "MdlCompiledArtifact": ...


def _toolchain_semantic_identity(
    backend_manifest_sha256: str,
    record: ReferencePlatformToolchain,
) -> str:
    return sha256_json(
        {
            "schema": "ncls.mdl-compiler-semantic@1",
            "sdk_build": MDL_SDK_BUILD,
            "backend_manifest_sha256": backend_manifest_sha256,
            "portable_bridge": {
                "main_cpp": sha256_file(
                    PROJECT_ROOT / "tools/reference/mdl_sdk_bridge/main.cpp"
                ),
                "cmake": sha256_file(
                    PROJECT_ROOT / "tools/reference/mdl_sdk_bridge/CMakeLists.txt"
                ),
            },
            "artifact_schema": sha256_file(
                PROJECT_ROOT
                / "references/mdl-vmaterials2-v1/schemas/mdl-compiled-artifact.schema.json"
            ),
            "stb": {
                "commit": STB_COMMIT,
                "stb_image_sha256": STB_IMAGE_SHA256,
            },
            "codegen_options": CODEGEN_OPTIONS,
            "target_code_types_path": record.mdl_sdk.target_code_types,
        }
    )


def resolve_mdl_program_toolchain(
    overrides: MdlProgramProviderOverrides | None = None,
) -> MdlProgramToolchainDescriptor:
    values = overrides or MdlProgramProviderOverrides()
    manifest = load_reference_backend_manifest(values.manifest_path)
    platform_id = values.platform_id or current_reference_platform_id()
    record = manifest.for_platform(platform_id)
    sdk = record.mdl_sdk
    if sdk.build != MDL_SDK_BUILD:
        raise ValueError("reference backend manifest changed the pinned MDL SDK build")
    sdk_root = (
        values.sdk_root or PROJECT_ROOT / sdk.archive.root
    ).resolve()
    executable = (
        values.executable or PROJECT_ROOT / record.mdl_bridge.executable
    ).resolve()
    cache_root = (
        values.cache_root or PROJECT_ROOT / "build/mdl-reference/cache"
    ).resolve()
    executable_sha256 = sha256_file(executable) if executable.is_file() else None
    semantic_identity = _toolchain_semantic_identity(
        manifest.sha256, record
    )
    build_identity = sha256_json(
        {
            "schema": "ncls.mdl-compiler-build@1",
            "platform": record.to_identity_dict(),
            "bridge_executable_sha256": executable_sha256,
        }
    )
    return MdlProgramToolchainDescriptor(
        platform_id,
        sdk.build,
        sdk_root,
        sdk_root / sdk.library,
        tuple(sdk_root / path for path in sdk.plugins),
        sdk_root / sdk.runtime_library_directory,
        sdk_root / sdk.target_code_types,
        executable,
        cache_root,
        sdk.archive.sha256,
        manifest.sha256,
        semantic_identity,
        build_identity,
        executable_sha256,
    )


def _contained(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"MDL artifact path escapes its root: {path}") from error
    return resolved


def _require_hex(name: str, value: object, length: int) -> str:
    result = str(value)
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} is not a {length}-character lowercase hexadecimal value")
    return result


def _argument_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping) and "path" in value:
        path = Path(str(value["path"]))
        gamma = float(value.get("effective_gamma", 1.0))
        if path.is_absolute() or ".." in path.parts or "|" in path.as_posix():
            raise ValueError("MDL texture argument must use a pack-relative path")
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("MDL texture argument gamma must be finite and positive")
        return f"/{path.as_posix()}|{gamma:.17g}"
    if isinstance(value, Mapping) and set(value) >= {"name", "value"}:
        return str(value["name"])
    if isinstance(value, (tuple, list)):
        return ",".join(_argument_text(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("MDL arguments must be finite")
    return str(value)


@dataclass(frozen=True)
class MdlCompiledArtifact:
    root: Path
    manifest: Mapping[str, Any]
    source_snapshot_id: str | None = None

    @classmethod
    def load(cls, root: Path, *, source_snapshot_id: str | None = None) -> "MdlCompiledArtifact":
        resolved = root.resolve()
        manifest_path = _contained(resolved, resolved / "manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"MDL compiled artifact has no manifest: {resolved}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("unsupported MDL compiled artifact schema")
        if manifest.get("mdl_sdk") != MDL_SDK_BUILD:
            raise ValueError("MDL compiled artifact was produced by a different SDK build")
        texture_payloads = manifest.get("texture_payloads")
        if texture_payloads not in {"decoded", "metadata-only"}:
            raise ValueError("MDL compiled artifact has an invalid texture payload role")
        capability_audit = manifest.get("capability_audit")
        if not isinstance(capability_audit, Mapping) or capability_audit != {
            "surface_bsdf_evaluate": True,
            "emission": False,
            "volume": False,
            "displacement": False,
            "cutout_opacity": bool(capability_audit.get("cutout_opacity", False)),
        }:
            raise ValueError("MDL compiled artifact has no valid V1 capability audit")
        _require_hex("MDL compiled material hash", manifest.get("compiled_material_hash"), 32)
        sub_expression_hashes = manifest.get("sub_expression_hashes")
        if not isinstance(sub_expression_hashes, Mapping) or set(sub_expression_hashes) != {
            "surface.scattering",
            "geometry.normal",
            "geometry.cutout_opacity",
        }:
            raise ValueError("MDL compiled artifact has no complete sub-expression hashes")
        for name, value in sub_expression_hashes.items():
            _require_hex(f"MDL {name} hash", value, 32)
        compiler_identity = manifest.get("compiler_identity")
        bridge_digest = (
            str(compiler_identity.get("bridge_executable_sha256", ""))
            if isinstance(compiler_identity, Mapping)
            else ""
        )
        if (
            not isinstance(compiler_identity, Mapping)
            or compiler_identity.get("mdl_sdk") != MDL_SDK_BUILD
            or compiler_identity.get("stb_commit") != STB_COMMIT
            or compiler_identity.get("stb_image_sha256") != STB_IMAGE_SHA256
            or compiler_identity.get("codegen_options") != CODEGEN_OPTIONS
        ):
            raise ValueError("MDL compiled artifact has an invalid compiler identity")
        try:
            _require_hex("MDL bridge executable hash", bridge_digest, 64)
            _require_hex(
                "MDL semantic compiler identity",
                compiler_identity.get("semantic_identity"),
                64,
            )
            _require_hex(
                "MDL build compiler identity",
                compiler_identity.get("build_identity"),
                64,
            )
            _require_hex(
                "reference backend manifest hash",
                compiler_identity.get("backend_manifest_sha256"),
                64,
            )
            _require_hex(
                "MDL SDK archive hash",
                compiler_identity.get("sdk_archive_sha256"),
                64,
            )
        except ValueError as error:
            raise ValueError("MDL compiled artifact has an invalid compiler identity") from error
        if compiler_identity.get("platform_id") not in {
            "windows-x86_64@1",
            "linux-x86_64@1",
        }:
            raise ValueError("MDL compiled artifact has an invalid compiler identity")
        if manifest.get("diagnostics"):
            raise ValueError(f"MDL compiled artifact contains diagnostics: {manifest['diagnostics']}")
        declared_files = manifest.get("files_sha256")
        if not isinstance(declared_files, dict) or not declared_files:
            raise ValueError("MDL compiled artifact has no finalized file hash table")
        actual_files = {
            path.relative_to(resolved).as_posix(): path
            for path in resolved.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if set(declared_files) != set(actual_files):
            raise ValueError("MDL compiled artifact file set differs from its manifest")
        for relative, path in actual_files.items():
            if sha256_file(path) != str(declared_files[relative]):
                raise ValueError(f"MDL compiled artifact file hash mismatch: {relative}")
        code = _contained(resolved, resolved / str(manifest.get("code", "")))
        if not code.is_file():
            raise ValueError("MDL compiled artifact is missing generated HLSL")
        argument_block = manifest.get("argument_block")
        if argument_block is not None:
            block = _contained(resolved, resolved / str(argument_block.get("path", "")))
            if not block.is_file() or block.stat().st_size != int(argument_block.get("size", -1)):
                raise ValueError("MDL argument block is missing or has the wrong size")
        for segment in manifest.get("ro_data", []):
            path = _contained(resolved, resolved / str(segment.get("path", "")))
            if not path.is_file() or path.stat().st_size != int(segment.get("size", -1)):
                raise ValueError("MDL read-only data segment is missing or has the wrong size")
        textures = manifest.get("textures", [])
        if [int(texture.get("index", -1)) for texture in textures] != list(range(1, len(textures) + 1)):
            raise ValueError("MDL texture indices must be contiguous and one-based")
        for texture in textures:
            if texture.get("shape") not in {"2d", "bsdf_data"}:
                raise ValueError(f"unsupported MDL texture shape: {texture.get('shape')}")
            data = texture.get("data")
            if data is not None:
                data_path = _contained(resolved, resolved / str(data))
                if not data_path.is_file():
                    raise ValueError("MDL texture payload is missing")
                texel_count = (
                    int(texture.get("width", 0))
                    * int(texture.get("height", 0))
                    * int(texture.get("depth", 0))
                )
                bytes_per_texel = {
                    "Sint8": 1,
                    "Rgb": 3,
                    "Rgba": 4,
                    "Rgb_16": 6,
                    "Rgba_16": 8,
                    "Float32": 4,
                    "Float32<2>": 8,
                    "Float32<3>": 12,
                    "Float32<4>": 16,
                    "Rgb_fp": 12,
                    "Color": 16,
                }.get(str(texture.get("pixel_type")))
                if bytes_per_texel is None or data_path.stat().st_size != texel_count * bytes_per_texel:
                    raise ValueError("MDL texture payload has the wrong pixel type or size")
            path = texture.get("path")
            if texture.get("shape") == "2d" and (not path or not Path(str(path)).is_file()):
                raise ValueError("MDL 2D texture resource is missing")
            if texture_payloads == "decoded" and data is None:
                raise ValueError("MDL runtime artifact is missing a decoded texture payload")
            if texture_payloads == "metadata-only" and texture.get("shape") == "2d" and data is not None:
                raise ValueError("MDL inspection artifact unexpectedly contains a decoded 2D texture")
            if data is not None and texture.get("shape") == "2d" and texture.get("data_origin") not in {
                "top_left",
                "lower_left",
            }:
                raise ValueError("MDL 2D texture has no supported decoded-data origin")
        return cls(resolved, manifest, source_snapshot_id)

    @property
    def hlsl(self) -> str:
        return (self.root / str(self.manifest["code"])).read_text(encoding="utf-8")

    @property
    def argument_block(self) -> bytes:
        descriptor = self.manifest.get("argument_block")
        return b"" if descriptor is None else (self.root / str(descriptor["path"])).read_bytes()

    @property
    def artifact_sha256(self) -> str:
        files: dict[str, str] = {}
        for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
            files[path.relative_to(self.root).as_posix()] = sha256_file(path)
        return sha256_json(files)

    @property
    def runtime_supported(self) -> bool:
        return not bool(self.manifest["capability_audit"]["cutout_opacity"])

    def require_runtime_supported(self) -> None:
        if not self.runtime_supported:
            raise ValueError(
                "MDL runtime does not support non-opaque geometry.cutout_opacity"
            )
        if self.manifest["texture_payloads"] != "decoded":
            raise ValueError("MDL inspection artifact has no decoded runtime texture payloads")


@dataclass(frozen=True)
class MdlModuleDiscovery:
    root: Path
    module: str
    materials: tuple[str, ...]
    bridge_executable_sha256: str

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        expected_module: str | None = None,
        expected_bridge_sha256: str | None = None,
    ) -> "MdlModuleDiscovery":
        resolved = root.resolve()
        path = _contained(resolved, resolved / "discovery.json")
        if not path.is_file():
            raise ValueError(f"MDL module discovery has no document: {resolved}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != DISCOVERY_SCHEMA or value.get("mdl_sdk") != MDL_SDK_BUILD:
            raise ValueError("unsupported MDL module discovery document")
        if not isinstance(value.get("diagnostics"), str):
            raise ValueError("MDL module discovery diagnostics must be text")
        module = str(value.get("module", ""))
        if expected_module is not None and module != expected_module:
            raise ValueError("MDL module discovery identity mismatch")
        materials_value = value.get("materials")
        if not isinstance(materials_value, list) or not materials_value:
            raise ValueError("MDL module discovery has no materials")
        materials = tuple(str(item) for item in materials_value)
        if materials != tuple(sorted(set(materials))):
            raise ValueError("MDL module discovery materials must be sorted and unique")
        if any(not item.startswith(module + "::") or "(" not in item for item in materials):
            raise ValueError("MDL module discovery contains an invalid exact export")
        bridge_digest = _require_hex(
            "MDL discovery bridge executable hash",
            value.get("bridge_executable_sha256"),
            64,
        )
        if expected_bridge_sha256 is not None and bridge_digest != expected_bridge_sha256:
            raise ValueError("MDL module discovery was produced by another bridge executable")
        return cls(resolved, module, materials, bridge_digest)


class MdlSdkProgramProvider:
    """锁定 MDL SDK 的内部 program provider；正式 reference 不依赖 falcor2。"""

    def __init__(
        self,
        module_root: Path,
        *,
        overrides: MdlProgramProviderOverrides | None = None,
    ) -> None:
        self.module_root = module_root.resolve()
        self.descriptor = resolve_mdl_program_toolchain(overrides)
        self.sdk_root = self.descriptor.sdk_root
        self.executable = self.descriptor.bridge_executable
        self.cache_root = self.descriptor.cache_root
        report = self.preflight()
        if not report.ready:
            raise FileNotFoundError(
                "MDL compiler capability is incomplete for "
                f"{report.platform_id}: {', '.join(report.missing)}"
            )

    def preflight(self) -> MdlCapabilityReport:
        descriptor = self.descriptor
        required = (
            self.module_root,
            descriptor.sdk_root,
            descriptor.sdk_library,
            *descriptor.plugin_libraries,
            descriptor.target_code_types,
            descriptor.bridge_executable,
        )
        missing = tuple(str(path) for path in required if not path.exists())
        return MdlCapabilityReport(
            descriptor.platform_id,
            not missing,
            missing,
            descriptor.semantic_identity,
            descriptor.build_identity,
        )

    @property
    def compiler_identity(self) -> dict[str, Any]:
        descriptor = self.descriptor
        bridge_digest = descriptor.bridge_executable_sha256
        if bridge_digest is None:
            raise RuntimeError("MDL bridge executable disappeared after capability resolution")
        return {
            "mdl_sdk": descriptor.sdk_build,
            "platform_id": descriptor.platform_id,
            "semantic_identity": descriptor.semantic_identity,
            "build_identity": descriptor.build_identity,
            "backend_manifest_sha256": descriptor.backend_manifest_sha256,
            "sdk_archive_sha256": descriptor.archive_sha256,
            "bridge_executable_sha256": bridge_digest,
            "stb_commit": STB_COMMIT,
            "stb_image_sha256": STB_IMAGE_SHA256,
            "codegen_options": CODEGEN_OPTIONS,
        }

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        runtime_root = str(self.descriptor.runtime_library_root)
        environment["PATH"] = runtime_root + os.pathsep + environment.get("PATH", "")
        if self.descriptor.platform_id.startswith("linux-"):
            current = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = runtime_root + (
                os.pathsep + current if current else ""
            )
        return environment

    def discover_module(self, module: str, *, output: Path) -> MdlModuleDiscovery:
        if output.exists():
            raise ValueError(f"MDL discovery output must be absent: {output}")
        command = [
            str(self.executable),
            "discover",
            "--sdk-root",
            str(self.sdk_root),
            "--sdk-library",
            str(self.descriptor.sdk_library),
            *(value for plugin in self.descriptor.plugin_libraries for value in ("--plugin", str(plugin))),
            "--module-root",
            str(self.module_root),
            "--module",
            module,
            "--output-dir",
            str(output.resolve()),
        ]
        environment = self._environment()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"MDL SDK bridge discovery failed: {message}")
        path = output / "discovery.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        bridge_digest = sha256_file(self.executable)
        value["bridge_executable_sha256"] = bridge_digest
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return MdlModuleDiscovery.load(
            output,
            expected_module=module,
            expected_bridge_sha256=bridge_digest,
        )

    def _run(
        self,
        module: str,
        material: str,
        arguments: Mapping[str, Any],
        output: Path,
        *,
        native_queries: Path | None = None,
        native_output: Path | None = None,
        metadata_only: bool = False,
    ) -> MdlCompiledArtifact:
        command = [
            str(self.executable),
            "native-evaluate" if native_queries is not None else "compile",
            "--sdk-root",
            str(self.sdk_root),
            "--sdk-library",
            str(self.descriptor.sdk_library),
            *(value for plugin in self.descriptor.plugin_libraries for value in ("--plugin", str(plugin))),
            "--module-root",
            str(self.module_root),
            "--material",
            material if material.startswith("::") else f"{module}::{material}",
            "--output-dir",
            str(output),
        ]
        if native_queries is not None:
            if native_output is None:
                raise ValueError("native result path is required")
            command.extend(("--native-queries", str(native_queries), "--native-output", str(native_output)))
        if metadata_only:
            command.append("--skip-texture-payloads")
        for name in sorted(arguments):
            command.extend(("--argument", f"{name}={_argument_text(arguments[name])}"))
        environment = self._environment()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"MDL SDK bridge failed: {message}")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["compiler_identity"] = self.compiler_identity
        manifest["files_sha256"] = {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return MdlCompiledArtifact.load(output)

    def inspect(
        self,
        module: str,
        material: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        output: Path,
    ) -> MdlCompiledArtifact:
        """V1 复用 class compilation 完成 authority-backed discovery。"""

        if output.exists():
            raise ValueError(f"MDL inspection output must be absent: {output}")
        return self._run(
            module,
            material,
            arguments or {},
            output.resolve(),
            metadata_only=True,
        )

    def native_evaluate(
        self,
        module: str,
        material: str,
        arguments: Mapping[str, Any],
        *,
        queries: Path,
        output: Path,
        result: Path,
    ) -> MdlCompiledArtifact:
        """验证专用的 SDK native backend；不进入正式 provider。"""

        if output.exists() or result.exists():
            raise ValueError("MDL native validation outputs must be absent")
        if not queries.is_file():
            raise FileNotFoundError(f"MDL native query packet is missing: {queries}")
        return self._run(
            module,
            material,
            arguments,
            output.resolve(),
            native_queries=queries.resolve(),
            native_output=result.resolve(),
        )

    def compile_snapshot(self, snapshot: "SourceSnapshot") -> MdlCompiledArtifact:
        from ncls.core.source import SourceSnapshot

        if not isinstance(snapshot, SourceSnapshot) or snapshot.family_id != "mdl.program@1":
            raise ValueError("MDL compiler bridge requires an mdl.program@1 snapshot")
        payload = json.loads(snapshot.native_payload.decode("utf-8"))
        if payload.get("schema") != "ncls.mdl-source@1":
            raise ValueError("unsupported MDL source payload")
        key = sha256_json(
            {
                "snapshot_id": snapshot.snapshot_id,
                "semantic_identity": self.descriptor.semantic_identity,
                "build_identity": self.descriptor.build_identity,
            }
        )
        target = self.cache_root / key
        if target.exists():
            return MdlCompiledArtifact.load(target, source_snapshot_id=snapshot.snapshot_id)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_root / f".{key}.{uuid.uuid4().hex}.partial"
        try:
            artifact = self._run(
                str(payload["module"]),
                str(payload["export"]),
                {
                    name: item["value"]
                    for name, item in payload.get("arguments", {}).items()
                    if item.get("editable", False)
                },
                temporary,
            )
            os.replace(temporary, target)
            return MdlCompiledArtifact.load(target, source_snapshot_id=snapshot.snapshot_id)
        except Exception:
            if temporary.is_dir():
                shutil.rmtree(temporary)
            raise


def create_mdl_program_provider(
    module_root: Path,
    *,
    overrides: MdlProgramProviderOverrides | None = None,
) -> MdlProgramProvider:
    return MdlSdkProgramProvider(module_root, overrides=overrides)


def canonical_mdl_payload(value: Mapping[str, Any]) -> bytes:
    return canonical_json(value).encode("utf-8")
