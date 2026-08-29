from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.core.identity import sha256_json
from ncls.core.scattering import (
    BackendCapability,
    MaterialPayload,
    ReferenceProgramDescriptor,
    ReferenceProgramProviderStatus,
    RuntimePayload,
)
from ncls.core.source import SourceSnapshot
from ncls.references.mdl import (
    CODEGEN_OPTIONS,
    MDL_SDK_BUILD,
    MdlProgramProviderOverrides,
    STB_COMMIT,
    STB_IMAGE_SHA256,
    create_mdl_program_provider,
    resolve_mdl_program_toolchain,
)
from ncls.references.backend_manifest import load_reference_backend_manifest
from ncls.source_materials.mdl import MdlMaterialSource

from .base import FileReferenceProgram, PROJECT_ROOT, implementation_identity, slang_module_closure


BACKEND_SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl.slang"
RUNTIME_SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
_CAPABILITIES = (
    BackendCapability.PREPARE
    | BackendCapability.EVALUATE
    | BackendCapability.SAMPLE
    | BackendCapability.PDF
    | BackendCapability.ANISOTROPIC_FRAME
    | BackendCapability.REVERSE_PDF
    | BackendCapability.DELTA_EVENTS
    | BackendCapability.TRANSMISSION
)
_IMPLEMENTATION = sha256_json(
    {
        "files": implementation_identity(
            (
                Path(__file__),
                PROJECT_ROOT / "src/ncls/references/mdl.py",
                BACKEND_SHADER,
                RUNTIME_SHADER,
            )
        ),
        "mdl_sdk": MDL_SDK_BUILD,
        "falcor": "8.0-9dc819c162b2070335c65060436041690b7937f8",
        "slang": "2024.1.34",
        "stb": {"commit": STB_COMMIT, "stb_image_sha256": STB_IMAGE_SHA256},
        "codegen_options": CODEGEN_OPTIONS,
        "mdl_program_provider": "ncls.mdl-sdk-program-provider@1",
    }
)

_MDL_PIXEL_LAYOUTS = {
    "Sint8": (np.dtype(np.uint8), 1),
    "Rgb": (np.dtype(np.uint8), 3),
    "Rgba": (np.dtype(np.uint8), 4),
    "Rgb_16": (np.dtype(np.uint16), 3),
    "Rgba_16": (np.dtype(np.uint16), 4),
    "Float32": (np.dtype(np.float32), 1),
    "Float32<2>": (np.dtype(np.float32), 2),
    "Float32<3>": (np.dtype(np.float32), 3),
    "Float32<4>": (np.dtype(np.float32), 4),
    "Rgb_fp": (np.dtype(np.float32), 3),
    "Color": (np.dtype(np.float32), 4),
}


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= np.float32(0.04045),
        values / np.float32(12.92),
        ((values + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
    ).astype(np.float32, copy=False)


def _decoded_texture_binding(
    path: Path, descriptor: dict[str, object]
) -> tuple[str, bytes, dict[str, object]]:
    pixel_type = str(descriptor.get("pixel_type", ""))
    layout = _MDL_PIXEL_LAYOUTS.get(pixel_type)
    if layout is None:
        raise ValueError(f"unsupported MDL decoded texture pixel type: {pixel_type}")
    dtype, channels = layout
    width = int(descriptor["width"])
    height = int(descriptor["height"])
    payload = path.read_bytes()
    expected_size = width * height * channels * dtype.itemsize
    if len(payload) != expected_size:
        raise ValueError("MDL decoded texture payload has the wrong size")
    source = np.frombuffer(payload, dtype=dtype).reshape(height, width, channels)
    if channels == 1:
        encoded = source[..., 0].copy()
    elif channels == 4:
        encoded = source.copy()
    else:
        one = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
        encoded = np.zeros((height, width, 4), dtype=dtype)
        encoded[..., :channels] = source
        encoded[..., 3] = one
    origin = descriptor.get("data_origin")
    if origin == "lower_left":
        encoded = encoded[::-1].copy()
    elif origin != "top_left":
        raise ValueError("MDL decoded texture has an unsupported row origin")

    gamma = str(descriptor.get("gamma", "linear"))
    scalar = encoded.ndim == 2
    hardware_srgb = gamma == "srgb" and dtype == np.dtype(np.uint8) and not scalar
    color_space = "srgb" if hardware_srgb else "linear"
    if gamma == "srgb" and not hardware_srgb:
        normalized = encoded.astype(np.float32)
        if np.issubdtype(dtype, np.integer):
            normalized /= np.float32(np.iinfo(dtype).max)
        if normalized.ndim == 2:
            normalized = _srgb_to_linear(normalized)
        else:
            normalized[..., :3] = _srgb_to_linear(normalized[..., :3])
        encoded = normalized
        dtype = np.dtype(np.float32)

    scalar = encoded.ndim == 2
    if dtype == np.dtype(np.uint8):
        suffix, format_name = ("r8", "r8-unorm") if scalar else ("rgba8", "rgba8-unorm")
    elif dtype == np.dtype(np.uint16):
        suffix, format_name = ("r16", "r16-unorm") if scalar else ("rgba16", "rgba16-unorm")
    else:
        suffix, format_name = ("r32f", "r32-float") if scalar else ("rgba32f", "rgba32-float")
    encoded = np.ascontiguousarray(encoded)
    return (
        suffix,
        encoded.tobytes(),
        {
            "kind": "texture2d",
            "dtype": dtype.name,
            "shape": [height, width] if scalar else [height, width, 4],
            "stride": dtype.itemsize if scalar else 4 * dtype.itemsize,
            "alignment": dtype.itemsize,
            "format": format_name,
            "color_space": color_space,
        },
    )


class MdlReferenceProgram(FileReferenceProgram):
    shader = BACKEND_SHADER
    descriptor = ReferenceProgramDescriptor(
        "ncls.mdl-vmaterials2",
        1,
        "Native MDL SDK reference",
        "mdl.program@1",
        1,
        _IMPLEMENTATION,
        "ncls.scattering-backend@1",
        int(_CAPABILITIES),
        {
            "maximum_prepare_steps": 1,
            "maximum_evaluate_steps": 1,
            "maximum_state_bytes": 512,
            "maximum_reads": 128,
        },
    )

    def preflight_provider(
        self, *, platform_id: str, project_root: Path
    ) -> tuple[ReferenceProgramProviderStatus, ...]:
        manifest = load_reference_backend_manifest()
        platform = manifest.for_platform(platform_id)
        descriptor = resolve_mdl_program_toolchain(
            MdlProgramProviderOverrides(
                platform_id=platform_id,
                sdk_root=project_root / platform.mdl_sdk.archive.root,
                executable=project_root / platform.mdl_bridge.executable,
                cache_root=project_root / "build/mdl-reference/cache",
            )
        )
        required = {
            "mdl-sdk-library": descriptor.sdk_library,
            "mdl-target-code-types": descriptor.target_code_types,
            "mdl-program-provider": descriptor.bridge_executable,
            **{
                f"mdl-resource-plugin-{index}": path
                for index, path in enumerate(descriptor.plugin_libraries)
            },
        }
        return tuple(
            ReferenceProgramProviderStatus(
                name,
                "ready" if path.is_file() else "missing",
                str(path),
            )
            for name, path in required.items()
        )

    def compile_runtime(self) -> RuntimePayload:
        closure = slang_module_closure(BACKEND_SHADER)
        return RuntimePayload(
            BACKEND_SHADER.relative_to(PROJECT_ROOT).as_posix(),
            closure,
            {},
            {},
            int(_CAPABILITIES),
        )

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        self.validate_snapshot(snapshot)
        source = MdlMaterialSource.from_snapshot(snapshot)
        compiler = create_mdl_program_provider(source.module_root)
        artifact = compiler.compile_snapshot(snapshot)
        artifact.require_runtime_supported()
        texture_descriptors = artifact.manifest.get("textures", [])
        if any(
            texture.get("shape") not in {"2d", "bsdf_data"}
            for texture in texture_descriptors
        ):
            raise ValueError("MDL runtime accepts 2D and SDK BSDF-data textures only")
        ro_segments = artifact.manifest.get("ro_data", [])
        if len(ro_segments) > 1:
            raise ValueError("MDL reference supports at most one read-only data segment")
        sdk_types = compiler.descriptor.target_code_types
        if not sdk_types.is_file():
            raise FileNotFoundError("锁定的 MDL SDK 未获取；无法构建 MDL reference runtime")
        generated_source = "\n".join(
            (
                "#define MDL_NUM_TEXTURE_RESULTS 16",
                "#define MDL_DF_HANDLE_SLOT_MODE -1",
                "struct NclsMdlRendererState { float3 view_direction; };",
                "#define RENDERER_STATE_TYPE NclsMdlRendererState",
                f"#define NCLS_MDL_TEXTURE_COUNT {max(1, len(texture_descriptors))}",
                sdk_types.read_text(encoding="utf-8"),
                RUNTIME_SHADER.read_text(encoding="utf-8"),
                artifact.hlsl,
            )
        ).encode("utf-8")
        argument_size = max(16, ((len(artifact.argument_block) + 15) // 16) * 16)
        argument_block = artifact.argument_block + bytes(
            argument_size - len(artifact.argument_block)
        )
        blobs = {
            "generated-module": generated_source,
            "argument-block": argument_block,
        }
        descriptors = {
            "generated-module": {
                "kind": "slang-module-source",
                "module_name": "NclsMdlGenerated",
                "dtype": "utf8-source",
                "shape": [len(generated_source)],
                "stride": 1,
                "alignment": 1,
                "usage": "NclsMdlGenerated",
            },
            "argument-block": {
                "kind": "structured-buffer",
                "dtype": "float32",
                "shape": [argument_size // 4],
                "stride": 16,
                "alignment": 16,
                "usage": "gMdlArgumentBlock",
            },
        }
        ro_payload = (
            (artifact.root / str(ro_segments[0]["path"])).read_bytes()
            if ro_segments
            else b""
        )
        ro_size = max(16, ((len(ro_payload) + 15) // 16) * 16)
        blobs["ro-data"] = ro_payload + bytes(ro_size - len(ro_payload))
        descriptors["ro-data"] = {
            "kind": "structured-buffer",
            "dtype": "float32",
            "shape": [ro_size // 4],
            "stride": 16,
            "alignment": 16,
            "usage": "gMdlRoData",
        }
        resources: dict[str, bytes] = {}
        resource_descriptors: dict[str, dict[str, object]] = {}
        for texture in texture_descriptors:
            index = int(texture["index"])
            if texture["shape"] == "2d":
                data = texture.get("data")
                if data is None:
                    raise ValueError("MDL runtime artifact has no decoded 2D texture payload")
                path = artifact.root / str(data)
                suffix, resource, resource_descriptor = _decoded_texture_binding(
                    path, dict(texture)
                )
                name = f"texture-{index}.{suffix}"
                resources[name] = resource
                resource_descriptors[name] = {
                    **resource_descriptor,
                    "usage": f"gMdlTexture2D{index - 1}",
                }
            else:
                path = artifact.root / str(texture["data"])
                name = f"texture-{index}.r32f"
                resources[name] = path.read_bytes()
                resource_descriptors[name] = {
                    "kind": "texture3d",
                    "dtype": "float32",
                    "shape": [int(texture["depth"]), int(texture["height"]), int(texture["width"])],
                    "stride": 4,
                    "alignment": 16,
                    "format": "r32-float",
                    "color_space": "linear",
                    "usage": f"gMdlTexture3D{index - 1}",
                }
        return MaterialPayload(
            snapshot.snapshot_id,
            blobs,
            descriptors,
            resources,
            resource_descriptors,
            {
                "mdl-textures": {
                    "kind": "sampler",
                    "usage": "gMdlTextureSampler",
                    "filter": "linear",
                    "address_mode": "wrap",
                }
            },
        )


REFERENCE_PROGRAM_DEFINITION = MdlReferenceProgram()
